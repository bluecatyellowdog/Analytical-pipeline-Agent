"""The whole workflow in one command: ingest, diagnose, clean, plan and run the analysis.

    uv run python -m pipeline.run briefs/telco.yaml

The brief names its data file. A data file with `--brief <brief>` works too.

Runs the steps back to back into one run folder and ends with a short summary
saying where the report is. It stops early only when a step cannot go on: Gate 1
blocks the file (before any LLM call, so it costs nothing), the diagnosis gives no
usable diagnosis, the coder no plan that runs, no analysis plan passes Gate 3,
or no write-up passes Gate 4. When Gate 2 keeps the clean data back, there is
nothing to analyse. Each step is the same code as its own command, so the run
folder is the same either way.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline import clean
from pipeline.agents import analysis, diagnostic, planner
from pipeline.agents.coder import CoderError
from pipeline.agents.decisions import DecisionsError
from pipeline.agents.llm import (
    LLMClient,
    LLMConfigError,
    connect_agents,
    describe_models,
    public_info,
)
from pipeline.contracts.context_brief import load_context_brief
from pipeline.ingest import IngestionResult, default_out_dir, ingest, update_index
from pipeline.ingestion.raw import RawStore
from pipeline.layout import REPORT_PDF, run_file
from pipeline.reports import build_reports
from pipeline.tools.context import AnalysisInputError


@dataclass
class RunSummary:
    run_dir: Path
    ingestion: IngestionResult
    stopped_at: str | None = None
    """'ingest', 'diagnosis', 'cleaning', 'planning' or 'analysis' when a step could not go on."""
    reason: str | None = None
    diagnosis: diagnostic.DiagnosisResult | None = None
    cleaning: clean.CleaningOutcome | None = None
    planning: planner.PlannerResult | None = None
    analysis: analysis.AnalysisResult | None = None
    pdf_error: str | None = None

    @property
    def cost_usd(self) -> float:
        steps = (self.diagnosis, self.cleaning, self.planning, self.analysis)
        return sum(step.cost_usd for step in steps if step is not None)


def run_all(
    source: str | Path,
    brief_path: str | Path | None,
    client: LLMClient,
    **options: Any,
) -> RunSummary:
    """Every step on one file, into one run folder, then the reports.

    `client` runs every agent that `clients` (agent name -> client) does not name.
    """
    summary = _run_steps(source, brief_path, client, **options)
    try:
        build_reports(summary.run_dir)
    except Exception as error:  # noqa: BLE001 - a failed PDF must not lose a finished run
        summary.pdf_error = f"{type(error).__name__}: {error}"
    return summary


def _run_steps(
    source: str | Path,
    brief_path: str | Path | None,
    client: LLMClient,
    *,
    clients: Mapping[str, LLMClient] | None = None,
    provider: dict[str, str] | None = None,
    reviewing: bool = True,
    planning: bool = True,
    budget_usd: float = clean.DEFAULT_BUDGET_USD,
    out: str | Path | None = None,
    sheet: str | None = None,
    seed: int = 42,
) -> RunSummary:
    def agent(name: str) -> tuple[LLMClient, dict[str, str]]:
        chosen = (clients or {}).get(name, client)
        return chosen, public_info(chosen, provider)

    brief = load_context_brief(brief_path) if brief_path else None
    with RawStore.open() as store:
        ingestion = ingest(source, brief, store=store, sheet=sheet, seed=seed)
        run_dir = ingestion.write(out or default_out_dir(source))
    if out is None:
        update_index(ingestion, run_dir, source)

    summary = RunSummary(run_dir=run_dir, ingestion=ingestion)
    if ingestion.blocked:
        summary.stopped_at = "ingest"
        summary.reason = f"Gate 1 blocked the file; {REPORT_PDF} says why."
        return summary

    try:
        summary.diagnosis = diagnostic.run_on_folder(run_dir, *agent("diagnostic"))
    except diagnostic.DiagnosisError as error:
        summary.stopped_at = "diagnosis"
        summary.reason = f"{error} Every attempt is logged in {diagnostic.DIAGNOSTIC_LOG}."
        return summary

    try:
        coder, coder_provider = agent("coder")
        reviewer, reviewer_provider = agent("reviewer")
        summary.cleaning = clean.clean_folder(
            run_dir,
            coder,
            coder_provider,
            review_client=reviewer,
            review_provider=reviewer_provider,
            reviewing=reviewing,
            budget_usd=budget_usd,
        )
    except CoderError as error:
        summary.stopped_at = "cleaning"
        summary.reason = f"{error} Every attempt is logged in {clean.CODER_LOG}."
        return summary
    except (clean.RunFolderError, DecisionsError) as error:
        summary.stopped_at = "cleaning"
        summary.reason = str(error)
        return summary

    if summary.cleaning.result.blocked or not planning:
        return summary

    try:
        summary.planning = planner.run_on_folder(run_dir, *agent("planner"))
    except planner.PlannerError as error:
        summary.stopped_at = "planning"
        summary.reason = f"{error} Every attempt is logged in {planner.PLANNER_LOG}."
        return summary
    except AnalysisInputError as error:
        summary.stopped_at = "planning"
        summary.reason = str(error)
        return summary

    try:
        summary.analysis = analysis.run_on_folder(run_dir, *agent("analysis"))
    except analysis.AnalysisError as error:
        summary.stopped_at = "analysis"
        summary.reason = f"{error} Every attempt is logged in {analysis.ANALYSIS_LOG}."
    except AnalysisInputError as error:
        summary.stopped_at = "analysis"
        summary.reason = str(error)
    return summary


def summarise(summary: RunSummary) -> str:
    """What happened at each step, what it cost, and which file to open."""
    folder = summary.run_dir
    load = summary.ingestion.load_result
    report = run_file(folder, REPORT_PDF)
    lines = [
        f"Start here: {report.resolve()}"
        if report.exists()
        else f"The report could not be written: {summary.pdf_error}",
        f"Run folder: {folder.resolve()}",
        "",
        f"1. Ingest    {load.n_loaded:,} rows loaded, {load.n_rejected:,} rejected. "
        f"Gate 1: {summary.ingestion.gate.outcome.upper()}",
    ]
    if summary.stopped_at == "ingest":
        return "\n".join([*lines, "", f"Stopped: {summary.reason}"])

    if summary.diagnosis is not None:
        diagnosis = summary.diagnosis.diagnosis
        calls = diagnosis.judgment_calls
        decided = [c for c in calls if c.ai_decision]
        low = [c for c in decided if c.ai_decision.confidence == "low"]
        calls_text = (
            f"{len(decided)} judgment call(s) decided by the AI ({len(low)} low confidence)"
            if decided
            else f"{len(calls)} judgment call(s) left for you in decisions.yaml"
        )
        lines.append(
            f"2. Diagnose  {len(diagnosis.issues)} issue(s), {calls_text}. "
            f"${summary.diagnosis.cost_usd:.2f}"
        )
    if summary.stopped_at == "diagnosis":
        return "\n".join([*lines, "", f"Stopped: {summary.reason}"])

    outcome = summary.cleaning
    if outcome is None:
        return "\n".join([*lines, "", f"Stopped: {summary.reason}"])
    result = outcome.result
    review = (
        f"Review: {outcome.status.upper()} after {len(outcome.reviews)} round(s)."
        if outcome.status not in {"not_reviewed", "blocked"}
        else "Review: not run."
    )
    lines.append(
        f"3. Clean     Gate 2: {result.gate.outcome.upper()}. {review} "
        f"{len(result.plan.steps)} step(s). ${outcome.cost_usd:.2f}"
    )
    lines += [f"   Note: {note}" for note in outcome.notes]
    if summary.planning is not None:
        plan = summary.planning.plan
        closed = sum(not q.feasible for q in plan.questions)
        lines.append(
            f"4. Plan      {len(plan.questions)} question(s), {closed} marked not possible. "
            f"Gate 3: {summary.planning.gate.outcome.upper()}, approved automatically. "
            f"${summary.planning.cost_usd:.2f}"
        )
    if summary.analysis is not None:
        outcome_ = summary.analysis.outcome
        answered = sum(a.status in ("answered", "partly") for a in outcome_.write_up.answers)
        lines.append(
            f"5. Analyse   {len(outcome_.calls)} tool call(s), {len(outcome_.write_up.findings)} "
            f"finding(s), {answered} question(s) answered. "
            f"Gate 4: {summary.analysis.gate.outcome.upper()}. ${summary.analysis.cost_usd:.2f}"
        )
    lines.append(f"   Total LLM cost: ${summary.cost_usd:.2f}")
    if summary.stopped_at in ("planning", "analysis"):
        lines += ["", f"Stopped: {summary.reason}"]

    lines.append("")
    if result.blocked:
        lines.append("No clean data was written; the report says why.")
    else:
        lines.append(f"Clean data: {run_file(folder, clean.CLEAN_CSV).resolve()}")
    lines.append(f"To change the cleaning, edit {run_file(folder, 'decisions.yaml').resolve()}")
    if summary.planning is not None:
        lines.append(
            f"To change the analysis, edit {run_file(folder, planner.PLAN_YAML).resolve()}"
        )
    return "\n".join(lines)


class RunInputError(ValueError):
    """What to run on is unclear, with what to do about it."""


BRIEF_SUFFIXES = (".yaml", ".yml")


def resolve_inputs(source: str | Path, brief: str | Path | None) -> tuple[Path, Path | None]:
    """The data file and the brief. Given only a brief, the data file is the one it names."""
    source = Path(source)
    if brief is None and source.suffix.lower() in BRIEF_SUFFIXES:
        data_file = load_context_brief(source).data_file
        if not data_file:
            raise RunInputError(
                f"{source} names no data file. Add a line such as "
                f"'data_file: data/raw/<your file>.csv' to it, or run with the data file first: "
                f"python -m pipeline.run <data file> --brief {source.as_posix()}"
            )
        data = Path(data_file)
        if not data.exists():
            raise RunInputError(
                f"the data file {data_file} named in {source.as_posix()} was not found "
                f"(paths are relative to the project folder)"
            )
        return data, source
    return source, Path(brief) if brief is not None else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.run",
        description="Ingest, diagnose, clean and analyse one file in one go. "
        "Usually: python -m pipeline.run briefs/<your brief>.yaml",
    )
    parser.add_argument(
        "source",
        help="A brief (briefs/<name>.yaml) that names its data file, or a data file "
        "(CSV, Excel, Parquet or JSON)",
    )
    parser.add_argument("--brief", help="The brief, when the first argument is a data file")
    parser.add_argument("--sheet", help="Excel sheet name (default: the first one)")
    parser.add_argument("--out", help="Run folder (default: runs/<file>_<timestamp>)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for profiling samples")
    parser.add_argument("--no-review", action="store_true", help="Skip the reviewer")
    parser.add_argument(
        "--budget",
        type=float,
        default=clean.DEFAULT_BUDGET_USD,
        help="Stop between review rounds once cleaning has spent this many dollars",
    )
    args = parser.parse_args(argv)

    try:
        clients = connect_agents()
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1

    try:
        source, brief = resolve_inputs(args.source, args.brief)
    except (RunInputError, FileNotFoundError, ValueError) as error:
        print(f"Cannot run: {error}")
        return 1
    sheet = args.sheet or (load_context_brief(brief).sheet if brief else None)

    print(f"Running {source}. This takes a few minutes...")
    print(f"Models: {describe_models(clients)}")
    try:
        summary = run_all(
            source,
            brief,
            clients["diagnostic"],
            clients=clients,
            reviewing=not args.no_review,
            budget_usd=args.budget,
            out=args.out,
            sheet=sheet,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot run {source}: {error}")
        return 1

    print()
    print(summarise(summary))
    finished = summary.stopped_at is None
    return 0 if finished and summary.cleaning and not summary.cleaning.result.blocked else 1


if __name__ == "__main__":
    sys.exit(main())
