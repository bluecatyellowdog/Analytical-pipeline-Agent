"""The whole workflow in one command: ingest, diagnose, clean.

    uv run python -m pipeline.run data/raw/Titanic-Dataset.csv --brief examples/titanic_brief.yaml

Runs the three steps back to back into one run folder and ends with a short
summary saying where the report is. It stops early only when a step cannot go
on: Gate 1 blocks the file (before any LLM call, so it costs nothing), the
analyst gives no usable diagnosis, or the coder no plan that runs. Each step is
the same code as its own command, so the run folder is the same either way.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline import clean
from pipeline.agents import analyst
from pipeline.agents.coder import CoderError
from pipeline.agents.decisions import DecisionsError
from pipeline.agents.llm import LLMClient, LLMConfigError, connect
from pipeline.contracts.context_brief import load_context_brief
from pipeline.ingest import INDEX_FILENAME, IngestionResult, default_out_dir, ingest, update_index
from pipeline.ingestion.raw import RawStore
from pipeline.report import REPORT_FILENAME


@dataclass
class RunSummary:
    run_dir: Path
    ingestion: IngestionResult
    stopped_at: str | None = None
    """'ingest', 'diagnosis' or 'cleaning' when a step could not go on; None otherwise."""
    reason: str | None = None
    diagnosis: analyst.AnalystResult | None = None
    cleaning: clean.CleaningOutcome | None = None

    @property
    def cost_usd(self) -> float:
        diagnosis = self.diagnosis.cost_usd if self.diagnosis else 0.0
        cleaning = self.cleaning.cost_usd if self.cleaning else 0.0
        return diagnosis + cleaning


def run_all(
    source: str | Path,
    brief_path: str | Path | None,
    client: LLMClient,
    *,
    provider: dict[str, str] | None = None,
    review_client: LLMClient | None = None,
    review_provider: dict[str, str] | None = None,
    reviewing: bool = True,
    budget_usd: float = clean.DEFAULT_BUDGET_USD,
    out: str | Path | None = None,
    sheet: str | None = None,
    seed: int = 42,
) -> RunSummary:
    """Ingest, diagnose and clean one file into one run folder."""
    provider = provider or {}
    brief = load_context_brief(brief_path) if brief_path else None
    with RawStore.open() as store:
        ingestion = ingest(source, brief, store=store, sheet=sheet, seed=seed)
        run_dir = ingestion.write(out or default_out_dir(source))
    if out is None:
        update_index(ingestion, run_dir, source)

    summary = RunSummary(run_dir=run_dir, ingestion=ingestion)
    if ingestion.blocked:
        summary.stopped_at = "ingest"
        summary.reason = f"Gate 1 blocked the file; {REPORT_FILENAME} says why."
        return summary

    try:
        summary.diagnosis = analyst.run_on_folder(run_dir, client, provider)
    except analyst.AnalystError as error:
        summary.stopped_at = "diagnosis"
        summary.reason = f"{error} Every attempt is logged in {analyst.ANALYST_LOG}."
        return summary

    try:
        summary.cleaning = clean.clean_folder(
            run_dir,
            client,
            provider,
            review_client=review_client,
            review_provider=review_provider,
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

    if out is None:
        _link_in_index(run_dir, "cleaning", clean.CLEANING_REPORT)
    return summary


def summarise(summary: RunSummary) -> str:
    """What happened at each step, what it cost, and which file to open."""
    folder = summary.run_dir
    load = summary.ingestion.load_result
    lines = [
        f"Run folder: {folder.resolve()}",
        "",
        f"1. Ingest    {load.n_loaded:,} rows loaded, {load.n_rejected:,} rejected. "
        f"Gate 1: {summary.ingestion.gate.outcome.upper()}",
    ]
    if summary.stopped_at == "ingest":
        return "\n".join(
            [
                *lines,
                "",
                f"Stopped: {summary.reason}",
                f"Report: {(folder / REPORT_FILENAME).resolve()}",
            ]
        )

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
    lines += [
        f"3. Clean     Gate 2: {result.gate.outcome.upper()}. {review} "
        f"{len(result.plan.steps)} step(s). ${outcome.cost_usd:.2f}",
        f"   Total LLM cost: ${summary.cost_usd:.2f}",
    ]
    lines += [f"   Note: {note}" for note in outcome.notes]
    lines += ["", f"Report:     {(folder / clean.CLEANING_REPORT).resolve()}"]
    if result.blocked:
        lines.append("No clean data was written; the report says why.")
    else:
        lines.append(f"Clean data: {(folder / clean.CLEAN_CSV).resolve()}")
    return "\n".join(lines)


def _link_in_index(run_dir: Path, label: str, filename: str) -> None:
    """Add a link to this run's row in runs/index.md, so the newest report is one click away."""
    index = run_dir.parent / INDEX_FILENAME
    if not index.exists():
        return
    marker = f"[{run_dir.name}]("
    lines = index.read_text(encoding="utf-8").splitlines(keepends=True)
    for number, line in enumerate(lines):
        if marker in line and line.rstrip().endswith("|"):
            body = line.rstrip()[:-1].rstrip()
            lines[number] = f"{body} · [{label}]({run_dir.name}/{filename}) |\n"
            break
    index.write_text("".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.run",
        description="Ingest, diagnose and clean one file in one go.",
    )
    parser.add_argument("source", help="CSV, Excel, Parquet or JSON file")
    parser.add_argument("--brief", help="Context brief YAML (see examples/)")
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
        client = connect()
        review_model = os.environ.get("LLM_REVIEW_MODEL", "").strip() or None
        review_client = connect(model=review_model) if review_model else client
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1

    print(f"Running {args.source} with {client.config.model}. This takes a few minutes...")
    try:
        summary = run_all(
            args.source,
            args.brief,
            client,
            provider=client.config.public(),
            review_client=review_client,
            review_provider=review_client.config.public(),
            reviewing=not args.no_review,
            budget_usd=args.budget,
            out=args.out,
            sheet=args.sheet,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot run {args.source}: {error}")
        return 1

    print()
    print(summarise(summary))
    finished = summary.stopped_at is None
    return 0 if finished and summary.cleaning and not summary.cleaning.result.blocked else 1


if __name__ == "__main__":
    sys.exit(main())
