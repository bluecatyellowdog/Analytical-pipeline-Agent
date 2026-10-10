"""Analyse a cleaned run folder: plan it if there is no plan yet, run the analysis, and
rebuild the PDF report.

    uv run python -m pipeline.analyze runs/<run folder>

After editing analysis_plan.yaml, run this again: your plan is checked by Gate 3
and run as you wrote it, and the report lists what you changed. Delete the file
to have the AI plan again.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pipeline.agents import analysis, planner
from pipeline.agents.llm import (
    LLMClient,
    LLMConfigError,
    connect_agents,
    describe_models,
    public_info,
)
from pipeline.layout import REPORT_HTML, is_run_folder, upgrade_layout
from pipeline.reports import build_reports
from pipeline.tools.context import AnalysisInputError


@dataclass
class AnalyzeSummary:
    run_dir: Path
    planning: planner.PlannerResult | None = None
    analysis: analysis.AnalysisResult | None = None
    stopped: str | None = None
    """Why the analysis could not run, when it could not."""
    report: Path | None = None
    pdf_error: str | None = None

    @property
    def cost_usd(self) -> float:
        steps = (self.planning, self.analysis)
        return sum(step.cost_usd for step in steps if step is not None)


def analyze_folder(
    run_dir: Path,
    client: LLMClient,
    provider: dict[str, str],
    clients: Mapping[str, LLMClient] | None = None,
) -> AnalyzeSummary:
    """Plan if needed, analyse, and rebuild the reports; a failed step is reported, not raised.

    `client` runs the planner and the analysis unless `clients` names their own.
    """

    def agent(name: str) -> tuple[LLMClient, dict[str, str]]:
        chosen = (clients or {}).get(name, client)
        return chosen, public_info(chosen, provider)

    upgrade_layout(run_dir)
    summary = AnalyzeSummary(run_dir=run_dir)
    report: Path | None = None
    try:
        if planner.plan_in_force(run_dir) is None:
            summary.planning = planner.run_on_folder(run_dir, *agent("planner"))
        summary.analysis = analysis.run_on_folder(run_dir, *agent("analysis"))
    except AnalysisInputError as error:
        summary.stopped = str(error)
    except planner.PlannerError as error:
        summary.stopped = f"{error} Every attempt is logged in {planner.PLANNER_LOG}."
    except analysis.AnalysisError as error:
        summary.stopped = f"{error} Every attempt is logged in {analysis.ANALYSIS_LOG}."
    try:
        report = build_reports(run_dir)
    except Exception as error:  # noqa: BLE001 - a failed PDF must not lose a finished analysis
        summary.pdf_error = f"{type(error).__name__}: {error}"
    summary.report = report
    return summary


def summarise(summary: AnalyzeSummary) -> str:
    lines = []
    if summary.planning is not None:
        lines.append(planner.render_text(summary.planning))
        lines.append("")
    if summary.analysis is not None:
        lines.append(analysis.render_text(summary.analysis))
        lines.append(f"LLM cost: ${summary.cost_usd:.2f}")
    if summary.stopped:
        lines.append(f"Stopped: {summary.stopped}")
    lines.append("")
    if summary.report is not None:
        lines.append(f"Start here: {summary.report.resolve()}")
        lines.append(f"In a browser: {summary.report.with_name(REPORT_HTML).resolve()}")
    else:
        lines.append(f"The report could not be written: {summary.pdf_error}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.analyze",
        description="Plan (if needed) and run the analysis of a cleaned run folder.",
    )
    parser.add_argument("run_dir", help="A cleaned run folder, e.g. runs/loans_...")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not is_run_folder(run_dir):
        print(f"{run_dir} is not a run folder (no logs/manifest.json).")
        return 1
    try:
        clients = connect_agents()
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1
    used = {name: clients[name] for name in ("planner", "analysis")}
    print(f"Analysing {run_dir} ({describe_models(used)}). This takes a minute or two...")
    summary = analyze_folder(run_dir, clients["analysis"], {}, clients=clients)
    print()
    print(summarise(summary))
    return 0 if summary.analysis is not None else 1


if __name__ == "__main__":
    sys.exit(main())
