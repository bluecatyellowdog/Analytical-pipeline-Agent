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
from dataclasses import dataclass
from pathlib import Path

from pipeline.agents import analysis, planner
from pipeline.agents.llm import LLMClient, LLMConfigError, connect
from pipeline.report_pdf import PDF_FILENAME, build_pdf
from pipeline.tools.context import AnalysisInputError


@dataclass
class AnalyzeSummary:
    run_dir: Path
    planning: planner.PlannerResult | None = None
    analysis: analysis.AnalysisResult | None = None
    stopped: str | None = None
    """Why the analysis could not run, when it could not."""
    pdf_error: str | None = None

    @property
    def cost_usd(self) -> float:
        steps = (self.planning, self.analysis)
        return sum(step.cost_usd for step in steps if step is not None)


def analyze_folder(run_dir: Path, client: LLMClient, provider: dict[str, str]) -> AnalyzeSummary:
    """Plan if needed, analyse, and rebuild the PDF; a failed step is reported, not raised."""
    summary = AnalyzeSummary(run_dir=run_dir)
    try:
        if planner.plan_in_force(run_dir) is None:
            summary.planning = planner.run_on_folder(run_dir, client, provider)
        summary.analysis = analysis.run_on_folder(run_dir, client, provider)
    except AnalysisInputError as error:
        summary.stopped = str(error)
    except planner.PlannerError as error:
        summary.stopped = f"{error} Every attempt is logged in {planner.PLANNER_LOG}."
    except analysis.AnalysisError as error:
        summary.stopped = f"{error} Every attempt is logged in {analysis.ANALYSIS_LOG}."
    try:
        build_pdf(run_dir)
    except Exception as error:  # noqa: BLE001 - a failed PDF must not lose a finished analysis
        summary.pdf_error = f"{type(error).__name__}: {error}"
    return summary


def summarise(summary: AnalyzeSummary) -> str:
    folder = summary.run_dir
    lines = []
    if summary.planning is not None:
        lines.append(planner.render_text(summary.planning))
        lines.append("")
    if summary.analysis is not None:
        lines.append(analysis.render_text(summary.analysis))
        lines.append(f"LLM cost: ${summary.cost_usd:.2f}")
    if summary.stopped:
        lines.append(f"Stopped: {summary.stopped}")
    pdf = folder / PDF_FILENAME
    lines += [
        "",
        f"PDF report: {pdf.resolve()}"
        if not summary.pdf_error
        else f"PDF failed: {summary.pdf_error}",
    ]
    if summary.analysis is not None:
        lines.append(f"Findings:   {(folder / analysis.ANALYSIS_REPORT).resolve()}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.analyze",
        description="Plan (if needed) and run the analysis of a cleaned run folder.",
    )
    parser.add_argument("run_dir", help="A cleaned run folder, e.g. runs/loans_...")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not (run_dir / "manifest.json").exists():
        print(f"{run_dir} is not a run folder (no manifest.json).")
        return 1
    try:
        client = connect()
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1
    print(f"Analysing {run_dir} with {client.config.model}. This takes a minute or two...")
    summary = analyze_folder(run_dir, client, client.config.public())
    print()
    print(summarise(summary))
    return 0 if summary.analysis is not None else 1


if __name__ == "__main__":
    sys.exit(main())
