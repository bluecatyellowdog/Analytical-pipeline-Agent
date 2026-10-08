"""The reports of a run folder: REPORT.pdf with the essentials, and one detail document per section.

    REPORT.pdf                  start here: the answer, data quality, decisions, a wrap-up
    details/1_analysis.pdf      every finding with its charts and tool results; the plan
    details/2_data_quality.pdf  the data as received, what was wrong, every change, the checks
    details/3_decisions.pdf     every judgment call: options, decision, why, what it changed
    details/4_audit.pdf         models, costs, prompt versions and the files in the folder

Everything is rebuilt from the run folder, so the reports can be regenerated at any time:

    uv run python -m pipeline.reports runs/<run folder>
"""

from __future__ import annotations

import sys
from pathlib import Path

from pipeline.layout import (
    ANALYSIS_PDF,
    AUDIT_PDF,
    DATA_QUALITY_PDF,
    DECISIONS_PDF,
    REPORT_PDF,
    ensure_layout,
    is_run_folder,
    run_file,
    upgrade_layout,
)
from pipeline.reports.common import Run, write_pdf
from pipeline.reports.details import (
    analysis_document,
    audit_document,
    data_quality_document,
    decisions_document,
)
from pipeline.reports.main import main_document, report_title


def build_reports(run_dir: str | Path) -> Path:
    """Write REPORT.pdf and the detail documents; return the path of REPORT.pdf."""
    run_dir = Path(run_dir)
    upgrade_layout(run_dir)
    ensure_layout(run_dir)
    run = Run(run_dir)
    name = run.file_name
    write_pdf(
        run_file(run_dir, ANALYSIS_PDF), f"What the analysis found - {name}", analysis_document(run)
    )
    write_pdf(
        run_file(run_dir, DATA_QUALITY_PDF),
        f"How good the data is - {name}",
        data_quality_document(run),
    )
    write_pdf(
        run_file(run_dir, DECISIONS_PDF),
        f"Decisions made for you - {name}",
        decisions_document(run),
    )
    report = write_pdf(run_file(run_dir, REPORT_PDF), report_title(run), main_document(run))
    write_pdf(run_file(run_dir, AUDIT_PDF), f"Audit - {name}", audit_document(Run(run_dir)))
    return report


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not is_run_folder(Path(args[0])):
        print("usage: python -m pipeline.reports runs/<run folder>")
        return 1
    print(f"Wrote {build_reports(args[0]).resolve()}")
    return 0


__all__ = ["build_reports", "main"]
