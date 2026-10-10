"""Where everything in a run folder lives, in one place so every step agrees.

    runs/<name>/
    ├── REPORT.pdf    start here: the answer in plain words, in two pages
    ├── REPORT.html   the same in a browser, with more to unfold under each part
    ├── details/      one document per section, the Excel column overview, the charts
    ├── edit/         the two files you change to change the result
    ├── data/         the clean data, and the rows that were dropped or could not be read
    └── logs/         the machine-readable audit trail

Code names a file ("diagnosis.json") and asks `run_file` where it lives; a file
not listed below belongs in logs/. A folder made before this layout, with
everything side by side, is moved into it the first time a step opens it
(`upgrade_layout`), or for all runs at once:

    uv run python -m pipeline.layout runs
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPORT_PDF = "REPORT.pdf"
REPORT_HTML = "REPORT.html"
DETAILS, EDIT, DATA, LOGS = "details", "edit", "data", "logs"
CHARTS = f"{DETAILS}/charts"
MANIFEST = "manifest.json"

TECHNICAL_PDF = "0_technical_summary.pdf"
ANALYSIS_PDF = "1_analysis.pdf"
DATA_QUALITY_PDF = "2_data_quality.pdf"
DATA_COLUMNS_XLSX = "2_data_columns.xlsx"
DECISIONS_PDF = "3_decisions.pdf"
AUDIT_PDF = "4_audit.pdf"

PLACES: dict[str, str] = {
    REPORT_PDF: "",
    REPORT_HTML: "",
    TECHNICAL_PDF: DETAILS,
    ANALYSIS_PDF: DETAILS,
    DATA_QUALITY_PDF: DETAILS,
    DATA_COLUMNS_XLSX: DETAILS,
    DECISIONS_PDF: DETAILS,
    AUDIT_PDF: DETAILS,
    "decisions.yaml": EDIT,
    "analysis_plan.yaml": EDIT,
    "clean_data.csv": DATA,
    "clean_data.parquet": DATA,
    "dropped_rows.csv": DATA,
    "rejected_lines.csv": DATA,
}
"""Where each file a person may open lives. Everything else is in logs/."""

RENAMED = {"ingestion_report.xlsx": DATA_COLUMNS_XLSX}
"""Old names, from before this layout, and what they are called now."""
REPLACED = ("run_report.pdf",)
"""Old outputs the new reports replace; removed when a folder is upgraded."""


def run_file(run_dir: Path, name: str) -> Path:
    """Where `name` lives in a run folder."""
    return Path(run_dir) / PLACES.get(name, LOGS) / name


def ensure_layout(run_dir: Path) -> Path:
    """Create the run folder and its sub-folders; return the folder."""
    run_dir = Path(run_dir)
    for folder in (DETAILS, EDIT, DATA, LOGS):
        (run_dir / folder).mkdir(parents=True, exist_ok=True)
    return run_dir


def is_run_folder(run_dir: Path) -> bool:
    run_dir = Path(run_dir)
    return run_file(run_dir, MANIFEST).exists() or (run_dir / MANIFEST).exists()


def upgrade_layout(run_dir: Path) -> list[str]:
    """Move a folder made before this layout into it. Returns what moved, as 'old -> new'.

    Only files move; their content never changes. A folder already in the
    layout is left alone.
    """
    run_dir = Path(run_dir)
    if not (run_dir / MANIFEST).exists():
        return []
    ensure_layout(run_dir)
    moved = []
    for path in sorted(p for p in run_dir.iterdir() if p.is_file()):
        if path.name in REPLACED:
            path.unlink(missing_ok=True)
            moved.append(f"{path.name} -> removed (replaced by {REPORT_PDF})")
            continue
        target = run_file(run_dir, RENAMED.get(path.name, path.name))
        path.replace(target)
        moved.append(f"{path.name} -> {target.relative_to(run_dir).as_posix()}")
    old_charts = run_dir / "analysis" / "charts"
    if old_charts.is_dir():
        (run_dir / CHARTS).mkdir(parents=True, exist_ok=True)
        for chart in old_charts.iterdir():
            chart.replace(run_dir / CHARTS / chart.name)
        shutil.rmtree(run_dir / "analysis", ignore_errors=True)
        moved.append(f"analysis/charts -> {CHARTS}")
    return moved


def main(argv: list[str] | None = None) -> int:
    """Upgrade every run folder below the given folders (default: runs)."""
    roots = [Path(a) for a in (sys.argv[1:] if argv is None else argv)] or [Path("runs")]
    upgraded = 0
    for root in roots:
        for manifest in sorted(root.rglob(MANIFEST)):
            if manifest.parent.name == LOGS:
                continue
            if upgrade_layout(manifest.parent):
                upgraded += 1
                print(f"Upgraded {manifest.parent}")
    print(f"{upgraded} run folder(s) moved into the new layout.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
