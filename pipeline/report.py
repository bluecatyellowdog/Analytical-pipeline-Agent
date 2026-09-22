"""A readable account of what ingestion read and what it did not.

Everything here is derived from the load result, the profile and the gate, so
the numbers in the report are the numbers in the audit trail, never restated.
The report is a Markdown file because that renders in an editor, on GitHub and
in a terminal without extra tooling.

Personal values are masked in the preview, as everywhere else. The rejected
lines are the exception: they are shown verbatim because the point of listing
them is to find them in the source file. They are written only under `--out`,
which is a local directory, and the report says so.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from pipeline.contracts.audit import GateResult
from pipeline.contracts.context_brief import ClaimResult
from pipeline.contracts.ingestion import LoadResult, RejectedLine
from pipeline.contracts.profile import DatasetProfile
from pipeline.ingestion.compact import compact_column
from pipeline.ingestion.pii import mask_shape
from pipeline.ingestion.raw import RawStore, RawTable, quote_ident

MAX_REJECTS_IN_REPORT = 50
PREVIEW_ROWS = 5
MAX_CELL_CHARS = 80

REJECTS_FILENAME = "rejected_lines.csv"
REPORT_FILENAME = "ingestion_report.md"
REJECT_COLUMNS = ("source_line", "reason", "raw")


def write_rejected_csv(rejects: Sequence[RejectedLine], path: str | Path) -> Path:
    """Every rejected line, in full, in a file Excel opens correctly.

    The BOM (`utf-8-sig`) is deliberate: without it Excel reads Danish text as
    cp1252 and shows `Ã¦` for `æ`. A header-only file is written when nothing
    was rejected, so the output directory always has the same shape.
    """
    target = Path(path)
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REJECT_COLUMNS)
        for reject in rejects:
            writer.writerow([reject.source_line, reject.reason, reject.raw])
    return target


def render_ingestion_report(
    load_result: LoadResult,
    profile: DatasetProfile,
    gate: GateResult,
    claim_results: Sequence[ClaimResult],
    *,
    store: RawStore,
    table: RawTable,
) -> str:
    """The whole report as Markdown."""
    sections = [
        _title(load_result, gate),
        _read_section(load_result),
        _not_read_section(load_result),
        _columns_section(profile),
        _flags_section(gate),
        _gate_section(gate),
    ]
    if claim_results:
        sections.append(_claims_section(claim_results))
    sections.append(_preview_section(store, table, profile))
    return "\n\n".join(sections) + "\n"


def _title(load_result: LoadResult, gate: GateResult) -> str:
    name = Path(load_result.file.path).name
    verdict = {
        "pass": "Passed. The file was read completely and nothing needs attention.",
        "flag": "Passed with flags. The file was read, but some things deserve a look.",
        "block": "Blocked. Do not use this load until the problems below are resolved.",
    }[gate.outcome]
    return f"# Ingestion report: {name}\n\n**Gate 1: {gate.outcome.upper()}.** {verdict}"


def _read_section(load_result: LoadResult) -> str:
    total = load_result.n_source_rows
    loaded_share = _share(load_result.n_loaded, total)
    rejected_share = _share(load_result.n_rejected, total)
    file = load_result.file

    lines = [
        "## What was read",
        "",
        "| | Rows | Share |",
        "| --- | ---: | ---: |",
        f"| Data rows in the file | {total:,} | 100.00% |",
        f"| Loaded | {load_result.n_loaded:,} | {loaded_share} |",
        f"| Rejected | {load_result.n_rejected:,} | {rejected_share} |",
        "",
        f"Loaded + rejected = {load_result.n_loaded + load_result.n_rejected:,} "
        f"of {total:,} source rows.",
        "",
        f"- **Format:** {file.format}"
        + (f", encoding `{file.encoding}`" if file.encoding else "")
        + (f", delimiter `{file.delimiter}`" if file.delimiter else "")
        + (f", sheet `{file.sheet}`" if file.sheet else ""),
        f"- **Size:** {file.size_bytes:,} bytes, SHA-256 `{file.file_hash[:12]}`",
        f"- **Columns:** {len(load_result.columns)}",
    ]
    if file.detection_notes:
        lines += ["", "How the file was interpreted:", ""]
        lines += [f"- {note}" for note in file.detection_notes]
    return "\n".join(lines)


def _not_read_section(load_result: LoadResult) -> str:
    lines = ["## What was not read", ""]

    if not load_result.rejects:
        lines.append("No lines were rejected. Every data row in the file was loaded.")
    else:
        shown = load_result.rejects[:MAX_REJECTS_IN_REPORT]
        lines += [
            f"{load_result.n_rejected:,} line(s) could not be parsed. They are set aside, "
            f"not lost: each is listed here with its line number in the original file "
            f"and stored in full in `{REJECTS_FILENAME}`.",
            "",
            "| Line in file | Reason | Text |",
            "| ---: | --- | --- |",
        ]
        lines += [f"| {r.source_line} | {_cell(r.reason)} | {_cell(r.raw)} |" for r in shown]
        hidden = len(load_result.rejects) - len(shown)
        if hidden > 0:
            lines += ["", f"...and {hidden:,} more in `{REJECTS_FILENAME}`."]
        lines += [
            "",
            f"The text above is shown as it appears in the file, unmasked, and may contain "
            f"personal data. `{REJECTS_FILENAME}` is written only to the `--out` directory; "
            f"keep it out of version control and shared reports.",
        ]

    if load_result.renames:
        lines += [
            "",
            "**Headers that had to be renamed** (the data was kept, only the name changed):",
            "",
            "| Column | In the file | Now called | Why |",
            "| ---: | --- | --- | --- |",
        ]
        lines += [
            f"| {r.position} | {_cell(repr(r.original))} | {_cell(r.final)} | {_cell(r.reason)} |"
            for r in load_result.renames
        ]
    if load_result.empty_columns:
        lines += ["", "**Columns with no values at all:** " + ", ".join(load_result.empty_columns)]
    return "\n".join(lines)


def _columns_section(profile: DatasetProfile) -> str:
    """What each column looks like, and which values did not fit its type."""
    lines = [
        "## Column by column",
        "",
        "Every value is stored as text exactly as it appeared. *Read as* is the type the "
        "data best supports; nothing has been converted. *Fits* is the share of non-missing "
        "values that parse as that type, and the last column shows values that do not.",
        "",
        "| Column | Read as | Fits | Missing | Distinct | Did not fit |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for column in profile.columns:
        entry = compact_column(column)
        unfit = ", ".join(f"`{_cell(v, 30)}`" for v in entry.get("unparsed_examples", []))
        read_as = entry["type"] + (" (ambiguous)" if entry.get("ambiguous") else "")
        lines.append(
            f"| {_cell(column.name)} | {read_as} | {entry['coverage']:.0%} | "
            f"{entry['missing']} | {entry['distinct']:,} | {unfit} |"
        )
    return "\n".join(lines)


def _flags_section(gate: GateResult) -> str:
    lines = ["## Things worth checking", ""]
    if not gate.flags:
        lines.append("Nothing was flagged.")
        return "\n".join(lines)

    severity_order = {"error": 0, "warn": 1, "info": 2}
    ordered = sorted(gate.flags, key=lambda f: severity_order.get(f.severity, 3))
    lines += [
        f"- **{flag.severity}** `{flag.code}`"
        + (f" ({flag.column})" if flag.column else "")
        + f": {flag.message}"
        for flag in ordered
    ]
    return "\n".join(lines)


def _gate_section(gate: GateResult) -> str:
    lines = ["## Gate 1 checks", ""]
    for check in gate.checks:
        mark = "pass" if check.passed else f"**FAIL ({check.severity})**"
        lines.append(f"- {mark} `{check.name}`: {check.message}")
    return "\n".join(lines)


def _claims_section(claim_results: Sequence[ClaimResult]) -> str:
    label = {"pass": "holds", "fail": "**does not hold**", "not_checkable": "could not be checked"}
    lines = [
        "## What the brief claimed",
        "",
        "Each statement in the context brief, tested against the data.",
        "",
    ]
    lines += [
        f"- {label[r.status]}: {r.claim.text or r.claim.kind}. {r.message}" for r in claim_results
    ]
    return "\n".join(lines)


def _preview_section(store: RawStore, table: RawTable, profile: DatasetProfile) -> str:
    """The first rows as loaded, with personal and identifier values masked."""
    masked = {
        column.name
        for column in profile.columns
        if column.pii_level != "none" or column.type_proposal.proposed == "id"
    }
    columns = table.columns
    schema, _, name = table.name.rpartition(".")
    selection = ", ".join(["_source_line", *(quote_ident(c) for c in columns)])
    source = f"{schema}.{quote_ident(name)}"
    rows = store.connection.execute(
        f"SELECT {selection} FROM {source} ORDER BY _row_id LIMIT {PREVIEW_ROWS}"
    ).fetchall()

    lines = [f"## First {len(rows)} loaded row(s)", ""]
    if masked:
        lines += [
            "Values in personal-data and identifier columns are shown as their shape only "
            f"(masked: {', '.join(sorted(masked))}).",
            "",
        ]
    if not rows:
        return "\n".join([*lines, "No rows were loaded."])

    lines += [
        "| Line in file | " + " | ".join(_cell(c) for c in columns) + " |",
        "| ---: | " + " | ".join("---" for _ in columns) + " |",
    ]
    for source_line, *values in rows:
        cells = [
            _cell(mask_shape(v) if column in masked and v is not None else v)
            for column, v in zip(columns, values, strict=True)
        ]
        lines.append(f"| {source_line} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _share(part: int, total: int) -> str:
    return f"{part / total:.2%}" if total else "n/a"


def _cell(value: object, limit: int = MAX_CELL_CHARS) -> str:
    """Make a value safe inside a Markdown table cell without changing what it says."""
    if value is None:
        return ""
    text = str(value).replace("\r", "\\r").replace("\n", "\\n").replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = [
    "REJECTS_FILENAME",
    "REPORT_FILENAME",
    "render_ingestion_report",
    "write_rejected_csv",
]
