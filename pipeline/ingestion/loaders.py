"""Load a file into the raw layer without changing anything in it.

Parsing is separated from storage on purpose: `parse_*` are pure functions over
bytes and can be tested against the fixture set directly, while `load` is the
thin layer that writes the result and returns the `LoadResult` Gate 1 checks.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pipeline.contracts.common import Flag
from pipeline.contracts.ingestion import ColumnRename, FileInfo, LoadResult, RejectedLine
from pipeline.ingestion.detect import detect
from pipeline.ingestion.raw import RESERVED_COLUMNS, RawStore, RawTable

MAX_REJECT_TEXT = 500
"""Rejected lines are stored truncated: enough to diagnose, not a data copy."""

MAX_STORED_REJECTS = 1_000
"""Beyond this the count still grows, but only the first lines are kept."""

_UNSAFE_TABLE_CHARS = re.compile(r"[^0-9a-zA-Z_]+")


def table_name_for(path: Path) -> str:
    """A stable, SQL-safe table name derived from the file name."""
    stem = _UNSAFE_TABLE_CHARS.sub("_", path.stem).strip("_").lower()
    return f"t_{stem}" if not stem or stem[0].isdigit() else stem


def normalize_headers(raw_headers: Sequence[str | None]) -> tuple[list[str], list[ColumnRename]]:
    """Make header names usable, and report every change.

    Blank headers become `column_<position>` and repeats get a numeric suffix.
    The rules are positional and deterministic so the same file always produces
    the same column names — otherwise nothing downstream could be compared
    across runs.
    """
    names: list[str] = []
    renames: list[ColumnRename] = []
    occurrences: dict[str, int] = {}
    used: set[str] = set()

    for position, header in enumerate(raw_headers, start=1):
        original = header if header is not None else ""
        name = original.strip().lstrip("﻿")
        reason: str | None = None

        if not name:
            name = f"column_{position}"
            reason = "Blank header replaced by its column position."
        elif name in RESERVED_COLUMNS:
            name = f"{name}_source"
            reason = f"{original!r} is a reserved pipeline column name."

        base = name
        count = occurrences.get(base, 0) + 1
        occurrences[base] = count
        if count > 1 or name in used:
            while f"{base}_{count}" in used:
                count += 1
            occurrences[base] = count
            name = f"{base}_{count}"
            reason = f"Duplicate header {base!r} renamed."

        used.add(name)
        names.append(name)
        if reason is not None:
            renames.append(
                ColumnRename(position=position, original=original, final=name, reason=reason)
            )

    return names, renames


@dataclass
class ParsedTable:
    """What parsing produced, before anything is stored."""

    columns: list[str]
    rows: list[list[Any]]
    source_lines: list[int]
    renames: list[ColumnRename] = field(default_factory=list)
    rejects: list[RejectedLine] = field(default_factory=list)
    n_rejected: int = 0
    typed: bool = False
    column_types: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)

    @property
    def n_source_rows(self) -> int:
        """Data rows in the source: what was loaded plus what was rejected."""
        return len(self.rows) + self.n_rejected

    def empty_columns(self) -> list[str]:
        """Columns where every value is missing — worth reporting, never dropped here."""
        return [
            name
            for index, name in enumerate(self.columns)
            if all(row[index] is None or str(row[index]).strip() == "" for row in self.rows)
        ]


def parse_csv(text: str, delimiter: str, *, quotechar: str = '"') -> ParsedTable:
    """Parse delimited text with real CSV quoting rules.

    Records, not lines: a quoted field may span several lines, and `_source_line`
    records where each record *starts* so a reject can be found in the file.
    """
    lines = text.splitlines(keepends=True)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter, quotechar=quotechar)

    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("file is empty: no header row") from None

    columns, renames = normalize_headers(header)
    width = len(columns)

    rows: list[list[str | None]] = []
    source_lines: list[int] = []
    rejects: list[RejectedLine] = []
    n_rejected = 0
    n_blank_lines = 0
    previous_line = reader.line_num

    while True:
        start_line = previous_line + 1
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as error:
            n_rejected += 1
            if len(rejects) < MAX_STORED_REJECTS:
                rejects.append(
                    RejectedLine(
                        source_line=start_line,
                        reason=f"CSV parse error: {error}",
                        raw="".join(lines[start_line - 1 : start_line])[:MAX_REJECT_TEXT],
                    )
                )
            previous_line = reader.line_num
            continue

        end_line = reader.line_num
        previous_line = end_line

        if not row:
            n_blank_lines += 1
            continue

        if len(row) != width:
            n_rejected += 1
            if len(rejects) < MAX_STORED_REJECTS:
                rejects.append(
                    RejectedLine(
                        source_line=start_line,
                        reason=f"{len(row)} fields, expected {width}",
                        raw="".join(lines[start_line - 1 : end_line])[:MAX_REJECT_TEXT],
                    )
                )
            continue

        rows.append(list(row))
        source_lines.append(start_line)

    notes = [f"Parsed {len(rows)} records with delimiter {delimiter!r}."]
    if n_blank_lines:
        notes.append(f"Skipped {n_blank_lines} blank line(s).")

    return ParsedTable(
        columns=columns,
        rows=rows,
        source_lines=source_lines,
        renames=renames,
        rejects=rejects,
        n_rejected=n_rejected,
        notes=notes,
    )


def _excel_value_to_text(value: Any) -> str | None:
    """Render one cell as the literal text it displays as.

    Excel has no text layer to preserve, so this is the one place where a
    rendering decision is unavoidable. It is kept narrow and documented: whole
    floats lose the '.0' Excel never showed, and datetimes become ISO.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, dt.datetime):
        if value.time() == dt.time(0, 0):
            return value.date().isoformat()
        return value.isoformat()
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    return str(value)


def parse_excel(path: Path, sheet: str) -> ParsedTable:
    """Read one sheet as text. A text-formatted '001' must stay '001'."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet]
        grid = worksheet.iter_rows(values_only=True)
        try:
            header = next(grid)
        except StopIteration:
            raise ValueError(f"sheet {sheet!r} is empty") from None

        columns, renames = normalize_headers([_excel_value_to_text(v) for v in header])
        width = len(columns)

        rows: list[list[str | None]] = []
        source_lines: list[int] = []
        rejects: list[RejectedLine] = []
        n_rejected = 0
        n_blank = 0

        for offset, raw_row in enumerate(grid, start=2):
            values = [_excel_value_to_text(v) for v in raw_row]
            if all(v is None or v == "" for v in values):
                n_blank += 1
                continue
            # Trailing empty cells are padding from the sheet's used range, not data.
            while len(values) > width and values[-1] is None:
                values.pop()
            if len(values) < width:
                values.extend([None] * (width - len(values)))
            if len(values) != width:
                n_rejected += 1
                if len(rejects) < MAX_STORED_REJECTS:
                    rejects.append(
                        RejectedLine(
                            source_line=offset,
                            reason=f"{len(values)} cells, expected {width}",
                            raw=str(values)[:MAX_REJECT_TEXT],
                        )
                    )
                continue
            rows.append(values)
            source_lines.append(offset)
    finally:
        workbook.close()

    notes = [f"Read sheet {sheet!r} as text: {len(rows)} rows."]
    if n_blank:
        notes.append(f"Skipped {n_blank} empty row(s).")

    return ParsedTable(
        columns=columns,
        rows=rows,
        source_lines=source_lines,
        renames=renames,
        rejects=rejects,
        n_rejected=n_rejected,
        notes=notes,
    )


def parse_json(text: str) -> ParsedTable:
    """Read records-oriented JSON or JSON Lines into text columns."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("file is empty")

    records: list[dict[str, Any]]
    if stripped.startswith("["):
        loaded = json.loads(stripped)
        records = loaded
        note = "Parsed as a JSON array of records."
    elif stripped.startswith("{") and "\n" not in stripped.strip().rstrip("}").rstrip():
        records = [json.loads(stripped)]
        note = "Parsed as a single JSON object."
    else:
        records = [json.loads(line) for line in stripped.splitlines() if line.strip()]
        note = "Parsed as JSON Lines."

    if not all(isinstance(record, dict) for record in records):
        raise ValueError("JSON must contain objects; nested arrays are not tabular")

    seen: dict[str, None] = {}
    for record in records:
        for key in record:
            seen.setdefault(key, None)

    columns, renames = normalize_headers(list(seen))
    original_keys = list(seen)

    rows: list[list[str | None]] = []
    for record in records:
        row: list[str | None] = []
        for key in original_keys:
            value = record.get(key)
            if value is None:
                row.append(None)
            elif isinstance(value, str):
                row.append(value)
            elif isinstance(value, bool):
                row.append("true" if value else "false")
            elif isinstance(value, int | float):
                row.append(repr(value) if isinstance(value, float) else str(value))
            else:
                row.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
        rows.append(row)

    return ParsedTable(
        columns=columns,
        rows=rows,
        source_lines=list(range(1, len(rows) + 1)),
        renames=renames,
        notes=[note, f"Parsed {len(rows)} records."],
    )


def parse_parquet(path: Path) -> ParsedTable:
    """Read Parquet with its own types intact.

    Spec 4.2: Parquet is the documented exception to text-first loading, because
    the file already carries the types someone chose deliberately. Rows are
    fetched in file order rather than assigned by a window function, so the row
    ids are deterministic.
    """
    import duckdb

    con = duckdb.connect()
    try:
        described = con.execute(
            "DESCRIBE SELECT * FROM read_parquet(?)", [path.as_posix()]
        ).fetchall()
        original = [row[0] for row in described]
        types = [row[1] for row in described]
        rows = [
            list(row)
            for row in con.execute("SELECT * FROM read_parquet(?)", [path.as_posix()]).fetchall()
        ]
    finally:
        con.close()

    columns, renames = normalize_headers(original)
    return ParsedTable(
        columns=columns,
        rows=rows,
        source_lines=list(range(1, len(rows) + 1)),
        renames=renames,
        typed=True,
        column_types=dict(zip(columns, types, strict=True)),
        notes=[f"Parquet types kept as declared: {dict(zip(columns, types, strict=True))}."],
    )


def parse(path: Path, info: FileInfo) -> ParsedTable:
    """Dispatch to the right parser for a detected file."""
    if info.format == "csv":
        text = path.read_bytes().decode(info.encoding or "utf-8")
        return parse_csv(text, info.delimiter or ",", quotechar=info.quotechar or '"')
    if info.format == "excel":
        assert info.sheet is not None  # detect() always resolves a sheet
        return parse_excel(path, info.sheet)
    if info.format == "json":
        return parse_json(path.read_bytes().decode(info.encoding or "utf-8"))
    if info.format == "parquet":
        return parse_parquet(path)
    raise ValueError(f"no parser for format {info.format!r}")


def load(
    path: str | Path,
    store: RawStore,
    *,
    table: str | None = None,
    sheet: str | None = None,
) -> tuple[LoadResult, RawTable]:
    """Detect, parse and store one file in the raw layer.

    Returns both the `LoadResult` that Gate 1 checks and the `RawTable` handle
    later stages read from.
    """
    path = Path(path)
    info = detect(path, sheet=sheet)
    parsed = parse(path, info)
    info.detection_notes.extend(parsed.notes)

    name = table or table_name_for(path)
    if parsed.typed:
        raw_table = store.create_typed_table(name, parsed.columns, parsed.column_types, parsed.rows)
    else:
        raw_table = store.create_text_table(name, parsed.columns, parsed.rows, parsed.source_lines)

    rejects_table = store.create_rejects_table(name, parsed.rejects)
    raw_table = RawTable(
        name=raw_table.name,
        columns=raw_table.columns,
        column_types=raw_table.column_types,
        typed=raw_table.typed,
        n_rows=raw_table.n_rows,
        data_hash=raw_table.data_hash,
        rejects_table=rejects_table,
    )

    flags = [*info.flags, *parsed.flags]
    empty_columns = parsed.empty_columns()
    if empty_columns:
        flags.append(
            Flag(
                code="empty_columns",
                severity="warn",
                message=f"{len(empty_columns)} column(s) contain no values: {empty_columns}.",
                stage="ingestion",
                details={"columns": empty_columns},
            )
        )
    if parsed.renames:
        flags.append(
            Flag(
                code="headers_renamed",
                severity="info",
                message=f"{len(parsed.renames)} header(s) were renamed to be usable.",
                stage="ingestion",
                details={"renames": [r.model_dump() for r in parsed.renames]},
            )
        )

    result = LoadResult(
        table=raw_table.name,
        file=info,
        n_source_rows=parsed.n_source_rows,
        n_loaded=len(parsed.rows),
        n_rejected=parsed.n_rejected,
        rejects=parsed.rejects,
        columns=parsed.columns,
        renames=parsed.renames,
        empty_columns=empty_columns,
        data_hash=raw_table.data_hash,
        flags=flags,
    )
    return result, raw_table
