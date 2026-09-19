"""Work out what a file is before reading a single value from it.

Every choice made here — encoding, delimiter, sheet — is a guess that can
corrupt data silently if it is wrong, so each one is recorded in
`FileInfo.detection_notes` and anything uncertain is flagged.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
from collections import Counter
from pathlib import Path

from pipeline.contracts.common import Flag
from pipeline.contracts.ingestion import FileFormat, FileInfo

DELIMITER_CANDIDATES = (",", ";", "\t", "|")
"""Spec 4.1: the delimiters worth considering for a tabular export."""

SAMPLE_LINES = 50
"""How many records to read when scoring delimiters."""

EXTENSIONS: dict[str, FileFormat] = {
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "csv",
    ".xlsx": "excel",
    ".xlsm": "excel",
    ".xls": "excel",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".json": "json",
    ".jsonl": "json",
    ".ndjson": "json",
}

_READ_BYTES_FOR_DETECTION = 1 << 20  # 1 MiB is plenty to decide encoding and delimiter.


def sha256_file(path: Path) -> str:
    """Hash of the file bytes: the provenance anchor for the whole run."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_format(path: Path) -> FileFormat:
    """Format from the extension. Unknown extensions are an error, not a guess."""
    suffix = path.suffix.lower()
    try:
        return EXTENSIONS[suffix]
    except KeyError:
        raise ValueError(
            f"unsupported file type {suffix!r}; supported: {sorted(set(EXTENSIONS))}"
        ) from None


def detect_encoding(head: bytes) -> tuple[str, list[str], list[Flag]]:
    """Decide how to decode a text file.

    Order matters: a byte-order mark is definitive, strict UTF-8 either works or
    does not, and cp1252 is the fallback because it is what older Danish exports
    use and it decodes any byte sequence — which is exactly why it is flagged
    rather than trusted.
    """
    notes: list[str] = []
    flags: list[Flag] = []

    for bom, encoding in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF32_LE, "utf-32"),
        (codecs.BOM_UTF32_BE, "utf-32"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if head.startswith(bom):
            notes.append(f"Byte-order mark found: decoding as {encoding}.")
            return encoding, notes, flags

    try:
        head.decode("utf-8")
    except UnicodeDecodeError as error:
        notes.append(
            f"Not valid UTF-8 (byte {error.start}: {head[error.start : error.start + 1]!r}); "
            f"falling back to cp1252."
        )
        flags.append(
            Flag(
                code="encoding_guessed",
                severity="warn",
                message="File is not UTF-8; decoded as cp1252. Check non-ASCII values.",
                stage="ingestion",
                details={"encoding": "cp1252", "first_bad_byte": error.start},
            )
        )
        return "cp1252", notes, flags

    notes.append("Decoded as UTF-8 without errors.")
    return "utf-8", notes, flags


def _field_counts(text: str, delimiter: str, max_rows: int = SAMPLE_LINES) -> list[int]:
    """Field counts for the first records, parsed with real CSV quoting rules."""
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    counts: list[int] = []
    try:
        for row in reader:
            if len(counts) >= max_rows:
                break
            if row:  # blank lines say nothing about the delimiter
                counts.append(len(row))
    except csv.Error:
        return []
    return counts


def detect_delimiter(text: str) -> tuple[str, list[str], list[Flag]]:
    """Pick the delimiter that splits the sample most consistently.

    Consistency beats field count: a Danish file full of '1.234,50' looks like
    it has two comma-separated fields on most lines, but the counts disagree,
    while the real ';' delimiter gives the same count on every line.
    """
    notes: list[str] = []
    flags: list[Flag] = []
    scores: dict[str, tuple[float, int]] = {}

    for candidate in DELIMITER_CANDIDATES:
        counts = _field_counts(text, candidate)
        if not counts:
            continue
        modal_count, modal_freq = Counter(counts).most_common(1)[0]
        if modal_count < 2:  # a single field means this delimiter is not present
            continue
        scores[candidate] = (modal_freq / len(counts), modal_count)

    if not scores:
        notes.append("No delimiter split the sample into multiple fields; assuming ','.")
        flags.append(
            Flag(
                code="delimiter_not_detected",
                severity="warn",
                message="Could not detect a delimiter; assumed ','. The file may be single-column.",
                stage="ingestion",
            )
        )
        return ",", notes, flags

    best = max(scores, key=lambda d: scores[d])
    consistency, modal_count = scores[best]
    notes.append(
        f"Delimiter {best!r} chosen: {modal_count} fields on "
        f"{consistency:.0%} of the first {SAMPLE_LINES} records."
    )

    runners_up = {d: s for d, s in scores.items() if d != best and s[0] >= consistency}
    if runners_up:
        flags.append(
            Flag(
                code="delimiter_ambiguous",
                severity="warn",
                message=f"Delimiters {sorted([best, *runners_up])} score equally; chose {best!r}.",
                stage="ingestion",
                details={"scores": {d: list(s) for d, s in scores.items()}},
            )
        )
    return best, notes, flags


def detect_sheets(path: Path) -> list[str]:
    """Sheet names in workbook order, without loading any cell values."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True)
    try:
        return list(workbook.sheetnames)
    finally:
        workbook.close()


def detect(path: str | Path, *, sheet: str | None = None) -> FileInfo:
    """Everything knowable about a file before its rows are read."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    fmt = detect_format(path)
    info = FileInfo(
        path=path.as_posix(),
        format=fmt,
        size_bytes=path.stat().st_size,
        file_hash=sha256_file(path),
    )

    if fmt in ("csv", "json"):
        head = path.read_bytes()[:_READ_BYTES_FOR_DETECTION]
        encoding, notes, flags = detect_encoding(head)
        info.encoding = encoding
        info.detection_notes.extend(notes)
        info.flags.extend(flags)

        if fmt == "csv":
            text = head.decode(encoding, errors="replace")
            delimiter, notes, flags = detect_delimiter(text)
            info.delimiter = delimiter
            info.quotechar = '"'
            info.detection_notes.extend(notes)
            info.flags.extend(flags)

    elif fmt == "parquet":
        info.detection_notes.append(
            "Parquet: types are read from the file, not inferred (spec 4.2)."
        )

    elif fmt == "excel":
        info.available_sheets = detect_sheets(path)
        if not info.available_sheets:
            raise ValueError(f"{path} contains no sheets")
        if sheet is not None and sheet not in info.available_sheets:
            raise ValueError(f"sheet {sheet!r} not in {info.available_sheets}")
        info.sheet = sheet or info.available_sheets[0]
        info.detection_notes.append(
            f"Loading sheet {info.sheet!r} of {info.available_sheets}."
            if sheet is None
            else f"Loading requested sheet {info.sheet!r}."
        )
        if sheet is None and len(info.available_sheets) > 1:
            info.flags.append(
                Flag(
                    code="multiple_sheets",
                    severity="warn",
                    message=(
                        f"Workbook has {len(info.available_sheets)} sheets; "
                        f"loaded the first one ({info.sheet!r})."
                    ),
                    stage="ingestion",
                    details={"sheets": info.available_sheets},
                )
            )

    return info
