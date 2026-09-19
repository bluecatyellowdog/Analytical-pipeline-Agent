"""Stage 1 structural contracts: what a file was, and what loading it produced.

Kept separate from the profile because these describe the *file*, while the
profile describes the *data*.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from pipeline.contracts.common import Flag, Strict, utcnow

FileFormat = Literal["csv", "excel", "parquet", "json"]


class FileInfo(Strict):
    """Everything detection worked out about a file, before a row was read."""

    path: str
    format: FileFormat
    size_bytes: int
    file_hash: str = Field(description="SHA-256 of the file bytes, for provenance.")
    encoding: str | None = Field(default=None, description="e.g. 'utf-8', 'utf-8-sig', 'cp1252'.")
    delimiter: str | None = Field(default=None, description="CSV only.")
    quotechar: str | None = None
    sheet: str | None = Field(default=None, description="Excel only.")
    available_sheets: list[str] = Field(default_factory=list)
    detection_notes: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)


class RejectedLine(Strict):
    """A source line that could not be parsed. Rejected, never silently dropped."""

    source_line: int
    reason: str
    raw: str = Field(description="The literal line, truncated for storage.")


class LoadResult(Strict):
    """The outcome of loading one file into the read-only raw layer."""

    table: str = Field(description="Table name in the DuckDB `raw` schema.")
    file: FileInfo
    n_source_rows: int = Field(description="Data rows in the source, excluding the header.")
    n_loaded: int
    n_rejected: int
    rejects: list[RejectedLine] = Field(default_factory=list)
    columns: list[str] = Field(description="Final column names, after deduplication.")
    renamed_columns: dict[str, str] = Field(
        default_factory=dict,
        description="Deterministic renames for duplicate or blank headers: original -> final.",
    )
    empty_columns: list[str] = Field(default_factory=list)
    data_hash: str = Field(description="SHA-256 of the loaded values, independent of the file.")
    loaded_at: datetime = Field(default_factory=utcnow)
    flags: list[Flag] = Field(default_factory=list)

    @property
    def reject_rate(self) -> float:
        """Share of source rows that could not be parsed. Gate 1 blocks above 1%."""
        return self.n_rejected / self.n_source_rows if self.n_source_rows else 0.0
