"""The raw layer: loaded once, hashed, never written to again.

DuckDB has no per-table read-only mode, so immutability is enforced the way the
rest of the pipeline enforces everything else — by hashing. `verify()` recomputes
a raw table's hash and fails loudly if anything touched it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb

from pipeline.hashing import hash_rows, hash_values

RAW_SCHEMA = "raw"
CLEAN_SCHEMA = "clean"
AUDIT_SCHEMA = "audit"
SCHEMAS = (RAW_SCHEMA, CLEAN_SCHEMA, AUDIT_SCHEMA)

ROW_ID = "_row_id"
SOURCE_LINE = "_source_line"
RESERVED_COLUMNS = (ROW_ID, SOURCE_LINE)
"""Bookkeeping columns the pipeline owns; a source column may not use these names."""


def quote_ident(name: str) -> str:
    """Quote an identifier for SQL. Column names come from files, so never trust them."""
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True)
class RawTable:
    """A loaded table in the raw layer, with the hash that pins it down."""

    name: str
    """Fully qualified, e.g. 'raw.danish_semicolon'."""
    columns: list[str]
    """Source columns, without the reserved bookkeeping columns."""
    column_types: dict[str, str]
    typed: bool
    """True only for Parquet, which carries its own types (spec 4.2)."""
    n_rows: int
    data_hash: str
    rejects_table: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class RawStore:
    """Owns the DuckDB connection and the three schemas the pipeline uses."""

    def __init__(self, connection: duckdb.DuckDBPyConnection) -> None:
        self._con = connection
        for schema in SCHEMAS:
            self._con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")

    @classmethod
    def open(cls, path: str | Path | None = None) -> RawStore:
        """Open a store on disk, or in memory when `path` is None (tests, dry runs)."""
        target = ":memory:" if path is None else str(Path(path))
        return cls(duckdb.connect(target))

    def __enter__(self) -> RawStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._con.close()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._con

    # -- writing ---------------------------------------------------------

    def create_text_table(
        self,
        table: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[str | None]],
        source_lines: Sequence[int],
    ) -> RawTable:
        """Store values as literal text (spec 4.2), plus row id and source line.

        Nothing is converted here. '007' stays '007' and '1.234,50' stays
        '1.234,50'; types are applied later as logged cast ops that the
        reconciliation checks can verify.
        """
        if len(rows) != len(source_lines):
            raise ValueError(f"{len(rows)} rows but {len(source_lines)} source lines")
        self._reject_reserved(columns)

        qualified = f"{RAW_SCHEMA}.{quote_ident(table)}"
        column_defs = ", ".join(f"{quote_ident(name)} VARCHAR" for name in columns)
        self._con.execute(f"DROP TABLE IF EXISTS {qualified}")
        self._con.execute(
            f"CREATE TABLE {qualified} "
            f"({ROW_ID} BIGINT, {SOURCE_LINE} BIGINT{', ' if column_defs else ''}{column_defs})"
        )

        if rows:
            placeholders = ", ".join("?" for _ in range(len(columns) + 2))
            paired = zip(rows, source_lines, strict=True)
            self._con.executemany(
                f"INSERT INTO {qualified} VALUES ({placeholders})",
                [
                    [row_id, source_line, *row]
                    for row_id, (row, source_line) in enumerate(paired, start=1)
                ],
            )

        return RawTable(
            name=f"{RAW_SCHEMA}.{table}",
            columns=list(columns),
            column_types=dict.fromkeys(columns, "VARCHAR"),
            typed=False,
            n_rows=len(rows),
            data_hash=hash_rows(columns, rows),
        )

    def create_typed_table(
        self,
        table: str,
        columns: Sequence[str],
        column_types: dict[str, str],
        rows: Sequence[Sequence[Any]],
    ) -> RawTable:
        """Store values with their own declared types. Parquet only."""
        self._reject_reserved(columns)

        qualified = f"{RAW_SCHEMA}.{quote_ident(table)}"
        column_defs = ", ".join(f"{quote_ident(name)} {column_types[name]}" for name in columns)
        self._con.execute(f"DROP TABLE IF EXISTS {qualified}")
        self._con.execute(
            f"CREATE TABLE {qualified} "
            f"({ROW_ID} BIGINT, {SOURCE_LINE} BIGINT, {column_defs})"
        )

        if rows:
            placeholders = ", ".join("?" for _ in range(len(columns) + 2))
            self._con.executemany(
                f"INSERT INTO {qualified} VALUES ({placeholders})",
                [[i, i, *row] for i, row in enumerate(rows, start=1)],
            )

        return RawTable(
            name=f"{RAW_SCHEMA}.{table}",
            columns=list(columns),
            column_types=dict(column_types),
            typed=True,
            n_rows=len(rows),
            data_hash=hash_rows(columns, rows),
        )

    def create_rejects_table(self, table: str, rejects: Sequence[Any]) -> str | None:
        """Store unparseable lines so nothing is lost, only set aside."""
        if not rejects:
            return None
        name = f"{table}__rejects"
        qualified = f"{RAW_SCHEMA}.{quote_ident(name)}"
        self._con.execute(f"DROP TABLE IF EXISTS {qualified}")
        self._con.execute(
            f"CREATE TABLE {qualified} ({SOURCE_LINE} BIGINT, reason VARCHAR, raw VARCHAR)"
        )
        self._con.executemany(
            f"INSERT INTO {qualified} VALUES (?, ?, ?)",
            [[r.source_line, r.reason, r.raw] for r in rejects],
        )
        return f"{RAW_SCHEMA}.{name}"

    @staticmethod
    def _reject_reserved(columns: Sequence[str]) -> None:
        clashes = sorted(set(columns) & set(RESERVED_COLUMNS))
        if clashes:
            raise ValueError(f"source columns may not be named {clashes}; these are reserved")

    # -- reading ---------------------------------------------------------

    def fetch(self, table: str, columns: Sequence[str] | None = None) -> list[tuple[Any, ...]]:
        """All rows in row-id order. Order is part of the contract: hashes depend on it."""
        selection = (
            ", ".join(quote_ident(name) for name in columns) if columns is not None else "*"
        )
        return self._con.execute(
            f"SELECT {selection} FROM {self._qualify(table)} ORDER BY {ROW_ID}"
        ).fetchall()

    def iter_column(self, table: str, column: str) -> Iterator[Any]:
        """One column's values in row order, streamed."""
        cursor = self._con.execute(
            f"SELECT {quote_ident(column)} FROM {self._qualify(table)} ORDER BY {ROW_ID}"
        )
        while batch := cursor.fetchmany(10_000):
            for (value,) in batch:
                yield value

    def column_values(self, table: str, column: str) -> list[Any]:
        return list(self.iter_column(table, column))

    def columns_of(self, table: str, *, include_reserved: bool = False) -> list[str]:
        rows = self._con.execute(f"DESCRIBE {self._qualify(table)}").fetchall()
        names = [row[0] for row in rows]
        if include_reserved:
            return names
        return [name for name in names if name not in RESERVED_COLUMNS]

    def column_types_of(self, table: str) -> dict[str, str]:
        rows = self._con.execute(f"DESCRIBE {self._qualify(table)}").fetchall()
        return {row[0]: row[1] for row in rows if row[0] not in RESERVED_COLUMNS}

    def row_count(self, table: str) -> int:
        return int(self._con.execute(f"SELECT count(*) FROM {self._qualify(table)}").fetchone()[0])

    def table_exists(self, table: str) -> bool:
        schema, _, name = table.rpartition(".")
        return bool(
            self._con.execute(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = ? AND table_name = ?",
                [schema or RAW_SCHEMA, name],
            ).fetchone()
        )

    # -- integrity -------------------------------------------------------

    def data_hash(self, table: str, columns: Sequence[str] | None = None) -> str:
        """Recompute a table's hash from what is stored right now."""
        names = list(columns) if columns is not None else self.columns_of(table)
        return hash_rows(names, self.fetch(table, names))

    def column_hash(self, table: str, column: str) -> str:
        """Hash of one column, for the untouched-column check."""
        return hash_values(self.iter_column(table, column))

    def verify(self, raw_table: RawTable) -> None:
        """Fail if anything changed the raw layer since it was loaded.

        The whole pipeline is built on raw being immutable; if that stops being
        true, no later check means anything.
        """
        actual = self.data_hash(raw_table.name, raw_table.columns)
        if actual != raw_table.data_hash:
            raise RuntimeError(
                f"raw table {raw_table.name} changed since it was loaded "
                f"(expected {raw_table.data_hash[:12]}, found {actual[:12]})"
            )

    @staticmethod
    def _qualify(table: str) -> str:
        schema, _, name = table.rpartition(".")
        return f"{schema or RAW_SCHEMA}.{quote_ident(name)}"
