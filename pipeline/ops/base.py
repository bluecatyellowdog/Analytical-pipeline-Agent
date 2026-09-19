"""The op contract: nothing changes the data except through here.

Every op declares *what* it wants (a projection, a filter, a rename) and this
module does all the accounting uniformly: hash before, hash after, count the
cells that changed, count the nulls that appeared, capture the rows that left
and why. An op cannot opt out of being audited, because it never touches the
database itself.

Each op writes a new versioned table rather than updating in place, so the
before/after comparison is a plain query and replaying the log from raw data
reproduces every intermediate state, not just the final one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic import BaseModel, ValidationError

from pipeline.contracts.audit import ColumnStats, OpRecord
from pipeline.contracts.common import Flag
from pipeline.hashing import hash_values
from pipeline.ingestion.raw import (
    AUDIT_SCHEMA,
    CLEAN_SCHEMA,
    ROW_ID,
    SOURCE_LINE,
    RawStore,
    RawTable,
    quote_ident,
)

NUMERIC_TYPES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
)

MAX_EXAMPLES_IN_ERROR = 5


class OpError(Exception):
    """An op refused to run, or refused to change data in a way it cannot justify."""


@dataclass(frozen=True)
class OpContext:
    """Where the data is right now, between ops."""

    store: RawStore
    run_id: str
    base_name: str
    """Logical dataset name; versioned tables are clean.<base_name>__v<N>."""
    table: str
    """Fully qualified current table."""
    columns: list[str]
    version: int = 0
    dropped_table: str | None = None

    @classmethod
    def from_raw(cls, store: RawStore, raw: RawTable, run_id: str) -> OpContext:
        """Start an op chain from the immutable raw table."""
        return cls(
            store=store,
            run_id=run_id,
            base_name=raw.name.split(".", 1)[-1],
            table=raw.name,
            columns=list(raw.columns),
            version=0,
        )

    def next_table(self) -> str:
        return f"{CLEAN_SCHEMA}.{self.base_name}__v{self.version + 1}"


@dataclass
class OpPlan:
    """What an op wants done, expressed declaratively so the executor can audit it."""

    expressions: dict[str, str] = field(default_factory=dict)
    """column -> SQL expression. Columns not listed are carried through unchanged."""
    drop_columns: list[str] = field(default_factory=list)
    renames: dict[str, str] = field(default_factory=dict)
    """old name -> new name, applied after expressions."""
    where: str | None = None
    """SQL predicate for the rows to KEEP. Everything else is captured as dropped."""
    distinct_on: list[str] | None = None
    """Keep the first row (by row id) per combination of these columns."""
    drop_reason: str | None = None
    target_columns: list[str] = field(default_factory=list)
    """Columns this op may change. Anything else must hash identically afterwards."""
    note: str | None = None
    flags: list[Flag] = field(default_factory=list)
    allow_new_nulls: bool = False
    """Set only by ops whose whole purpose is to introduce nulls, e.g. replace_sentinel."""


@dataclass(frozen=True)
class OpSpec:
    name: str
    params_model: type[BaseModel]
    build: Callable[[OpContext, Any], OpPlan]
    doc: str


REGISTRY: dict[str, OpSpec] = {}


def register(name: str, params_model: type[BaseModel]) -> Callable[[Callable], Callable]:
    """Add an op to the library. The name is what an agent may ask for."""

    def decorator(build: Callable[[OpContext, Any], OpPlan]) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"op {name!r} is already registered")
        REGISTRY[name] = OpSpec(
            name=name, params_model=params_model, build=build, doc=build.__doc__ or ""
        )
        return build

    return decorator


def op_names() -> list[str]:
    return sorted(REGISTRY)


def validate_params(name: str, params: dict[str, Any]) -> BaseModel:
    """Reject a malformed op before it can touch anything."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise OpError(f"unknown op {name!r}; available: {op_names()}")
    try:
        return spec.params_model.model_validate(params)
    except ValidationError as error:
        raise OpError(f"invalid parameters for op {name!r}: {error}") from error


def require_columns(ctx: OpContext, *columns: str) -> None:
    """An op may only name columns that exist; a typo must not create one."""
    missing = [c for c in columns if c not in ctx.columns]
    if missing:
        raise OpError(f"column(s) {missing} not in table {ctx.table} (have: {ctx.columns})")


def apply_op(
    ctx: OpContext,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    rationale: str | None = None,
) -> tuple[OpContext, OpRecord]:
    """Run one op and return the new state plus the record that accounts for it."""
    params = dict(params or {})
    spec = REGISTRY.get(name)
    if spec is None:
        raise OpError(f"unknown op {name!r}; available: {op_names()}")

    validated = validate_params(name, params)
    plan = spec.build(ctx, validated)

    store = ctx.store
    started = time.perf_counter()

    columns_after = _columns_after(ctx.columns, plan)
    touched = plan.target_columns or list(plan.expressions)
    unknown_targets = [c for c in touched if c not in ctx.columns]
    if unknown_targets:
        raise OpError(f"op {name!r} targets unknown column(s) {unknown_targets}")

    before = _column_stats(store, ctx.table, ctx.columns)
    rows_before = store.row_count(ctx.table)
    input_hash = store.data_hash(ctx.table, ctx.columns)

    new_table = ctx.next_table()
    sql = _build_sql(ctx, plan, new_table)
    store.connection.execute(f"DROP TABLE IF EXISTS {_qualified(new_table)}")
    store.connection.execute(sql)

    rows_after = store.row_count(new_table)
    rows_dropped = rows_before - rows_after
    if rows_dropped < 0:
        raise OpError(
            f"op {name!r} produced {rows_after} rows from {rows_before}: ops may not create rows"
        )

    dropped_table = ctx.dropped_table
    if rows_dropped:
        dropped_table = _capture_dropped_rows(ctx, new_table, name, plan.drop_reason or name)

    after = _column_stats(store, new_table, columns_after)
    cells_changed, new_nulls = _diff(store, ctx, new_table, touched, columns_after)

    if new_nulls and not plan.allow_new_nulls:
        _reject_silent_coercion(name, new_nulls)

    record = OpRecord(
        op_name=name,
        params=validated.model_dump(mode="json"),
        code=sql,
        target_columns=touched,
        rationale=rationale,
        rows_before=rows_before,
        rows_after=rows_after,
        rows_dropped=rows_dropped,
        cells_changed=cells_changed,
        new_nulls=new_nulls,
        before=before,
        after=after,
        input_hash=input_hash,
        output_hash=store.data_hash(new_table, columns_after),
        duration_ms=(time.perf_counter() - started) * 1000,
        flags=list(plan.flags),
    )

    new_ctx = replace(
        ctx,
        table=new_table,
        columns=columns_after,
        version=ctx.version + 1,
        dropped_table=dropped_table,
    )
    return new_ctx, record


def _reject_silent_coercion(name: str, new_nulls: dict[str, int]) -> None:
    """Spec: no silent coercion. Values that stop parsing must be handled, not lost."""
    detail = ", ".join(f"{column}: {count}" for column, count in new_nulls.items())
    raise OpError(
        f"op {name!r} would introduce nulls ({detail}) without saying so. "
        f"Handle the failing values explicitly, or use an op that declares it "
        f"creates nulls."
    )


def _columns_after(columns: Sequence[str], plan: OpPlan) -> list[str]:
    kept = [c for c in columns if c not in plan.drop_columns]
    return [plan.renames.get(c, c) for c in kept]


def _build_sql(ctx: OpContext, plan: OpPlan, new_table: str) -> str:
    """Assemble the exact statement that will run. This text becomes the record."""
    projection = [ROW_ID, SOURCE_LINE]
    for column in ctx.columns:
        if column in plan.drop_columns:
            continue
        expression = plan.expressions.get(column, quote_ident(column))
        alias = plan.renames.get(column, column)
        projection.append(f"{expression} AS {quote_ident(alias)}")

    source = _qualified(ctx.table)
    clauses = [f"SELECT {', '.join(projection)}", f"FROM {source}"]
    if plan.where:
        clauses.append(f"WHERE {plan.where}")

    if plan.distinct_on:
        keys = ", ".join(quote_ident(c) for c in plan.distinct_on)
        # DISTINCT ON keeps the first row per key; ordering by row id makes
        # "first" mean the first row in the file rather than whatever the
        # scan happened to produce.
        clauses[0] = f"SELECT DISTINCT ON ({keys}) {', '.join(projection)}"
        clauses.append(f"ORDER BY {keys}, {ROW_ID}")

    return f"CREATE TABLE {_qualified(new_table)} AS\n" + "\n".join(clauses)


def _capture_dropped_rows(ctx: OpContext, new_table: str, op_name: str, reason: str) -> str:
    """Every dropped row is kept with the op and reason that removed it."""
    store = ctx.store
    dropped = f"{AUDIT_SCHEMA}.{ctx.base_name}__dropped"
    qualified = _qualified(dropped)
    columns = ", ".join(quote_ident(c) for c in ctx.columns)

    store.connection.execute(
        f"CREATE TABLE IF NOT EXISTS {qualified} AS "
        f"SELECT CAST(NULL AS VARCHAR) AS op_name, CAST(NULL AS VARCHAR) AS reason, "
        f"{ROW_ID}, {SOURCE_LINE}, {columns} FROM {_qualified(ctx.table)} WHERE false"
    )
    store.connection.execute(
        f"INSERT INTO {qualified} "
        f"SELECT ?, ?, {ROW_ID}, {SOURCE_LINE}, {columns} FROM {_qualified(ctx.table)} "
        f"WHERE {ROW_ID} NOT IN (SELECT {ROW_ID} FROM {_qualified(new_table)})",
        [op_name, reason],
    )
    return dropped


def _column_stats(store: RawStore, table: str, columns: Sequence[str]) -> dict[str, ColumnStats]:
    """Before/after snapshot of every column, so a change cannot hide."""
    types = store.column_types_of(table)
    stats: dict[str, ColumnStats] = {}

    for column in columns:
        identifier = quote_ident(column)
        dtype = types.get(column, "UNKNOWN")
        numeric = any(dtype.upper().startswith(t) for t in NUMERIC_TYPES)
        mean_expression = f"avg({identifier})" if numeric else "NULL"

        row = store.connection.execute(
            f"SELECT count(*), count({identifier}), count(DISTINCT {identifier}), "
            f"min({identifier}), max({identifier}), {mean_expression} "
            f"FROM {_qualified(table)}"
        ).fetchone()

        stats[column] = ColumnStats(
            name=column,
            dtype=dtype,
            n_rows=int(row[0]),
            n_null=int(row[0]) - int(row[1]),
            n_distinct=int(row[2]),
            min=_scalar(row[3]),
            max=_scalar(row[4]),
            mean=float(row[5]) if row[5] is not None else None,
            value_hash=hash_values(store.iter_column(table, column)),
        )
    return stats


def _diff(
    store: RawStore,
    ctx: OpContext,
    new_table: str,
    touched: Sequence[str],
    columns_after: Sequence[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Count changed cells and newly-null cells, for rows present in both tables.

    Rows that left are accounted for separately by row accounting; counting them
    here as well would double-count the same change.
    """
    cells_changed: dict[str, int] = {}
    new_nulls: dict[str, int] = {}

    for column in touched:
        alias = column
        if alias not in columns_after:
            continue  # dropped columns are reported as a column change, not cell changes

        old, new = quote_ident(column), quote_ident(alias)
        row = store.connection.execute(
            f"SELECT "
            f"  count(*) FILTER (WHERE CAST(a.{old} AS VARCHAR) IS DISTINCT FROM "
            f"                         CAST(b.{new} AS VARCHAR)), "
            f"  count(*) FILTER (WHERE a.{old} IS NOT NULL AND b.{new} IS NULL) "
            f"FROM {_qualified(ctx.table)} a "
            f"JOIN {_qualified(new_table)} b USING ({ROW_ID})"
        ).fetchone()

        if row[0]:
            cells_changed[column] = int(row[0])
        if row[1]:
            new_nulls[column] = int(row[1])

    return cells_changed, new_nulls


def _scalar(value: Any) -> Any:
    """Keep stats JSON-serialisable without losing the value."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _qualified(table: str) -> str:
    schema, _, name = table.rpartition(".")
    return f"{schema or CLEAN_SCHEMA}.{quote_ident(name)}"


def finalize(ctx: OpContext) -> str:
    """Publish the last version as clean.<base_name>, the table later stages read."""
    target = f"{CLEAN_SCHEMA}.{ctx.base_name}"
    ctx.store.connection.execute(f"DROP TABLE IF EXISTS {_qualified(target)}")
    ctx.store.connection.execute(
        f"CREATE TABLE {_qualified(target)} AS SELECT * FROM {_qualified(ctx.table)}"
    )
    return target
