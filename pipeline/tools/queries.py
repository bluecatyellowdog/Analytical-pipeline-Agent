"""The reviewer's query tool: a closed vocabulary of aggregate questions about the data.

The reviewer may ask how many rows match a condition, what values a column
holds, a column's summary, counts per group (or per year or month of a date),
and how raw values became clean ones. It never gets a row, and it never writes
SQL: each `Query` is a small structured request, column names are checked
against the table and quoted, and values are bound as parameters, so no text
from the model reaches the database as SQL (the same rule as `drop_rows`, D015).

Columns holding personal data or identifiers are masked inside the query
itself: their values are grouped and shown only as their shape ('99999'), and
they can be filtered only with is_null / is_not_null. Otherwise a question such
as "how many rows have email = anna@..." would confirm that a person is there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipeline.contracts.review import Condition, Query
from pipeline.ingestion.pii import mask_shape
from pipeline.ingestion.raw import ROW_ID, RawStore, quote_ident
from pipeline.ops.base import NUMERIC_TYPES

MAX_RESULT_CHARS = 1500
DATE_TYPES = ("DATE", "TIMESTAMP")


class QueryError(ValueError):
    """The query was refused; the message says why, for the reviewer to correct it."""


@dataclass(frozen=True)
class QueryContext:
    store: RawStore
    raw_table: str
    clean_table: str
    masked: frozenset[str] = frozenset()
    """Columns (raw or clean names) whose values are shown only as their shape."""
    clean_to_raw: dict[str, str] = field(default_factory=dict)
    """Clean column name -> raw name, for columns an op renamed."""


def run_query(ctx: QueryContext, query: Query) -> str:
    """Answer one query as a few lines of text. Raises QueryError when it is refused."""
    table = ctx.raw_table if query.table == "raw" else ctx.clean_table
    types = ctx.store.column_types_of(table)
    if query.column is not None:
        _require(query.column, types, query.table)
    where, params = _where(query.where, types, ctx.masked, query.table)

    answer = {
        "count": _count,
        "value_counts": _value_counts,
        "summary": _summary,
        "group_count": _group_count,
        "compare": _compare,
    }[query.kind](ctx, query, table, types, where, params)
    if len(answer) > MAX_RESULT_CHARS:
        answer = answer[: MAX_RESULT_CHARS - 1] + "…"
    return answer


# -- the five kinds ---------------------------------------------------------


def _count(ctx, query, table, types, where, params) -> str:
    source = _qualified(table)
    total = ctx.store.connection.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
    matched = ctx.store.connection.execute(
        f"SELECT count(*) FROM {source}{where}", params
    ).fetchone()[0]
    return f"{query.id} count ({_scope(query)}): {matched} of {total} rows"


def _value_counts(ctx, query, table, types, where, params) -> str:
    column = query.column
    masked = column in ctx.masked
    value = _text_value(quote_ident(column), masked)
    source = _qualified(table)
    rows, missing, distinct = ctx.store.connection.execute(
        f"SELECT count(*), count(*) FILTER (WHERE {quote_ident(column)} IS NULL), "
        f"count(DISTINCT {value}) FROM {source}{where}",
        params,
    ).fetchone()
    top = ctx.store.connection.execute(
        f"SELECT {value} AS v, count(*) AS n FROM {source}{where} "
        f"GROUP BY v HAVING v IS NOT NULL ORDER BY n DESC, v LIMIT {query.limit}",
        params,
    ).fetchall()
    lines = [
        f"{query.id} value_counts of {column} ({_scope(query)}): {rows} rows, "
        f"{missing} missing, {distinct} distinct" + (" (shown as shape)" if masked else "")
    ]
    lines += [f"  {_quote(v)}: {n}" for v, n in top]
    if distinct > len(top):
        lines.append(f"({distinct - len(top)} more value(s) not shown)")
    return "\n".join(lines)


def _summary(ctx, query, table, types, where, params) -> str:
    column = query.column
    identifier = quote_ident(column)
    dtype = types[column]
    masked = column in ctx.masked
    source = _qualified(table)
    rows, missing, distinct, low, high = ctx.store.connection.execute(
        f"SELECT count(*), count(*) FILTER (WHERE {identifier} IS NULL), "
        f"count(DISTINCT {identifier}), min({identifier}), max({identifier}) FROM {source}{where}",
        params,
    ).fetchone()
    shown = (lambda v: _quote(mask_shape(v))) if masked else (lambda v: _format(v, dtype))
    parts = [
        f"{rows} rows",
        f"missing {missing}",
        f"distinct {distinct}",
        f"min {shown(low)}",
        f"max {shown(high)}",
    ]
    if _is_numeric(dtype) and not masked:
        mean, p25, median, p75 = ctx.store.connection.execute(
            f"SELECT avg({identifier}), quantile_cont({identifier}, 0.25), "
            f"quantile_cont({identifier}, 0.5), quantile_cont({identifier}, 0.75) "
            f"FROM {source}{where}",
            params,
        ).fetchone()
        parts += [
            f"mean {_format(mean, 'DOUBLE')}",
            f"p25 {_format(p25, 'DOUBLE')}",
            f"median {_format(median, 'DOUBLE')}",
            f"p75 {_format(p75, 'DOUBLE')}",
        ]
    elif _is_text(dtype):
        shortest, longest = ctx.store.connection.execute(
            f"SELECT min(length({identifier})), max(length({identifier})) FROM {source}{where}",
            params,
        ).fetchone()
        parts.append(f"length {shortest}-{longest}")
    return f"{query.id} summary of {column} ({_scope(query)}, {dtype}): " + ", ".join(parts)


def _group_count(ctx, query, table, types, where, params) -> str:
    column = query.column
    identifier = quote_ident(column)
    dtype = types[column]
    if query.bucket:
        if not dtype.upper().startswith(DATE_TYPES):
            raise QueryError(
                f"bucket {query.bucket!r} needs a date column; {column} is {dtype} in the "
                f"{query.table} table"
            )
        pattern = "%Y" if query.bucket == "year" else "%Y-%m"
        group, label = f"strftime({identifier}, '{pattern}')", f"{query.bucket}({column})"
        show = str
    else:
        group, label = _text_value(identifier, column in ctx.masked), column
        show = _quote
    source = _qualified(table)
    groups = ctx.store.connection.execute(
        f"SELECT {group} AS g, count(*) FROM {source}{where} GROUP BY g ORDER BY g NULLS LAST",
        params,
    ).fetchall()
    total = sum(n for _, n in groups)
    header = f"{query.id} group_count by {label} ({_scope(query)})"
    lines = [f"{header}: {total} rows in {len(groups)} group(s)"]
    lines += [
        f"  {show(g) if g is not None else '(missing)'}: {n}" for g, n in groups[: query.limit]
    ]
    if len(groups) > query.limit:
        lines.append(f"({len(groups) - query.limit} more group(s) not shown)")
    return "\n".join(lines)


def _compare(ctx, query, table, types, where, params) -> str:
    column = query.column
    clean_types = ctx.store.column_types_of(ctx.clean_table)
    raw_types = ctx.store.column_types_of(ctx.raw_table)
    _require(column, clean_types, "clean")
    source_column = ctx.clean_to_raw.get(column, column)
    if source_column not in raw_types:
        raise QueryError(f"compare needs a column that exists in raw too; {column} does not")
    masked = column in ctx.masked or source_column in ctx.masked
    where, params = _where(query.where, clean_types, ctx.masked, "clean", prefix="c.")
    before = _text_value(f"r.{quote_ident(source_column)}", masked)
    after = _text_value(f"c.{quote_ident(column)}", masked)
    joined = (
        f"FROM {_qualified(ctx.raw_table)} r JOIN {_qualified(ctx.clean_table)} c "
        f"USING ({ROW_ID}){where}"
    )
    rows, changed = ctx.store.connection.execute(
        f"SELECT count(*), count(*) FILTER (WHERE CAST(r.{quote_ident(source_column)} AS VARCHAR) "
        f"IS DISTINCT FROM CAST(c.{quote_ident(column)} AS VARCHAR)) {joined}",
        params,
    ).fetchone()
    pairs = ctx.store.connection.execute(
        f"SELECT {before} AS b, {after} AS a, count(*) AS n {joined} "
        f"GROUP BY b, a ORDER BY n DESC, b LIMIT {query.limit + 1}",
        params,
    ).fetchall()
    clean_type = "VARCHAR" if masked else clean_types[column]
    lines = [
        f"{query.id} compare {column} raw -> clean ({rows} rows kept, {changed} changed"
        + (", shown as shape" if masked else "")
        + ")"
    ]
    lines += [
        f"  {_quote(b)} -> {_format_text(a, clean_type)}: {n}" for b, a, n in pairs[: query.limit]
    ]
    if len(pairs) > query.limit:
        lines.append("(more pairs not shown)")
    return "\n".join(lines)


# -- conditions -------------------------------------------------------------


def _where(
    conditions: list[Condition],
    types: dict[str, str],
    masked: frozenset[str],
    table_label: str,
    *,
    prefix: str = "",
) -> tuple[str, list[Any]]:
    parts: list[str] = []
    params: list[Any] = []
    for condition in conditions:
        _require(condition.column, types, table_label)
        if condition.column in masked and condition.op not in {"is_null", "is_not_null"}:
            raise QueryError(
                f"{condition.column} holds personal data or identifiers; it can only be "
                f"filtered with is_null or is_not_null"
            )
        identifier = prefix + quote_ident(condition.column)
        dtype = types[condition.column]
        if condition.op == "is_null":
            parts.append(f"{identifier} IS NULL")
            continue
        if condition.op == "is_not_null":
            parts.append(f"{identifier} IS NOT NULL")
            continue
        value = str(condition.value)
        if condition.op in {"equals", "not_equals"}:
            operator = "=" if condition.op == "equals" else "IS DISTINCT FROM"
            target = "?" if _is_text(dtype) else f"TRY_CAST(? AS {dtype})"
            parts.append(f"{identifier} {operator} {target}")
        else:
            operator = ">" if condition.op == "greater_than" else "<"
            if _is_text(dtype):
                parts.append(f"TRY_CAST({identifier} AS DOUBLE) {operator} TRY_CAST(? AS DOUBLE)")
            else:
                parts.append(f"{identifier} {operator} TRY_CAST(? AS {dtype})")
        params.append(value)
    return (" WHERE " + " AND ".join(parts) if parts else ""), params


def _require(column: str, types: dict[str, str], table_label: str) -> None:
    if column not in types:
        raise QueryError(
            f"no column {column!r} in the {table_label} table; its columns are {sorted(types)}"
        )


# -- formatting -------------------------------------------------------------


def _text_value(expression: str, masked: bool) -> str:
    """The value as text, or as its shape, computed inside the query."""
    text = f"CAST({expression} AS VARCHAR)"
    if not masked:
        return text
    for pattern, replacement in (
        (r"\p{Lu}", "A"),
        (r"[\p{Lt}\p{Lo}\p{Lm}]", "A"),
        (r"\p{Ll}", "a"),
        (r"\p{Nd}", "9"),
    ):
        text = f"regexp_replace({text}, '{pattern}', '{replacement}', 'g')"
    return text


def _scope(query: Query) -> str:
    conditions = " and ".join(
        f"{c.column} {c.op.replace('_', ' ')}" + (f" {c.value!r}" if c.value is not None else "")
        for c in query.where
    )
    return query.table + (f", where {conditions}" if conditions else "")


def _quote(value: Any) -> str:
    return "'" + str(value) + "'"


def _format(value: Any, dtype: str) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else format(value, ".10g")
    if _is_text(dtype):
        return _quote(value)
    return str(value)


def _format_text(value: str | None, dtype: str) -> str:
    """A value that arrives as text, shown quoted only if the column is text."""
    if value is None:
        return "none"
    return _quote(value) if _is_text(dtype) else value


def _is_text(dtype: str) -> bool:
    return dtype.upper().startswith("VARCHAR")


def _is_numeric(dtype: str) -> bool:
    return any(dtype.upper().startswith(t) for t in NUMERIC_TYPES)


def _qualified(table: str) -> str:
    schema, _, name = table.rpartition(".")
    return f"{schema}.{quote_ident(name)}"


__all__ = ["QueryContext", "QueryError", "run_query"]
