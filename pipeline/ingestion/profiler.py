"""Describe the data accurately, and change none of it.

Spec 4.6: counts run on the full data because that is what later checks are
compared against; expensive distribution statistics may run on a sample, and
when they do the sample is recorded in the profile so no number is quoted
without saying where it came from.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pipeline.contracts.common import Flag
from pipeline.contracts.profile import (
    ColumnProfile,
    DatasetProfile,
    LengthSummary,
    NumericSummary,
    SentinelCandidate,
    ValueCount,
)
from pipeline.ingestion.pii import detect_pii, mask_shape, strongest_level
from pipeline.ingestion.raw import ROW_ID, RawStore, RawTable, quote_ident
from pipeline.ingestion.types import (
    NULL_TOKENS,
    detect_number_format,
    is_missing,
    parse_number,
    propose_type,
    sentinel_candidates,
)

SAMPLE_LIMIT = 100_000
"""Above this many rows, distribution statistics use a sample (spec 4.6)."""

TOP_VALUES = 5
OUTLIER_IQR_FACTOR = 3.0
"""Spec 4.1: 'far outliers' are beyond 3x the interquartile range."""

HIGH_MISSING_SHARE = 0.5
"""Worth flagging: half a column missing changes what can be asked of it."""


def profile_table(
    store: RawStore,
    table: RawTable,
    *,
    sample_limit: int = SAMPLE_LIMIT,
    seed: int = 42,
) -> DatasetProfile:
    """Profile one raw table into the object every later stage reads."""
    n_rows = store.row_count(table.name)
    sampled = n_rows > sample_limit

    columns: list[ColumnProfile] = []
    flags: list[Flag] = []
    for position, name in enumerate(table.columns):
        values = store.column_values(table.name, name)
        sample = _sample_column(store, table.name, name, sample_limit, seed) if sampled else values
        column = profile_column(
            name,
            values,
            position=position,
            sample=sample,
            declared_type=table.column_types.get(name) if table.typed else None,
        )
        columns.append(column)
        flags.extend(column.flags)

    profile = DatasetProfile(
        table=table.name,
        n_rows=n_rows,
        n_columns=len(table.columns),
        n_duplicate_rows=_count_duplicate_rows(store, table),
        candidate_keys=[c.name for c in columns if _is_candidate_key(c, n_rows)],
        columns=columns,
        sampled=sampled,
        sample_rows=sample_limit if sampled else None,
        flags=[],
    )

    if sampled:
        profile.flags.append(
            Flag(
                code="profiled_on_sample",
                severity="info",
                message=(
                    f"Distribution statistics use a {sample_limit:,}-row sample of "
                    f"{n_rows:,} rows (seed {seed}). Counts are on the full data."
                ),
                stage="ingestion",
                details={"sample_rows": sample_limit, "n_rows": n_rows, "seed": seed},
            )
        )
    if not profile.candidate_keys:
        profile.flags.append(
            Flag(
                code="no_candidate_key",
                severity="info",
                message="No column uniquely identifies a row; duplicates cannot be judged by key.",
                stage="ingestion",
            )
        )
    return profile


def profile_column(
    name: str,
    values: Sequence[Any],
    *,
    position: int,
    sample: Sequence[Any] | None = None,
    declared_type: str | None = None,
) -> ColumnProfile:
    """Everything observable about one column, with nothing inferred about intent."""
    sample = values if sample is None else sample
    n_rows = len(values)

    null_tokens = _count_null_tokens(values)
    n_missing = sum(null_tokens.values())
    present = [v for v in values if not is_missing(v)]
    n_blank = sum(1 for v in values if isinstance(v, str) and v.strip() == "")

    proposal, flags = propose_type(name, values, declared_type=declared_type)
    pii = detect_pii(name, values)
    mask = strongest_level(pii) != "none"

    distinct = len({_key(v) for v in present})
    counts = Counter(str(v) for v in present)
    top_values = [
        ValueCount(
            value=mask_shape(value) if mask else value,
            count=count,
            share=count / len(present) if present else 0.0,
        )
        for value, count in counts.most_common(TOP_VALUES)
    ]

    present_sample = [v for v in sample if not is_missing(v)]
    numeric = (
        _numeric_summary(present_sample) if proposal.proposed in {"integer", "float"} else None
    )
    length = _length_summary(present_sample) if proposal.proposed in {"text", "id"} else None

    sentinels = [
        SentinelCandidate(
            value=value,
            count=count,
            share=count / len(present) if present else 0.0,
            reason="Common coded-missing value; only the brief or the cleaning agent can decide.",
        )
        for value, count in sentinel_candidates(values).items()
    ]

    column = ColumnProfile(
        name=name,
        position=position,
        raw_dtype=declared_type or "text",
        n_rows=n_rows,
        n_missing=n_missing,
        missing_share=n_missing / n_rows if n_rows else 0.0,
        null_tokens=null_tokens,
        n_distinct=distinct,
        distinct_share=distinct / len(present) if present else 0.0,
        n_blank=n_blank,
        top_values=top_values,
        numeric=numeric,
        length=length,
        type_proposal=proposal,
        sentinel_candidates=sentinels,
        pii=pii,
        flags=flags,
    )
    column.flags.extend(_column_flags(column))
    return column


def _column_flags(column: ColumnProfile) -> list[Flag]:
    """Observations worth surfacing. None of them change anything."""
    flags: list[Flag] = []

    if column.missing_share >= HIGH_MISSING_SHARE:
        flags.append(
            Flag(
                code="high_missing_share",
                severity="warn",
                message=f"Column {column.name!r} is {column.missing_share:.0%} missing.",
                column=column.name,
                stage="ingestion",
                details={"missing_share": column.missing_share},
            )
        )

    for sentinel in column.sentinel_candidates:
        flags.append(
            Flag(
                code="sentinel_candidate",
                severity="warn",
                message=(
                    f"Value {sentinel.value!r} appears {sentinel.count} times in "
                    f"{column.name!r} and may be a coded missing value."
                ),
                column=column.name,
                stage="ingestion",
                details={"value": sentinel.value, "count": sentinel.count},
            )
        )

    for finding in column.pii:
        flags.append(
            Flag(
                code="pii_detected",
                severity="warn" if finding.level == "likely" else "info",
                message=(
                    f"Column {column.name!r} {finding.level} contains {finding.kind} data "
                    f"({finding.share:.0%} of values). Masked before any LLM call."
                ),
                column=column.name,
                stage="ingestion",
                details={"kind": finding.kind, "detector": finding.detector},
            )
        )

    if column.numeric and column.numeric.n_outliers:
        flags.append(
            Flag(
                code="far_outliers",
                severity="info",
                message=(
                    f"Column {column.name!r} has {column.numeric.n_outliers} value(s) beyond "
                    f"{OUTLIER_IQR_FACTOR:g}x IQR."
                ),
                column=column.name,
                stage="ingestion",
                details={"bounds": list(column.numeric.outlier_bounds or ())},
            )
        )
    return flags


def _count_null_tokens(values: Sequence[Any]) -> dict[str, int]:
    """Which token stood in for missing, and how often. Blanks count as ''."""
    counts: Counter[str] = Counter()
    for value in values:
        if value is None:
            counts["<null>"] += 1
        elif isinstance(value, str) and value.strip().lower() in NULL_TOKENS:
            counts[value.strip()] += 1
    return dict(counts)


def _numeric_summary(values: Sequence[Any]) -> NumericSummary | None:
    fmt = detect_number_format([str(v) for v in values])
    numbers = [n for n in (parse_number(v, fmt) for v in values) if n is not None]
    if not numbers:
        return None

    ordered = sorted(numbers)
    quartiles = _quantiles(ordered)
    iqr = quartiles[2] - quartiles[0]
    low, high = (
        quartiles[0] - OUTLIER_IQR_FACTOR * iqr,
        quartiles[2] + OUTLIER_IQR_FACTOR * iqr,
    )

    return NumericSummary(
        min=ordered[0],
        q1=quartiles[0],
        median=quartiles[1],
        q3=quartiles[2],
        max=ordered[-1],
        mean=statistics.fmean(ordered),
        std=statistics.stdev(ordered) if len(ordered) > 1 else None,
        n_outliers=sum(1 for n in ordered if n < low or n > high),
        outlier_bounds=(low, high),
    )


def _quantiles(ordered: Sequence[float]) -> tuple[float, float, float]:
    """Q1, median and Q3 by linear interpolation, defined for any length."""

    def at(fraction: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        position = fraction * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return float(ordered[low]) * (1 - weight) + float(ordered[high]) * weight

    return at(0.25), at(0.5), at(0.75)


def _length_summary(values: Sequence[Any]) -> LengthSummary | None:
    lengths = [len(str(v)) for v in values]
    if not lengths:
        return None
    return LengthSummary(min=min(lengths), max=max(lengths), mean=statistics.fmean(lengths))


def _key(value: Any) -> Any:
    """Distinct-count key. Text values compare as text, typed values as themselves."""
    return value


def _is_candidate_key(column: ColumnProfile, n_rows: int) -> bool:
    return n_rows > 0 and column.n_missing == 0 and column.n_distinct == n_rows


def _count_duplicate_rows(store: RawStore, table: RawTable) -> int:
    """Rows that are an exact repeat of another row, ignoring bookkeeping columns."""
    if not table.columns:
        return 0
    selection = ", ".join(quote_ident(name) for name in table.columns)
    distinct = store.connection.execute(
        f"SELECT count(*) FROM (SELECT DISTINCT {selection} FROM {table.name})"
    ).fetchone()[0]
    return int(store.row_count(table.name) - distinct)


def _sample_column(store: RawStore, table: str, column: str, n: int, seed: int) -> list[Any]:
    """A repeatable sample, so the same data always yields the same statistics."""
    rows = store.connection.execute(
        f"SELECT {quote_ident(column)} FROM {table} "
        f"USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})"
    ).fetchall()
    return [row[0] for row in rows]


__all__ = ["ROW_ID", "profile_column", "profile_table"]
