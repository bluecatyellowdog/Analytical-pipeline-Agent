"""Test what the brief claims against what the data shows.

Spec stage 0: every checkable statement is a hypothesis, not a fact. A brief
that says 'cust_id is unique' is useful precisely because it can be wrong —
a mismatch is usually the most valuable thing Stage 1 finds.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pipeline.contracts.context_brief import Claim, ClaimResult, ContextBrief
from pipeline.contracts.profile import ColumnProfile, DatasetProfile
from pipeline.ingestion.types import (
    detect_number_format,
    is_missing,
    parse_date,
    parse_number,
)

MAX_REPORTED_VALUES = 5


def derive_claims(brief: ContextBrief) -> list[Claim]:
    """Turn the brief's structured fields into claims that can be tested.

    The user wrote 'id_column: kunde_id' as a fact about the data; treating it
    as a claim costs nothing and catches the case where it is not true.
    """
    claims: list[Claim] = []

    if brief.target:
        claims.append(
            Claim(
                kind="column_exists",
                column=brief.target,
                text=f"the target column {brief.target!r} exists",
                source="derived",
            )
        )

    if brief.id_column:
        claims.extend(
            [
                Claim(
                    kind="column_exists",
                    column=brief.id_column,
                    text=f"the ID column {brief.id_column!r} exists",
                    source="derived",
                ),
                Claim(
                    kind="column_unique",
                    column=brief.id_column,
                    text=f"{brief.id_column!r} uniquely identifies a row",
                    source="derived",
                ),
                Claim(
                    kind="column_not_null",
                    column=brief.id_column,
                    text=f"{brief.id_column!r} is never missing",
                    source="derived",
                ),
            ]
        )

    for column in brief.excluded_variables:
        claims.append(
            Claim(
                kind="column_exists",
                column=column,
                text=f"the excluded column {column!r} exists",
                source="derived",
            )
        )

    for note in brief.column_notes:
        if note.allowed_values:
            claims.append(
                Claim(
                    kind="allowed_values",
                    column=note.name,
                    params={"values": list(note.allowed_values)},
                    text=f"{note.name} only contains {note.allowed_values}",
                    source="derived",
                )
            )

    return claims


def all_claims(brief: ContextBrief) -> list[Claim]:
    """Stated claims first, then derived ones that are not already stated."""
    stated = {(c.kind, c.column) for c in brief.claims}
    return [*brief.claims, *(c for c in derive_claims(brief) if (c.kind, c.column) not in stated)]


def verify_claims(brief: ContextBrief, profile: DatasetProfile) -> list[ClaimResult]:
    """Check every claim the brief makes, stated or derived."""
    return [verify_claim(claim, profile) for claim in all_claims(brief)]


def verify_claim(claim: Claim, profile: DatasetProfile) -> ClaimResult:
    """One claim against the profile. Unverifiable is a third outcome, not a pass."""
    column = profile.column(claim.column) if claim.column else None

    if claim.kind == "row_count":
        return _check_row_count(claim, profile)

    if claim.column is None:
        return ClaimResult(
            claim=claim,
            status="not_checkable",
            message=f"Claim of kind {claim.kind!r} needs a column.",
        )

    if column is None:
        available = ", ".join(profile.column_names[:10])
        return ClaimResult(
            claim=claim,
            status="fail" if claim.kind == "column_exists" else "not_checkable",
            message=f"Column {claim.column!r} is not in the data. Columns: {available}.",
            observed={"columns": profile.column_names},
        )

    checker = {
        "column_exists": _check_exists,
        "column_unique": _check_unique,
        "column_not_null": _check_not_null,
        "binary_column": _check_binary,
        "value_range": _check_value_range,
        "date_range": _check_date_range,
        "allowed_values": _check_allowed_values,
    }.get(claim.kind)

    if checker is None:
        return ClaimResult(
            claim=claim,
            status="not_checkable",
            message=f"No test implemented for claim kind {claim.kind!r}.",
        )
    return checker(claim, column, profile)


def _check_exists(claim: Claim, column: ColumnProfile, _: DatasetProfile) -> ClaimResult:
    return ClaimResult(
        claim=claim,
        status="pass",
        message=f"Column {column.name!r} is present.",
        observed={"position": column.position},
    )


def _check_unique(claim: Claim, column: ColumnProfile, profile: DatasetProfile) -> ClaimResult:
    n_present = column.n_rows - column.n_missing
    duplicates = n_present - column.n_distinct
    if duplicates == 0 and column.n_missing == 0:
        return ClaimResult(
            claim=claim,
            status="pass",
            message=f"All {column.n_distinct:,} values of {column.name!r} are distinct.",
            observed={"n_distinct": column.n_distinct, "n_rows": column.n_rows},
        )
    parts = []
    if duplicates:
        parts.append(f"{duplicates:,} duplicate value(s)")
    if column.n_missing:
        parts.append(f"{column.n_missing:,} missing value(s)")
    return ClaimResult(
        claim=claim,
        status="fail",
        message=f"Column {column.name!r} is not a unique key: {' and '.join(parts)}.",
        observed={
            "n_rows": column.n_rows,
            "n_distinct": column.n_distinct,
            "n_missing": column.n_missing,
            "n_duplicates": duplicates,
        },
    )


def _check_not_null(claim: Claim, column: ColumnProfile, _: DatasetProfile) -> ClaimResult:
    if column.n_missing == 0:
        return ClaimResult(
            claim=claim, status="pass", message=f"Column {column.name!r} has no missing values."
        )
    return ClaimResult(
        claim=claim,
        status="fail",
        message=(
            f"Column {column.name!r} is missing in {column.n_missing:,} row(s) "
            f"({column.missing_share:.1%})."
        ),
        observed={"n_missing": column.n_missing, "null_tokens": column.null_tokens},
    )


def _check_binary(claim: Claim, column: ColumnProfile, _: DatasetProfile) -> ClaimResult:
    values = [v.value for v in column.top_values]
    if column.n_distinct == 2:
        return ClaimResult(
            claim=claim,
            status="pass",
            message=f"Column {column.name!r} has exactly two distinct values: {values}.",
            observed={"values": values, "n_missing": column.n_missing},
        )
    return ClaimResult(
        claim=claim,
        status="fail",
        message=(
            f"Column {column.name!r} has {column.n_distinct} distinct values, not 2. "
            f"Most common: {values}."
        ),
        observed={"n_distinct": column.n_distinct, "top_values": values},
    )


def _check_value_range(claim: Claim, column: ColumnProfile, _: DatasetProfile) -> ClaimResult:
    low, high = claim.params.get("min"), claim.params.get("max")
    if column.numeric is None:
        return ClaimResult(
            claim=claim,
            status="not_checkable",
            message=(
                f"Column {column.name!r} was not read as a number "
                f"(proposed {column.type_proposal.proposed!r}), so a range cannot be checked."
            ),
        )

    observed = {"min": column.numeric.min, "max": column.numeric.max}
    problems = []
    if low is not None and column.numeric.min < float(low):
        problems.append(f"minimum {column.numeric.min:g} is below {low}")
    if high is not None and column.numeric.max > float(high):
        problems.append(f"maximum {column.numeric.max:g} is above {high}")

    if problems:
        return ClaimResult(
            claim=claim,
            status="fail",
            message=f"Column {column.name!r}: {', and '.join(problems)}.",
            observed=observed,
        )
    return ClaimResult(
        claim=claim,
        status="pass",
        message=(
            f"Column {column.name!r} ranges from {column.numeric.min:g} to "
            f"{column.numeric.max:g}, within the stated range."
        ),
        observed=observed,
    )


def _check_date_range(claim: Claim, column: ColumnProfile, _: DatasetProfile) -> ClaimResult:
    """Checked against the profile's top values, which is all Stage 1 keeps.

    A date range therefore says whether the *observed* values fit, and the
    result reports how many values it could actually test.
    """
    fmt = column.type_proposal.details.get("date_format")
    if column.type_proposal.proposed not in {"date", "datetime"} or not fmt:
        return ClaimResult(
            claim=claim,
            status="not_checkable",
            message=(
                f"Column {column.name!r} was not read as a date "
                f"(proposed {column.type_proposal.proposed!r})."
            ),
        )

    low = _as_date(claim.params.get("min"))
    high = _as_date(claim.params.get("max"))
    sampled = [parse_date(v.value, fmt) for v in column.top_values]
    dates = [d for d in sampled if d is not None]
    if not dates:
        return ClaimResult(
            claim=claim,
            status="not_checkable",
            message=f"No parseable dates available in the profile for {column.name!r}.",
        )

    outside = [d for d in dates if (low and d < low) or (high and d > high)]
    observed = {
        "min_seen": min(dates).isoformat(),
        "max_seen": max(dates).isoformat(),
        "n_checked": len(dates),
    }
    if outside:
        return ClaimResult(
            claim=claim,
            status="fail",
            message=(
                f"Column {column.name!r} contains dates outside the stated period, "
                f"e.g. {outside[0].isoformat()}."
            ),
            observed=observed | {"examples": [d.isoformat() for d in outside[:3]]},
        )
    return ClaimResult(
        claim=claim,
        status="pass",
        message=(
            f"The {len(dates)} most common dates in {column.name!r} fall between "
            f"{observed['min_seen']} and {observed['max_seen']}, inside the stated period."
        ),
        observed=observed,
    )


def _check_allowed_values(claim: Claim, column: ColumnProfile, _: DatasetProfile) -> ClaimResult:
    allowed = {str(v) for v in claim.params.get("values", [])}
    if not allowed:
        return ClaimResult(
            claim=claim, status="not_checkable", message="No allowed values were given."
        )
    if column.n_distinct > len(column.top_values):
        return ClaimResult(
            claim=claim,
            status="not_checkable",
            message=(
                f"Column {column.name!r} has {column.n_distinct} distinct values; "
                f"the profile keeps only the {len(column.top_values)} most common."
            ),
        )

    unexpected = [v.value for v in column.top_values if v.value not in allowed]
    if unexpected:
        return ClaimResult(
            claim=claim,
            status="fail",
            message=f"Column {column.name!r} contains unexpected values: "
            f"{unexpected[:MAX_REPORTED_VALUES]}.",
            observed={"unexpected": unexpected, "allowed": sorted(allowed)},
        )
    return ClaimResult(
        claim=claim,
        status="pass",
        message=f"Column {column.name!r} only contains the stated values.",
        observed={"allowed": sorted(allowed)},
    )


def _check_row_count(claim: Claim, profile: DatasetProfile) -> ClaimResult:
    expected = claim.params.get("expected")
    low, high = claim.params.get("min"), claim.params.get("max")
    observed = {"n_rows": profile.n_rows}

    if expected is not None and profile.n_rows != int(expected):
        return ClaimResult(
            claim=claim,
            status="fail",
            message=f"Expected {int(expected):,} rows, loaded {profile.n_rows:,}.",
            observed=observed,
        )
    if low is not None and profile.n_rows < int(low):
        return ClaimResult(
            claim=claim,
            status="fail",
            message=f"Loaded {profile.n_rows:,} rows, fewer than the stated minimum {low}.",
            observed=observed,
        )
    if high is not None and profile.n_rows > int(high):
        return ClaimResult(
            claim=claim,
            status="fail",
            message=f"Loaded {profile.n_rows:,} rows, more than the stated maximum {high}.",
            observed=observed,
        )
    return ClaimResult(
        claim=claim,
        status="pass",
        message=f"Loaded {profile.n_rows:,} rows, matching the brief.",
        observed=observed,
    )


def _as_date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return parse_date(str(value), "%Y-%m-%d")


def check_values_in_range(values: list[Any], low: float, high: float) -> int:
    """Count values outside a range, used where full data is available."""
    fmt = detect_number_format([str(v) for v in values])
    outside = 0
    for value in values:
        if is_missing(value):
            continue
        number = parse_number(value, fmt)
        if number is not None and (number < low or number > high):
            outside += 1
    return outside
