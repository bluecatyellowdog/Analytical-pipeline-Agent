"""Propose a type for each column without converting anything.

Spec 4.3. Every proposal carries the coverage that justifies it and, when it
falls short, the values that failed — so the cleaning agent decides with the
evidence in front of it. Nothing here writes data: a proposal is an opinion,
and the cast that acts on it is a separate, logged op.
"""

from __future__ import annotations

import datetime as dt
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pipeline.contracts.common import Flag
from pipeline.contracts.profile import ProposedType, TypeProposal

COVERAGE_THRESHOLD = 0.95
"""Spec 4.1: propose the most specific type that parses at least 95% of values."""

NULL_TOKENS = frozenset({"", "na", "n/a", "null", "none", "-", "?", "missing", "nan"})
"""Spec 4.3: counted as missing, never converted."""

SENTINEL_NUMBERS = frozenset({"999", "9999", "-1", "-999", "-9999", "99999", "999999"})
SENTINEL_DATES = frozenset({"1900-01-01", "9999-12-31", "0001-01-01", "1970-01-01"})
"""Flagged only. Whether a sentinel means missing is a judgment call for later."""

BOOLEAN_TOKENS: dict[str, bool] = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "ja": True,
    "nej": False,
    "y": True,
    "n": False,
    "1": True,
    "0": False,
    "t": True,
    "f": False,
}

ID_NAME_HINTS = ("id", "nr", "key", "nummer", "number", "kode", "code", "cpr", "cvr")

CURRENCY_NOISE = re.compile(r"[\s  ]|kr\.?|dkk|eur|usd|[€$£]", re.IGNORECASE)

_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_LEADING_ZERO_RE = re.compile(r"^0\d+$")
_CODE_LIKE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-/.]*$")

DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%Y%m%d",
    "%d-%m-%y",
    "%d/%m/%y",
)

DATETIME_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
)

DAY_FIRST_VS_MONTH_FIRST = (("%d-%m-%Y", "%m-%d-%Y"), ("%d/%m/%Y", "%m/%d/%Y"))
"""Pairs that only differ in day/month order; the source of the 03/04 bug."""

DATE_NAME_HINTS = ("dato", "date", "dag", "day", "tid", "time", "stamp", "aar", "year")


def is_missing(value: Any) -> bool:
    """True for nulls, blanks and recognised null tokens."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in NULL_TOKENS
    return False


def non_missing(values: Sequence[Any]) -> list[Any]:
    return [v for v in values if not is_missing(v)]


# -- numbers ------------------------------------------------------------


@dataclass(frozen=True)
class NumberFormat:
    """How one column writes its numbers."""

    decimal: str
    thousands: str | None
    evidence: int
    """How many values unambiguously showed this convention."""


def detect_number_format(values: Sequence[str]) -> NumberFormat:
    """Decide decimal comma vs dot from the column's own unambiguous values.

    '1.234,50' and '1,234.50' both contain a dot and a comma, and whichever
    comes last is the decimal separator. A lone separator followed by exactly
    three digits ('1.234') could be either, so it is not counted as evidence.
    """
    comma_votes = 0
    dot_votes = 0

    for raw in values:
        text = CURRENCY_NOISE.sub("", str(raw))
        has_dot, has_comma = "." in text, "," in text
        if has_dot and has_comma:
            if text.rfind(",") > text.rfind("."):
                comma_votes += 1
            else:
                dot_votes += 1
        elif has_comma and text.count(",") == 1:
            if len(text.rsplit(",", 1)[1]) != 3:
                comma_votes += 1
        elif has_dot and text.count(".") == 1:
            if len(text.rsplit(".", 1)[1]) != 3:
                dot_votes += 1

    if comma_votes > dot_votes:
        return NumberFormat(decimal=",", thousands=".", evidence=comma_votes)
    if dot_votes > comma_votes:
        return NumberFormat(decimal=".", thousands=",", evidence=dot_votes)
    return NumberFormat(decimal=".", thousands=",", evidence=0)


def parse_number(raw: Any, fmt: NumberFormat) -> float | None:
    """Parse one value under a known convention, or return None. Never guesses."""
    if isinstance(raw, int | float | Decimal) and not isinstance(raw, bool):
        return float(raw)
    text = CURRENCY_NOISE.sub("", str(raw))
    if not text:
        return None
    if fmt.thousands:
        text = text.replace(fmt.thousands, "")
    if fmt.decimal != ".":
        text = text.replace(fmt.decimal, ".")
    try:
        return float(text)
    except ValueError:
        return None


def looks_integer(raw: Any, fmt: NumberFormat) -> bool:
    """True when a value is a whole number written without a decimal part."""
    if isinstance(raw, bool):
        return False
    if isinstance(raw, int):
        return True
    text = CURRENCY_NOISE.sub("", str(raw))
    if fmt.thousands:
        text = text.replace(fmt.thousands, "")
    return bool(_INTEGER_RE.match(text))


# -- dates --------------------------------------------------------------


def parse_date(raw: Any, fmt: str) -> dt.date | None:
    if isinstance(raw, dt.datetime):
        return raw.date()
    if isinstance(raw, dt.date):
        return raw
    try:
        return dt.datetime.strptime(str(raw).strip(), fmt).date()
    except (ValueError, TypeError):
        return None


def date_format_coverage(values: Sequence[Any]) -> dict[str, float]:
    """Coverage for every candidate date format, so ambiguity is visible."""
    if not values:
        return {}
    coverage: dict[str, float] = {}
    for fmt in (*DATE_FORMATS, *DATETIME_FORMATS):
        matched = sum(1 for value in values if parse_date(value, fmt) is not None)
        if matched:
            coverage[fmt] = matched / len(values)
    return coverage


def day_month_is_ambiguous(values: Sequence[Any], fmt: str) -> tuple[bool, str | None]:
    """Spec 5.3: with no day above 12, 03/04 could be 3 April or 4 March.

    The data cannot settle it, so it is flagged rather than guessed.
    """
    for day_first, month_first in DAY_FIRST_VS_MONTH_FIRST:
        if fmt not in (day_first, month_first):
            continue
        other = month_first if fmt == day_first else day_first
        both_parse = all(parse_date(v, other) is not None for v in values)
        if both_parse:
            return True, other
    return False, None


# -- names --------------------------------------------------------------


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^0-9a-zA-Z]+", name.lower()) if t]


def has_id_name(name: str) -> bool:
    """Whether the column name itself suggests an identifier."""
    return any(
        token == hint or token.endswith(hint) for token in _tokens(name) for hint in ID_NAME_HINTS
    )


def has_date_name(name: str) -> bool:
    return any(hint in token for token in _tokens(name) for hint in DATE_NAME_HINTS)


# -- the proposal -------------------------------------------------------

_DUCKDB_TYPE_MAP: tuple[tuple[re.Pattern[str], ProposedType], ...] = (
    (re.compile(r"^BOOLEAN$"), "boolean"),
    (
        re.compile(
            r"^(TINYINT|SMALLINT|INTEGER|BIGINT|HUGEINT|U?INTEGER|UBIGINT|USMALLINT|UTINYINT)$"
        ),
        "integer",
    ),
    (re.compile(r"^(DECIMAL.*|DOUBLE|FLOAT|REAL)$"), "float"),
    (re.compile(r"^DATE$"), "date"),
    (re.compile(r"^TIMESTAMP.*$"), "datetime"),
)


def propose_type(
    column: str,
    values: Sequence[Any],
    *,
    declared_type: str | None = None,
) -> tuple[TypeProposal, list[Flag]]:
    """Propose one column's type, with the evidence behind the proposal.

    `declared_type` short-circuits the heuristics for Parquet, which already
    carries types someone chose on purpose.
    """
    flags: list[Flag] = []
    present = non_missing(values)

    if not present:
        return (
            TypeProposal(
                column=column,
                proposed="empty",
                coverage=0.0,
                notes=["Every value is missing; no type can be proposed."],
            ),
            flags,
        )

    if declared_type is not None:
        declared = _from_declared_type(column, declared_type)
        if declared is not None:
            return declared, flags

    texts = [str(v) for v in present]
    n = len(present)
    distinct = len(set(texts))
    number_format = detect_number_format(texts)

    boolean_coverage = (
        sum(1 for t in texts if t.strip().lower() in BOOLEAN_TOKENS) / n if distinct <= 2 else 0.0
    )
    integer_coverage = sum(1 for v in present if looks_integer(v, number_format)) / n
    float_coverage = sum(1 for v in present if parse_number(v, number_format) is not None) / n
    date_coverage = date_format_coverage(present)
    best_date_format, best_date_coverage = (
        max(date_coverage.items(), key=lambda kv: kv[1]) if date_coverage else (None, 0.0)
    )

    alternatives: dict[str, float] = {
        "boolean": boolean_coverage,
        "integer": integer_coverage,
        "float": float_coverage,
        "date": best_date_coverage,
    }

    # Strong identifier evidence outranks every parser: '007' is unique with a
    # leading zero and must stay text however cleanly it reads as the integer 7.
    # Weaker evidence (merely code-like values) is checked further down, after
    # the real types have had their chance — otherwise a date column of unique
    # values like '03-04-2021' would be mistaken for an identifier.
    leading_zeros = any(_LEADING_ZERO_RE.match(t) for t in texts)
    is_unique = distinct == n
    if is_unique and (leading_zeros or has_id_name(column)):
        reasons = []
        if leading_zeros:
            reasons.append("leading zeros would be lost by a cast")
        if has_id_name(column):
            reasons.append("the column name reads as an identifier")
        return (
            _id_proposal(column, leading_zeros, alternatives, reasons),
            flags,
        )

    if boolean_coverage >= COVERAGE_THRESHOLD:
        return (
            TypeProposal(
                column=column,
                proposed="boolean",
                coverage=boolean_coverage,
                details={"values": sorted(set(t.lower() for t in texts))},
            ),
            flags,
        )

    # A date is checked before integers only when it could not be mistaken for
    # one; otherwise '20210403' would be proposed as a number.
    date_first = best_date_format is not None and (
        has_date_name(column) or not _INTEGER_RE.match(texts[0].strip())
    )

    if date_first and best_date_coverage >= COVERAGE_THRESHOLD:
        return _date_proposal(
            column, present, best_date_format, best_date_coverage, date_coverage, flags
        )

    if integer_coverage >= COVERAGE_THRESHOLD:
        proposal = TypeProposal(
            column=column,
            proposed="integer",
            coverage=integer_coverage,
            details=_number_details(number_format, thousands_only=True),
            failing_examples=_failures(present, lambda v: looks_integer(v, number_format)),
        )
        _note_partial(proposal, integer_coverage, flags, column)
        return proposal, flags

    if float_coverage >= COVERAGE_THRESHOLD:
        proposal = TypeProposal(
            column=column,
            proposed="float",
            coverage=float_coverage,
            details=_number_details(number_format),
            failing_examples=_failures(
                present, lambda v: parse_number(v, number_format) is not None
            ),
        )
        if number_format.decimal == ",":
            proposal.notes.append(
                f"Decimal comma decided from {number_format.evidence} unambiguous value(s)."
            )
        elif number_format.evidence == 0:
            proposal.notes.append("No value showed a decimal separator; assumed '.'.")
        _note_partial(proposal, float_coverage, flags, column)
        return proposal, flags

    if best_date_format is not None and best_date_coverage >= COVERAGE_THRESHOLD:
        return _date_proposal(
            column, present, best_date_format, best_date_coverage, date_coverage, flags
        )

    best_name, best_value = max(alternatives.items(), key=lambda kv: kv[1])

    # Weak identifier evidence: unique, code-like values containing digits, such
    # as 'A1' or 'DK-0042'. It applies only when no real type has meaningful
    # support, so a date column of unique values like '03-04-2021' — which is
    # also code-like — is still read as dates.
    if (
        is_unique
        and n >= 3
        and best_value < 0.5
        and all(_CODE_LIKE_RE.match(t) and any(c.isdigit() for c in t) for t in texts)
        and not all(looks_integer(t, number_format) for t in texts)
    ):
        return (
            _id_proposal(column, leading_zeros, alternatives, ["values are code-like"]),
            flags,
        )

    # A date column with mixed formats stays a date proposal and reports the
    # mix (spec 4.4), rather than collapsing into a shrug.
    if best_name == "date" and best_date_format is not None and best_value > 0.5:
        return _date_proposal(
            column, present, best_date_format, best_date_coverage, date_coverage, flags
        )

    if best_value > 0.5:
        proposal = TypeProposal(
            column=column,
            proposed="ambiguous",
            coverage=best_value,
            ambiguous=True,
            alternatives={k: v for k, v in alternatives.items() if v > 0},
            failing_examples=_failures_for(best_name, present, number_format, best_date_format),
            notes=[
                f"Best candidate {best_name!r} parses only {best_value:.1%} of values, "
                f"below the {COVERAGE_THRESHOLD:.0%} threshold."
            ],
        )
        flags.append(
            Flag(
                code="ambiguous_type",
                severity="warn",
                message=(
                    f"Column {column!r} has no confident type: "
                    f"{best_name} covers {best_value:.1%} of values."
                ),
                column=column,
                stage="ingestion",
                details={"alternatives": alternatives},
            )
        )
        return proposal, flags

    if distinct < n and distinct <= 50 and distinct / n < 0.5:
        return (
            TypeProposal(
                column=column,
                proposed="categorical",
                coverage=1.0,
                details={"n_distinct": distinct},
                notes=[f"{distinct} distinct values over {n} rows."],
            ),
            flags,
        )

    return (
        TypeProposal(
            column=column,
            proposed="text",
            coverage=1.0,
            details={"n_distinct": distinct},
            alternatives={k: v for k, v in alternatives.items() if v > 0},
        ),
        flags,
    )


def _from_declared_type(column: str, declared: str) -> TypeProposal | None:
    """Restate a type the file already declares, rather than re-inferring it."""
    upper = declared.upper()
    for pattern, proposed in _DUCKDB_TYPE_MAP:
        if pattern.match(upper):
            return TypeProposal(
                column=column,
                proposed=proposed,
                coverage=1.0,
                details={"declared_type": declared},
                notes=["Type declared by the file, not inferred."],
            )
    return None  # VARCHAR and friends still need the text heuristics


def _id_proposal(
    column: str, leading_zeros: bool, alternatives: dict[str, float], reasons: list[str]
) -> TypeProposal:
    return TypeProposal(
        column=column,
        proposed="id",
        coverage=1.0,
        details={"leading_zeros": leading_zeros, "unique": True},
        alternatives={k: v for k, v in alternatives.items() if v > 0},
        notes=[f"Unique and {', '.join(reasons)}; kept as text."],
    )


def _date_proposal(
    column: str,
    present: Sequence[Any],
    fmt: str,
    coverage: float,
    all_coverage: dict[str, float],
    flags: list[Flag],
) -> tuple[TypeProposal, list[Flag]]:
    day_month_ambiguous, other = day_month_is_ambiguous(present, fmt)
    proposed: ProposedType = "datetime" if fmt in DATETIME_FORMATS else "date"
    # Below the coverage threshold the proposal still stands, but it is not a
    # confident one: a cast would silently drop whatever did not parse.
    ambiguous = day_month_ambiguous or coverage < COVERAGE_THRESHOLD

    proposal = TypeProposal(
        column=column,
        proposed=proposed,
        coverage=coverage,
        ambiguous=ambiguous,
        details={"date_format": fmt},
        alternatives={k: v for k, v in all_coverage.items() if v >= COVERAGE_THRESHOLD},
        failing_examples=_failures(present, lambda v: parse_date(v, fmt) is not None),
    )

    if day_month_ambiguous:
        proposal.notes.append(
            f"No day above 12, so {fmt} and {other} both fit; day/month order is undecidable."
        )
        flags.append(
            Flag(
                code="ambiguous_date_order",
                severity="warn",
                message=(
                    f"Column {column!r}: no day above 12, so day/month order cannot be "
                    f"determined from the data ({fmt} vs {other})."
                ),
                column=column,
                stage="ingestion",
                details={"candidates": [fmt, other]},
            )
        )
    elif coverage < 1.0:
        matched = int(round(coverage * len(present)))
        proposal.notes.append(f"Mixed formats: {matched}/{len(present)} match {fmt}.")
        flags.append(
            Flag(
                code="mixed_date_formats",
                severity="warn",
                message=f"Column {column!r}: only {coverage:.1%} of values match {fmt}.",
                column=column,
                stage="ingestion",
                details={"coverage": all_coverage},
            )
        )
    return proposal, flags


def _number_details(fmt: NumberFormat, *, thousands_only: bool = False) -> dict[str, Any]:
    details: dict[str, Any] = {}
    if fmt.evidence or not thousands_only:
        details["decimal_separator"] = fmt.decimal
    if fmt.thousands:
        details["thousands_separator"] = fmt.thousands
    return details


def _failures(values: Sequence[Any], accepts) -> list[str]:
    """Up to five values the chosen parser rejected, for the report."""
    failures: list[str] = []
    for value in values:
        if not accepts(value):
            text = str(value)
            if text not in failures:
                failures.append(text)
            if len(failures) == 5:
                break
    return failures


def _failures_for(
    candidate: str, present: Sequence[Any], fmt: NumberFormat, date_format: str | None
) -> list[str]:
    if candidate == "integer":
        return _failures(present, lambda v: looks_integer(v, fmt))
    if candidate == "float":
        return _failures(present, lambda v: parse_number(v, fmt) is not None)
    if candidate == "date" and date_format:
        return _failures(present, lambda v: parse_date(v, date_format) is not None)
    if candidate == "boolean":
        return _failures(present, lambda v: str(v).strip().lower() in BOOLEAN_TOKENS)
    return []


def _note_partial(proposal: TypeProposal, coverage: float, flags: list[Flag], column: str) -> None:
    """A proposal above the threshold but below 100% still loses values on a cast."""
    if coverage < 1.0:
        proposal.notes.append(
            f"{(1 - coverage):.1%} of values would not parse and must be handled explicitly."
        )
        flags.append(
            Flag(
                code="partial_type_coverage",
                severity="warn",
                message=(
                    f"Column {column!r}: {(1 - coverage):.1%} of values do not parse as "
                    f"{proposal.proposed}; casting would drop them unless handled."
                ),
                column=column,
                stage="ingestion",
                details={"coverage": coverage, "examples": proposal.failing_examples},
            )
        )


def sentinel_candidates(values: Sequence[Any]) -> Counter[str]:
    """Count values that look like coded missing markers. Flagged, never acted on."""
    counts: Counter[str] = Counter()
    for value in values:
        if is_missing(value):
            continue
        text = str(value).strip()
        if text in SENTINEL_NUMBERS or text in SENTINEL_DATES:
            counts[text] += 1
    return counts
