"""Scoring one cleaning against the answer key: found, fixed, false alarms, damage.

- **Found:** the diagnosis names the planted problem (an issue of the right kind,
  or a judgment call about it).
- **Fixed:** the cleaned cell holds the right value: the clean original, or
  missing where the original cannot be recovered; a duplicate row is gone.
- **False alarms:** problems the diagnosis raised where nothing was planted.
- **Damage:** cells nobody broke that the cleaning changed, and good rows it
  dropped. Leaving text as text is not damage; changing what a value means is.

Values are compared by meaning, not spelling: 31, 31.0 and "31" are the same
age; 1 and True are the same target; "nord" may stand for "Nord" as long as
every Nord row became the same value.
"""

from __future__ import annotations

import datetime as dt
import math
import numbers
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from benchmark.datasets.synthetic import CleanDataset
from benchmark.injectors import Corrupted, Injection
from pipeline.contracts.diagnosis import Diagnosis, Issue, JudgmentCall

MAX_EXAMPLES = 10
PROBLEM_KINDS = {
    "missing_tokens",
    "sentinel",
    "duplicates",
    "whitespace",
    "case",
    "categories",
    "outliers",
    "invalid_values",
}
"""Issue kinds that claim a problem. Conversions ('type', formats) are routine, never alarms."""

_ISSUE_KINDS = {
    "null_tokens": {"missing_tokens"},
    "decimal_comma": {"decimal_format"},
    "date_format": {"date_format"},
    "sentinel": {"sentinel"},
    "whitespace_case": {"whitespace", "case", "categories"},
    "currency_suffix": {"invalid_values"},
    "outliers": {"outliers"},
    "leaky_column": {"other", "invalid_values"},
}
_ISSUE_OPS = {
    "null_tokens": {"tokens_to_null"},
    "decimal_comma": {"normalize_decimal"},
    "date_format": {"parse_dates"},
    "sentinel": {"replace_sentinel"},
    "whitespace_case": {"trim_whitespace", "normalize_case", "standardize_categories"},
    "currency_suffix": {"strip_currency"},
    "outliers": {"clip_outliers"},
    "leaky_column": set(),
}
_WORDS = {
    "null_tokens": ("missing", "null", "token"),
    "decimal_comma": ("decimal", "comma", "format"),
    "date_format": ("date", "format"),
    "whitespace_case": ("case", "spelling", "space", "variant"),
    "currency_suffix": (" kr", "kr.", "currency", "suffix"),
    "outliers": ("outlier", "extreme", "implausib", "unrealistic", "too large", "1000"),
    "leaky_column": ("leak", "target", "outcome", "after the"),
}


@dataclass
class InjectionScore:
    kind: str
    column: str | None
    expected: str
    cells: int
    found: bool
    fixed: int | None
    """None for problems that only have to be raised (judgment calls)."""
    wrong_examples: list[str] = field(default_factory=list)

    @property
    def fix_rate(self) -> float | None:
        if self.fixed is None:
            return None
        return self.fixed / self.cells if self.cells else 1.0


@dataclass
class CaseScore:
    injections: list[InjectionScore]
    clean_available: bool
    damaged_cells: int = 0
    damaged_examples: list[str] = field(default_factory=list)
    wrongly_dropped_rows: int = 0
    false_alarms: list[str] = field(default_factory=list)
    checked_cells: int = 0

    @property
    def detection_rate(self) -> float:
        return (
            sum(s.found for s in self.injections) / len(self.injections) if self.injections else 1.0
        )

    @property
    def repair_rate(self) -> float:
        fixable = [s for s in self.injections if s.fixed is not None]
        cells = sum(s.cells for s in fixable)
        return sum(s.fixed for s in fixable) / cells if cells else 1.0


def score_case(
    data: CleanDataset,
    corrupted: Corrupted,
    clean_rows: list[dict[str, Any]] | None,
    diagnosis: Diagnosis | None,
    *,
    renames: dict[str, str] | None = None,
) -> CaseScore:
    """Score one pipeline run on one broken file."""
    renames = renames or {}
    by_position = {int(r["_source_line"]) - 2: r for r in clean_rows or []}
    canonical = _canonical_categories(data, corrupted, by_position, renames)
    items: list[Issue | JudgmentCall] = (
        [*diagnosis.issues, *diagnosis.judgment_calls] if diagnosis else []
    )

    scores = [
        _score_injection(injection, data, corrupted, by_position, items, canonical, renames)
        for injection in corrupted.injections
    ]
    result = CaseScore(injections=scores, clean_available=clean_rows is not None)
    result.false_alarms = [
        _describe(item)
        for item in items
        if (not isinstance(item, Issue) or item.kind in PROBLEM_KINDS)
        and not any(_matches(item, injection) for injection in corrupted.injections)
    ]
    if clean_rows is not None:
        _score_damage(result, data, corrupted, by_position, renames)
    return result


def _score_injection(injection, data, corrupted, by_position, items, canonical, renames):
    found = any(_matches(item, injection) for item in items)
    score = InjectionScore(
        kind=injection.kind,
        column=injection.column,
        expected=injection.expected,
        cells=len(injection.rows),
        found=found,
        fixed=None,
    )
    if injection.expected == "flag":
        return score
    if injection.expected == "removed":
        sources = injection.detail.get("copies_of", [])
        score.fixed = sum(
            1
            for row, source in zip(injection.rows, sources, strict=False)
            if row not in by_position and source in by_position
        )
        return score

    column = renames.get(injection.column, injection.column)
    kind = data.kinds[injection.column]
    fixed = 0
    for row in injection.rows:
        if row not in by_position:
            continue
        value = by_position[row].get(column)
        original = data.frame.loc[corrupted.origin[row], injection.column]
        if injection.expected == "missing":
            ok = _missing(value)
        elif kind == "category":
            ok = value == canonical.get(injection.column, {}).get(original, original)
        else:
            ok = same(original, value, kind)
        if ok:
            fixed += 1
        elif len(score.wrong_examples) < MAX_EXAMPLES:
            shown = "missing" if injection.expected == "missing" else repr(original)
            score.wrong_examples.append(
                f"line {row + 2}: expected {shown}, got {value!r} "
                f"(file had {corrupted.table.loc[row, injection.column]!r})"
            )
    score.fixed = fixed
    return score


def _score_damage(result, data, corrupted, by_position, renames) -> None:
    injected = {(row, i.column) for i in corrupted.injections if i.column for row in i.rows}
    for position in range(len(data.frame)):
        row = by_position.get(position)
        if row is None:
            result.wrongly_dropped_rows += 1
            continue
        for column, kind in data.kinds.items():
            if (position, column) in injected:
                continue
            result.checked_cells += 1
            name = renames.get(column, column)
            original = data.frame.loc[position, column]
            value = row.get(name, _ABSENT)
            if value is not _ABSENT and same(original, value, kind):
                continue
            result.damaged_cells += 1
            if len(result.damaged_examples) < MAX_EXAMPLES:
                shown = "column removed" if value is _ABSENT else repr(value)
                result.damaged_examples.append(
                    f"line {position + 2} {column}: {original!r} became {shown}"
                )


def _canonical_categories(data, corrupted, by_position, renames) -> dict[str, dict[Any, Any]]:
    """For each category column, what each original value became where nobody broke it."""
    injected = {(row, i.column) for i in corrupted.injections if i.column for row in i.rows}
    result: dict[str, dict[Any, Any]] = {}
    for column, kind in data.kinds.items():
        if kind != "category":
            continue
        seen: dict[Any, Counter] = {}
        for position, row in by_position.items():
            if position >= len(data.frame) or (position, column) in injected:
                continue
            original = data.frame.loc[corrupted.origin[position], column]
            seen.setdefault(original, Counter())[row.get(renames.get(column, column))] += 1
        result[column] = {orig: counts.most_common(1)[0][0] for orig, counts in seen.items()}
    return result


# -- matching the diagnosis to the answer key --------------------------------


def _matches(item: Issue | JudgmentCall, injection: Injection) -> bool:
    is_issue = isinstance(item, Issue)
    text = " ".join(
        str(part or "")
        for part in (
            getattr(item, "evidence", ""),
            getattr(item, "fix", ""),
            getattr(item, "question", ""),
            getattr(item, "recommendation", ""),
        )
    ).lower()
    op = getattr(item, "suggested_op", None)

    if injection.kind == "duplicates":
        if is_issue:
            return item.kind == "duplicates" or op == "drop_duplicates"
        return "duplicate" in text
    if item.column != injection.column:
        return False

    words = list(_WORDS.get(injection.kind, ()))
    if injection.kind == "sentinel":
        words += [str(injection.detail.get("code", "")), "sentinel", "code"]
    hinted = any(word and word in text for word in words)
    if not is_issue:
        return hinted
    return (
        item.kind in _ISSUE_KINDS.get(injection.kind, set())
        or op in _ISSUE_OPS.get(injection.kind, set())
        or (injection.kind in {"currency_suffix", "leaky_column", "sentinel"} and hinted)
    )


def _describe(item: Issue | JudgmentCall) -> str:
    if isinstance(item, Issue):
        return f"{item.id} ({item.kind}) on {item.column or 'the table'}: {item.fix}"
    return f"{item.id} on {item.column or 'the table'}: {item.question}"


# -- comparing values by meaning ---------------------------------------------

_ABSENT = object()


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(isinstance(value, float) and math.isnan(value))
    except TypeError:
        return False


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, numbers.Number):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _truth(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Number):
        return int(value) if float(value) in (0.0, 1.0) else None
    if isinstance(value, str):
        return {"1": 1, "0": 0, "true": 1, "false": 0}.get(value.strip().lower())
    return None


def _date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().date()
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def same(original: Any, value: Any, kind: str) -> bool:
    """Do these two values mean the same thing in a column of this kind?"""
    if _missing(value) or _missing(original):
        return _missing(value) and _missing(original)
    if kind in ("integer", "float"):
        number = _number(value)
        tolerance = 0.005 if kind == "float" else 1e-9
        return number is not None and abs(number - float(original)) <= tolerance
    if kind == "boolean":
        return _truth(value) == int(original)
    if kind == "date":
        return _date(value) == original
    if kind == "id":
        return str(value).strip() == str(original)
    return str(value).strip().casefold() == str(original).strip().casefold()


__all__ = ["CaseScore", "InjectionScore", "same", "score_case"]
