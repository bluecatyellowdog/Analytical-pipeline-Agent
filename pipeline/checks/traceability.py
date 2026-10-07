"""Number traceability: every number in AI-written text must be one the tools logged.

Spec Gate 4: 'every number in the narrative matches a logged value'. Numbers
are read with the precision they were written in, so rounding is fine (74%
matches a logged 0.742), but a number the AI worked out itself (a difference,
a ratio, a sum) matches nothing and is reported. Numbers inside logged labels
count too ('5,308-86,851', 'per 100,000'). Whole numbers up to ten are
ordinary words ('3 classes'), and the 95% interval and the usual
significance levels are conventions, not results.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

SMALL_COUNT = 10
"""Whole numbers up to this ('3 classes', 'the 2 groups') are ordinary words, not results."""
CONVENTIONS = (0.95, 0.05, 0.01, 0.001)
"""The 95% interval and the usual significance levels."""

_NUMBER = re.compile(
    r"(?<![\w.])"  # not inside a word, an id or a longer number
    r"(?P<prefix>[x×])?"  # x2.99: an odds ratio or a multiplier
    r"(?P<num>[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<suffix>\s?%|[x×](?![A-Za-z]))?"  # 74.2% or 3x
    r"(?!\w)"  # not followed by letters: 2nd, 3rd, art_12a
)


@dataclass(frozen=True)
class Number:
    """One number as written: its text, its value (a fraction for a percentage) and kind."""

    raw: str
    value: float
    kind: str
    """number | percent | multiplier | threshold (the 0.001 in p<0.001)."""
    decimals: int

    @property
    def tolerance(self) -> float:
        """Half a unit of the last digit written: 74.2% covers 0.7415 to 0.7425."""
        half = 10.0**-self.decimals / 2
        return (half / 100 if self.kind == "percent" else half) + 1e-9


def numbers_in(text: str) -> list[Number]:
    found = []
    for match in _NUMBER.finditer(text):
        digits = match["num"].replace(",", "").replace("−", "-")
        value = float(digits)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        suffix = (match["suffix"] or "").strip()
        if suffix == "%":
            kind, value = "percent", value / 100
        elif match["prefix"] or suffix in ("x", "×"):
            kind = "multiplier"
        elif text[: match.start()].rstrip()[-1:] in ("<", ">", "≤", "≥"):
            kind = "threshold"
        else:
            kind = "number"
        found.append(Number(match.group(0), value, kind, decimals))
    return found


def values_of(*sources: Any) -> list[float]:
    """Every number in logged data: values, and numbers written inside labels and text."""
    values: list[float] = []

    def walk(item: Any) -> None:
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, int | float):
            if math.isfinite(item):
                values.append(float(item))
        elif isinstance(item, str):
            values.extend(number.value for number in numbers_in(item))
        elif isinstance(item, dict):
            for value in item.values():
                walk(value)
        elif isinstance(item, list | tuple | set):
            for value in item:
                walk(value)

    for source in sources:
        walk(source)
    return values


def untraced(text: str, values: Iterable[float], *, allowed: Iterable[float] = ()) -> list[str]:
    """The numbers in `text` that match no logged value, as written."""
    logged = [abs(v) for v in values]
    as_percent = [v * 100 for v in logged if v <= 1]  # a rate written without its % sign
    accepted = [*CONVENTIONS, *(abs(v) for v in allowed)]
    missing = []
    for number in numbers_in(text):
        if number.kind == "number" and number.decimals == 0 and 0 <= number.value <= SMALL_COUNT:
            continue
        if _matches(number, accepted) or _matches(number, logged):
            continue
        if number.kind == "number" and _matches(number, as_percent):
            continue
        missing.append(number.raw)
    return missing


def _matches(number: Number, values: list[float]) -> bool:
    target = abs(number.value)
    return any(abs(target - value) <= number.tolerance for value in values)
