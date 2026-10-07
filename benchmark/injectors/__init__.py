"""Breakers: plant known problems in clean data, and write down the answer key.

Each breaker changes one kind of thing, in the text form a file would hold,
and records an `Injection`: which rows of which column it touched and what the
cleaned value should be there ("original": the clean value, "missing": it can
only become missing, "removed": the row is a duplicate and should go, "flag":
a judgment call the pipeline should raise, not necessarily fix). The scorer
compares the pipeline's output with the clean data using this key.

Breakers never touch a cell another breaker already changed, so every planted
problem is scored on its own.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from benchmark.datasets.synthetic import CleanDataset, to_text

Expected = Literal["original", "missing", "removed", "flag"]


@dataclass
class Injection:
    kind: str
    column: str | None
    """None for table-level problems such as duplicate rows."""
    rows: list[int]
    """Positions of the affected data rows in the broken file (0 = first data row)."""
    expected: Expected
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Corrupted:
    table: pd.DataFrame
    """Every cell as text, exactly as the broken file will hold it."""
    origin: list[int]
    """For each row of the table, the row of the clean data it came from."""
    injections: list[Injection] = field(default_factory=list)
    column_notes: list[dict[str, str]] = field(default_factory=list)
    """What a user would write in the brief about the planted problems they know of."""

    @classmethod
    def from_clean(cls, dataset: CleanDataset) -> Corrupted:
        table = pd.DataFrame(
            {
                column: pd.Series(
                    [to_text(v, dataset.kinds[column]) for v in dataset.frame[column]],
                    dtype=object,
                )
                for column in dataset.frame.columns
            }
        )
        return cls(table=table, origin=list(range(len(table))))

    def taken(self, column: str) -> set[int]:
        return {row for i in self.injections if i.column == column for row in i.rows}

    def pick(self, rng: np.random.Generator, column: str, share: float) -> list[int]:
        candidates = [row for row in range(len(self.table)) if row not in self.taken(column)]
        count = min(len(candidates), max(1, round(share * len(candidates))))
        return sorted(int(r) for r in rng.choice(candidates, size=count, replace=False))

    def free_rows(self, column: str) -> list[int]:
        return [row for row in range(len(self.table)) if row not in self.taken(column)]

    def write_csv(self, path: Path, delimiter: str = ",") -> Path:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(list(self.table.columns))
            writer.writerows(self.table.astype(str).values.tolist())
        return path


# -- the breakers -----------------------------------------------------------


def _null_tokens(c, data, rng, *, column, share=0.08, tokens=("NA", "-")) -> Injection:
    rows = c.pick(rng, column, share)
    for n, row in enumerate(rows):
        c.table.loc[row, column] = tokens[n % len(tokens)]
    return Injection("null_tokens", column, rows, "missing", {"tokens": list(tokens)})


def _decimal_comma(c, data, rng, *, column) -> Injection:
    rows = c.free_rows(column)
    for row in rows:
        whole, cents = f"{float(data.frame.loc[c.origin[row], column]):.2f}".split(".")
        c.table.loc[row, column] = f"{int(whole):,}".replace(",", ".") + "," + cents
    return Injection("decimal_comma", column, rows, "original")


def _date_format(c, data, rng, *, column, fmt="%d-%m-%Y") -> Injection:
    rows = c.free_rows(column)
    for row in rows:
        c.table.loc[row, column] = data.frame.loc[c.origin[row], column].strftime(fmt)
    return Injection("date_format", column, rows, "original", {"format": fmt})


def _sentinel(c, data, rng, *, column, code="999", share=0.04, documented=True) -> Injection:
    rows = c.pick(rng, column, share)
    for row in rows:
        c.table.loc[row, column] = code
    if documented:
        c.column_notes.append(
            {"name": column, "known_issues": f"{code} was used for unknown values."}
        )
    expected: Expected = "missing" if documented else "flag"
    return Injection("sentinel", column, rows, expected, {"code": code, "documented": documented})


def _whitespace_case(c, data, rng, *, column, share=0.15) -> Injection:
    variants = (
        lambda v: f" {v}",
        lambda v: f"{v}  ",
        lambda v: v.upper(),
        lambda v: v.lower(),
    )
    rows = c.pick(rng, column, share)
    for row in rows:
        value = str(data.frame.loc[c.origin[row], column])
        c.table.loc[row, column] = variants[int(rng.integers(len(variants)))](value)
    return Injection("whitespace_case", column, rows, "original")


def _currency_suffix(c, data, rng, *, column, suffix=" kr") -> Injection:
    rows = c.free_rows(column)
    for row in rows:
        c.table.loc[row, column] = c.table.loc[row, column] + suffix
    return Injection("currency_suffix", column, rows, "original", {"suffix": suffix})


def _outliers(c, data, rng, *, column, share=0.01, factor=1000) -> Injection:
    rows = c.pick(rng, column, share)
    kind = data.kinds[column]
    for row in rows:
        value = data.frame.loc[c.origin[row], column] * factor
        c.table.loc[row, column] = to_text(value, kind)
    return Injection("outliers", column, rows, "flag", {"factor": factor})


def _duplicates(c, data, rng, *, share=0.03) -> Injection:
    originals = len(data.frame)
    count = max(1, round(share * originals))
    sources = sorted(int(r) for r in rng.choice(originals, size=count, replace=False))
    copies = c.table.iloc[sources].copy()
    start = len(c.table)
    c.table = pd.concat([c.table, copies], ignore_index=True)
    c.origin += [c.origin[s] for s in sources]
    rows = list(range(start, start + count))
    return Injection("duplicates", None, rows, "removed", {"copies_of": sources})


def _leaky_column(c, data, rng, *, name="sag_lukket", noise=0.03) -> Injection:
    target = data.frame[data.target]
    values = []
    for row in range(len(c.table)):
        value = int(target.iloc[c.origin[row]])
        values.append(str(1 - value if rng.random() < noise else value))
    c.table[name] = pd.Series(values, dtype=object)
    return Injection(
        "leaky_column", name, list(range(len(c.table))), "flag", {"copies": data.target}
    )


BREAKERS: dict[str, Callable[..., Injection]] = {
    "null_tokens": _null_tokens,
    "decimal_comma": _decimal_comma,
    "date_format": _date_format,
    "sentinel": _sentinel,
    "whitespace_case": _whitespace_case,
    "currency_suffix": _currency_suffix,
    "outliers": _outliers,
    "duplicates": _duplicates,
    "leaky_column": _leaky_column,
}


def inject(
    corrupted: Corrupted, data: CleanDataset, kind: str, *, seed: int, **params: Any
) -> Injection:
    """Plant one kind of problem and add it to the answer key."""
    if any(i.kind == "duplicates" for i in corrupted.injections):
        raise ValueError("plant duplicates last: rows copied earlier would be scored as originals")
    rng = np.random.default_rng(seed)
    injection = BREAKERS[kind](corrupted, data, rng, **params)
    corrupted.injections.append(injection)
    return injection


__all__ = ["BREAKERS", "Corrupted", "Injection", "inject"]
