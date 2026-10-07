"""A clean, reproducible loan table: ground truth the benchmark can break and score against.

Generated from a seed, so every benchmark run can be repeated exactly and the
right value of every cell is known. Danish column names and regions (with æ and
ø) keep it close to the data the pipeline is built for. The target depends on
the loan-to-income ratio and age, so the later analysis stages have a real
signal to find.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIONS = ("Hovedstaden", "Sjælland", "Syddanmark", "Midtjylland", "Nordjylland")


@dataclass(frozen=True)
class CleanDataset:
    name: str
    frame: pd.DataFrame
    """The clean values, typed: int, float, datetime.date, str."""
    kinds: dict[str, str]
    """Column -> id | integer | float | date | category | boolean."""
    target: str
    id_column: str
    description: str
    goal: str
    column_notes: list[dict[str, str]] = field(default_factory=list)


def make_loans(rows: int = 500, seed: int = 0) -> CleanDataset:
    """One row per loan application, every value valid and present."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 81, rows)
    income = np.round(rng.lognormal(mean=np.log(420_000), sigma=0.4, size=rows), 2)
    amount = np.round(rng.uniform(5_000, 400_000, rows), 2)
    start = dt.date(2019, 1, 1)
    span = (dt.date(2024, 12, 31) - start).days
    created = [start + dt.timedelta(days=int(d)) for d in rng.integers(0, span + 1, rows)]
    risk = -3.0 + 4.0 * (amount / income) - 0.02 * (age - 40)
    defaulted = (rng.random(rows) < 1.0 / (1.0 + np.exp(-risk))).astype(int)

    frame = pd.DataFrame(
        {
            "kunde_id": pd.Series([f"{i:05d}" for i in range(1, rows + 1)], dtype=object),
            "alder": age.astype(int),
            "indkomst": income,
            "beloeb": amount,
            "region": pd.Series(rng.choice(REGIONS, rows).tolist(), dtype=object),
            "oprettet": pd.Series(created, dtype=object),
            "gebyr": rng.integers(0, 2001, rows).astype(int),
            "misligholdt": defaulted,
        }
    )
    return CleanDataset(
        name="loans",
        frame=frame,
        kinds={
            "kunde_id": "id",
            "alder": "integer",
            "indkomst": "float",
            "beloeb": "float",
            "region": "category",
            "oprettet": "date",
            "gebyr": "integer",
            "misligholdt": "boolean",
        },
        target="misligholdt",
        id_column="kunde_id",
        description=(
            "One row per consumer loan application at a Danish bank, exported from the core "
            "system. Amounts, income and fees are in DKK."
        ),
        goal="Understand which applicant characteristics are associated with default.",
        column_notes=[
            {"name": "kunde_id", "description": "Customer number. Has leading zeros."},
            {"name": "alder", "description": "Applicant age in years."},
            {"name": "indkomst", "description": "Yearly income in DKK."},
            {"name": "beloeb", "description": "Loan amount in DKK."},
            {"name": "gebyr", "description": "Application fee in DKK."},
            {"name": "misligholdt", "description": "1 if the loan defaulted, 0 if not."},
        ],
    )


def to_text(value: object, kind: str) -> str:
    """How a clean value is written in a plain CSV export."""
    if kind == "integer" or kind == "boolean":
        return str(int(value))
    if kind == "float":
        return f"{float(value):.2f}"
    if kind == "date":
        return value.isoformat()
    return str(value)
