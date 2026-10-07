"""Benchmark scenarios: realistic mixes of planted problems.

Each scenario is the kind of mess one kind of source produces. Order matters
for two reasons: a breaker never touches a cell an earlier one changed, so
codes and tokens are planted before a format change sweeps the rest of a
column; and duplicates come last, so copied rows carry every other problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Plant:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    plants: tuple[Plant, ...]
    delimiter: str = ","
    guidance: str = (
        "Codes documented as meaning unknown must become missing. Never fill in missing values. "
        "Keep real values even when they are extreme, but flag values that are impossible."
    )


SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario
    for scenario in (
        Scenario(
            name="danish_export",
            description=(
                "An export from a Danish core system: amounts with decimal commas, day-first "
                "dates, 'NA' for missing ages and a documented 999 code for unknown income, "
                "separated by semicolons."
            ),
            plants=(
                Plant("null_tokens", {"column": "alder", "share": 0.06, "tokens": ("NA",)}),
                Plant("sentinel", {"column": "indkomst", "code": "999", "share": 0.04}),
                Plant("decimal_comma", {"column": "beloeb"}),
                Plant("decimal_comma", {"column": "indkomst"}),
                Plant("date_format", {"column": "oprettet", "fmt": "%d-%m-%Y"}),
            ),
            delimiter=";",
        ),
        Scenario(
            name="manual_entry",
            description=(
                "Typed in by hand: stray spaces and mixed case in the regions, 'kr' after the "
                "fee, '-' and '?' for unknown ages, and some applications entered twice."
            ),
            plants=(
                Plant("null_tokens", {"column": "alder", "share": 0.06, "tokens": ("-", "?")}),
                Plant("whitespace_case", {"column": "region", "share": 0.15}),
                Plant("currency_suffix", {"column": "gebyr", "suffix": " kr"}),
                Plant("duplicates", {"share": 0.03}),
            ),
        ),
        Scenario(
            name="mixed",
            description=(
                "A bit of everything at lower rates, plus fees a thousand times too large and a "
                "column that gives the answer away."
            ),
            plants=(
                Plant("null_tokens", {"column": "indkomst", "share": 0.04, "tokens": ("NA",)}),
                Plant("sentinel", {"column": "alder", "code": "999", "share": 0.03}),
                Plant("outliers", {"column": "gebyr", "share": 0.02, "factor": 1000}),
                Plant("whitespace_case", {"column": "region", "share": 0.08}),
                Plant("date_format", {"column": "oprettet", "fmt": "%m/%d/%Y"}),
                Plant("leaky_column", {"name": "sag_lukket"}),
                Plant("duplicates", {"share": 0.02}),
            ),
        ),
    )
}
