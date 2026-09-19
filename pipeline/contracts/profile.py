"""The dataset profile: what Stage 1 hands to every later stage.

Two audiences, one source. `DatasetProfile` is the full, logged detail; the
compact `profile_llm.json` view is derived from it and is the *only* data
description an agent ever sees. No raw row ever appears in either.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field

from pipeline.contracts.common import Flag, Strict, utcnow

ProposedType = Literal[
    "id",
    "integer",
    "float",
    "boolean",
    "date",
    "datetime",
    "categorical",
    "text",
    "empty",
    "ambiguous",
]

PIIKind = Literal["email", "cpr", "phone", "iban", "name", "address", "ip"]
PIILevel = Literal["none", "possible", "likely"]


class ValueCount(Strict):
    """One of a column's most frequent values, masked if the column holds PII."""

    value: str
    count: int
    share: float


class NumericSummary(Strict):
    """Distribution summary for columns that parse as numbers."""

    min: float
    q1: float
    median: float
    q3: float
    max: float
    mean: float
    std: float | None = None
    n_outliers: int = Field(default=0, description="Values beyond 3x IQR from the quartiles.")
    outlier_bounds: tuple[float, float] | None = None


class LengthSummary(Strict):
    """Character-length summary for text columns, a cheap corruption signal."""

    min: int
    max: int
    mean: float


class SentinelCandidate(Strict):
    """A value that looks like a coded missing marker (999, 1900-01-01, ...).

    Flagged only: whether it *means* missing is a judgment call for the brief
    or the cleaning agent, never for the profiler.
    """

    value: str
    count: int
    share: float
    reason: str


class PIIFinding(Strict):
    """Detected personal data in one column. Flag and mask; never auto-drop."""

    kind: PIIKind
    level: PIILevel
    n_matches: int
    share: float
    detector: str = Field(description="e.g. 'regex:cpr' or 'presidio:PERSON'.")
    matched_column_name: bool = Field(
        default=False, description="True when the column *name* was the hint."
    )


class TypeProposal(Strict):
    """A proposed type for one column. Nothing is converted at this stage.

    `coverage` is the share of non-null values the candidate parser accepted;
    below the threshold the proposal is `ambiguous` with failing examples so the
    cleaning agent can see exactly what did not parse.
    """

    column: str
    proposed: ProposedType
    coverage: float = Field(ge=0.0, le=1.0)
    ambiguous: bool = False
    failing_examples: list[str] = Field(default_factory=list, max_length=5)
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g. {'decimal_separator': ',', 'date_format': '%d-%m-%Y'}.",
    )
    alternatives: dict[str, float] = Field(
        default_factory=dict, description="Other candidate types and their coverage."
    )
    notes: list[str] = Field(default_factory=list)


class ColumnProfile(Strict):
    """Everything Stage 1 observed about one column."""

    name: str
    position: int
    raw_dtype: str = Field(description="How the value is stored now, usually 'text'.")

    n_rows: int
    n_missing: int = Field(description="Nulls plus recognised null tokens.")
    missing_share: float
    null_tokens: dict[str, int] = Field(
        default_factory=dict, description="Which null token appeared how often, e.g. {'NA': 41}."
    )
    n_distinct: int
    distinct_share: float
    n_blank: int = Field(default=0, description="Empty or whitespace-only strings.")

    top_values: list[ValueCount] = Field(default_factory=list)
    numeric: NumericSummary | None = None
    length: LengthSummary | None = None

    type_proposal: TypeProposal
    sentinel_candidates: list[SentinelCandidate] = Field(default_factory=list)
    pii: list[PIIFinding] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)

    @property
    def pii_level(self) -> PIILevel:
        """The strongest PII signal on this column; drives masking."""
        levels = [f.level for f in self.pii]
        if "likely" in levels:
            return "likely"
        if "possible" in levels:
            return "possible"
        return "none"


class DatasetProfile(Strict):
    """The full profile of one table. Logged in full, summarised for the LLM."""

    table: str
    n_rows: int
    n_columns: int
    n_duplicate_rows: int = 0
    candidate_keys: list[str] = Field(
        default_factory=list, description="Columns that are unique and non-null."
    )
    columns: list[ColumnProfile] = Field(default_factory=list)

    sampled: bool = Field(
        default=False, description="True when expensive stats used a sample, per spec 4.6."
    )
    sample_rows: int | None = None
    profiled_at: datetime = Field(default_factory=utcnow)
    flags: list[Flag] = Field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]
