"""Stage 0 — the context brief: domain knowledge the system cannot infer.

Every checkable statement the user makes is treated as a hypothesis and verified
in Stage 1 (see `pipeline/ingestion/claims.py`), never trusted outright.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator

from pipeline.contracts.common import Strict

ClaimKind = Literal[
    "column_exists",
    "column_unique",
    "column_not_null",
    "binary_column",
    "value_range",
    "date_range",
    "row_count",
    "allowed_values",
]
"""The claim kinds Stage 1 knows how to test. Anything else stays prose."""

ClaimStatus = Literal["pass", "fail", "not_checkable"]


class Claim(Strict):
    """One checkable statement from the brief, e.g. 'cust_id is unique'."""

    kind: ClaimKind
    column: str | None = None
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Kind-specific arguments, e.g. {'min': 18, 'max': 120}.",
    )
    text: str | None = Field(default=None, description="The user's original phrasing.")
    source: Literal["brief", "derived"] = Field(
        default="brief",
        description="'derived' when the pipeline inferred it from a structured field.",
    )


class ClaimResult(Strict):
    """The verdict on one claim after Stage 1 tested it against the raw data."""

    claim: Claim
    status: ClaimStatus
    message: str
    observed: dict[str, Any] = Field(
        default_factory=dict,
        description="What the data actually showed, injected into the report.",
    )


class ColumnNote(Strict):
    """What the user knows about one column that the values do not show."""

    name: str
    description: str | None = None
    unit: str | None = None
    allowed_values: list[str] | None = None
    known_issues: str | None = None


class ContextBrief(Strict):
    """Stage 0 output. Stored in the run manifest and passed to every agent."""

    data_file: str | None = Field(
        default=None,
        description="The data file this brief is about, relative to the project folder, e.g. "
        "data/raw/loans.csv. With it, `pipeline.run <brief>` needs nothing else.",
    )
    sheet: str | None = Field(default=None, description="The Excel sheet, for an Excel file.")
    title: str | None = Field(
        default=None,
        description="The report's title, e.g. 'What drives customer churn?'. Default: the file.",
    )
    description: str = Field(description="What the rows are and where they come from.")
    goal: str = Field(description="The question or decision the analysis should serve.")
    target: str | None = Field(default=None, description="Target column, if known.")
    unit_of_observation: str | None = Field(
        default=None, description="What one row represents, e.g. 'one loan application'."
    )
    id_column: str | None = None
    column_notes: list[ColumnNote] = Field(default_factory=list)
    known_issues: list[str] = Field(default_factory=list)
    focus_areas: list[str] = Field(
        default_factory=list, description="Segments or themes the user cares about."
    )
    excluded_variables: list[str] = Field(
        default_factory=list,
        description="Columns that must not be used, e.g. legally sensitive features.",
    )
    claims: list[Claim] = Field(
        default_factory=list,
        description="Explicit checkable claims; more are derived in Stage 1.",
    )
    notes: str | None = Field(default=None, description="Anything else, free text.")
    guidance: str | None = Field(
        default=None,
        description=(
            "How unclear cases should be handled, in plain words, e.g. '999 always means "
            "unknown; never fill in missing values'. Read when judgment calls are decided."
        ),
    )
    judgment_calls: Literal["decide", "ask"] = Field(
        default="decide",
        description=(
            "'decide': the AI settles each judgment call from this brief, visibly and "
            "reversibly; 'ask': every call waits for the user in decisions.yaml."
        ),
    )

    @field_validator("description", "goal")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    def note_for(self, column: str) -> ColumnNote | None:
        """The user's note for one column, if they wrote one."""
        return next((n for n in self.column_notes if n.name == column), None)


def load_context_brief(path: str | Path) -> ContextBrief:
    """Read a context brief from YAML.

    The brief is human-written, so a bad field must produce a clear validation
    error here rather than a confusing failure three stages later.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"{path} is empty")
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return ContextBrief.model_validate(raw)
