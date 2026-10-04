"""The reviewer's contracts: the questions it may ask of the data, and what it replies.

The reviewer judges whether a cleaning is right, not just accounted for. It may
ask questions of the data, but only from a closed vocabulary of aggregate
queries (`Query`), never SQL and never rows. Its final reply (`ReviewOutcome`)
must ground every claim in numbers it was shown; the official verdict that
Stage 3 reads is the `ReviewVerdict` in `contracts/plan.py`, built from it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from pipeline.contracts.common import Strict
from pipeline.contracts.plan import FixRequest

QueryKind = Literal["count", "value_counts", "summary", "group_count", "compare"]
ConditionOp = Literal["is_null", "is_not_null", "equals", "not_equals", "greater_than", "less_than"]

MAX_QUERIES_PER_REPLY = 8
MAX_CONDITIONS = 4
MAX_QUERY_ROWS = 20


class Condition(Strict):
    """One filter: a column, a comparison, a value. Combined with AND."""

    column: str
    op: ConditionOp
    value: Any = None

    @model_validator(mode="after")
    def _comparison_needs_value(self) -> Condition:
        if self.op not in {"is_null", "is_not_null"} and self.value is None:
            raise ValueError(f"condition {self.op!r} on {self.column!r} needs a value")
        return self


class Query(Strict):
    """One aggregate question about the raw or the clean table."""

    id: str = Field(description="Short id, e.g. 'q1'.")
    kind: QueryKind
    table: Literal["raw", "clean"] = "clean"
    column: str | None = Field(
        default=None, description="Required for value_counts, summary, group_count and compare."
    )
    bucket: Literal["year", "month"] | None = Field(
        default=None, description="group_count on a date column: group by year or month."
    )
    where: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)
    limit: int = Field(default=10, ge=1, le=MAX_QUERY_ROWS)
    purpose: str = Field(min_length=1, description="What this question is meant to settle.")

    @model_validator(mode="after")
    def _column_where_needed(self) -> Query:
        if self.kind != "count" and not self.column:
            raise ValueError(f"query {self.id}: kind {self.kind!r} needs a column")
        if self.bucket and self.kind != "group_count":
            raise ValueError(f"query {self.id}: bucket only applies to group_count")
        return self


class Verification(Strict):
    """Something the reviewer checked and found right, with the numbers that show it."""

    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Numbers quoted from what it was shown.")


class ReviewFinding(Strict):
    """Something wrong or doubtful about the cleaning."""

    step: int | None = Field(default=None, description="The plan step it concerns, if any.")
    column: str | None = None
    severity: Literal["warn", "error"] = Field(
        description="'error' when the cleaning did the wrong thing; 'warn' when it deserves a look."
    )
    problem: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Numbers quoted from what it was shown.")
    fix: FixRequest | None = Field(
        default=None, description="What the coder should change; only for problems ops can fix."
    )


class Ambiguity(Strict):
    """A question about the data that only the user can settle. Reported, never decided."""

    column: str | None = None
    question: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class ReviewOutcome(Strict):
    accepted: bool = Field(description="False exactly when a finding carries a fix.")
    summary: str = Field(min_length=1)
    verified: list[Verification] = Field(default_factory=list)
    findings: list[ReviewFinding] = Field(default_factory=list)
    ambiguities: list[Ambiguity] = Field(default_factory=list)


class ReviewReply(Strict):
    """Either more questions, or the verdict. Never both."""

    queries: list[Query] | None = Field(default=None, max_length=MAX_QUERIES_PER_REPLY)
    verdict: ReviewOutcome | None = None

    @model_validator(mode="after")
    def _one_of(self) -> ReviewReply:
        if (self.queries is None) == (self.verdict is None):
            raise ValueError("reply with either 'queries' or 'verdict', not both and not neither")
        if self.queries is not None and not self.queries:
            raise ValueError("'queries' must contain at least one query")
        return self
