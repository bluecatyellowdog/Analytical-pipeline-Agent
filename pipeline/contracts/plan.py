"""Stage 3 outputs: the review verdict and the analysis plan.

The plan is the contract the analysis agent executes. It is deliberately narrow:
every column must exist, every method must be in the tool registry, and the
question count is capped so plans stay executable (spec 11, 'overambitious plans').
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from pipeline.contracts.common import Flag, Strict, utcnow

TaskType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "none",
]

QuestionKind = Literal[
    "distribution",
    "relationship",
    "segment",
    "drivers",
    "data_quality",
]
"""No 'prediction': the analysis describes, it does not build prediction models (D033)."""

MIN_QUESTIONS = 3
MAX_QUESTIONS = 5


class FixRequest(Strict):
    """A specific, actionable change the review agent sends back to cleaning."""

    column: str | None = None
    issue: str
    suggested_op: str | None = Field(
        default=None, description="Op library name, when the reviewer can name one."
    )
    params: dict[str, object] = Field(default_factory=dict)
    reason: str


class ReviewVerdict(Strict):
    """Stage 3's first half: is the cleaned data fit to analyse?"""

    accepted: bool
    round: int = Field(default=1, ge=1, le=3, description="Loop-back is capped at 2 rounds.")
    summary: str
    findings: list[Flag] = Field(default_factory=list)
    fix_requests: list[FixRequest] = Field(default_factory=list)
    decided_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _rejection_is_actionable(self) -> ReviewVerdict:
        """A rejection without a fix request would loop with nothing to change."""
        if not self.accepted and not self.fix_requests:
            raise ValueError("a rejected review must carry at least one fix request")
        return self


class ExcludedFeature(Strict):
    """A column the analysis must not use, and why."""

    column: str
    reason: str
    source: Literal["brief", "leakage", "quality", "planner"] = "planner"


class PlanQuestion(Strict):
    """One prioritised question, tied to columns and to registry methods."""

    id: str = Field(description="Stable id, e.g. 'q1'. Referenced by every analysis artifact.")
    question: str
    kind: QuestionKind
    columns: list[str] = Field(
        default_factory=list, description="Must all exist in the clean data (Gate 3)."
    )
    methods: list[str] = Field(
        default_factory=list, description="Must all exist in the tool registry (Gate 3)."
    )
    priority: int = Field(ge=1, le=5)
    success_criteria: str = Field(description="What would count as answering it.")
    rationale: str | None = None
    feasible: bool = True
    infeasible_reason: str | None = None

    @model_validator(mode="after")
    def _infeasible_needs_reason(self) -> PlanQuestion:
        if not self.feasible and not self.infeasible_reason:
            raise ValueError(f"question {self.id} is marked infeasible without a reason")
        return self


class AnalysisPlan(Strict):
    """Stage 3's second half, approved by a human in Stage 4."""

    run_id: str
    objective: str = Field(description="The user's goal, restated by the planner.")
    interpretation: str | None = Field(
        default=None,
        description="How the planner read a thin brief, so the human can correct it.",
    )
    assumptions: list[str] = Field(default_factory=list)

    target: str | None = None
    task_type: TaskType
    task_type_reason: str = Field(
        description="Why this task type, or why prediction is not supported by this data."
    )

    questions: list[PlanQuestion] = Field(default_factory=list)
    excluded_features: list[ExcludedFeature] = Field(default_factory=list)
    success_criteria: str
    limitations: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=utcnow)
    approved: bool = False
    approved_at: datetime | None = None
    human_edits: list[str] = Field(
        default_factory=list, description="What the human changed at approval, for the audit trail."
    )

    @model_validator(mode="after")
    def _plan_shape(self) -> AnalysisPlan:
        """Gate 3 shape rules that are cheap enough to enforce in the contract itself."""
        n = len(self.questions)
        if not MIN_QUESTIONS <= n <= MAX_QUESTIONS:
            raise ValueError(f"a plan needs {MIN_QUESTIONS}-{MAX_QUESTIONS} questions, got {n}")
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"question ids must be unique, got {ids}")
        if self.task_type != "none" and not self.target:
            raise ValueError(f"task_type {self.task_type!r} requires a target column")
        if self.target and self.target in {e.column for e in self.excluded_features}:
            raise ValueError(f"target {self.target!r} cannot also be an excluded feature")
        return self

    @property
    def excluded_columns(self) -> set[str]:
        return {e.column for e in self.excluded_features}
