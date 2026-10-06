"""The analyst agent's output: what is wrong with the data and what it should become.

The diagnosis changes nothing. It is the brief the coder agent works from, and
the record of *why* each later change was made. Clear errors go in `issues`;
anything a reasonable analyst could decide either way goes in `judgment_calls`,
to be decided by a human or the brief, never silently fixed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from pipeline.contracts.common import Strict

IssueKind = Literal[
    "type",
    "missing_tokens",
    "sentinel",
    "decimal_format",
    "date_format",
    "whitespace",
    "case",
    "categories",
    "duplicates",
    "outliers",
    "invalid_values",
    "identifier",
    "personal_data",
    "other",
]
ColumnRole = Literal["feature", "target", "identifier", "excluded", "drop"]
TargetType = Literal["integer", "float", "boolean", "date", "datetime", "categorical", "text", "id"]


class Issue(Strict):
    """A clear error with a clear fix, grounded in the profile."""

    id: str = Field(description="Short id, e.g. 'I1'.")
    column: str | None = Field(default=None, description="None for table-level issues.")
    kind: IssueKind
    evidence: str = Field(min_length=1, description="The profile facts that show the problem.")
    fix: str = Field(min_length=1, description="What should be done, in plain words.")
    suggested_op: str | None = Field(
        default=None,
        description="A library op that fits, or 'custom_code' when none does.",
    )


class Alternative(Strict):
    """An option the AI did not choose, and what choosing it would have changed."""

    option: str
    effect: str = Field(min_length=1, description="The analyst's estimate, not a measurement.")


class AIDecision(Strict):
    """How the AI settled a judgment call from the brief. Always shown, always reversible."""

    choice: str = Field(description="Exactly one of the call's options.")
    confidence: Literal["high", "medium", "low"]
    reasoning: str = Field(
        min_length=1, description="What in the brief, the guidance or the profile decided it."
    )
    expected_effect: str = Field(
        min_length=1, description="What the choice will change, as the analyst expects it."
    )
    least_destructive: bool = Field(
        description="True when the brief did not settle the call, so the option that changes "
        "the least was taken."
    )
    alternatives: list[Alternative] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unsure_means_cautious(self) -> AIDecision:
        if self.confidence == "low" and not self.least_destructive:
            raise ValueError(
                "a low-confidence decision must take the least destructive option "
                "(least_destructive: true)"
            )
        return self


class JudgmentCall(Strict):
    """A decision the data cannot settle on its own: decided by the user, or by the AI from
    the brief, but never silently."""

    id: str = Field(description="Short id, e.g. 'J1'.")
    column: str | None = None
    question: str = Field(min_length=1)
    options: list[str] = Field(min_length=2)
    recommendation: str = Field(min_length=1)
    why_it_matters: str = Field(min_length=1)
    ai_decision: AIDecision | None = Field(
        default=None, description="Present when the brief says judgment_calls: decide."
    )

    @model_validator(mode="after")
    def _choice_is_an_option(self) -> JudgmentCall:
        decision = self.ai_decision
        if decision is None:
            return self
        if decision.choice not in self.options:
            raise ValueError(
                f"{self.id}: the AI's choice must be exactly one of the options {self.options}"
            )
        stray = [a.option for a in decision.alternatives if a.option not in self.options]
        if stray or any(a.option == decision.choice for a in decision.alternatives):
            raise ValueError(
                f"{self.id}: alternatives must be the other options, each with its effect"
            )
        return self


class ColumnTarget(Strict):
    """What one column should look like once it is clean, and what it is for."""

    name: str
    role: ColumnRole
    target_type: TargetType
    notes: str | None = None


class Diagnosis(Strict):
    summary: str = Field(min_length=1, description="Two or three sentences.")
    issues: list[Issue] = Field(default_factory=list)
    judgment_calls: list[JudgmentCall] = Field(default_factory=list)
    columns: list[ColumnTarget] = Field(description="Exactly one entry per column in the data.")

    @model_validator(mode="after")
    def _unique_ids(self) -> Diagnosis:
        ids = [i.id for i in self.issues] + [j.id for j in self.judgment_calls]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate issue/judgment ids: {duplicates}")
        names = [c.name for c in self.columns]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"columns listed more than once: {repeated}")
        return self

    def column(self, name: str) -> ColumnTarget | None:
        return next((c for c in self.columns if c.name == name), None)
