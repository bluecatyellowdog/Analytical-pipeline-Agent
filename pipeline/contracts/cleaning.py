"""The coder agent's output: an ordered cleaning plan, and the user's decisions it may act on.

A plan is data, not code that runs by itself. Every step names an op from the
library (or `custom_code`) with its parameters, says which diagnosis items it
addresses, and why. The executor in `pipeline/ops/plan.py` runs it; nothing in
here touches the data.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from pipeline.contracts.common import Strict

MAX_STEPS = 60
"""A plan longer than this is a runaway, not a cleaning plan."""


class PlanStep(Strict):
    op: str = Field(description="A library op name, or 'custom_code'.")
    params: dict[str, Any] = Field(default_factory=dict)
    addresses: list[str] = Field(
        min_length=1, description="Ids of the issues or decided judgment calls this step serves."
    )
    rationale: str = Field(min_length=1, description="Why this step, in one sentence.")


class Skipped(Strict):
    """An issue or decision the plan deliberately does not implement."""

    id: str
    reason: str = Field(min_length=1)


class CleaningPlan(Strict):
    summary: str = Field(min_length=1, description="One or two sentences.")
    steps: list[PlanStep] = Field(default_factory=list, max_length=MAX_STEPS)
    skipped: list[Skipped] = Field(default_factory=list)

    @model_validator(mode="after")
    def _skipped_once(self) -> CleaningPlan:
        ids = [s.id for s in self.skipped]
        repeated = sorted({i for i in ids if ids.count(i) > 1})
        if repeated:
            raise ValueError(f"skipped lists the same id more than once: {repeated}")
        return self


UNDECIDED = "undecided"


class Decision(Strict):
    """The answer to one judgment call: the user's, or the AI's when the brief lets it decide."""

    id: str
    choice: str = Field(min_length=1)
    decided_by: Literal["user", "ai"] = "user"

    @property
    def decided(self) -> bool:
        return self.choice.strip().lower() != UNDECIDED
