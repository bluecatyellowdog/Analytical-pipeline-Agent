"""Stage 5: the analysis agent's moves, the record of its tool calls, and its write-up.

The agent works in rounds. In a round it asks for tool calls (RunRound); the
code runs them and shows the aggregate results back, each under an id (T1,
T2, ...). When it has seen enough it writes (WriteUp): an answer per plan
question, findings that each cite the tool calls behind them, limitations and
what to look at next. Gate 4 checks the write-up against the logged results
before anything reaches a report.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from pipeline.contracts.common import Strict, utcnow
from pipeline.contracts.plan import AnalysisPlan

MAX_CALLS_PER_ROUND = 8

Strength = Literal["strong", "moderate", "weak", "none"]
AnswerStatus = Literal["answered", "partly", "not_answered", "infeasible"]


class ToolRequest(Strict):
    """One tool call the agent asks for."""

    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    question: str | None = Field(
        default=None, description="The plan question it serves, e.g. 'q2'; null for a side look."
    )
    why: str = Field(min_length=1, description="What this call should show, in one sentence.")


class RunRound(Strict):
    """A round of tool calls."""

    action: Literal["run"]
    calls: list[ToolRequest] = Field(min_length=1, max_length=MAX_CALLS_PER_ROUND)
    notes: str | None = Field(default=None, description="What you are looking for.")


class Finding(Strict):
    """One insight, with the tool calls it rests on."""

    id: str = Field(description="Short id, e.g. 'F1'.")
    question_id: str | None = Field(
        default=None, description="The plan question it answers; null for an insight beyond it."
    )
    headline: str = Field(min_length=1, description="The insight in one sentence.")
    detail: str = Field(
        min_length=1, description="The numbers behind it and what they mean, in 2-3 sentences."
    )
    evidence: list[str] = Field(
        min_length=1, description="Ids of the tool calls it rests on, e.g. ['T2', 'T5']."
    )
    strength: Strength = Field(
        description="How clear the evidence is: strong, moderate, weak, or none (no pattern)."
    )
    caveats: list[str] = Field(default_factory=list)


class QuestionAnswer(Strict):
    """The answer to one plan question."""

    question_id: str
    status: AnswerStatus
    answer: str = Field(min_length=1, description="The question answered in plain words.")
    finding_ids: list[str] = Field(default_factory=list)
    reason: str | None = Field(default=None, description="Why it is not fully answered.")

    @model_validator(mode="after")
    def _explained(self) -> QuestionAnswer:
        if self.status != "answered" and not self.reason:
            raise ValueError(f"{self.question_id} is {self.status}: give the reason")
        if self.status in ("answered", "partly") and not self.finding_ids:
            raise ValueError(f"{self.question_id} is {self.status}: cite at least one finding")
        return self


class WriteUp(Strict):
    """What the analysis found, for a reader, checked by Gate 4."""

    action: Literal["write"]
    summary: str = Field(
        min_length=1,
        description="The most interesting things learned, in plain words, in 3-6 sentences.",
    )
    answers: list[QuestionAnswer]
    findings: list[Finding]
    limitations: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(
        default_factory=list, description="What would be worth looking at next."
    )

    @model_validator(mode="after")
    def _ids(self) -> WriteUp:
        ids = [f.id for f in self.findings]
        if len(set(ids)) != len(ids):
            raise ValueError(f"finding ids must be unique, got {ids}")
        questions = [a.question_id for a in self.answers]
        if len(set(questions)) != len(questions):
            raise ValueError(f"one answer per question, got {questions}")
        unknown = sorted({f for a in self.answers for f in a.finding_ids} - set(ids))
        if unknown:
            raise ValueError(f"answers cite findings that do not exist: {unknown}")
        return self

    def finding(self, finding_id: str) -> Finding | None:
        return next((f for f in self.findings if f.id == finding_id), None)


AgentMove = Annotated[RunRound | WriteUp, Field(discriminator="action")]


class CallRecord(Strict):
    """One tool call as the agent asked for it, and what came back."""

    call_id: str = Field(description="The id the agent cites, e.g. 'T3'.")
    round: int
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    question_id: str | None = None
    why: str = ""
    summary: str = Field(default="", description="The aggregate result the agent was shown.")
    artifact_ids: list[str] = Field(default_factory=list)
    tool_call_id: str | None = Field(default=None, description="The registry's id for the call.")
    error: str | None = Field(default=None, description="Why the call was refused, as shown.")
    internal_error: str | None = Field(
        default=None, description="An unexpected failure, kept for the audit log only."
    )
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


class AnalysisOutcome(Strict):
    """Everything the analysis did and found, as written to analysis.json."""

    run_id: str
    plan: AnalysisPlan = Field(description="The plan the analysis ran: the AI's, or your edit.")
    write_up: WriteUp
    calls: list[CallRecord] = Field(default_factory=list)
    finding_artifacts: dict[str, str] = Field(
        default_factory=dict, description="Finding id -> its artifact in artifacts.json."
    )
    created_at: datetime = Field(default_factory=utcnow)

    def call(self, call_id: str) -> CallRecord | None:
        return next((c for c in self.calls if c.call_id == call_id), None)
