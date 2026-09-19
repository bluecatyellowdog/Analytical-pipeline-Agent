"""The run manifest: enough to reproduce a run's *decisions*, not just its ops.

Spec 6: git SHA, prompt versions, model name and version, seeds, config and the
context brief. Two runs with the same manifest should make the same choices.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from pipeline.contracts.common import Strict, new_id, utcnow
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.ingestion import FileInfo


class ModelInfo(Strict):
    """Which model answered, and how it was configured."""

    provider: str = Field(description="e.g. 'anthropic'.")
    name: str = Field(description="Exact model id, e.g. 'claude-sonnet-5'.")
    version: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    seed: int | None = None


class CostRecord(Strict):
    """Token and money cost for one run. Spec 6 targets under $0.50 per dataset."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    wall_clock_s: float = 0.0

    def add(
        self, *, prompt_tokens: int, completion_tokens: int, usd: float, wall_clock_s: float = 0.0
    ) -> None:
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.usd += usd
        self.wall_clock_s += wall_clock_s


class SamplingRecord(Strict):
    """Recorded whenever a statistic came from a sample rather than the full data."""

    stage: str
    reason: str
    n_rows_total: int
    n_rows_sampled: int
    seed: int


class RunManifest(Strict):
    """One record per run, written at the start and completed as stages finish."""

    run_id: str = Field(default_factory=lambda: new_id("run"))
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None

    git_sha: str | None = None
    git_dirty: bool = Field(
        default=False, description="True when the working tree had uncommitted changes."
    )
    pipeline_version: str
    python_version: str
    platform: str

    context_brief: ContextBrief
    inputs: list[FileInfo] = Field(default_factory=list)

    models: dict[str, ModelInfo] = Field(
        default_factory=dict, description="Per agent: 'cleaning', 'review_planning', 'analysis'."
    )
    prompt_versions: dict[str, str] = Field(
        default_factory=dict, description="Per agent, so a prompt change is visible in the diff."
    )
    seeds: dict[str, int] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    sampling: list[SamplingRecord] = Field(default_factory=list)

    cost: CostRecord = Field(default_factory=CostRecord)
    stage_status: dict[str, str] = Field(
        default_factory=dict, description="Stage name -> 'pending' | 'done' | 'blocked'."
    )
