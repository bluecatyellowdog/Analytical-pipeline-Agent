"""Artifacts and publish configs.

An artifact is a standalone unit — a chart, table, metric or finding — that
carries its own aggregated data and its audit references. This is what makes
publishing safe: the dashboard renders artifacts, never the dataset.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from pipeline.contracts.common import Flag, Strict, new_id, utcnow

ArtifactKind = Literal["chart", "table", "metric", "finding"]
AuditRefKind = Literal["op", "gate", "check", "model", "plan_question", "claim"]


class AuditRef(Strict):
    """A pointer from an artifact back into the audit trail."""

    kind: AuditRefKind
    id: str
    note: str | None = None


class Artifact(Strict):
    """One publishable result, with its evidence attached.

    `data` holds *aggregated* values only. The publish gate enforces that, plus
    a minimum group size, because anything here can end up on a public page.
    """

    artifact_id: str = Field(default_factory=lambda: new_id("art"))
    run_id: str
    kind: ArtifactKind
    title: str
    description: str | None = None
    stage: str = Field(description="Which stage produced it, e.g. 'analysis'.")
    question_id: str | None = Field(
        default=None, description="The plan question this answers, when it answers one."
    )

    data: dict[str, Any] | list[dict[str, Any]] = Field(
        default_factory=dict, description="Aggregated data behind the artifact. Never row-level."
    )
    spec: dict[str, Any] = Field(
        default_factory=dict, description="Render spec, e.g. a Plotly figure definition."
    )
    min_group_n: int | None = Field(
        default=None, description="Smallest group size in `data`; the publish gate needs >= 10."
    )
    row_level: bool = Field(
        default=False, description="True if `data` contains individual records. Never publishable."
    )
    restricted: bool = Field(
        default=False, description="Marked as not for publication, for any reason."
    )

    audit_refs: list[AuditRef] = Field(default_factory=list)
    interpretation: str | None = Field(
        default=None, description="LLM text. Rendered visibly separated from computed values."
    )
    flags: list[Flag] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _traceable(self) -> Artifact:
        """Spec 7: every artifact has audit references. Enforced at creation, not at publish."""
        if not self.audit_refs:
            raise ValueError(f"artifact {self.title!r} has no audit references")
        return self


class ArtifactCatalog(Strict):
    """Every artifact one run produced. The publish builder selects from this."""

    run_id: str
    artifacts: list[Artifact] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    def add(self, artifact: Artifact) -> None:
        if artifact.run_id != self.run_id:
            raise ValueError(f"artifact belongs to run {artifact.run_id}, not {self.run_id}")
        self.artifacts.append(artifact)

    def get(self, artifact_id: str) -> Artifact | None:
        return next((a for a in self.artifacts if a.artifact_id == artifact_id), None)

    def of_kind(self, kind: ArtifactKind) -> list[Artifact]:
        return [a for a in self.artifacts if a.kind == kind]


class PublishSelection(Strict):
    """One artifact chosen for a dashboard, in the order the user wants it."""

    artifact_id: str
    order: int
    commentary: str | None = Field(
        default=None, description="Human-written, and labelled as such on the page."
    )


class PublishConfig(Strict):
    """The record of what went live. Committed under publish/ after a successful gate."""

    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$", description="URL segment for the dashboard.")
    title: str
    version: int = Field(ge=1, description="Immutable snapshot number: dashboards/<slug>/v<N>/.")
    run_id: str
    summary: str | None = None
    selections: list[PublishSelection] = Field(default_factory=list)
    headline_metrics: list[str] = Field(
        default_factory=list, description="artifact_ids shown on the project page."
    )
    acknowledged_flags: list[str] = Field(
        default_factory=list,
        description="Flag codes the user explicitly accepted. Open flags block publishing.",
    )
    author: str | None = None
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _selection_order(self) -> PublishConfig:
        ids = [s.artifact_id for s in self.selections]
        if len(set(ids)) != len(ids):
            raise ValueError("the same artifact is selected twice")
        orders = [s.order for s in self.selections]
        if len(set(orders)) != len(orders):
            raise ValueError("two selections share the same order")
        return self

    @property
    def ordered_selections(self) -> list[PublishSelection]:
        return sorted(self.selections, key=lambda s: s.order)
