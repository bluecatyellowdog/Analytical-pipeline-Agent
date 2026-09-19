"""Shared primitives for every contract.

Nothing here knows about pandas, DuckDB or the LLM: contracts are the interface
between stages, and stay importable without the heavy stack.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["info", "warn", "block"]
"""How much a flag matters. `block` stops a run at the next gate."""

GateOutcome = Literal["pass", "flag", "retry", "block"]
"""Spec 5.1: the one consistent failure policy shared by every gate."""

CheckLayer = Literal["contract", "invariant", "judgment"]
"""Spec 5.1: layers run in this order; judgment only after the first two pass."""


class Strict(BaseModel):
    """Base for all contracts: unknown fields are an error, not a silent drop.

    A typo in a handoff or plan field must fail loudly at the stage boundary
    rather than quietly reaching a later stage as a missing value.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Flag(Strict):
    """A single issue raised by any stage, carried forward into reports.

    Flags are how 'flag rather than fix' is represented: a judgment call is
    recorded here and decided later, with the context brief in mind.
    """

    code: str = Field(description="Stable machine key, e.g. 'ambiguous_date_order'.")
    severity: Severity = "warn"
    message: str = Field(description="One human-readable sentence.")
    column: str | None = None
    stage: str | None = Field(default=None, description="Stage that raised it, e.g. 'ingestion'.")
    details: dict[str, object] = Field(default_factory=dict)
    acknowledged: bool = Field(
        default=False,
        description="Set when a human or the publish config explicitly accepts the flag.",
    )


def utcnow() -> datetime:
    """Timezone-aware UTC now. Every timestamp in the audit trail uses this."""
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    """Short, unique, sortable-enough identifier for runs, ops and artifacts."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
