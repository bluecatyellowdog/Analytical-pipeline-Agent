"""The audit trail: op records, check results and gate results.

This is the product. Reports are assembled from these objects, and every number
in a report must be traceable to one of them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from pipeline.contracts.common import (
    CheckLayer,
    Flag,
    GateOutcome,
    Severity,
    Strict,
    new_id,
    utcnow,
)


class ColumnStats(Strict):
    """A column summary taken before and after an op, so changes are provable."""

    name: str
    dtype: str
    n_rows: int
    n_null: int
    n_distinct: int
    min: Any | None = None
    max: Any | None = None
    mean: float | None = None
    value_hash: str | None = Field(
        default=None,
        description="Hash of the column's values; equal hashes mean nothing changed.",
    )


class OpRecord(Strict):
    """One logged data transformation. Every op in `pipeline/ops/` returns one.

    `code` is the exact code that ran, not a description of it: the replay check
    and the report appendix both depend on that being literally true.
    """

    op_id: str = Field(default_factory=lambda: new_id("op"))
    op_name: str = Field(description="Op library name, e.g. 'cast' or 'drop_duplicates'.")
    params: dict[str, Any] = Field(default_factory=dict)
    code: str = Field(description="The exact code that was executed.")
    target_columns: list[str] = Field(default_factory=list)
    rationale: str | None = Field(
        default=None, description="Why the agent chose this op. LLM text, never a number."
    )

    rows_before: int
    rows_after: int
    rows_dropped: int = 0
    cells_changed: dict[str, int] = Field(
        default_factory=dict, description="Changed cells per column, for the cell-diff check."
    )
    new_nulls: dict[str, int] = Field(
        default_factory=dict,
        description="Nulls introduced per column. Non-zero must be explained, never silent.",
    )

    before: dict[str, ColumnStats] = Field(default_factory=dict)
    after: dict[str, ColumnStats] = Field(default_factory=dict)
    input_hash: str | None = None
    output_hash: str | None = None

    started_at: datetime = Field(default_factory=utcnow)
    duration_ms: float | None = None
    flags: list[Flag] = Field(default_factory=list)

    @model_validator(mode="after")
    def _row_accounting(self) -> OpRecord:
        """rows_before = rows_after + rows_dropped, exactly. No op may lose a row silently."""
        if self.rows_after + self.rows_dropped != self.rows_before:
            raise ValueError(
                f"row accounting broken for op {self.op_name}: "
                f"{self.rows_before} before != {self.rows_after} after "
                f"+ {self.rows_dropped} dropped"
            )
        return self


class CheckResult(Strict):
    """One deterministic check inside a gate."""

    name: str
    layer: CheckLayer
    passed: bool
    severity: Severity = "block"
    message: str = ""
    observed: dict[str, Any] = Field(default_factory=dict)


class GateResult(Strict):
    """The verdict of one quality gate, assembled from its checks.

    Checks run contract → invariant → judgment; the outcome is the worst
    severity among the failures.
    """

    gate: str = Field(description="Gate name, e.g. 'gate_1_ingestion'.")
    outcome: GateOutcome
    checks: list[CheckResult] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    attempt: int = Field(default=1, description="Retry attempt number; max 2 per spec 5.1.")
    evaluated_at: datetime = Field(default_factory=utcnow)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def blocked(self) -> bool:
        return self.outcome == "block"


class AuditLog(Strict):
    """Everything one run did, in order. The final report reads only from here."""

    run_id: str
    ops: list[OpRecord] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)

    def add_op(self, record: OpRecord) -> None:
        self.ops.append(record)
        self.flags.extend(record.flags)

    def add_gate(self, result: GateResult) -> None:
        self.gates.append(result)
        self.flags.extend(result.flags)

    def ops_touching(self, column: str) -> list[OpRecord]:
        """Every op that declared this column as a target — used by the untouched-column check."""
        return [o for o in self.ops if column in o.target_columns]

    @property
    def open_flags(self) -> list[Flag]:
        """Flags nobody has accepted yet. The publish gate blocks on these."""
        return [f for f in self.flags if not f.acknowledged]
