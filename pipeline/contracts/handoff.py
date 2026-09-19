"""Stage 2 → Stage 3 handoff: what the cleaning agent guarantees about its output.

Machine-readable on purpose. The review agent reads this, not the data.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from pipeline.contracts.common import Flag, Strict, utcnow
from pipeline.contracts.profile import DatasetProfile


class ColumnSchema(Strict):
    """The post-cleaning contract for one column."""

    name: str
    dtype: str
    nullable: bool
    null_rate: float = Field(ge=0.0, le=1.0)
    n_distinct: int
    source_column: str | None = Field(
        default=None, description="Original name, when an op renamed it."
    )
    transformations: list[str] = Field(
        default_factory=list, description="op_ids that touched this column, in order."
    )


class Handoff(Strict):
    """Clean data + profile + flags, with the row accounting that justifies it."""

    run_id: str
    table: str = Field(description="Table name in the DuckDB `clean` schema.")
    source_table: str = Field(description="The raw table it derives from.")

    rows_raw: int
    rows_clean: int
    rows_dropped: int
    dropped_rows_table: str | None = Field(
        default=None, description="Table holding every dropped row with its op and reason."
    )

    schema_: list[ColumnSchema] = Field(
        default_factory=list,
        alias="schema",
        description="Named `schema` on the wire; `schema_` in Python to avoid shadowing pydantic.",
    )
    dropped_columns: dict[str, str] = Field(
        default_factory=dict, description="column -> reason, for columns an op removed."
    )
    applied_ops: list[str] = Field(default_factory=list, description="op_ids in execution order.")
    profile: DatasetProfile | None = Field(
        default=None, description="Profile of the cleaned data, for the review agent."
    )
    flags: list[Flag] = Field(default_factory=list)
    data_hash: str
    created_at: datetime = Field(default_factory=utcnow)

    model_config = Strict.model_config | {"populate_by_name": True}

    @model_validator(mode="after")
    def _row_accounting(self) -> Handoff:
        """Spec 5.2: rows_raw = rows_clean + rows_dropped, exactly."""
        if self.rows_clean + self.rows_dropped != self.rows_raw:
            raise ValueError(
                f"row accounting broken: {self.rows_raw} raw != "
                f"{self.rows_clean} clean + {self.rows_dropped} dropped"
            )
        return self

    @property
    def row_loss(self) -> float:
        """Share of rows lost in cleaning. Gate 2: <5% pass, 5-20% warn, >20% block."""
        return self.rows_dropped / self.rows_raw if self.rows_raw else 0.0

    def column(self, name: str) -> ColumnSchema | None:
        return next((c for c in self.schema_ if c.name == name), None)
