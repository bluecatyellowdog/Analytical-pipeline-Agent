"""The `custom_code` op: an agent's own pandas code, audited like every other op.

The library ops cover the common problems. Real data has a long tail they do
not, so an agent may write a `transform(df)` function instead. What it may not
do is change data without saying so: it declares every column it changes, adds
or drops, whether it creates nulls and why, and whether it drops rows and why.
`apply_op` then measures what actually changed and refuses any difference from
the declaration. The code is stored in the record, so replay re-runs it.

    def transform(df):          # df: one row per data row, indexed by row id
        df["beloeb"] = ...      # change declared columns only
        return df               # keep the index; never add rows
"""

from __future__ import annotations

from pydantic import Field, model_validator

from pipeline.contracts.common import Flag
from pipeline.ops.base import (
    OpContext,
    OpError,
    OpPlan,
    _qualified,
    register,
    require_columns,
)
from pipeline.ops.library import Params
from pipeline.ops.sandbox import SandboxError, check_code, run_transform


class CustomCodeParams(Params):
    code: str = Field(min_length=1, description="Python defining transform(df) -> DataFrame.")
    changes: list[str] = Field(default_factory=list, description="Existing columns it modifies.")
    adds: list[str] = Field(default_factory=list, description="New columns it creates.")
    drops: list[str] = Field(default_factory=list, description="Columns it removes.")
    may_create_nulls: list[str] = Field(
        default_factory=list, description="Columns (in changes or adds) that may gain nulls."
    )
    null_reason: str | None = Field(default=None, description="Why those nulls are correct.")
    drop_reason: str | None = Field(
        default=None, description="Set only if the code drops rows: why those rows go."
    )

    @model_validator(mode="after")
    def _consistent(self) -> CustomCodeParams:
        lists = {"changes": self.changes, "adds": self.adds, "drops": self.drops}
        seen: dict[str, str] = {}
        for label, columns in lists.items():
            for column in columns:
                if column in seen:
                    raise ValueError(f"column {column!r} is in both {seen[column]} and {label}")
                seen[column] = label
        stray = [c for c in self.may_create_nulls if c not in self.changes + self.adds]
        if stray:
            raise ValueError(f"may_create_nulls lists {stray}, which are not in changes or adds")
        if self.may_create_nulls and not (self.null_reason or "").strip():
            raise ValueError("null_reason is required when may_create_nulls is set")
        return self


@register("custom_code", CustomCodeParams)
def custom_code(ctx: OpContext, params: CustomCodeParams) -> OpPlan:
    """Run agent-written pandas code in the sandbox, within declared bounds."""
    require_columns(ctx, *params.changes, *params.drops)
    problems = check_code(params.code)
    if problems:
        raise OpError("custom code refused before running: " + "; ".join(problems))

    def materialize(context: OpContext, new_table: str) -> str:
        try:
            run_transform(
                context.store,
                _qualified(context.table),
                list(context.columns),
                params.code,
                new_table,
            )
        except SandboxError as error:
            raise OpError(f"custom code failed: {error}") from error
        return params.code

    flags = []
    if params.may_create_nulls:
        flags.append(
            Flag(
                code="custom_code_new_nulls",
                severity="warn",
                message=f"Custom code may set values to null in {params.may_create_nulls}: "
                f"{params.null_reason}",
                stage="cleaning",
                details={"columns": params.may_create_nulls},
            )
        )

    return OpPlan(
        target_columns=[*params.changes, *params.drops],
        drop_columns=list(params.drops),
        new_columns=list(params.adds),
        allow_new_nulls_in=list(params.may_create_nulls),
        may_drop_rows=params.drop_reason is not None,
        drop_reason=params.drop_reason,
        materialize=materialize,
        note="Custom code (not a library op).",
        flags=flags,
    )
