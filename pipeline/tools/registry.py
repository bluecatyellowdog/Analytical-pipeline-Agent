"""The analysis tool registry: what the analysis agent may run, and how every call is logged.

The rule from cleaning carries over: the LLM decides, tools compute. A tool is a
tested function over the clean data that returns aggregates only (rates, counts,
summaries, regression effects), never rows. Every call is validated against the
tool's parameters, refused if it would use the target, an ID, an excluded or
personal column as a feature, and turned into artifacts that point back to the
call and the plan question they answer. Groups smaller than MIN_GROUP_N are
merged or hidden, the same threshold the publish gate applies later.

The frame is never changed: tools work on copies of the columns they need.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from pipeline.contracts.artifact import Artifact, ArtifactCatalog, AuditRef
from pipeline.contracts.common import Flag, new_id
from pipeline.layout import CHARTS

MIN_GROUP_N = 10
"""Smallest group shown on its own, to the agent or in a report."""

FEATURE_KINDS = ("numeric", "categorical", "boolean")
"""Column kinds a model or a comparison may use. IDs, dates, free text and personal data may not."""


class ToolError(ValueError):
    """The call was refused; the message says why, for the agent to correct it."""


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass
class ToolResult:
    """One result of a tool, before it becomes an artifact."""

    title: str
    data: dict[str, Any]
    summary: str
    """One or two lines with the key numbers, for the agent and the report."""
    kind: str = "table"
    figure: Any = None
    min_group_n: int | None = None
    flags: list[Flag] = field(default_factory=list)
    description: str | None = None


@dataclass
class ToolCall:
    call_id: str
    tool: str
    params: dict[str, Any]
    question_id: str | None
    artifacts: list[Artifact]
    summary: str
    seconds: float


@dataclass
class AnalysisContext:
    """The clean data and what is known about it, shared by every tool call in a run."""

    run_id: str
    frame: pd.DataFrame
    target: str | None
    task_type: str
    kinds: dict[str, str]
    """Column -> numeric | categorical | boolean | date | id | text | personal."""
    excluded: set[str]
    out_dir: Path
    min_group_n: int = MIN_GROUP_N
    seed: int = 42
    catalog: ArtifactCatalog | None = None
    calls: list[ToolCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.catalog is None:
            self.catalog = ArtifactCatalog(run_id=self.run_id)

    def require(self, column: str) -> str:
        """The column's kind; refused if it does not exist."""
        if column not in self.frame.columns or column not in self.kinds:
            raise ToolError(
                f"no column {column!r} in the clean data; columns are {sorted(self.kinds)}"
            )
        return self.kinds[column]

    def feature(self, column: str) -> str:
        """A column that may be used to describe or predict the target."""
        kind = self.require(column)
        if column == self.target:
            raise ToolError(f"{column} is the target; it cannot be used to explain itself")
        if column in self.excluded:
            raise ToolError(f"{column} is excluded from the analysis and cannot be used")
        if kind not in FEATURE_KINDS:
            raise ToolError(f"{column} is a {kind} column and cannot be used as a feature")
        return kind

    def features(self, requested: list[str] | None) -> list[str]:
        if requested:
            for column in requested:
                self.feature(column)
            return list(requested)
        usable = [
            column
            for column, kind in self.kinds.items()
            if kind in FEATURE_KINDS and column != self.target and column not in self.excluded
        ]
        if not usable:
            raise ToolError("no usable feature columns")
        return usable

    def with_outcome(self, outcome: str | None) -> AnalysisContext:
        """This context with another column as the thing to explain, for a side question.

        Results still go to this run's catalog and call log. The analysis target
        itself may then be used like any other column.
        """
        if outcome is None or outcome == self.target:
            return self
        kind = self.require(outcome)
        if outcome in self.excluded:
            raise ToolError(f"{outcome} is excluded from the analysis and cannot be used")
        if kind not in FEATURE_KINDS:
            raise ToolError(f"{outcome} is a {kind} column and cannot be an outcome")
        return replace(self, target=outcome)

    def target_type(self) -> str:
        """'binary' for a yes/no target, 'numeric' for a number; refused otherwise."""
        if not self.target:
            raise ToolError("there is no target column to compare against")
        try:
            self.binary_target()
        except ToolError:
            if self.kinds.get(self.target) == "numeric":
                return "numeric"
            raise ToolError(f"the target {self.target} is neither yes/no nor a number") from None
        return "binary"

    def numeric_target(self) -> pd.Series:
        """The target as numbers, missing where it is missing. Refused if it is not numeric."""
        if not self.target:
            raise ToolError("there is no target column to compare against")
        if self.kinds.get(self.target) != "numeric":
            raise ToolError(f"the target {self.target} is not a number")
        return pd.to_numeric(self.frame[self.target]).astype("Float64")

    def binary_target(self) -> pd.Series:
        """The target as 0/1, missing where it is missing. Refused if it is not binary."""
        if not self.target:
            raise ToolError("there is no target column to compare against")
        values = as_yes_no(self.frame[self.target])
        if values is None:
            raise ToolError(
                f"the target {self.target} is not binary; this tool needs a yes/no target"
            )
        return values


YES_NO = {1: 1, 0: 0, "1": 1, "0": 0, "true": 1, "false": 0}  # True == 1, False == 0
YES_NO |= {"yes": 1, "no": 0, "ja": 1, "nej": 0}


def as_yes_no(values: pd.Series) -> pd.Series | None:
    """The values as 0/1, missing where missing; None if they do not all read as yes/no."""
    mapped = values.map(lambda v: YES_NO.get(v.lower() if isinstance(v, str) else v))
    if mapped[values.notna()].isna().any():
        return None
    return mapped.astype("Float64")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    params_model: type[Params]
    run: Callable[[AnalysisContext, Any], list[ToolResult]]
    doc: str
    question_kinds: tuple[str, ...]


REGISTRY: dict[str, ToolSpec] = {}


def register(
    name: str, params_model: type[Params], question_kinds: tuple[str, ...]
) -> Callable[[Callable], Callable]:
    """Add a tool. The name is what a plan and the analysis agent may ask for."""

    def decorator(run: Callable) -> Callable:
        if name in REGISTRY:
            raise ValueError(f"tool {name!r} is already registered")
        REGISTRY[name] = ToolSpec(name, params_model, run, run.__doc__ or "", question_kinds)
        return run

    return decorator


def tool_names() -> list[str]:
    return sorted(REGISTRY)


def run_tool(
    ctx: AnalysisContext, name: str, params: dict[str, Any], *, question_id: str | None
) -> ToolCall:
    """Validate, run, turn the results into artifacts, and log the call."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise ToolError(f"unknown tool {name!r}; available: {tool_names()}")
    try:
        validated = spec.params_model.model_validate(params)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in error.errors()
        )
        raise ToolError(f"invalid parameters for {name}: {problems}") from error

    call_id = new_id("call")
    started = time.perf_counter()
    results = spec.run(ctx, validated)
    artifacts = [_artifact(ctx, result, name, call_id, question_id) for result in results]
    call = ToolCall(
        call_id=call_id,
        tool=name,
        params=validated.model_dump(mode="json"),
        question_id=question_id,
        artifacts=artifacts,
        summary="\n".join(result.summary for result in results),
        seconds=round(time.perf_counter() - started, 3),
    )
    ctx.calls.append(call)
    return call


def _artifact(
    ctx: AnalysisContext, result: ToolResult, tool: str, call_id: str, question_id: str | None
) -> Artifact:
    refs = [AuditRef(kind="tool", id=call_id, note=tool)]
    if question_id:
        refs.insert(0, AuditRef(kind="plan_question", id=question_id))
    artifact = Artifact(
        run_id=ctx.run_id,
        kind="chart" if result.figure is not None else result.kind,
        title=result.title,
        description=result.description,
        stage="analysis",
        question_id=question_id,
        data=result.data,
        min_group_n=result.min_group_n,
        audit_refs=refs,
        flags=list(result.flags),
    )
    if result.figure is not None:
        from pipeline.tools.charts import save

        relative = Path(CHARTS) / f"{artifact.artifact_id}.png"
        save(result.figure, ctx.out_dir / relative)
        artifact.spec = {"chart_file": relative.as_posix()}
    ctx.catalog.add(artifact)
    return artifact


def describe_tools() -> str:
    """Every tool with its purpose and parameters, for a prompt. Generated, so it cannot drift."""
    lines = []
    for name in tool_names():
        spec = REGISTRY[name]
        lines.append(f"- {name}: {(spec.doc.strip().splitlines() or [''])[0]}")
        schema = spec.params_model.model_json_schema()
        required = set(schema.get("required", []))
        for field_name, field_schema in schema.get("properties", {}).items():
            default = "" if field_name in required else f", default {field_schema.get('default')!r}"
            description = field_schema.get("description", "")
            lines.append(f"    {field_name}{default}" + (f": {description}" if description else ""))
    return "\n".join(lines)
