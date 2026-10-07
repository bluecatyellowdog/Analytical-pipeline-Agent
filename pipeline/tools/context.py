"""The analysis context of a run folder: the clean data, and what each column is for.

Built only from what cleaning delivered and Gate 2 verified: the clean Parquet
file (re-checked against its contract before use), handoff.json (types, roles
and the clean profile with its personal-data findings) and the brief in the
manifest. Every column gets one analysis kind. Identifiers, personal data and
free text are never analysed, and columns the brief or the diagnosis excluded
stay out.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from pipeline.agents.common import load_manifest
from pipeline.checks.schema import type_fits, validate_against_contract
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.handoff import ColumnSchema, Handoff
from pipeline.ingestion.raw import RESERVED_COLUMNS
from pipeline.tools.eda import fmt
from pipeline.tools.registry import AnalysisContext, as_yes_no

HANDOFF_FILE = "handoff.json"
MAX_CATEGORIES = 50
"""A text column with more distinct values than this is free text, not a category."""

KIND_LABELS = {
    "numeric": "number",
    "categorical": "category",
    "boolean": "yes/no",
    "date": "date",
    "id": "identifier",
    "personal": "personal data",
    "text": "free text",
}


class AnalysisInputError(RuntimeError):
    """The run folder has no clean data the analysis may use, with what to do about it."""


def column_kind(column: ColumnSchema, *, personal: bool) -> str:
    """numeric | categorical | boolean | date | id | text | personal, from the delivered type."""
    if personal:
        return "personal"
    if column.role == "identifier" or column.expected_type == "id" or column.unique:
        return "id"
    if type_fits("boolean", column.dtype):
        return "boolean"
    if type_fits("float", column.dtype):
        return "numeric"
    if type_fits("date", column.dtype):
        return "date"
    if column.expected_type == "text" or column.n_distinct > MAX_CATEGORIES:
        return "text"
    return "categorical"


def analysis_kinds(handoff: Handoff) -> dict[str, str]:
    profile = handoff.profile
    personal = {c.name for c in profile.columns if c.pii_level != "none"} if profile else set()
    return {c.name: column_kind(c, personal=c.name in personal) for c in handoff.schema_}


def exclusions(handoff: Handoff, brief: ContextBrief) -> dict[str, str]:
    """Columns kept out of the analysis, and why."""
    reasons = {c.name: "excluded in the diagnosis" for c in handoff.schema_ if c.role == "excluded"}
    for name in brief.excluded_variables:
        if handoff.column(name) is not None:
            reasons[name] = "excluded by the brief"
    return reasons


def target_of(handoff: Handoff, brief: ContextBrief) -> str | None:
    """The brief's target, or else the one column the diagnosis gave the role 'target'."""
    if brief.target and handoff.column(brief.target) is not None:
        return brief.target
    targets = [c.name for c in handoff.schema_ if c.role == "target"]
    return targets[0] if len(targets) == 1 else None


def task_type_of(ctx: AnalysisContext) -> str:
    """What the target makes the analysis: yes/no, a number, one of several categories."""
    target = ctx.target
    if not target or target not in ctx.kinds:
        return "none"
    kind = ctx.kinds[target]
    if kind in ("id", "personal", "text", "date"):
        return "none"
    if as_yes_no(ctx.frame[target]) is not None:
        return "binary_classification"
    if kind == "numeric":
        return "regression"
    distinct = ctx.frame[target].dropna().nunique()
    return "binary_classification" if distinct == 2 else "multiclass_classification"


def read_handoff(run_dir: Path) -> Handoff:
    path = run_dir / HANDOFF_FILE
    if not path.exists():
        raise AnalysisInputError(
            "there is no clean data to analyse (no handoff.json): clean the data first, or "
            "read cleaning_report.md for why Gate 2 stopped it"
        )
    return Handoff.model_validate_json(path.read_text(encoding="utf-8"))


def load_context(
    run_dir: Path, *, handoff: Handoff | None = None, out_dir: Path | None = None
) -> AnalysisContext:
    """The clean data of a run folder, checked against its contract, with every column's kind."""
    handoff = handoff or read_handoff(run_dir)
    brief = load_manifest(run_dir).context_brief
    data = run_dir / (handoff.data_file or "clean_data.parquet")
    if not data.exists():
        raise AnalysisInputError(f"the clean data file {data.name} is missing from {run_dir}")

    source = "read_parquet('{}')".format(data.resolve().as_posix().replace("'", "''"))
    with duckdb.connect() as connection:
        broken = [
            check.message
            for check in validate_against_contract(connection, source, handoff.schema_)
            if not check.passed
        ]
        if broken:
            raise AnalysisInputError(
                "the clean data no longer matches its contract: " + "; ".join(broken)
            )
        frame = connection.execute(f"SELECT * FROM {source}").df()
    frame = frame.drop(columns=[c for c in RESERVED_COLUMNS if c in frame.columns])

    ctx = AnalysisContext(
        run_id=handoff.run_id,
        frame=frame,
        target=target_of(handoff, brief),
        task_type="none",
        kinds=analysis_kinds(handoff),
        excluded=set(exclusions(handoff, brief)),
        out_dir=out_dir or run_dir,
    )
    ctx.task_type = task_type_of(ctx)
    return ctx


def describe_data(ctx: AnalysisContext) -> str:
    """One line per column for a prompt, from aggregates only.

    No value of an identifier, personal or free-text column is shown, and a
    category value appears only when at least the minimum group size has it.
    """
    lines = [f"{len(ctx.frame)} rows."]
    for column, kind in ctx.kinds.items():
        values = ctx.frame[column]
        marks = (" (TARGET)" if column == ctx.target else "") + (
            " (EXCLUDED)" if column in ctx.excluded else ""
        )
        head = f"- {column}: {KIND_LABELS[kind]}{marks}"
        if kind in ("id", "personal", "text"):
            lines.append(f"{head}; never analysed")
            continue
        parts = []
        missing = float(values.isna().mean())
        if missing:
            parts.append(f"{missing:.1%} missing")
        present = values.dropna()
        flags = as_yes_no(values) if kind == "boolean" or column == ctx.target else None
        if flags is not None and len(present):
            yes = int(flags.sum())
            parts.append(
                f"yes in {yes} rows ({yes / len(present):.1%}), no in {len(present) - yes}"
            )
        elif kind == "numeric" and len(present):
            x = pd.to_numeric(present).astype(float)
            parts.append(f"median {fmt(x.median())}, range {fmt(x.min())} to {fmt(x.max())}")
        elif kind == "date" and len(present):
            earliest, latest = (
                pd.Timestamp(v).date().isoformat() for v in (present.min(), present.max())
            )
            parts.append(f"{earliest} to {latest}")
        elif len(present):
            counts = present.astype(str).value_counts()
            shown = [f"{value} ({n})" for value, n in counts.items() if n >= ctx.min_group_n][:8]
            parts.append(
                f"{len(counts)} values" + (f", most common: {', '.join(shown)}" if shown else "")
            )
        lines.append(head + (f"; {'; '.join(parts)}" if parts else ""))
    return "\n".join(lines)
