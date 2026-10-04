"""Run a cleaning plan: every step through `apply_op`, finalised only if all of them succeed.

The executor is deterministic and knows nothing about agents. It checks the
whole plan before anything runs, starts every run from raw (clearing what an
earlier attempt left in the clean and audit schemas), stops at the first step
an op refuses, and only then finalises and reconciles. Two invariants are
enforced here rather than in any op, because they are about the plan as a
whole: protected columns (the target) are never filled in or removed, and a
step that changes nothing is reported, since it is almost always a mistake.

Error messages are kept in full, for the audit log. Masking them before an LLM
sees them is the caller's job.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from pipeline.checks.reconciliation import reconcile
from pipeline.contracts.audit import CheckResult, OpRecord
from pipeline.contracts.cleaning import CleaningPlan
from pipeline.ingestion.raw import (
    AUDIT_SCHEMA,
    CLEAN_SCHEMA,
    ROW_ID,
    RawStore,
    RawTable,
    quote_ident,
)
from pipeline.ops.base import OpContext, OpError, apply_op, finalize, validate_params
from pipeline.ops.sandbox import check_code


@dataclass
class PlanRun:
    ok: bool
    records: list[OpRecord] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    """Blocking: the plan was not run, a step was refused, or reconciliation failed."""
    warnings: list[str] = field(default_factory=list)
    """Suspicious but not blocking, e.g. a step that changed nothing."""
    failed_step: int | None = None
    """1-based number of the step that stopped the plan."""
    clean_table: str | None = None
    dropped_table: str | None = None
    checks: list[CheckResult] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)


def validate_steps(plan: CleaningPlan) -> list[str]:
    """Everything wrong with the plan that can be known without running it."""
    problems: list[str] = []
    for number, step in enumerate(plan.steps, start=1):
        label = f"step {number} ({step.op})"
        try:
            validate_params(step.op, step.params)
        except OpError as error:
            problems.append(f"{label}: {error}")
            continue
        if step.op == "custom_code":
            problems.extend(f"{label}: {p}" for p in check_code(step.params.get("code", "")))
    return problems


def execute_plan(
    store: RawStore,
    raw: RawTable,
    plan: CleaningPlan,
    *,
    run_id: str,
    protected: Iterable[str] = (),
) -> PlanRun:
    """Run the plan from raw. Nothing is finalised unless every step and check passes."""
    problems = validate_steps(plan)
    if problems:
        return PlanRun(ok=False, problems=problems)

    base_name = raw.name.split(".", 1)[-1]
    reset_derived_tables(store, base_name)
    ctx = OpContext.from_raw(store, raw, run_id)
    protected_names = set(protected)
    records: list[OpRecord] = []
    warnings: list[str] = []

    for number, step in enumerate(plan.steps, start=1):
        label = f"step {number} ({step.op})"
        rationale = f"[{', '.join(step.addresses)}] {step.rationale}"
        try:
            new_ctx, record = apply_op(ctx, step.op, step.params, rationale=rationale)
        except Exception as error:  # noqa: BLE001 - any failure stops the plan, with its reason
            reason = (
                str(error) if isinstance(error, OpError) else f"{type(error).__name__}: {error}"
            )
            return PlanRun(
                ok=False,
                records=records,
                problems=[f"{label} was refused: {reason}"],
                warnings=warnings,
                failed_step=number,
            )

        violation = _protected_violation(store, ctx, new_ctx, step, protected_names)
        if violation:
            return PlanRun(
                ok=False,
                records=records,
                problems=[f"{label} {violation}"],
                warnings=warnings,
                failed_step=number,
            )

        if _is_no_op(record):
            warnings.append(f"{label} changed nothing; check its parameters or remove it")
        protected_names = _follow_renames(protected_names, step)
        records.append(record)
        ctx = new_ctx

    clean_table = finalize(ctx)
    checks = reconcile(
        store, raw, clean_table, records, run_id=run_id, dropped_table=ctx.dropped_table
    )
    blocking = [c for c in checks if not c.passed and c.severity == "block"]
    if blocking:
        store.connection.execute(f"DROP TABLE IF EXISTS {_qualified(clean_table)}")
        return PlanRun(
            ok=False,
            records=records,
            problems=[f"reconciliation check {c.name} failed: {c.message}" for c in blocking],
            warnings=warnings,
            checks=checks,
        )

    return PlanRun(
        ok=True,
        records=records,
        warnings=warnings,
        clean_table=clean_table,
        dropped_table=ctx.dropped_table,
        checks=checks,
        columns=list(ctx.columns),
    )


def reset_derived_tables(store: RawStore, base_name: str) -> None:
    """Remove what an earlier run of this dataset left in the clean and audit schemas.

    Never touches raw. Without this, rows dropped by a failed attempt would stay
    in the dropped-rows table and break the row accounting of the next one.
    """
    rows = store.connection.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema IN (?, ?) AND (table_name = ? OR starts_with(table_name, ?))",
        [CLEAN_SCHEMA, AUDIT_SCHEMA, base_name, f"{base_name}__"],
    ).fetchall()
    for schema, name in rows:
        store.connection.execute(f"DROP TABLE IF EXISTS {schema}.{quote_ident(name)}")


def _protected_violation(
    store: RawStore, before: OpContext, after: OpContext, step, protected: set[str]
) -> str | None:
    """The target may be cleaned, renamed or have rows dropped; never filled in or removed."""
    for column in sorted(protected):
        if column not in before.columns:
            continue
        if step.op == "rename_column" and step.params.get("column") == column:
            continue
        if column not in after.columns:
            return f"removed the target column {column!r}; the target is never dropped"
        filled = store.connection.execute(
            f"SELECT count(*) FROM {_qualified(before.table)} a "
            f"JOIN {_qualified(after.table)} b USING ({ROW_ID}) "
            f"WHERE a.{quote_ident(column)} IS NULL AND b.{quote_ident(column)} IS NOT NULL"
        ).fetchone()[0]
        if filled:
            return (
                f"filled {filled} missing value(s) in the target {column!r}; "
                f"the target is never imputed"
            )
    return None


def _follow_renames(protected: set[str], step) -> set[str]:
    if step.op == "rename_column" and step.params.get("column") in protected:
        return (protected - {step.params["column"]}) | {step.params["to"]}
    return protected


def _is_no_op(record: OpRecord) -> bool:
    if record.cells_changed or record.new_nulls or record.rows_dropped:
        return False
    if list(record.before) != list(record.after):
        return False
    return all(record.before[c].dtype == record.after[c].dtype for c in record.before)


def _qualified(table: str) -> str:
    schema, _, name = table.rpartition(".")
    return f"{schema or CLEAN_SCHEMA}.{quote_ident(name)}"


__all__ = ["PlanRun", "execute_plan", "reset_derived_tables", "validate_steps"]
