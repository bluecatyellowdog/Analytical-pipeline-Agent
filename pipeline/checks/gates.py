"""Quality gates.

Spec 5.1: every gate runs contract checks, then invariant checks, then judgment,
and reports one outcome under one policy. A gate never repairs anything — it
decides whether the run may continue, and says exactly why.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.checks.schema import (
    build_contract,
    check_columns,
    check_primary_key,
    check_row_loss,
    check_target,
    check_types,
    key_column,
)
from pipeline.contracts.audit import CheckResult, GateResult, OpRecord
from pipeline.contracts.cleaning import CleaningPlan
from pipeline.contracts.common import CheckLayer, Flag, GateOutcome, Severity
from pipeline.contracts.context_brief import ClaimResult, ContextBrief
from pipeline.contracts.diagnosis import Diagnosis
from pipeline.contracts.handoff import ColumnSchema
from pipeline.contracts.ingestion import LoadResult
from pipeline.contracts.profile import DatasetProfile
from pipeline.ingestion.raw import RawStore, RawTable
from pipeline.ingestion.types import COVERAGE_THRESHOLD

REJECT_RATE_WARN = 0.001
"""Spec Gate 1: parse errors below 0.1% pass quietly."""

REJECT_RATE_BLOCK = 0.01
"""Above 1% the file is not what it was taken to be; stop."""

LAYER_ORDER = {"contract": 0, "invariant": 1, "judgment": 2}


def outcome_for(checks: Sequence[CheckResult]) -> GateOutcome:
    """One policy for every gate: the worst failure decides.

    A failed `block` check stops the run, a failed `warn` check lets it continue
    with the issue prominent in the report, and `retry` is reserved for agent
    stages where sending the error back can plausibly fix it.
    """
    failed = [c for c in checks if not c.passed]
    if any(c.severity == "block" for c in failed):
        return "block"
    if any(c.severity == "warn" for c in failed):
        return "flag"
    return "pass"


def build_gate(
    name: str,
    checks: Sequence[CheckResult],
    *,
    flags: Sequence[Flag] = (),
    attempt: int = 1,
) -> GateResult:
    """Assemble a gate result with its checks in layer order."""
    ordered = sorted(checks, key=lambda c: LAYER_ORDER[c.layer])
    return GateResult(
        gate=name,
        outcome=outcome_for(ordered),
        checks=list(ordered),
        flags=list(flags),
        attempt=attempt,
    )


def checked(
    name: str, layer: CheckLayer, severity: Severity, problems: list[str], ok: str
) -> CheckResult:
    """A check from the problems it found: passed when there are none, each one listed."""
    return CheckResult(
        name=name,
        layer=layer,
        passed=not problems,
        severity=severity,
        message="; ".join(problems) if problems else ok,
        observed={"problems": problems} if problems else {},
    )


def problems_of(gate: GateResult) -> list[str]:
    """Every failure as one line, for an agent or the user to act on."""
    return [p for check in gate.failures for p in check.observed.get("problems", [check.message])]


def gate_1_ingestion(
    load_result: LoadResult,
    profile: DatasetProfile,
    claim_results: Sequence[ClaimResult] = (),
) -> GateResult:
    """Gate 1: is what was loaded faithfully the file that was handed in?

    Everything downstream assumes this gate passed, so its checks are about
    completeness and provenance rather than data quality.
    """
    checks: list[CheckResult] = [
        _rows_accounted_for(load_result),
        _reject_rate(load_result),
        _unique_column_names(load_result),
        _provenance_hashes(load_result),
        _row_count_matches_profile(load_result, profile),
        _type_coverage(profile),
    ]
    if claim_results:
        checks.append(_brief_claims(claim_results))

    flags = [*load_result.flags, *profile.flags]
    for column in profile.columns:
        flags.extend(column.flags)
    return build_gate("gate_1_ingestion", checks, flags=flags)


def _rows_accounted_for(load_result: LoadResult) -> CheckResult:
    total = load_result.n_loaded + load_result.n_rejected
    passed = total == load_result.n_source_rows
    return CheckResult(
        name="rows_accounted_for",
        layer="invariant",
        passed=passed,
        severity="block",
        message=(
            f"{load_result.n_loaded:,} loaded + {load_result.n_rejected:,} rejected "
            f"= {total:,} of {load_result.n_source_rows:,} source rows."
        ),
        observed={
            "n_loaded": load_result.n_loaded,
            "n_rejected": load_result.n_rejected,
            "n_source_rows": load_result.n_source_rows,
        },
    )


def _reject_rate(load_result: LoadResult) -> CheckResult:
    rate = load_result.reject_rate
    if rate > REJECT_RATE_BLOCK:
        severity, passed = "block", False
    elif rate > REJECT_RATE_WARN:
        severity, passed = "warn", False
    else:
        severity, passed = "block", True
    return CheckResult(
        name="reject_rate_within_threshold",
        layer="invariant",
        passed=passed,
        severity=severity,  # type: ignore[arg-type]
        message=(
            f"{load_result.n_rejected:,} of {load_result.n_source_rows:,} lines could not be "
            f"parsed ({rate:.2%}); warn above {REJECT_RATE_WARN:.1%}, "
            f"block above {REJECT_RATE_BLOCK:.0%}."
        ),
        observed={"reject_rate": rate, "n_rejected": load_result.n_rejected},
    )


def _unique_column_names(load_result: LoadResult) -> CheckResult:
    duplicates = sorted(
        {name for name in load_result.columns if load_result.columns.count(name) > 1}
    )
    return CheckResult(
        name="unique_column_names",
        layer="contract",
        passed=not duplicates,
        severity="block",
        message=(
            f"{len(load_result.columns)} column names are unique."
            if not duplicates
            else f"Duplicate column names survived loading: {duplicates}."
        ),
        observed={"columns": load_result.columns, "renamed": len(load_result.renames)},
    )


def _provenance_hashes(load_result: LoadResult) -> CheckResult:
    """Without both hashes a run cannot prove which bytes it read."""
    has_file = len(load_result.file.file_hash) == 64
    has_data = len(load_result.data_hash) == 64
    return CheckResult(
        name="provenance_hashes_present",
        layer="contract",
        passed=has_file and has_data,
        severity="block",
        message=(
            "File and data hashes recorded."
            if has_file and has_data
            else f"Missing hashes (file: {has_file}, data: {has_data})."
        ),
        observed={
            "file_hash": load_result.file.file_hash[:12],
            "data_hash": load_result.data_hash[:12],
        },
    )


def _row_count_matches_profile(load_result: LoadResult, profile: DatasetProfile) -> CheckResult:
    passed = profile.n_rows == load_result.n_loaded
    return CheckResult(
        name="profile_matches_loaded_rows",
        layer="contract",
        passed=passed,
        severity="block",
        message=(
            f"Profile describes {profile.n_rows:,} rows; {load_result.n_loaded:,} were loaded."
        ),
        observed={"profile_rows": profile.n_rows, "loaded_rows": load_result.n_loaded},
    )


def _type_coverage(profile: DatasetProfile) -> CheckResult:
    """Spec Gate 1: a type must cover 95% of values, or the column must be flagged.

    Low coverage is not a failure by itself — it is a failure to *notice*, so
    this check passes as long as every uncertain column carries a flag.
    """
    unflagged = [
        column.name
        for column in profile.columns
        if column.type_proposal.coverage < COVERAGE_THRESHOLD
        and column.type_proposal.proposed not in {"id", "text", "categorical", "empty"}
        and not column.flags
    ]
    covered = sum(1 for c in profile.columns if c.type_proposal.coverage >= COVERAGE_THRESHOLD)
    return CheckResult(
        name="type_coverage_or_flagged",
        layer="invariant",
        passed=not unflagged,
        severity="block",
        message=(
            f"{covered}/{len(profile.columns)} columns have a type covering "
            f"{COVERAGE_THRESHOLD:.0%} of values; the rest are flagged."
            if not unflagged
            else f"Columns below the coverage threshold without a flag: {unflagged}."
        ),
        observed={"n_columns": len(profile.columns), "n_confident": covered},
    )


def _brief_claims(claim_results: Sequence[ClaimResult]) -> CheckResult:
    """A wrong brief is a finding, not a blocker: the human corrects it at approval."""
    failed = [r for r in claim_results if r.status == "fail"]
    unchecked = [r for r in claim_results if r.status == "not_checkable"]
    return CheckResult(
        name="brief_claims_hold",
        layer="judgment",
        passed=not failed,
        severity="warn",
        message=(
            f"All {len(claim_results)} checkable claims in the brief hold."
            if not failed
            else f"{len(failed)} claim(s) from the brief do not hold: "
            + "; ".join(r.message for r in failed[:3])
        ),
        observed={
            "n_claims": len(claim_results),
            "n_failed": len(failed),
            "n_not_checkable": len(unchecked),
        },
    )


def gate_2_cleaning(
    store: RawStore,
    raw: RawTable,
    *,
    clean_table: str,
    records: Sequence[OpRecord],
    reconciliation: Sequence[CheckResult],
    plan: CleaningPlan,
    diagnosis: Diagnosis,
    brief: ContextBrief,
    null_tokens: dict[str, list[str]],
    attempt: int = 1,
) -> tuple[GateResult, list[ColumnSchema]]:
    """Gate 2: is the clean data what the diagnosis said it should become?

    Contract layer: types and columns against the diagnosis. Invariant layer:
    primary key, target never imputed, row loss, plus every reconciliation check
    the executor already ran (row accounting, cell diff, replay). Returns the
    gate and the verified schema contract that travels with the handoff.
    """
    key = key_column(brief, diagnosis, raw.columns)
    contract = build_contract(
        store,
        clean_table,
        raw_columns=raw.columns,
        records=records,
        diagnosis=diagnosis,
        key=key,
    )
    contract, types = check_types(contract, plan, diagnosis)
    target = brief.target if brief.target in raw.columns else None
    checks = [
        types,
        check_columns(contract, diagnosis, records),
        check_primary_key(store, raw, clean_table, key, records),
        check_target(store, raw, clean_table, target, records, null_tokens.get(target or "", [])),
        check_row_loss(raw.n_rows, records),
        *reconciliation,
    ]
    flags = [flag for record in records for flag in record.flags]
    return build_gate("gate_2_cleaning", checks, flags=flags, attempt=attempt), contract
