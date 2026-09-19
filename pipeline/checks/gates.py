"""Quality gates.

Spec 5.1: every gate runs contract checks, then invariant checks, then judgment,
and reports one outcome under one policy. A gate never repairs anything — it
decides whether the run may continue, and says exactly why.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.contracts.audit import CheckResult, GateResult
from pipeline.contracts.common import Flag, GateOutcome
from pipeline.contracts.context_brief import ClaimResult
from pipeline.contracts.ingestion import LoadResult
from pipeline.contracts.profile import DatasetProfile
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
