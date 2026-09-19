"""Prove that every change is accounted for, and that nothing else changed.

Spec 5.2. These checks cannot show that the cleaned data is *correct* — nothing
can — but they can show that the cleaned data is exactly what the op log says
it is. What is left over after that is wrong-but-documented decisions, which is
what the review agent and the benchmark are for.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pipeline.contracts.audit import CheckResult, OpRecord
from pipeline.ingestion.raw import ROW_ID, RawStore, RawTable, quote_ident
from pipeline.ops.base import OpContext, apply_op

PSI_FLAG_THRESHOLD = 0.2
"""Spec Gate 2: a population stability index above 0.2 is a real distribution shift."""

PSI_BINS = 10


def _qualified(table: str) -> str:
    schema, _, name = table.rpartition(".")
    return f"{schema or 'clean'}.{quote_ident(name)}"


def final_name_of(column: str, records: Sequence[OpRecord]) -> str | None:
    """Follow a raw column through renames; None if an op dropped it."""
    current = column
    for record in records:
        if record.op_name == "rename_column" and record.params.get("column") == current:
            current = record.params["to"]
        elif record.op_name == "drop_column" and record.params.get("column") == current:
            return None
    return current


def touched_columns(records: Sequence[OpRecord]) -> set[str]:
    """Every column any op declared as a target."""
    return {column for record in records for column in record.target_columns}


def check_row_accounting(
    store: RawStore, raw: RawTable, final_table: str, records: Sequence[OpRecord]
) -> CheckResult:
    """rows_raw = rows_clean + rows_dropped, exactly, with a reason for each drop."""
    rows_raw = store.row_count(raw.name)
    rows_clean = store.row_count(final_table)
    claimed = sum(record.rows_dropped for record in records)
    balanced = rows_raw == rows_clean + claimed

    return CheckResult(
        name="row_accounting",
        layer="invariant",
        passed=balanced,
        severity="block",
        message=(
            f"{rows_raw:,} raw = {rows_clean:,} clean + {claimed:,} dropped."
            if balanced
            else f"{rows_raw:,} raw != {rows_clean:,} clean + {claimed:,} dropped "
            f"({rows_raw - rows_clean - claimed:+,} unaccounted)."
        ),
        observed={"rows_raw": rows_raw, "rows_clean": rows_clean, "rows_dropped": claimed},
    )


def check_dropped_rows_have_reasons(
    store: RawStore, dropped_table: str | None, records: Sequence[OpRecord]
) -> CheckResult:
    """A dropped row without an op and a reason is a row that vanished."""
    expected = sum(record.rows_dropped for record in records)
    if not expected:
        return CheckResult(
            name="dropped_rows_have_reasons",
            layer="invariant",
            passed=True,
            message="No rows were dropped.",
        )
    if dropped_table is None:
        return CheckResult(
            name="dropped_rows_have_reasons",
            layer="invariant",
            passed=False,
            severity="block",
            message=f"{expected:,} row(s) were dropped but no dropped-rows table exists.",
        )

    stored, without_reason = store.connection.execute(
        f"SELECT count(*), count(*) FILTER (WHERE reason IS NULL OR op_name IS NULL) "
        f"FROM {_qualified(dropped_table)}"
    ).fetchone()
    passed = stored == expected and without_reason == 0

    return CheckResult(
        name="dropped_rows_have_reasons",
        layer="invariant",
        passed=passed,
        severity="block",
        message=(
            f"All {stored:,} dropped row(s) are stored with an op and a reason."
            if passed
            else f"{stored:,} row(s) stored for {expected:,} dropped; "
            f"{without_reason:,} without a reason."
        ),
        observed={"n_stored": stored, "n_expected": expected},
    )


def check_cell_diff(
    store: RawStore, raw: RawTable, final_table: str, records: Sequence[OpRecord]
) -> CheckResult:
    """Cells that differ from raw must belong to a column some op claimed to change.

    Per-op counts can exceed the net difference — an op may change a value and a
    later one change it back — so the test is that the net change is covered by
    what the ops claimed, never that a column changed with nothing claiming it.
    """
    unaccounted: dict[str, int] = {}
    undercounted: dict[str, dict[str, int]] = {}
    targeted = touched_columns(records)

    for column in raw.columns:
        final_column = final_name_of(column, records)
        if final_column is None:
            continue

        differing = _count_differing_cells(store, raw.name, final_table, column, final_column)
        if not differing:
            continue

        claimed = sum(
            record.cells_changed.get(column, 0) + record.cells_changed.get(final_column, 0)
            for record in records
        )
        if column not in targeted and final_column not in targeted:
            unaccounted[column] = differing
        elif claimed < differing:
            undercounted[column] = {"claimed": claimed, "observed": differing}

    passed = not unaccounted and not undercounted
    return CheckResult(
        name="cell_diff_accounted_for",
        layer="invariant",
        passed=passed,
        severity="block",
        message=(
            "Every changed cell belongs to a column an op declared."
            if passed
            else f"Unexplained changes: {unaccounted or ''} {undercounted or ''}".strip()
        ),
        observed={"unaccounted": unaccounted, "undercounted": undercounted},
    )


def check_untouched_columns(
    store: RawStore, raw: RawTable, final_table: str, records: Sequence[OpRecord]
) -> CheckResult:
    """Columns no op touched must be identical to raw, row for surviving row."""
    targeted = touched_columns(records)
    changed: dict[str, int] = {}
    checked: list[str] = []

    for column in raw.columns:
        final_column = final_name_of(column, records)
        if final_column is None or column in targeted or final_column in targeted:
            continue
        checked.append(column)
        differing = _count_differing_cells(store, raw.name, final_table, column, final_column)
        if differing:
            changed[column] = differing

    return CheckResult(
        name="untouched_columns_unchanged",
        layer="invariant",
        passed=not changed,
        severity="block",
        message=(
            f"{len(checked)} untouched column(s) are byte-identical to raw."
            if not changed
            else f"Columns changed without any op touching them: {changed}."
        ),
        observed={"n_checked": len(checked), "changed": changed},
    )


def check_replay(
    store: RawStore, raw: RawTable, final_table: str, records: Sequence[OpRecord], run_id: str
) -> CheckResult:
    """Re-run the op log from raw and require the same output hash.

    This is the strongest claim the pipeline makes: the log is not a description
    of what happened, it *is* what happened, and running it again proves it.
    """
    expected = store.data_hash(final_table)
    replay_ctx = OpContext(
        store=store,
        run_id=run_id,
        base_name=f"{raw.name.split('.', 1)[-1]}__replay",
        table=raw.name,
        columns=list(raw.columns),
    )

    try:
        for record in records:
            replay_ctx, _ = apply_op(replay_ctx, record.op_name, record.params)
    except Exception as error:  # noqa: BLE001 - any failure is a failed replay
        return CheckResult(
            name="replay_reproduces_output",
            layer="invariant",
            passed=False,
            severity="block",
            message=f"Replaying the op log failed at {record.op_name!r}: {error}",
            observed={"failed_op": record.op_name},
        )

    actual = store.data_hash(replay_ctx.table)
    passed = actual == expected
    return CheckResult(
        name="replay_reproduces_output",
        layer="invariant",
        passed=passed,
        severity="block",
        message=(
            f"Replaying {len(records)} op(s) from raw reproduces the output hash."
            if passed
            else f"Replay produced {actual[:12]}, expected {expected[:12]}."
        ),
        observed={"expected": expected[:12], "actual": actual[:12], "n_ops": len(records)},
    )


def check_no_unexplained_nulls(records: Sequence[OpRecord]) -> CheckResult:
    """New nulls are allowed only where an op declared it would create them.

    The executor already refuses to run such an op, so this is a second line of
    defence over the log itself, which is what a reviewer actually reads.
    """
    declaring = {"tokens_to_null", "replace_sentinel"}
    offenders = {
        record.op_name: record.new_nulls
        for record in records
        if record.new_nulls
        and record.op_name not in declaring
        and record.params.get("on_error") != "null"
    }
    return CheckResult(
        name="no_unexplained_new_nulls",
        layer="invariant",
        passed=not offenders,
        severity="block",
        message=(
            "No op introduced nulls without declaring it."
            if not offenders
            else f"Ops introduced undeclared nulls: {offenders}."
        ),
        observed={"offenders": offenders},
    )


def population_stability_index(
    store: RawStore,
    raw_table: str,
    final_table: str,
    raw_column: str,
    final_column: str,
    bins: int = PSI_BINS,
) -> float | None:
    """PSI between the raw and cleaned distribution of one column.

    Implemented directly rather than pulled from scipy: it is a dozen lines, and
    the pipeline reports the number in an audit trail, so being able to read the
    definition matters more than the dependency.
    """
    expected = _numeric_values(store, raw_table, raw_column)
    actual = _numeric_values(store, final_table, final_column)
    if len(expected) < bins or not actual:
        return None

    ordered = sorted(expected)
    edges = [ordered[min(int(i * len(ordered) / bins), len(ordered) - 1)] for i in range(1, bins)]
    edges = sorted(set(edges))
    if not edges:
        return None

    def distribute(values: list[float]) -> list[float]:
        counts = [0] * (len(edges) + 1)
        for value in values:
            index = 0
            while index < len(edges) and value > edges[index]:
                index += 1
            counts[index] += 1
        total = len(values)
        # A zero bucket would make the log term infinite; nudge it instead, which
        # is the standard treatment and keeps the index finite and comparable.
        return [max(count / total, 1e-6) for count in counts]

    expected_share = distribute(expected)
    actual_share = distribute(actual)
    return sum((a - e) * math.log(a / e) for e, a in zip(expected_share, actual_share, strict=True))


def check_distribution_shift(
    store: RawStore, raw: RawTable, final_table: str, records: Sequence[OpRecord]
) -> CheckResult:
    """Flag columns whose distribution moved more than cleaning should move it."""
    shifted: dict[str, float] = {}
    measured = 0

    for column in raw.columns:
        final_column = final_name_of(column, records)
        if final_column is None:
            continue
        psi = population_stability_index(store, raw.name, final_table, column, final_column)
        if psi is None:
            continue
        measured += 1
        if psi > PSI_FLAG_THRESHOLD:
            shifted[final_column] = round(psi, 4)

    return CheckResult(
        name="distribution_shift_within_threshold",
        layer="invariant",
        passed=not shifted,
        severity="warn",
        message=(
            f"No large distribution shift in {measured} numeric column(s)."
            if not shifted
            else f"Distribution shifted (PSI > {PSI_FLAG_THRESHOLD}): {shifted}. "
            f"Check that the ops explain it."
        ),
        observed={"psi": shifted, "n_measured": measured},
    )


def reconcile(
    store: RawStore,
    raw: RawTable,
    final_table: str,
    records: Sequence[OpRecord],
    *,
    run_id: str,
    dropped_table: str | None = None,
    include_replay: bool = True,
) -> list[CheckResult]:
    """Run every reconciliation check, cheapest first."""
    checks = [
        check_row_accounting(store, raw, final_table, records),
        check_dropped_rows_have_reasons(store, dropped_table, records),
        check_no_unexplained_nulls(records),
        check_untouched_columns(store, raw, final_table, records),
        check_cell_diff(store, raw, final_table, records),
        check_distribution_shift(store, raw, final_table, records),
    ]
    if include_replay:
        checks.append(check_replay(store, raw, final_table, records, run_id))
    return checks


def _count_differing_cells(
    store: RawStore, raw_table: str, final_table: str, raw_column: str, final_column: str
) -> int:
    """Cells that differ between raw and final, over rows that survived.

    Values are compared as text so that a cast from '31' to 31 does not read as
    a change of value: the check is about *content*, and type changes are
    already visible in the op record's before/after stats.
    """
    row = store.connection.execute(
        f"SELECT count(*) FROM {_qualified(raw_table)} a "
        f"JOIN {_qualified(final_table)} b USING ({ROW_ID}) "
        f"WHERE CAST(a.{quote_ident(raw_column)} AS VARCHAR) "
        f"IS DISTINCT FROM CAST(b.{quote_ident(final_column)} AS VARCHAR)"
    ).fetchone()
    return int(row[0])


def _numeric_values(store: RawStore, table: str, column: str) -> list[float]:
    rows = store.connection.execute(
        f"SELECT TRY_CAST({quote_ident(column)} AS DOUBLE) FROM {_qualified(table)} "
        f"WHERE TRY_CAST({quote_ident(column)} AS DOUBLE) IS NOT NULL"
    ).fetchall()
    return [float(row[0]) for row in rows]
