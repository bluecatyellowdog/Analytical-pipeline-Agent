"""Reconciliation (spec 5.2).

Each test does the same thing twice: runs a legitimate cleaning sequence and
checks it reconciles, then breaks one thing behind the ops' backs and checks
the corresponding check notices. A check that never fails proves nothing.
"""

from __future__ import annotations

import pytest

from pipeline.checks.reconciliation import (
    check_cell_diff,
    check_distribution_shift,
    check_dropped_rows_have_reasons,
    check_no_unexplained_nulls,
    check_replay,
    check_row_accounting,
    check_untouched_columns,
    final_name_of,
    population_stability_index,
    reconcile,
)
from pipeline.ingestion.loaders import load
from pipeline.ingestion.raw import RawStore
from pipeline.ops import OpContext, apply_op, finalize

pytestmark = pytest.mark.integration

RUN_ID = "run_recon"

CLEANING = [
    ("tokens_to_null", {"columns": ["alder", "email"]}),
    ("cast", {"column": "alder", "to": "integer"}),
    ("normalize_decimal", {"column": "beloeb"}),
    ("parse_dates", {"column": "dato", "format": "%d-%m-%Y"}),
    ("replace_sentinel", {"column": "beloeb", "values": ["999.0"]}),
    ("drop_column", {"column": "email", "reason": "41% missing and flagged as PII"}),
]


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


@pytest.fixture
def cleaned(fixtures_dir, store):
    """A realistic cleaning run over the Danish fixture."""
    _, raw = load(fixtures_dir / "danish_semicolon.csv", store)
    ctx = OpContext.from_raw(store, raw, RUN_ID)
    records = []
    for name, params in CLEANING:
        ctx, record = apply_op(ctx, name, params)
        records.append(record)
    final = finalize(ctx)
    return raw, ctx, records, final


class TestHappyPath:
    def test_every_check_passes_on_a_legitimate_run(self, store, cleaned) -> None:
        raw, ctx, records, final = cleaned
        checks = reconcile(
            store, raw, final, records, run_id=RUN_ID, dropped_table=ctx.dropped_table
        )
        failures = [c for c in checks if not c.passed]
        assert not failures, [f"{c.name}: {c.message}" for c in failures]

    def test_replay_from_raw_reproduces_the_output(self, store, cleaned) -> None:
        """The op log is not a description of the run; it is the run."""
        raw, _, records, final = cleaned
        check = check_replay(store, raw, final, records, RUN_ID)
        assert check.passed, check.message
        assert check.observed["n_ops"] == len(CLEANING)

    def test_the_raw_layer_survives_the_whole_run(self, store, cleaned) -> None:
        raw, _, _, _ = cleaned
        store.verify(raw)

    def test_cleaning_actually_did_something(self, store, cleaned) -> None:
        """Guard against the checks passing because nothing happened."""
        _, ctx, records, final = cleaned
        assert "email" not in ctx.columns
        assert sum(sum(r.cells_changed.values()) for r in records) > 0
        assert store.column_values(final, "beloeb")[0] == 12345.50


class TestBrokenRuns:
    def test_a_row_deleted_outside_an_op_breaks_row_accounting(self, store, cleaned) -> None:
        raw, _, records, final = cleaned
        store.connection.execute(f"DELETE FROM {final} WHERE _row_id = 1")

        check = check_row_accounting(store, raw, final, records)
        assert not check.passed
        assert "unaccounted" in check.message

    def test_a_value_changed_outside_an_op_is_caught(self, store, cleaned) -> None:
        """The check the whole design exists for."""
        raw, _, records, final = cleaned
        store.connection.execute(f"UPDATE {final} SET kunde_id = '99999' WHERE _row_id = 1")

        check = check_untouched_columns(store, raw, final, records)
        assert not check.passed
        assert check.observed["changed"] == {"kunde_id": 1}

    def test_an_untargeted_change_is_unaccounted_for_in_the_cell_diff(self, store, cleaned) -> None:
        raw, _, records, final = cleaned
        store.connection.execute(f"UPDATE {final} SET kunde_id = '99999' WHERE _row_id = 2")

        check = check_cell_diff(store, raw, final, records)
        assert not check.passed
        assert check.observed["unaccounted"] == {"kunde_id": 1}

    def test_a_record_understating_its_changes_is_caught(self, store, cleaned) -> None:
        """An op that changed 12 cells but logged 1 is exactly what this catches."""
        raw, _, records, final = cleaned
        for record in records:
            if "beloeb" in record.cells_changed:
                record.cells_changed["beloeb"] = 1

        check = check_cell_diff(store, raw, final, records)
        assert not check.passed
        assert "beloeb" in check.observed["undercounted"]

    def test_replay_notices_a_tampered_result(self, store, cleaned) -> None:
        raw, _, records, final = cleaned
        store.connection.execute(f"UPDATE {final} SET alder = 99 WHERE _row_id = 1")

        check = check_replay(store, raw, final, records, RUN_ID)
        assert not check.passed
        assert check.observed["expected"] != check.observed["actual"]

    def test_replay_notices_a_missing_op(self, store, cleaned) -> None:
        raw, _, records, final = cleaned
        check = check_replay(store, raw, final, records[:-1], f"{RUN_ID}_short")
        assert not check.passed

    def test_a_record_claiming_undeclared_nulls_is_caught(self, cleaned) -> None:
        _, _, records, _ = cleaned
        trim = next(r for r in records if r.op_name == "normalize_decimal")
        trim.new_nulls = {"beloeb": 3}

        check = check_no_unexplained_nulls(records)
        assert not check.passed
        assert "normalize_decimal" in check.observed["offenders"]


class TestDroppedRows:
    def test_dropped_rows_are_stored_with_op_and_reason(self, store, fixtures_dir) -> None:
        _, raw = load(fixtures_dir / "danish_semicolon.csv", store)
        ctx = OpContext.from_raw(store, raw, RUN_ID)
        ctx, record = apply_op(
            ctx,
            "drop_rows",
            {"column": "alder", "condition": "equals", "value": "NA", "reason": "age required"},
        )
        final = finalize(ctx)

        check = check_dropped_rows_have_reasons(store, ctx.dropped_table, [record])
        assert check.passed
        assert check.observed["n_stored"] == record.rows_dropped

        rows = store.connection.execute(
            f"SELECT op_name, reason, kunde_id FROM {ctx.dropped_table}"
        ).fetchall()
        assert all(row[0] == "drop_rows" and row[1] == "age required" for row in rows)
        assert store.row_count(final) + len(rows) == store.row_count(raw.name)

    def test_dropped_rows_without_a_table_is_a_failure(self, store) -> None:
        table = store.create_text_table("t", ["a"], [["1"], ["1"]], [2, 3])
        ctx = OpContext.from_raw(store, table, RUN_ID)
        _, record = apply_op(ctx, "drop_duplicates", {})

        check = check_dropped_rows_have_reasons(store, None, [record])
        assert not check.passed


class TestDistributionShift:
    def test_a_faithful_cleaning_does_not_shift_distributions(self, store, cleaned) -> None:
        raw, _, records, final = cleaned
        assert check_distribution_shift(store, raw, final, records).passed

    def test_psi_rises_when_a_column_is_transformed_heavily(self, store) -> None:
        table = store.create_text_table(
            "psi", ["v"], [[str(i)] for i in range(100)], list(range(2, 102))
        )
        ctx = OpContext.from_raw(store, table, RUN_ID)
        ctx, _ = apply_op(ctx, "cast", {"column": "v", "to": "integer"})
        ctx, _ = apply_op(ctx, "clip_outliers", {"column": "v", "upper": 10})
        final = finalize(ctx)

        psi = population_stability_index(store, table.name, final, "v", "v")
        assert psi is not None and psi > 0.2

    def test_psi_is_none_when_there_is_too_little_to_compare(self, store) -> None:
        table = store.create_text_table("tiny", ["v"], [["1"], ["2"]], [2, 3])
        assert population_stability_index(store, table.name, table.name, "v", "v") is None


class TestRenameTracking:
    def test_a_renamed_column_is_still_reconciled(self, store, fixtures_dir) -> None:
        """Renames must not let a column escape the untouched-column check."""
        _, raw = load(fixtures_dir / "danish_semicolon.csv", store)
        ctx = OpContext.from_raw(store, raw, RUN_ID)
        ctx, record = apply_op(ctx, "rename_column", {"column": "beloeb", "to": "amount"})
        final = finalize(ctx)

        assert final_name_of("beloeb", [record]) == "amount"
        checks = reconcile(store, raw, final, [record], run_id=RUN_ID, include_replay=True)
        assert all(c.passed for c in checks), [c.message for c in checks if not c.passed]

        store.connection.execute(f"UPDATE {final} SET kunde_id = 'x' WHERE _row_id = 1")
        assert not check_untouched_columns(store, raw, final, [record]).passed

    def test_a_dropped_column_is_not_reconciled_against(self, cleaned) -> None:
        _, _, records, _ = cleaned
        assert final_name_of("email", records) is None


class TestShiftOnlyWhereComparable:
    """Raw text that mostly does not read as numbers cannot be compared with clean numbers."""

    def test_danish_formatted_raw_values_are_not_compared(self, store) -> None:
        values = [[f"{1000 + i}.{i:03d},50"] for i in range(90)] + [["999"]] * 12
        table = store.create_text_table("dk", ["v"], values, list(range(2, 104)))
        ctx = OpContext.from_raw(store, table, RUN_ID)
        ctx, _ = apply_op(ctx, "normalize_decimal", {"column": "v"})
        final = finalize(ctx)
        # Only the twelve 999 codes read as numbers in raw: comparing those with the
        # parsed amounts would report a huge, meaningless shift.
        assert population_stability_index(store, table.name, final, "v", "v") is None
