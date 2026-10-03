"""The op library and its accounting.

The point of these tests is not that each op computes the right value — though
they check that too — but that an op cannot change data without the record
saying so, and cannot lose values without being told to.
"""

from __future__ import annotations

import pytest

from pipeline.ingestion.raw import RawStore
from pipeline.ops import OpContext, OpError, apply_op, finalize, op_names, validate_params

RUN_ID = "run_test"


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


@pytest.fixture
def ctx(store):
    """A small table with one of everything the ops care about."""
    table = store.create_text_table(
        "t",
        ["id", "alder", "beloeb", "dato", "by", "note"],
        [
            ["001", "31", "12.345,50", "03-04-2021", "Nord", "a"],
            ["002", "NA", "999", "15-07-2021", "nord", "b"],
            ["003", "54", "7.800,00", "28-02-2020", "SYD", " c "],
            ["004", "23", "250.000,00", "13-11-2022", "syd", "d"],
            ["004", "23", "250.000,00", "13-11-2022", "syd", "d"],
        ],
        [2, 3, 4, 5, 6],
    )
    return OpContext.from_raw(store, table, RUN_ID), table


def run(ctx, name, params=None, **kwargs):
    return apply_op(ctx, name, params or {}, **kwargs)


class TestLibrary:
    def test_the_library_has_the_planned_ops(self) -> None:
        """15 library ops plus custom_code for the long tail (D022)."""
        assert len(op_names()) == 16
        assert "custom_code" in op_names()
        assert {"cast", "impute", "drop_duplicates", "clip_outliers", "drop_column"} <= set(
            op_names()
        )

    def test_an_unknown_op_is_refused(self, ctx) -> None:
        with pytest.raises(OpError, match="unknown op"):
            run(ctx[0], "delete_everything")

    def test_bad_parameters_are_refused_before_anything_runs(self, ctx) -> None:
        with pytest.raises(OpError, match="invalid parameters"):
            validate_params("cast", {"column": "alder", "to": "not_a_type"})

    def test_unknown_parameters_are_refused(self) -> None:
        with pytest.raises(OpError, match="invalid parameters"):
            validate_params("drop_column", {"column": "x", "reason": "y", "force": True})

    def test_an_op_cannot_name_a_column_that_does_not_exist(self, ctx) -> None:
        with pytest.raises(OpError, match="not in table"):
            run(ctx[0], "trim_whitespace", {"columns": ["nonexistent"]})


class TestAccounting:
    def test_every_op_records_the_sql_that_ran(self, ctx) -> None:
        new_ctx, record = run(ctx[0], "normalize_decimal", {"column": "beloeb"})
        assert record.code.startswith("CREATE TABLE")
        assert "beloeb" in record.code
        assert record.output_hash != record.input_hash

    def test_before_and_after_stats_cover_every_column(self, ctx) -> None:
        _, record = run(ctx[0], "trim_whitespace", {"columns": ["note"]})
        assert set(record.before) == set(ctx[0].columns)
        assert set(record.after) == set(ctx[0].columns)
        assert record.before["note"].value_hash != record.after["note"].value_hash
        assert record.before["by"].value_hash == record.after["by"].value_hash

    def test_only_the_targeted_column_changes(self, ctx) -> None:
        _, record = run(ctx[0], "trim_whitespace", {"columns": ["note"]})
        assert record.cells_changed == {"note": 1}
        assert record.target_columns == ["note"]

    def test_row_accounting_balances_on_every_op(self, ctx) -> None:
        current, records = ctx[0], []
        for name, params in [
            ("normalize_decimal", {"column": "beloeb"}),
            ("drop_duplicates", {}),
            ("drop_column", {"column": "note", "reason": "not needed"}),
        ]:
            current, record = run(current, name, params)
            records.append(record)
            assert record.rows_before == record.rows_after + record.rows_dropped

    def test_rationale_is_stored_but_never_a_number(self, ctx) -> None:
        _, record = run(
            ctx[0], "trim_whitespace", {"columns": ["note"]}, rationale="Export padded the field."
        )
        assert record.rationale == "Export padded the field."


class TestNoSilentCoercion:
    def test_a_cast_that_would_lose_values_is_refused(self, ctx) -> None:
        """'NA' does not parse as an integer, so the cast must not quietly null it."""
        with pytest.raises(OpError, match="would introduce nulls"):
            run(ctx[0], "cast", {"column": "alder", "to": "integer"})

    def test_the_same_cast_is_allowed_when_the_loss_is_declared(self, ctx) -> None:
        new_ctx, record = run(
            ctx[0], "cast", {"column": "alder", "to": "integer", "on_error": "null"}
        )
        assert record.new_nulls == {"alder": 1}
        assert "cast_drops_unparseable" in {f.code for f in record.flags}

    def test_a_clean_cast_needs_no_declaration(self, ctx) -> None:
        current, _ = run(ctx[0], "tokens_to_null", {"columns": ["alder"]})
        _, record = run(current, "cast", {"column": "alder", "to": "integer"})
        assert record.new_nulls == {}

    def test_a_date_cast_must_state_its_format(self) -> None:
        with pytest.raises(OpError, match="requires an explicit format"):
            validate_params("cast", {"column": "dato", "to": "date"})


class TestValueOps:
    def test_normalize_decimal_reads_danish_amounts(self, ctx, store) -> None:
        new_ctx, _ = run(ctx[0], "normalize_decimal", {"column": "beloeb"})
        assert store.column_values(new_ctx.table, "beloeb")[:3] == [12345.50, 999.0, 7800.0]

    def test_parse_dates_uses_the_given_format(self, ctx, store) -> None:
        import datetime as dt

        new_ctx, _ = run(ctx[0], "parse_dates", {"column": "dato", "format": "%d-%m-%Y"})
        assert store.column_values(new_ctx.table, "dato")[0] == dt.date(2021, 4, 3)

    def test_tokens_to_null_declares_the_nulls_it_creates(self, ctx) -> None:
        _, record = run(ctx[0], "tokens_to_null", {"columns": ["alder"]})
        assert record.new_nulls == {"alder": 1}

    def test_standardize_categories_folds_spellings(self, ctx, store) -> None:
        new_ctx, record = run(
            ctx[0],
            "standardize_categories",
            {"column": "by", "mapping": {"nord": "Nord", "syd": "Syd"}},
        )
        assert store.column_values(new_ctx.table, "by") == ["Nord", "Nord", "Syd", "Syd", "Syd"]
        # 'Nord' was already correct; the other four spellings changed.
        assert record.cells_changed == {"by": 4}

    def test_replace_sentinel_is_flagged_as_a_judgment_call(self, ctx) -> None:
        _, record = run(ctx[0], "replace_sentinel", {"column": "beloeb", "values": ["999"]})
        assert record.new_nulls == {"beloeb": 1}
        assert "sentinel_reinterpreted" in {f.code for f in record.flags}

    def test_strip_currency_then_normalize(self, ctx, store) -> None:
        table = store.create_text_table("m", ["v"], [["1.234,50 kr"], ["800,00 DKK"]], [2, 3])
        current = OpContext.from_raw(store, table, RUN_ID)
        current, _ = run(current, "strip_currency", {"column": "v"})
        current, _ = run(current, "normalize_decimal", {"column": "v"})
        assert store.column_values(current.table, "v") == [1234.50, 800.0]


class TestJudgmentCalls:
    def test_imputation_records_the_value_it_used(self, ctx, store) -> None:
        current, _ = run(ctx[0], "tokens_to_null", {"columns": ["alder"]})
        current, _ = run(current, "cast", {"column": "alder", "to": "integer"})
        current, record = run(current, "impute", {"column": "alder", "strategy": "median"})

        flag = next(f for f in record.flags if f.code == "imputation")
        assert flag.details["n_filled"] == 1
        assert flag.details["strategy"] == "median"
        assert str(flag.details["fill_value"]) in record.code

    def test_imputation_on_an_empty_column_is_refused(self, store) -> None:
        table = store.create_text_table("e", ["a", "b"], [[None, "1"], [None, "2"]], [2, 3])
        current = OpContext.from_raw(store, table, RUN_ID)
        with pytest.raises(OpError, match="no value could be computed"):
            run(current, "impute", {"column": "a", "strategy": "mean"})

    def test_clipping_reports_how_many_values_moved(self, ctx, store) -> None:
        current, _ = run(ctx[0], "normalize_decimal", {"column": "beloeb"})
        current, record = run(current, "clip_outliers", {"column": "beloeb", "upper": 100000})

        flag = next(f for f in record.flags if f.code == "outliers_clipped")
        assert flag.details["n_clipped"] == 2
        assert max(store.column_values(current.table, "beloeb")) == 100000

    def test_dropping_a_column_needs_a_reason_and_raises_a_flag(self, ctx) -> None:
        with pytest.raises(OpError, match="invalid parameters"):
            validate_params("drop_column", {"column": "note"})

        new_ctx, record = run(ctx[0], "drop_column", {"column": "note", "reason": "92% missing"})
        assert "note" not in new_ctx.columns
        assert "column_dropped" in {f.code for f in record.flags}

    def test_the_last_column_cannot_be_dropped(self, store) -> None:
        table = store.create_text_table("one", ["only"], [["x"]], [2])
        current = OpContext.from_raw(store, table, RUN_ID)
        with pytest.raises(OpError, match="last remaining column"):
            run(current, "drop_column", {"column": "only", "reason": "why not"})


class TestRowOps:
    def test_duplicates_are_dropped_keeping_the_first_row(self, ctx, store) -> None:
        new_ctx, record = run(ctx[0], "drop_duplicates", {})
        assert record.rows_dropped == 1
        assert store.column_values(new_ctx.table, "id") == ["001", "002", "003", "004"]

    def test_dropped_rows_are_kept_with_their_reason(self, ctx, store) -> None:
        new_ctx, _ = run(ctx[0], "drop_duplicates", {})
        rows = store.connection.execute(
            f"SELECT op_name, reason FROM {new_ctx.dropped_table}"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "drop_duplicates"
        assert "duplicate" in rows[0][1]

    def test_drop_rows_takes_a_condition_not_sql(self, ctx, store) -> None:
        new_ctx, record = run(
            ctx[0],
            "drop_rows",
            {"column": "by", "condition": "equals", "value": "syd", "reason": "out of scope"},
        )
        assert record.rows_dropped == 2
        assert "syd" not in store.column_values(new_ctx.table, "by")
        assert "rows_dropped" in {f.code for f in record.flags}

    def test_a_comparison_without_a_value_is_refused(self) -> None:
        with pytest.raises(OpError, match="needs a value"):
            validate_params(
                "drop_rows", {"column": "alder", "condition": "greater_than", "reason": "x"}
            )

    def test_dropping_rows_by_null(self, ctx, store) -> None:
        current, _ = run(ctx[0], "tokens_to_null", {"columns": ["alder"]})
        current, record = run(
            current,
            "drop_rows",
            {"column": "alder", "condition": "is_null", "reason": "age is required"},
        )
        assert record.rows_dropped == 1


class TestColumnOps:
    def test_renaming_keeps_the_values(self, ctx, store) -> None:
        new_ctx, record = run(ctx[0], "rename_column", {"column": "by", "to": "region"})
        assert "region" in new_ctx.columns and "by" not in new_ctx.columns
        assert store.column_values(new_ctx.table, "region")[0] == "Nord"

    def test_renaming_onto_an_existing_column_is_refused(self, ctx) -> None:
        with pytest.raises(OpError, match="that column exists"):
            run(ctx[0], "rename_column", {"column": "by", "to": "note"})


class TestVersioning:
    def test_each_op_writes_a_new_version(self, ctx) -> None:
        """Versioned tables are what make before/after diffs and replay possible."""
        current, _ = run(ctx[0], "trim_whitespace", {"columns": ["note"]})
        assert current.table.endswith("__v1")
        current, _ = run(current, "normalize_decimal", {"column": "beloeb"})
        assert current.table.endswith("__v2")

    def test_the_raw_table_is_never_written_to(self, ctx, store) -> None:
        current, table = ctx
        for name, params in [
            ("trim_whitespace", {"columns": ["note"]}),
            ("drop_duplicates", {}),
        ]:
            current, _ = run(current, name, params)
        store.verify(table)

    def test_finalize_publishes_the_last_version(self, ctx, store) -> None:
        current, _ = run(ctx[0], "drop_duplicates", {})
        name = finalize(current)
        assert name == "clean.t"
        assert store.row_count(name) == 4
