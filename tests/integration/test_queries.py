"""The reviewer's query tool: aggregate answers only, personal values masked, nothing changed.

Run against the Danish fixture before and after a real cleaning plan. The point
is less that each query kind computes the right number (it does) than that no
query can return a row, reveal a personal value, filter on one, or touch data.
"""

from __future__ import annotations

import pytest

from pipeline.contracts.cleaning import CleaningPlan
from pipeline.contracts.review import Query
from pipeline.ingestion.loaders import load
from pipeline.ingestion.raw import RawStore
from pipeline.ops.plan import execute_plan
from pipeline.tools.queries import QueryContext, QueryError, run_query
from tests.unit.test_coder import GOOD_STEPS

pytestmark = pytest.mark.integration


@pytest.fixture
def setup(fixtures_dir):
    with RawStore.open() as store:
        _, raw = load(fixtures_dir / "danish_semicolon.csv", store)
        run = execute_plan(store, raw, CleaningPlan(summary="t", steps=GOOD_STEPS), run_id="q")
        ctx = QueryContext(
            store=store,
            raw_table=raw.name,
            clean_table=run.clean_table,
            masked=frozenset({"kunde_id", "email"}),
        )
        yield store, raw, ctx


def ask(ctx, **fields) -> str:
    return run_query(ctx, Query(id="q1", purpose="test", **fields))


class TestAnswers:
    def test_count_with_a_condition(self, setup) -> None:
        _, _, ctx = setup
        assert "3 of 12 rows" in ask(
            ctx, kind="count", where=[{"column": "beloeb", "op": "equals", "value": 999}]
        )

    def test_count_on_raw_compares_text(self, setup) -> None:
        _, _, ctx = setup
        result = ask(
            ctx,
            kind="count",
            table="raw",
            where=[{"column": "alder", "op": "equals", "value": "NA"}],
        )
        assert "2 of 12 rows" in result

    def test_summary_of_a_date_column(self, setup) -> None:
        _, _, ctx = setup
        result = ask(ctx, kind="summary", column="dato")
        assert "min 2019-01-30" in result and "max 2024-05-25" in result

    def test_summary_of_a_number_includes_quartiles(self, setup) -> None:
        _, _, ctx = setup
        result = ask(ctx, kind="summary", column="beloeb")
        assert "median" in result and "max 250000" in result

    def test_group_count_by_year(self, setup) -> None:
        _, _, ctx = setup
        result = ask(ctx, kind="group_count", column="dato", bucket="year")
        counts = [int(part.split(": ")[1]) for part in result.split("\n")[1:] if ": " in part]
        assert sum(counts) == 12

    def test_compare_shows_how_raw_values_became_clean_ones(self, setup) -> None:
        _, _, ctx = setup
        result = ask(ctx, kind="compare", column="beloeb")
        assert "'12.345,50' -> 12345.5" in result

    def test_results_are_capped(self, setup) -> None:
        _, _, ctx = setup
        result = ask(ctx, kind="value_counts", column="beloeb", limit=2)
        assert len([line for line in result.split("\n") if line.startswith("  ")]) == 2
        assert "more" in result


class TestPrivacy:
    def test_personal_values_come_back_as_their_shape(self, setup) -> None:
        _, _, ctx = setup
        result = ask(ctx, kind="value_counts", column="email", limit=20)
        assert "@example.dk" not in result
        assert "aaaa.aaaaaa@aaaaaaa.aa" in result

    def test_identifiers_are_masked_in_comparisons(self, setup) -> None:
        _, _, ctx = setup
        result = ask(ctx, kind="compare", column="kunde_id")
        assert "00417" not in result and "99999" in result

    def test_personal_columns_cannot_be_filtered_by_value(self, setup) -> None:
        """Otherwise 'how many rows have email = anna@...' would confirm a person is there."""
        _, _, ctx = setup
        with pytest.raises(QueryError, match="personal"):
            ask(ctx, kind="count", where=[{"column": "email", "op": "equals", "value": "a@b.dk"}])
        assert "of 12 rows" in ask(ctx, kind="count", where=[{"column": "email", "op": "is_null"}])

    def test_injection_in_a_value_changes_nothing(self, setup) -> None:
        store, raw, ctx = setup
        hostile = "x'; DROP TABLE raw.danish_semicolon; --"
        result = ask(
            ctx, kind="count", where=[{"column": "dato", "op": "equals", "value": hostile}]
        )
        assert "0 of 12 rows" in result
        store.verify(raw)

    def test_unknown_columns_are_refused_with_the_real_ones(self, setup) -> None:
        _, _, ctx = setup
        with pytest.raises(QueryError, match="kunde_id"):
            ask(ctx, kind="summary", column='x"; DROP TABLE t; --')


class TestRefusals:
    def test_bucket_needs_a_date_column(self, setup) -> None:
        _, _, ctx = setup
        with pytest.raises(QueryError, match="date"):
            ask(ctx, kind="group_count", table="raw", column="dato", bucket="year")
