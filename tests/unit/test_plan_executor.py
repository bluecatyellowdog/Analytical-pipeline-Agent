"""The plan executor: runs a whole cleaning plan, or none of it.

A plan is checked before anything runs, executed step by step through
`apply_op`, stopped at the first refused step, and only finalised and
reconciled when every step succeeded. Each run starts from raw, so a failed
attempt leaves nothing behind that a later attempt could be confused by.
"""

from __future__ import annotations

import pytest

from pipeline.contracts.cleaning import CleaningPlan
from pipeline.ingestion.loaders import load
from pipeline.ingestion.raw import RawStore
from pipeline.ops.plan import execute_plan, validate_steps

RUN_ID = "run_plan"


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


@pytest.fixture
def raw(store, fixtures_dir):
    _, raw_table = load(fixtures_dir / "danish_semicolon.csv", store)
    return raw_table


def step(op: str, addresses=("I1",), rationale="test", **params) -> dict:
    return {"op": op, "params": params, "addresses": list(addresses), "rationale": rationale}


def plan(*steps: dict, skipped=()) -> CleaningPlan:
    return CleaningPlan(summary="test plan", steps=list(steps), skipped=list(skipped))


GOOD = (
    step("tokens_to_null", columns=["alder"], tokens=["NA"]),
    step("cast", column="alder", to="integer"),
    step("normalize_decimal", ("I2",), column="beloeb"),
    step("parse_dates", ("I3",), column="dato", format="%d-%m-%Y"),
)


class TestValidation:
    def test_a_good_plan_has_no_problems(self) -> None:
        assert validate_steps(plan(*GOOD)) == []

    def test_unknown_ops_and_bad_params_are_reported_per_step(self) -> None:
        problems = validate_steps(
            plan(step("magic"), step("cast", column="alder", to="not_a_type"))
        )
        assert any("step 1" in p and "magic" in p for p in problems)
        assert any("step 2" in p and "cast" in p for p in problems)

    def test_custom_code_is_checked_before_anything_runs(self) -> None:
        code = "import os\ndef transform(df):\n    return df\n"
        problems = validate_steps(plan(step("custom_code", code=code)))
        assert any("step 1" in p and "import" in p for p in problems)


class TestExecution:
    def test_a_good_plan_runs_finalises_and_reconciles(self, store, raw) -> None:
        run = execute_plan(store, raw, plan(*GOOD), run_id=RUN_ID)

        assert run.ok, run.problems
        assert len(run.records) == len(GOOD)
        assert run.clean_table == "clean.danish_semicolon"
        assert store.column_values(run.clean_table, "alder")[:2] == [31, None]
        blocking = [c for c in run.checks if not c.passed and c.severity == "block"]
        assert not blocking, blocking
        assert any(c.name == "replay_reproduces_output" for c in run.checks)
        store.verify(raw)

    def test_the_rationale_and_addresses_reach_the_record(self, store, raw) -> None:
        run = execute_plan(store, raw, plan(*GOOD), run_id=RUN_ID)
        assert run.records[2].rationale == "[I2] test"

    def test_an_invalid_plan_runs_nothing(self, store, raw) -> None:
        run = execute_plan(store, raw, plan(step("magic")), run_id=RUN_ID)
        assert not run.ok
        assert run.records == []
        assert not store.table_exists("clean.danish_semicolon")

    def test_a_refused_step_stops_the_plan(self, store, raw) -> None:
        bad = plan(
            GOOD[0],
            step("cast", ("I2",), column="beloeb", to="float"),  # '12.345,50' does not parse
            GOOD[1],
        )
        run = execute_plan(store, raw, bad, run_id=RUN_ID)

        assert not run.ok
        assert run.failed_step == 2
        assert len(run.records) == 1
        assert any("step 2" in p and "nulls" in p for p in run.problems)
        assert run.clean_table is None
        assert not store.table_exists("clean.danish_semicolon")
        store.verify(raw)

    def test_a_step_that_changes_nothing_is_a_warning(self, store, raw) -> None:
        run = execute_plan(
            store, raw, plan(*GOOD, step("trim_whitespace", columns=["kunde_id"])), run_id=RUN_ID
        )
        assert run.ok
        assert any("step 5" in w and "changed nothing" in w for w in run.warnings)

    def test_a_pure_type_change_is_not_a_no_op(self, store, raw) -> None:
        run = execute_plan(store, raw, plan(*GOOD), run_id=RUN_ID)
        assert not any("step 2" in w for w in run.warnings)

    def test_filling_the_target_is_refused(self, store, raw) -> None:
        filling = plan(GOOD[0], GOOD[1], step("impute", column="alder", strategy="median"))
        run = execute_plan(store, raw, filling, run_id=RUN_ID, protected=["alder"])
        assert not run.ok
        assert run.failed_step == 3
        assert any("target" in p for p in run.problems)

    def test_each_run_starts_clean(self, store, raw) -> None:
        """A failed attempt that dropped rows must not leave them in the dropped-rows table."""
        drop = step("drop_rows", ("J1",), column="alder", condition="is_null", reason="age unknown")
        failing = plan(GOOD[0], drop, step("cast", ("I2",), column="beloeb", to="float"))
        assert not execute_plan(store, raw, failing, run_id=RUN_ID).ok

        run = execute_plan(store, raw, plan(GOOD[0], drop, GOOD[2]), run_id=RUN_ID)
        assert run.ok, run.problems
        dropped = next(c for c in run.checks if c.name == "dropped_rows_have_reasons")
        assert dropped.passed, dropped.message
        assert store.row_count(run.dropped_table) == 2
