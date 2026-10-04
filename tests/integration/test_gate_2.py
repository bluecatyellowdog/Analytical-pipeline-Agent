"""Gate 2: is the cleaned data what the diagnosis said it should become?

Run on real plans against the Danish fixture. The reconciliation checks already
prove every change is accounted for; these tests pin down what Gate 2 adds: the
schema contract (types and roles from the diagnosis), row loss thresholds, the
primary key, and the target, including the loophole where a null token is
turned straight into a value. Plus the contract as a portable file that later
stages can re-check the data against.
"""

from __future__ import annotations

import pytest

from pipeline.checks.gates import gate_2_cleaning
from pipeline.checks.schema import validate_against_contract
from pipeline.contracts.cleaning import CleaningPlan
from pipeline.contracts.context_brief import ContextBrief
from pipeline.ingestion.loaders import load
from pipeline.ingestion.raw import RawStore
from pipeline.ops.plan import execute_plan
from tests.unit.test_coder import DIAGNOSIS, GOOD_STEPS, step

pytestmark = pytest.mark.integration

RUN_ID = "run_gate2"
NULL_TOKENS = {"alder": ["NA"], "email": ["", "NA"]}


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


@pytest.fixture
def raw(store, fixtures_dir):
    _, raw_table = load(fixtures_dir / "danish_semicolon.csv", store)
    return raw_table


def brief(**overrides) -> ContextBrief:
    fields = {
        "description": "Loan applications.",
        "goal": "Understand loan size.",
        "target": "beloeb",
        "id_column": "kunde_id",
        "excluded_variables": ["email"],
    }
    fields.update(overrides)
    return ContextBrief(**fields)


def gate(store, raw, steps, *, skipped=(), the_brief=None):
    plan = CleaningPlan(summary="test", steps=steps, skipped=list(skipped))
    run = execute_plan(store, raw, plan, run_id=RUN_ID)
    assert run.ok, run.problems
    return gate_2_cleaning(
        store,
        raw,
        clean_table=run.clean_table,
        records=run.records,
        reconciliation=run.checks,
        plan=plan,
        diagnosis=DIAGNOSIS,
        brief=the_brief or brief(),
        null_tokens=NULL_TOKENS,
    )


def failed(result, severity: str) -> dict[str, str]:
    return {c.name: c.message for c in result.checks if not c.passed and c.severity == severity}


class TestAGoodCleaning:
    def test_passes_and_describes_every_column(self, store, raw) -> None:
        result, contract = gate(store, raw, GOOD_STEPS)
        assert result.outcome in {"pass", "flag"}, failed(result, "block")
        assert not failed(result, "block")

        by_name = {c.name: c for c in contract}
        assert list(by_name) == ["kunde_id", "alder", "beloeb", "dato", "email"]
        assert by_name["kunde_id"].unique and not by_name["kunde_id"].nullable
        assert by_name["kunde_id"].dtype == "VARCHAR"
        assert by_name["alder"].dtype == "BIGINT" and by_name["alder"].expected_type == "integer"
        assert by_name["alder"].nullable and by_name["alder"].transformations
        assert by_name["beloeb"].role == "target"
        assert by_name["email"].role == "excluded"

    def test_the_gate_includes_the_reconciliation_checks(self, store, raw) -> None:
        result, _ = gate(store, raw, GOOD_STEPS)
        assert "replay_reproduces_output" in {c.name for c in result.checks}


class TestSchemaContract:
    def test_a_column_left_in_the_wrong_type_blocks(self, store, raw) -> None:
        no_cast = [s for s in GOOD_STEPS if s["op"] != "cast"]
        result, _ = gate(store, raw, no_cast)
        assert result.outcome == "block"
        assert "alder" in failed(result, "block")["schema_types_match_diagnosis"]

    def test_a_type_left_for_a_stated_reason_is_only_flagged(self, store, raw) -> None:
        no_cast = [s for s in GOOD_STEPS if s["op"] != "cast"]
        skipped = [{"id": "I2", "reason": "ages are needed as text for now"}]
        result, contract = gate(store, raw, no_cast, skipped=skipped)
        assert "schema_types_match_diagnosis" in failed(result, "warn")
        alder = next(c for c in contract if c.name == "alder")
        assert "ages are needed as text for now" in alder.note

    def test_an_id_turned_into_a_number_blocks(self, store, raw) -> None:
        """Casting '00417' to 417 loses the leading zeros: the ID is a different value."""
        result, _ = gate(
            store, raw, [*GOOD_STEPS, step("cast", ["I1"], column="kunde_id", to="integer")]
        )
        assert "kunde_id" in failed(result, "block")["schema_types_match_diagnosis"]

    def test_a_dropped_column_is_reported(self, store, raw) -> None:
        drop = step("drop_column", ["I5"], column="email", reason="not needed")
        result, contract = gate(store, raw, [*GOOD_STEPS, drop])
        assert "email" in failed(result, "warn")["columns_match_diagnosis"]
        assert "email" not in {c.name for c in contract}


class TestPrimaryKey:
    def test_a_key_made_non_unique_by_cleaning_blocks(self, store, raw) -> None:
        merge = step(
            "standardize_categories", ["I1"], column="kunde_id", mapping={"00417": "00418"}
        )
        result, _ = gate(store, raw, [*GOOD_STEPS, merge])
        assert "primary_key_intact" in failed(result, "block")

    def test_a_key_that_was_never_unique_is_only_flagged(self, store, raw) -> None:
        result, _ = gate(store, raw, GOOD_STEPS, the_brief=brief(id_column="beloeb"))
        assert "primary_key_intact" in failed(result, "warn")


class TestTarget:
    def test_a_null_token_turned_into_a_value_counts_as_imputation(self, store, raw) -> None:
        """'NA' -> '0' never passes through null, so only a comparison with raw catches it."""
        fill = step("standardize_categories", ["I1"], column="alder", mapping={"NA": "0"})
        cast = step("cast", ["I2"], column="alder", to="integer")
        result, _ = gate(store, raw, [fill, cast], the_brief=brief(target="alder"))
        assert "2" in failed(result, "block")["target_not_imputed"]


class TestRowLoss:
    def test_moderate_row_loss_is_flagged(self, store, raw) -> None:
        drop = step("drop_rows", ["J1"], column="alder", condition="is_null", reason="age unknown")
        result, _ = gate(store, raw, [*GOOD_STEPS, drop])
        assert "row_loss_within_threshold" in failed(result, "warn")

    def test_heavy_row_loss_blocks(self, store, raw) -> None:
        drop = step(
            "drop_rows", ["J1"], column="beloeb", condition="equals", value=999, reason="coded"
        )
        result, _ = gate(store, raw, [*GOOD_STEPS, drop])
        assert "row_loss_within_threshold" in failed(result, "block")


class TestContractAsAFile:
    def test_exported_data_validates_against_its_contract(self, store, raw, tmp_path) -> None:
        _, contract = gate(store, raw, GOOD_STEPS)
        path = (tmp_path / "clean.parquet").as_posix()
        store.connection.execute(
            f"COPY (SELECT * EXCLUDE (_row_id) FROM clean.danish_semicolon) TO '{path}'"
        )
        checks = validate_against_contract(store.connection, f"read_parquet('{path}')", contract)
        assert all(c.passed for c in checks), [c.message for c in checks if not c.passed]

    def test_data_that_drifted_from_its_contract_fails(self, store, raw, tmp_path) -> None:
        _, contract = gate(store, raw, GOOD_STEPS)
        path = (tmp_path / "drifted.parquet").as_posix()
        store.connection.execute(
            "COPY (SELECT * EXCLUDE (_row_id) REPLACE (CAST(alder AS VARCHAR) AS alder, "
            "'same' AS kunde_id) FROM clean.danish_semicolon) TO '" + path + "'"
        )
        checks = validate_against_contract(store.connection, f"read_parquet('{path}')", contract)
        problems = " ".join(c.message for c in checks if not c.passed)
        assert "alder" in problems and "kunde_id" in problems
