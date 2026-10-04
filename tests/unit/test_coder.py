"""The coder agent: diagnosis + decisions in, an executed and checked cleaning plan out.

FakeLLM throughout. What is pinned down: the prompt carries the diagnosis and
the decisions and no raw values; a plan is checked against the diagnosis before
it runs (every issue handled or skipped, no undecided call acted on, the target
never imputed); a failing plan goes back with its errors masked; a step that
changes nothing earns a second chance; and the run gives up after two retries.
"""

from __future__ import annotations

import json

import pytest

from pipeline.agents.coder import (
    CoderError,
    build_messages,
    check_plan,
    parse_reply,
    write_and_run,
)
from pipeline.agents.llm import FakeLLM
from pipeline.contracts.cleaning import Decision
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.diagnosis import Diagnosis
from pipeline.ingest import ingest
from pipeline.ingestion.compact import compact_profile
from pipeline.ingestion.loaders import load
from pipeline.ingestion.raw import RawStore

RUN_ID = "run_coder"

DIAGNOSIS = Diagnosis.model_validate(
    {
        "summary": "Danish formats and NA tokens; 999 may mean unknown.",
        "issues": [
            {
                "id": "I1",
                "column": "alder",
                "kind": "missing_tokens",
                "evidence": "null_tokens {'NA': 2}",
                "fix": "Turn 'NA' into missing.",
                "suggested_op": "tokens_to_null",
            },
            {
                "id": "I2",
                "column": "alder",
                "kind": "type",
                "evidence": "integer, coverage 1.0",
                "fix": "Cast to integer.",
                "suggested_op": "cast",
            },
            {
                "id": "I3",
                "column": "beloeb",
                "kind": "decimal_format",
                "evidence": "decimal_separator ','",
                "fix": "Parse Danish decimals.",
                "suggested_op": "normalize_decimal",
            },
            {
                "id": "I4",
                "column": "dato",
                "kind": "date_format",
                "evidence": "date_format '%d-%m-%Y'",
                "fix": "Parse dates.",
                "suggested_op": "parse_dates",
            },
            {
                "id": "I5",
                "column": "email",
                "kind": "personal_data",
                "evidence": "pii likely",
                "fix": "Keep as it is.",
                "suggested_op": None,
            },
        ],
        "judgment_calls": [
            {
                "id": "J1",
                "column": "beloeb",
                "question": "Does 999 mean unknown?",
                "options": ["Treat 999 as missing", "Keep 999"],
                "recommendation": "Treat as missing.",
                "why_it_matters": "It distorts amounts.",
            },
            {
                "id": "J2",
                "column": "alder",
                "question": "Impute missing ages?",
                "options": ["Leave missing", "Impute median"],
                "recommendation": "Leave missing.",
                "why_it_matters": "Under-25 focus.",
            },
        ],
        "columns": [
            {"name": "kunde_id", "role": "identifier", "target_type": "id"},
            {"name": "alder", "role": "feature", "target_type": "integer"},
            {"name": "beloeb", "role": "target", "target_type": "float"},
            {"name": "dato", "role": "feature", "target_type": "date"},
            {"name": "email", "role": "excluded", "target_type": "text"},
        ],
    }
)

UNDECIDED = [Decision(id="J1", choice="undecided"), Decision(id="J2", choice="undecided")]
J1_DECIDED = [Decision(id="J1", choice="Treat 999 as missing"), UNDECIDED[1]]


def step(op, addresses, **params) -> dict:
    return {"op": op, "params": params, "addresses": addresses, "rationale": "test"}


GOOD_STEPS = [
    step("tokens_to_null", ["I1"], columns=["alder"], tokens=["NA"]),
    step("cast", ["I2"], column="alder", to="integer"),
    step("normalize_decimal", ["I3"], column="beloeb"),
    step("parse_dates", ["I4"], column="dato", format="%d-%m-%Y"),
]


def plan_json(steps=None, skipped=()) -> str:
    return json.dumps(
        {
            "summary": "Clean the Danish formats.",
            "steps": GOOD_STEPS if steps is None else steps,
            "skipped": list(skipped),
        }
    )


@pytest.fixture
def brief():
    return ContextBrief(
        description="Loan applications.",
        goal="Understand loan size.",
        target="beloeb",
        excluded_variables=["email"],
    )


@pytest.fixture(scope="module")
def profile_llm(fixtures_dir):
    with RawStore.open() as store:
        result = ingest(fixtures_dir / "danish_semicolon.csv", store=store)
    return compact_profile(result.profile, result.claim_results)


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


@pytest.fixture
def raw(store, fixtures_dir):
    _, raw_table = load(fixtures_dir / "danish_semicolon.csv", store)
    return raw_table


def run(llm, profile_llm, brief, store, raw, decisions=UNDECIDED):
    return write_and_run(
        profile_llm, brief, DIAGNOSIS, decisions, llm, store=store, raw=raw, run_id=RUN_ID
    )


class TestPrompt:
    def test_the_prompt_carries_diagnosis_decisions_and_profile(self, profile_llm, brief) -> None:
        text = "\n".join(
            m["content"] for m in build_messages(profile_llm, brief, DIAGNOSIS, J1_DECIDED)
        )
        for item in ("I1", "I4", "J1", "J2", "Treat 999 as missing", "<profile>", "custom_code"):
            assert item in text
        assert "J2" in text.split("Undecided")[1]

    def test_no_personal_value_reaches_the_prompt(self, fixtures_dir, brief) -> None:
        with RawStore.open() as store:
            result = ingest(fixtures_dir / "cpr_numbers.csv", store=store)
        profile = compact_profile(result.profile, result.claim_results)
        text = "\n".join(m["content"] for m in build_messages(profile, brief, DIAGNOSIS, []))
        for cpr in ("010190-1234", "311299-5678", "290204-1111", "150385-9999"):
            assert cpr not in text


class TestPlanChecks:
    def check(self, steps=None, skipped=(), decisions=UNDECIDED, brief=None):
        brief = brief or ContextBrief(description="d", goal="g", target="beloeb")
        plan = parse_reply(plan_json(steps, skipped))
        return check_plan(plan, DIAGNOSIS, decisions, brief)

    def test_a_good_plan_has_no_problems(self) -> None:
        assert self.check() == []

    def test_every_issue_with_an_op_is_handled_or_skipped(self) -> None:
        problems = self.check(GOOD_STEPS[:3])
        assert any("I4" in p for p in problems)
        assert self.check(GOOD_STEPS[:3], skipped=[{"id": "I4", "reason": "no dates needed"}]) == []

    def test_unknown_ids_are_reported(self) -> None:
        problems = self.check([*GOOD_STEPS, step("trim_whitespace", ["I9"], columns=["email"])])
        assert any("I9" in p for p in problems)

    def test_undecided_judgment_calls_are_left_alone(self) -> None:
        impute = step("impute", ["J2"], column="alder", strategy="median")
        problems = self.check([*GOOD_STEPS, impute])
        assert any("J2" in p and "undecided" in p for p in problems)

    def test_decided_judgment_calls_must_be_carried_out(self) -> None:
        problems = self.check(decisions=J1_DECIDED)
        assert any("J1" in p for p in problems)
        sentinel = step("replace_sentinel", ["J1"], column="beloeb", values=["999.0"])
        assert self.check([*GOOD_STEPS, sentinel], decisions=J1_DECIDED) == []

    def test_the_target_is_never_imputed_or_dropped(self) -> None:
        impute = step("impute", ["I3"], column="beloeb", strategy="median")
        drop = step("drop_column", ["I3"], column="beloeb", reason="no")
        assert any("target" in p for p in self.check([*GOOD_STEPS, impute]))
        assert any("target" in p for p in self.check([*GOOD_STEPS, drop]))

    def test_invalid_params_are_reported_before_anything_runs(self) -> None:
        bad = [*GOOD_STEPS[:1], step("cast", ["I2"], column="alder", to="nope"), *GOOD_STEPS[2:]]
        assert any("step 2" in p for p in self.check(bad))


class TestWriteAndRun:
    def test_a_good_plan_is_run_on_the_first_attempt(self, profile_llm, brief, store, raw) -> None:
        result = run(FakeLLM([plan_json()]), profile_llm, brief, store, raw)
        assert result.run.ok
        assert len(result.attempts) == 1
        assert store.column_values(result.run.clean_table, "beloeb")[0] == 12345.5

    def test_errors_go_back_masked_and_stay_complete_in_the_log(
        self, profile_llm, brief, store, raw
    ) -> None:
        crash = (
            "def transform(df):\n"
            "    df['beloeb'] = [float(v) for v in df['beloeb']]\n"
            "    return df\n"
        )
        bad_steps = [*GOOD_STEPS[:2], step("custom_code", ["I3"], code=crash, changes=["beloeb"])]
        bad_steps.append(GOOD_STEPS[3])
        llm = FakeLLM([plan_json(bad_steps), plan_json()])
        result = run(llm, profile_llm, brief, store, raw)

        assert result.run.ok
        sent_back = llm.calls[1][-1]["content"]
        assert "step 3" in sent_back
        assert "12.345,50" not in sent_back
        assert "99.999,99" in sent_back
        assert any("12.345,50" in p for p in result.attempts[0].problems)

    def test_a_step_that_changes_nothing_gets_a_second_chance(
        self, profile_llm, brief, store, raw
    ) -> None:
        noop = [*GOOD_STEPS, step("trim_whitespace", ["I1"], columns=["kunde_id"])]
        llm = FakeLLM([plan_json(noop), plan_json()])
        result = run(llm, profile_llm, brief, store, raw)
        assert len(result.attempts) == 2
        assert "changed nothing" in llm.calls[1][-1]["content"]
        assert result.run.warnings == []

    def test_a_plan_with_only_warnings_is_kept_if_later_attempts_fail(
        self, profile_llm, brief, store, raw
    ) -> None:
        noop = [*GOOD_STEPS, step("trim_whitespace", ["I1"], columns=["kunde_id"])]
        llm = FakeLLM([plan_json(noop), "not json", "still not json"])
        result = run(llm, profile_llm, brief, store, raw)
        assert result.run.ok
        assert result.run.warnings
        assert len(result.plan.steps) == 5
        assert store.table_exists(result.run.clean_table)

    def test_it_gives_up_after_two_retries(self, profile_llm, brief, store, raw) -> None:
        llm = FakeLLM(["no", "no", "no", "never asked"])
        with pytest.raises(CoderError) as error:
            run(llm, profile_llm, brief, store, raw)
        assert len(error.value.attempts) == 3
        assert len(llm.calls) == 3

    def test_a_decided_sentinel_is_carried_out(self, profile_llm, brief, store, raw) -> None:
        sentinel = step("replace_sentinel", ["J1"], column="beloeb", values=["999.0"])
        llm = FakeLLM([plan_json([*GOOD_STEPS, sentinel])])
        result = run(llm, profile_llm, brief, store, raw, decisions=J1_DECIDED)
        assert result.run.ok
        assert store.column_values(result.run.clean_table, "beloeb").count(None) == 3
