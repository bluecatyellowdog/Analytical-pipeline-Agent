"""The reviewer agent: evidence in, queries if it needs them, a grounded verdict out.

FakeLLM throughout, on a real cleaned table. What is pinned down: the evidence
carries what each step actually did and no personal value; queries run and
their answers come back; claims citing numbers that appear nowhere in the
evidence are dropped; an accept needs two verified checks; fixes and
ambiguities end up in the official verdict; budgets and retries hold.
"""

from __future__ import annotations

import json

import pytest

from pipeline.agents.llm import FakeLLM
from pipeline.agents.reviewer import (
    ReviewerError,
    ReviewInputs,
    build_messages,
    masked_from_profile,
    review,
)
from pipeline.checks.gates import gate_2_cleaning
from pipeline.contracts.cleaning import CleaningPlan, Decision
from pipeline.contracts.context_brief import ContextBrief
from pipeline.ingest import ingest
from pipeline.ingestion.compact import compact_profile
from pipeline.ingestion.profiler import profile_table
from pipeline.ingestion.raw import RawStore, RawTable
from pipeline.ops.plan import execute_plan
from pipeline.tools.queries import QueryContext
from tests.unit.test_coder import DIAGNOSIS, GOOD_STEPS, step

EMAIL_STEP = step("tokens_to_null", ["I5"], columns=["email"], tokens=["", "NA"])
DECISIONS = [Decision(id="J1", choice="undecided"), Decision(id="J2", choice="undecided")]


@pytest.fixture
def setup(fixtures_dir):
    brief = ContextBrief(
        description="Loan applications.",
        goal="Understand loan size.",
        target="beloeb",
        excluded_variables=["email"],
    )
    with RawStore.open() as store:
        ingested = ingest(fixtures_dir / "danish_semicolon.csv", brief, store=store)
        raw = ingested.table
        plan = CleaningPlan(summary="Danish formats.", steps=[*GOOD_STEPS, EMAIL_STEP])
        run = execute_plan(store, raw, plan, run_id="rv")
        assert run.ok, run.problems
        profile_raw = compact_profile(ingested.profile, ingested.claim_results)
        null_tokens = {c["name"]: list(c.get("null_tokens") or {}) for c in profile_raw["columns"]}
        gate, contract = gate_2_cleaning(
            store,
            raw,
            clean_table=run.clean_table,
            records=run.records,
            reconciliation=run.checks,
            plan=plan,
            diagnosis=DIAGNOSIS,
            brief=brief,
            null_tokens=null_tokens,
        )
        clean = RawTable(
            name=run.clean_table,
            columns=store.columns_of(run.clean_table),
            column_types=store.column_types_of(run.clean_table),
            typed=True,
            n_rows=store.row_count(run.clean_table),
            data_hash=store.data_hash(run.clean_table),
        )
        profile_clean = compact_profile(profile_table(store, clean))
        masked = masked_from_profile(profile_raw) | masked_from_profile(profile_clean)
        inputs = ReviewInputs(
            brief=brief,
            diagnosis=DIAGNOSIS,
            decisions=DECISIONS,
            plan=plan,
            records=run.records,
            gate=gate,
            contract=contract,
            profile_raw=profile_raw,
            profile_clean=profile_clean,
            masked=masked,
        )
        ctx = QueryContext(
            store=store, raw_table=raw.name, clean_table=run.clean_table, masked=masked
        )
        yield inputs, ctx


def verdict(verified=None, findings=(), ambiguities=(), accepted=True) -> str:
    if verified is None:
        verified = [
            {"claim": "No rows were lost.", "evidence": "12 raw = 12 clean"},
            {"claim": "Ages parse as integers.", "evidence": "alder range 19..71"},
        ]
    return json.dumps(
        {
            "verdict": {
                "accepted": accepted,
                "summary": "Checked.",
                "verified": verified,
                "findings": list(findings),
                "ambiguities": list(ambiguities),
            }
        }
    )


def queries(*items) -> str:
    return json.dumps({"queries": list(items)})


COUNT_999 = {
    "id": "q1",
    "kind": "count",
    "where": [{"column": "beloeb", "op": "equals", "value": 999}],
    "purpose": "are coded 999s left?",
}


class TestEvidence:
    def test_the_evidence_shows_what_each_step_did(self, setup) -> None:
        inputs, _ = setup
        text = build_messages(inputs)[-1]["content"]
        for expected in ("tokens_to_null", "normalize_decimal", "Gate 2", "<profile_raw>"):
            assert expected in text
        assert "<profile_clean>" in text and "<steps>" in text
        assert "alder: VARCHAR -> BIGINT" in text

    def test_no_personal_value_reaches_the_reviewer(self, setup) -> None:
        inputs, _ = setup
        text = "\n".join(m["content"] for m in build_messages(inputs))
        assert "@example.dk" not in text
        assert "00417" not in text


class TestQueries:
    def test_queries_run_and_their_answers_come_back(self, setup) -> None:
        inputs, ctx = setup
        cited = [
            {"claim": "The 999s are untouched, J1 is open.", "evidence": "3 of 12 rows"},
            {"claim": "No rows were lost.", "evidence": "12 raw = 12 clean"},
        ]
        llm = FakeLLM([queries(COUNT_999), verdict(verified=cited)])
        result = review(llm, inputs, ctx)

        assert len(result.queries) == 1
        assert "3 of 12 rows" in result.queries[0].answer
        assert "<query_results>" in llm.calls[1][-1]["content"]
        assert len(result.outcome.verified) == 2
        assert result.verdict.accepted

    def test_a_refused_query_is_explained_not_fatal(self, setup) -> None:
        inputs, ctx = setup
        nosy = {
            "id": "q1",
            "kind": "count",
            "where": [{"column": "email", "op": "equals", "value": "anna@x.dk"}],
            "purpose": "who is anna",
        }
        llm = FakeLLM([queries(nosy), verdict()])
        result = review(llm, inputs, ctx)
        assert "refused" in result.queries[0].answer
        assert result.verdict.accepted

    def test_the_query_budget_holds(self, setup) -> None:
        inputs, ctx = setup
        llm = FakeLLM([queries(COUNT_999)] * 3 + [verdict()])
        result = review(llm, inputs, ctx)
        assert len(result.queries) == 2
        assert "verdict now" in llm.calls[3][-1]["content"]


class TestGrounding:
    def test_claims_with_invented_numbers_are_dropped(self, setup) -> None:
        inputs, ctx = setup
        cited = [
            {"claim": "No rows were lost.", "evidence": "12 raw = 12 clean"},
            {"claim": "Ages parse as integers.", "evidence": "alder range 19..71"},
            {"claim": "Amounts look right.", "evidence": "mean amount is 777777"},
        ]
        result = review(FakeLLM([verdict(verified=cited)] * 3), inputs, ctx)
        assert len(result.outcome.verified) == 2
        assert any("777777" in d for d in result.dropped)

    def test_an_accept_needs_two_verified_checks(self, setup) -> None:
        inputs, ctx = setup
        one = [{"claim": "No rows were lost.", "evidence": "12 raw = 12 clean"}]
        llm = FakeLLM([verdict(verified=one), verdict()])
        result = review(llm, inputs, ctx)
        assert len(result.attempts) == 2
        assert "at least 2" in llm.calls[1][-1]["content"]

    def test_a_review_that_never_verifies_enough_is_flagged(self, setup) -> None:
        inputs, ctx = setup
        one = [{"claim": "No rows were lost.", "evidence": "12 raw = 12 clean"}]
        result = review(FakeLLM([verdict(verified=one)] * 3), inputs, ctx)
        assert any(f.code == "review_unverified" for f in result.verdict.findings)


class TestVerdict:
    FIX = {
        "step": 3,
        "column": "beloeb",
        "severity": "error",
        "problem": "999 kept as an amount although it means unknown.",
        "evidence": "3 of 12 rows",
        "fix": {
            "column": "beloeb",
            "issue": "replace 999 with missing",
            "suggested_op": "replace_sentinel",
            "params": {"column": "beloeb", "values": ["999.0"]},
            "reason": "999 is a coded missing value",
        },
    }

    def test_a_finding_with_a_fix_rejects(self, setup) -> None:
        inputs, ctx = setup
        llm = FakeLLM([queries(COUNT_999), verdict(findings=[self.FIX], accepted=False)])
        result = review(llm, inputs, ctx)
        assert not result.verdict.accepted
        assert result.verdict.fix_requests[0].suggested_op == "replace_sentinel"

    def test_accepted_is_derived_from_the_fixes_not_taken_on_trust(self, setup) -> None:
        inputs, ctx = setup
        llm = FakeLLM([queries(COUNT_999), verdict(findings=[self.FIX], accepted=True)])
        assert not review(llm, inputs, ctx).verdict.accepted

    def test_ambiguities_are_reported_not_fixed(self, setup) -> None:
        inputs, ctx = setup
        question = {
            "column": "beloeb",
            "question": "Does 999 mean unknown in every year?",
            "evidence": "3 of 12 rows",
        }
        llm = FakeLLM([queries(COUNT_999), verdict(ambiguities=[question])])
        result = review(llm, inputs, ctx)
        assert result.verdict.accepted
        assert any(f.code == "review_ambiguity" for f in result.verdict.findings)

    def test_a_fix_naming_an_unknown_op_is_sent_back(self, setup) -> None:
        inputs, ctx = setup
        bad = json.loads(json.dumps(self.FIX))
        bad["fix"]["suggested_op"] = "make_it_right"
        llm = FakeLLM(
            [
                queries(COUNT_999),
                verdict(findings=[bad], accepted=False),
                verdict(findings=[self.FIX], accepted=False),
            ]
        )
        result = review(llm, inputs, ctx)
        assert "make_it_right" in llm.calls[2][-1]["content"]
        assert result.verdict.fix_requests[0].suggested_op == "replace_sentinel"

    def test_it_gives_up_after_two_retries(self, setup) -> None:
        inputs, ctx = setup
        llm = FakeLLM(["no", "no", "no", "never asked"])
        with pytest.raises(ReviewerError):
            review(llm, inputs, ctx)
        assert len(llm.calls) == 3
