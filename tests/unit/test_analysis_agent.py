"""The analysis agent: rounds of tool calls, then a write-up that must pass Gate 4.

The AI is scripted; the tools are real, on the synthetic loans. What is pinned
down: calls run and their aggregate results go back under ids; a refused or
crashed call is reported, not fatal, and cannot be cited; a write-up with a
number the tools never reported goes back with the problem; the rounds are
capped; and every finding becomes an artifact tied to its question and calls.
"""

from __future__ import annotations

import json

import pytest

from pipeline.agents.analysis import AnalysisError, analyse, build_messages
from pipeline.agents.llm import FakeLLM
from pipeline.contracts.context_brief import ContextBrief
from pipeline.tools import REGISTRY
from tests.unit.test_analysis_tools import loans_context
from tests.unit.test_plan_gate import plan

BRIEF = ContextBrief(
    description="One row per loan application.",
    goal="Understand which applicants default.",
    target="misligholdt",
)
PLAN = plan()


def run(*calls: dict) -> str:
    return json.dumps({"action": "run", "calls": list(calls), "notes": "First look."})


def call(tool: str, question: str | None, **params) -> dict:
    return {"tool": tool, "params": params, "question": question, "why": "To see."}


FIRST_ROUND = run(
    call("target_by_group", "q1", column="region"),
    call("logistic_regression", "q2"),
    call("missingness", "q3"),
)


def write(**changes) -> str:
    up = {
        "action": "write",
        "summary": "Larger loans and lower incomes go with more defaults; region matters little.",
        "answers": [
            {
                "question_id": "q1",
                "status": "answered",
                "answer": "Default rates are similar across regions.",
                "finding_ids": ["F1"],
            },
            {
                "question_id": "q2",
                "status": "answered",
                "answer": "Loan size and income go with default, others held fixed.",
                "finding_ids": ["F2"],
            },
            {
                "question_id": "q3",
                "status": "not_answered",
                "answer": "Nothing is missing.",
                "reason": "There are no missing values.",
            },
        ],
        "findings": [
            {
                "id": "F1",
                "question_id": "q1",
                "headline": "Region makes little difference.",
                "detail": "The rates overlap within their intervals.",
                "evidence": ["T1"],
                "strength": "weak",
            },
            {
                "id": "F2",
                "question_id": "q2",
                "headline": "Bigger loans go with more defaults.",
                "detail": "The odds rise with the loan amount, income held fixed.",
                "evidence": ["T2"],
                "strength": "strong",
            },
        ],
        "limitations": ["The data is observational."],
        "next_questions": ["Does the loan-to-income ratio explain both effects?"],
    }
    return json.dumps(up | changes)


@pytest.fixture
def ctx(tmp_path):
    return loans_context(tmp_path)


class TestTheLoop:
    def test_a_round_of_calls_then_a_write_up(self, ctx) -> None:
        llm = FakeLLM([FIRST_ROUND, write()])
        result = analyse(ctx, PLAN, BRIEF, llm)
        assert result.gate.outcome == "pass" and len(result.attempts) == 2
        calls = result.outcome.calls
        assert [c.call_id for c in calls] == ["T1", "T2", "T3"] and all(c.ok for c in calls)
        shown = llm.calls[1][-1]["content"]
        assert shown.startswith("<results>") and "T1 target_by_group" in shown
        assert "Hovedstaden" in shown and "odds" in shown.lower()
        assert "Rounds left: 2" in shown

    def test_findings_become_artifacts_tied_to_their_question_and_calls(self, ctx) -> None:
        result = analyse(ctx, PLAN, BRIEF, FakeLLM([FIRST_ROUND, write()]))
        artifact = result.catalog.get(result.outcome.finding_artifacts["F2"])
        assert artifact.kind == "finding" and artifact.question_id == "q2"
        refs = {(r.kind, r.note) for r in artifact.audit_refs}
        assert ("tool", "T2") in refs and ("plan_question", None) in refs
        assert artifact.interpretation.startswith("The odds rise")
        assert artifact.data["evidence"] == result.outcome.calls[1].artifact_ids

    def test_a_refused_call_is_shown_and_cannot_be_cited(self, ctx) -> None:
        bad = run(call("target_by_group", "q1", column="kunde_alder"), call("missingness", "q3"))
        cites_it = json.loads(write())
        cites_it["findings"][0]["evidence"] = ["T1"]
        fixed = json.loads(write())
        fixed["findings"][0]["evidence"] = ["T2"]
        llm = FakeLLM([bad, json.dumps(cites_it), json.dumps(fixed)])
        result = analyse(ctx, PLAN, BRIEF, llm)
        assert result.gate.outcome == "pass" and not result.outcome.calls[0].ok
        assert "refused: no column 'kunde_alder'" in llm.calls[1][-1]["content"]
        assert "T1, which the tool refused" in llm.calls[2][-1]["content"]

    def test_a_made_up_number_goes_back_with_the_problem(self, ctx) -> None:
        findings = json.loads(write())["findings"]
        findings[1]["detail"] = "The odds are x9.99 per 100,000 of loan."
        llm = FakeLLM([FIRST_ROUND, write(findings=findings), write()])
        result = analyse(ctx, PLAN, BRIEF, llm)
        assert len(result.attempts) == 3
        assert "F2: x9.99 is not in the results of T2" in result.attempts[1].feedback

    def test_the_rounds_are_capped(self, ctx) -> None:
        again = run(call("correlation", None))
        llm = FakeLLM([FIRST_ROUND, again, write()])
        result = analyse(ctx, PLAN, BRIEF, llm, max_rounds=1)
        assert "That was the last round" in llm.calls[1][-1]["content"]
        assert "No tool rounds are left" in llm.calls[2][-1]["content"]
        assert len(result.outcome.calls) == 3

    def test_no_usable_write_up_after_the_retries_is_an_error(self, ctx) -> None:
        llm = FakeLLM([FIRST_ROUND, "no json", write(answers=[]), "still no json"])
        with pytest.raises(AnalysisError) as caught:
            analyse(ctx, PLAN, BRIEF, llm)
        assert len(caught.value.attempts) == 4 and len(caught.value.calls) == 3

    def test_a_tool_that_crashes_does_not_end_the_analysis(self, ctx, monkeypatch) -> None:
        def broken(ctx, params):
            raise RuntimeError("secret value 00042 inside")

        spec = REGISTRY["missingness"]
        monkeypatch.setitem(
            REGISTRY, "missingness", spec.__class__(**{**vars(spec), "run": broken})
        )
        result = analyse(ctx, PLAN, BRIEF, FakeLLM([FIRST_ROUND, write()]))
        crashed = result.outcome.calls[2]
        assert crashed.error == "the tool failed (RuntimeError); try other parameters"
        assert "00042" not in crashed.error and "00042" in crashed.internal_error


def test_the_prompt_holds_the_plan_and_the_tools_but_no_row_values(ctx) -> None:
    messages = build_messages(ctx, PLAN, BRIEF)
    text = "\n".join(m["content"] for m in messages)
    assert "Question q2" in text and "logistic_regression" in text and "<data>" in text
    assert "insight, not complexity" in text
    assert not any(i in text for i in ctx.frame["kunde_id"].head(20))


FIXTURE_ANALYSIS_REPLIES = [
    run(call("summary_table", "q1"), call("missingness", "q2")),
    json.dumps(
        {
            "action": "write",
            "summary": "The loans vary widely in size, and some ages are missing.",
            "answers": [
                {
                    "question_id": "q1",
                    "status": "answered",
                    "answer": "Amounts vary widely.",
                    "finding_ids": ["F1"],
                },
                {
                    "question_id": "q2",
                    "status": "answered",
                    "answer": "Some ages are missing.",
                    "finding_ids": ["F2"],
                },
                {
                    "question_id": "q3",
                    "status": "infeasible",
                    "answer": "Too few rows for a regression.",
                    "reason": "Only 12 rows.",
                },
            ],
            "findings": [
                {
                    "id": "F1",
                    "question_id": "q1",
                    "headline": "Loan amounts vary widely.",
                    "detail": "The summary shows a wide range of amounts.",
                    "evidence": ["T1"],
                    "strength": "moderate",
                },
                {
                    "id": "F2",
                    "question_id": "q2",
                    "headline": "Some ages are missing.",
                    "detail": "The missingness check lists the missing ages.",
                    "evidence": ["T2"],
                    "strength": "weak",
                },
            ],
            "limitations": ["Only 12 rows."],
            "doubts": ["With so few loans, one unusual amount can move every summary."],
            "recommendations": ["Collect more loans before reading much into the amounts."],
        }
    ),
]
"""The analysis replies that pass Gate 4 on the cleaned danish_semicolon fixture."""
