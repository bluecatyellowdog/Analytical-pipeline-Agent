"""Gate 4: the write-up answers every plan question, rests on calls that ran, and every
number in it is one the tools logged.

The calls here are real tool calls on the synthetic loans, recorded the way the
analysis agent records them, so the numbers a write-up may use are real ones.
"""

from __future__ import annotations

import pytest

from pipeline.checks.analysis_gate import gate_4_analysis
from pipeline.checks.gates import problems_of
from pipeline.contracts.analysis import CallRecord, WriteUp
from pipeline.tools import run_tool
from tests.unit.test_analysis_tools import loans_context
from tests.unit.test_plan_gate import GOOD_QUESTIONS, plan

PLAN = plan()


def record(ctx, call_id: str, tool: str, params: dict, question: str | None) -> CallRecord:
    call = run_tool(ctx, tool, params, question_id=question)
    return CallRecord(
        call_id=call_id,
        round=1,
        tool=tool,
        params=call.params,
        question_id=question,
        why="test",
        summary=call.summary,
        artifact_ids=[a.artifact_id for a in call.artifacts],
        tool_call_id=call.call_id,
    )


@pytest.fixture
def ctx(tmp_path):
    return loans_context(tmp_path)


@pytest.fixture
def calls(ctx):
    return [
        record(ctx, "T1", "target_by_group", {"column": "region"}, "q1"),
        record(ctx, "T2", "logistic_regression", {}, "q2"),
        record(ctx, "T3", "missingness", {}, "q3"),
    ]


def numbers(ctx, calls) -> dict:
    groups = ctx.catalog.get(calls[0].artifact_ids[0]).data["groups"]
    top = max(groups, key=lambda g: g["rate"])
    effects = next(
        ctx.catalog.get(a).data["effects"]
        for a in calls[1].artifact_ids
        if "effects" in ctx.catalog.get(a).data
    )
    loan = next(e for e in effects if e["feature"] == "beloeb")
    return {"top": top, "loan": loan}


def write_up(ctx, calls, **changes) -> WriteUp:
    n = numbers(ctx, calls)
    top, loan = n["top"], n["loan"]
    base = {
        "action": "write",
        "summary": f"Default goes with the loan amount: x{loan['odds_ratio']:.2f} per 100,000.",
        "answers": [
            {
                "question_id": "q1",
                "status": "answered",
                "answer": f"{top['group']} has the highest rate, {top['rate']:.1%}.",
                "finding_ids": ["F1"],
            },
            {
                "question_id": "q2",
                "status": "answered",
                "answer": "Larger loans go with more defaults.",
                "finding_ids": ["F2"],
            },
            {
                "question_id": "q3",
                "status": "not_answered",
                "answer": "Nothing is missing.",
                "reason": "There are no missing values to compare.",
            },
        ],
        "findings": [
            {
                "id": "F1",
                "question_id": "q1",
                "headline": f"{top['group']} defaults most often.",
                "detail": f"{top['rate']:.1%} of {top['n']} loans there defaulted.",
                "evidence": ["T1"],
                "strength": "weak",
            },
            {
                "id": "F2",
                "question_id": "q2",
                "headline": "Larger loans default more often.",
                "detail": f"The odds of default are x{loan['odds_ratio']:.2f} per 100,000 of loan.",
                "evidence": ["T2"],
                "strength": "strong",
            },
        ],
        "limitations": ["The data is observational."],
    }
    return WriteUp.model_validate(base | changes)


def failed(gate) -> set[str]:
    return {check.name for check in gate.failures}


def test_a_traced_write_up_passes(ctx, calls) -> None:
    gate = gate_4_analysis(write_up(ctx, calls), PLAN, calls, ctx.catalog)
    assert gate.outcome == "pass", problems_of(gate)
    assert gate.gate == "gate_4_analysis"


class TestQuestionsAndEvidence:
    def test_every_plan_question_gets_an_answer(self, ctx, calls) -> None:
        answers = write_up(ctx, calls).model_dump()["answers"][:2]
        gate = gate_4_analysis(write_up(ctx, calls, answers=answers), PLAN, calls, ctx.catalog)
        assert gate.blocked and "every_question_handled" in failed(gate)
        assert "q3" in " ".join(problems_of(gate))

    def test_a_question_the_plan_marked_infeasible_stays_infeasible(self, ctx, calls) -> None:
        closed = [
            *GOOD_QUESTIONS[:2],
            {**GOOD_QUESTIONS[2], "feasible": False, "infeasible_reason": "No missing values."},
        ]
        gate = gate_4_analysis(write_up(ctx, calls), plan(questions=closed), calls, ctx.catalog)
        assert "every_question_handled" in failed(gate)

        answers = write_up(ctx, calls).model_dump()["answers"]
        answers[2] |= {"status": "infeasible"}
        fixed = write_up(ctx, calls, answers=answers)
        assert gate_4_analysis(fixed, plan(questions=closed), calls, ctx.catalog).outcome == "pass"

    def test_evidence_must_be_a_call_that_ran(self, ctx, calls) -> None:
        refused = CallRecord(call_id="T4", round=1, tool="linear_regression", error="refused")
        findings = write_up(ctx, calls).model_dump()["findings"]
        findings[0]["evidence"] = ["T9"]
        findings[1]["evidence"] = ["T2", "T4"]
        gate = gate_4_analysis(
            write_up(ctx, calls, findings=findings), PLAN, [*calls, refused], ctx.catalog
        )
        assert "evidence_exists" in failed(gate)
        text = " ".join(problems_of(gate))
        assert "T9" in text and "T4" in text


class TestNumbers:
    def test_a_number_the_tools_never_reported_is_caught(self, ctx, calls) -> None:
        findings = write_up(ctx, calls).model_dump()["findings"]
        findings[1]["detail"] = "The odds of default are x3.47 per 100,000 of loan."
        gate = gate_4_analysis(write_up(ctx, calls, findings=findings), PLAN, calls, ctx.catalog)
        assert gate.blocked and "numbers_traceable" in failed(gate)
        assert "F2: x3.47" in " ".join(problems_of(gate))

    def test_a_number_from_another_call_must_cite_that_call(self, ctx, calls) -> None:
        loan = numbers(ctx, calls)["loan"]
        findings = write_up(ctx, calls).model_dump()["findings"]
        findings[0]["detail"] += f" Bigger loans: x{loan['odds_ratio']:.2f}."
        gate = gate_4_analysis(write_up(ctx, calls, findings=findings), PLAN, calls, ctx.catalog)
        assert "numbers_traceable" in failed(gate)

        findings[0]["evidence"] = ["T1", "T2"]
        gate = gate_4_analysis(write_up(ctx, calls, findings=findings), PLAN, calls, ctx.catalog)
        assert gate.outcome == "pass", problems_of(gate)

    def test_numbers_in_the_summary_must_be_logged_somewhere(self, ctx, calls) -> None:
        gate = gate_4_analysis(
            write_up(ctx, calls, summary="A 42.5% gap between regions."), PLAN, calls, ctx.catalog
        )
        assert "summary: 42.5%" in " ".join(problems_of(gate))

    def test_numbers_from_the_brief_and_plan_are_allowed(self, ctx, calls) -> None:
        summary = "Loans from 2019 to 2024 were studied."
        up = write_up(ctx, calls, summary=summary)
        assert gate_4_analysis(up, PLAN, calls, ctx.catalog).blocked
        allowed = gate_4_analysis(up, PLAN, calls, ctx.catalog, context="From 2019 to 2024.")
        assert allowed.outcome == "pass"


class TestHonesty:
    def test_a_model_no_better_than_none_cannot_back_a_strong_finding(self, ctx, calls) -> None:
        noise = record(
            ctx,
            "T4",
            "linear_regression",
            {"outcome": "gebyr", "features": ["alder", "indkomst", "beloeb"]},
            None,
        )
        finding = {
            "id": "F3",
            "headline": "Fees do not follow age, income or loan size.",
            "detail": "The fee model explains nothing.",
            "evidence": ["T4"],
            "strength": "strong",
        }
        findings = [*write_up(ctx, calls).model_dump()["findings"], finding]
        up = write_up(ctx, calls, findings=findings)
        gate = gate_4_analysis(up, PLAN, [*calls, noise], ctx.catalog)
        assert gate.blocked and "no_overclaiming" in failed(gate)

        findings[-1]["strength"] = "none"
        up = write_up(ctx, calls, findings=findings)
        assert gate_4_analysis(up, PLAN, [*calls, noise], ctx.catalog).outcome == "pass"

    def test_model_warnings_reach_the_report(self, ctx, calls) -> None:
        ctx.frame["sag_lukket"] = ctx.frame["misligholdt"]
        ctx.kinds["sag_lukket"] = "numeric"
        leaky = record(ctx, "T4", "logistic_regression", {}, "q2")
        gate = gate_4_analysis(write_up(ctx, calls), PLAN, [*calls, leaky], ctx.catalog)
        assert gate.outcome == "flag" and failed(gate) == {"model_warnings"}
        assert [f.code for f in gate.flags] == ["suspiciously_good_model"]
        assert "T4" in " ".join(problems_of(gate))
