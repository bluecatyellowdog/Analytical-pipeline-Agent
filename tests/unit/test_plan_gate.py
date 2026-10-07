"""Gate 3: an analysis plan is executable, safe and feasible on this data, or it does not run.

Checked on the plan and the clean data, never on the planner's word: columns and
tools exist, the target is the brief's, nothing personal or excluded is
analysed, a column that gives the answer away is excluded, and every question
has enough rows or is marked infeasible. For the AI's plan every failure goes
back to the planner; a user's own edit may keep a judgment call, with a warning.
"""

from __future__ import annotations

import pytest

from pipeline.checks.gates import problems_of
from pipeline.checks.plan_gate import gate_3_plan, leak_suspects, task_type_of
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.plan import AnalysisPlan
from tests.unit.test_analysis_tools import loans_context

BRIEF = ContextBrief(
    description="One row per loan application.",
    goal="Understand which applicants default.",
    target="misligholdt",
)


def question(qid: str, kind: str, columns: list[str], methods: list[str], **extra) -> dict:
    return {
        "id": qid,
        "question": f"Question {qid}",
        "kind": kind,
        "columns": columns,
        "methods": methods,
        "priority": int(qid[1:]),
        "success_criteria": "Answered with numbers.",
        **extra,
    }


GOOD_QUESTIONS = [
    question("q1", "segment", ["region", "misligholdt"], ["target_by_group"]),
    question(
        "q2", "drivers", ["alder", "indkomst", "beloeb", "misligholdt"], ["logistic_regression"]
    ),
    question("q3", "data_quality", ["indkomst"], ["missingness"]),
]


def plan(**changes) -> AnalysisPlan:
    base = {
        "run_id": "run_plan",
        "objective": "Find what goes with default.",
        "target": "misligholdt",
        "task_type": "binary_classification",
        "task_type_reason": "misligholdt is yes/no.",
        "questions": GOOD_QUESTIONS,
        "excluded_features": [],
        "success_criteria": "Each question answered or marked infeasible.",
    }
    return AnalysisPlan.model_validate(base | changes)


def failed(gate) -> set[str]:
    return {check.name for check in gate.failures}


@pytest.fixture
def ctx(tmp_path):
    return loans_context(tmp_path)


def test_a_good_plan_passes(ctx) -> None:
    gate = gate_3_plan(plan(), ctx, BRIEF)
    assert gate.outcome == "pass", problems_of(gate)
    assert gate.gate == "gate_3_plan"


class TestInvariants:
    def test_unknown_columns_and_tools_are_caught(self, ctx) -> None:
        questions = [
            *GOOD_QUESTIONS[:2],
            question("q3", "drivers", ["kunde_alder", "misligholdt"], ["shap_summary"]),
        ]
        gate = gate_3_plan(plan(questions=questions), ctx, BRIEF)
        assert gate.outcome == "block"
        assert {"columns_exist", "methods_exist"} <= failed(gate)
        text = " ".join(problems_of(gate))
        assert "kunde_alder" in text and "shap_summary" in text

    def test_the_target_is_the_briefs(self, ctx) -> None:
        gate = gate_3_plan(plan(target="beloeb", task_type="regression"), ctx, BRIEF)
        assert "target_as_briefed" in failed(gate)

    def test_the_task_type_fits_the_target(self, ctx) -> None:
        gate = gate_3_plan(plan(task_type="regression"), ctx, BRIEF)
        assert "task_type_fits_target" in failed(gate)
        assert task_type_of(ctx) == "binary_classification"

    def test_ids_personal_data_and_free_text_are_never_analysed(self, ctx) -> None:
        ctx.frame["email"] = "x@example.dk"
        ctx.kinds["email"] = "personal"
        questions = [
            *GOOD_QUESTIONS[:2],
            question("q3", "distribution", ["kunde_id", "email"], ["distribution"]),
        ]
        gate = gate_3_plan(plan(questions=questions), ctx, BRIEF)
        assert "usable_columns" in failed(gate)
        text = " ".join(problems_of(gate))
        assert "kunde_id" in text and "email" in text

    def test_excluded_columns_are_listed_and_never_used(self, ctx) -> None:
        brief = BRIEF.model_copy(update={"excluded_variables": ["region"]})
        ctx.excluded = {"region"}
        gate = gate_3_plan(plan(), ctx, brief)
        assert {"exclusions_listed", "usable_columns"} <= failed(gate)

        listed = [{"column": "region", "reason": "Excluded by the brief.", "source": "brief"}]
        questions = [question("q1", "segment", ["alder", "misligholdt"], ["target_by_group"])]
        gate = gate_3_plan(
            plan(excluded_features=listed, questions=questions + GOOD_QUESTIONS[1:]), ctx, brief
        )
        assert gate.outcome == "pass", problems_of(gate)

    def test_methods_fit_the_target(self, ctx) -> None:
        questions = [
            *GOOD_QUESTIONS[:2],
            question("q3", "drivers", ["region", "misligholdt"], ["linear_regression"]),
        ]
        gate = gate_3_plan(plan(questions=questions), ctx, BRIEF)
        assert "methods_fit_target" in failed(gate)
        assert "linear_regression" in " ".join(problems_of(gate))

    def test_linear_regression_may_explain_another_number_in_a_side_question(self, ctx) -> None:
        questions = [
            *GOOD_QUESTIONS[:2],
            question("q3", "relationship", ["beloeb", "indkomst"], ["linear_regression"]),
        ]
        gate = gate_3_plan(plan(questions=questions), ctx, BRIEF)
        assert gate.outcome == "pass", problems_of(gate)


class TestLeakage:
    def test_a_column_that_gives_the_answer_away_is_found(self, ctx) -> None:
        ctx.frame["sag_lukket"] = ctx.frame["misligholdt"]
        ctx.kinds["sag_lukket"] = "numeric"
        suspects = leak_suspects(ctx)
        assert suspects == {"sag_lukket": 1.0}

    def test_a_categorical_leak_is_found_but_real_drivers_are_not(self, ctx) -> None:
        defaulted = ctx.frame["misligholdt"] == 1
        ctx.frame["status"] = defaulted.map({True: "inkasso", False: "aktiv"})
        ctx.kinds["status"] = "categorical"
        suspects = leak_suspects(ctx)
        assert set(suspects) == {"status"}, "beloeb predicts default, but not perfectly"

    def test_the_planner_must_exclude_a_leak(self, ctx) -> None:
        ctx.frame["sag_lukket"] = ctx.frame["misligholdt"]
        ctx.kinds["sag_lukket"] = "numeric"
        gate = gate_3_plan(plan(), ctx, BRIEF)
        assert gate.outcome == "block" and "leaky_columns_excluded" in failed(gate)
        assert "sag_lukket" in " ".join(problems_of(gate))

        excluded = [
            {"column": "sag_lukket", "reason": "Gives the answer away.", "source": "leakage"}
        ]
        assert gate_3_plan(plan(excluded_features=excluded), ctx, BRIEF).outcome == "pass"

    def test_a_users_edit_may_keep_a_leaky_column_with_a_warning(self, ctx) -> None:
        ctx.frame["sag_lukket"] = ctx.frame["misligholdt"]
        ctx.kinds["sag_lukket"] = "numeric"
        gate = gate_3_plan(plan(), ctx, BRIEF, edited_by_user=True)
        assert gate.outcome == "flag"


class TestFeasibility:
    def test_a_regression_needs_ten_rarer_outcomes_per_feature(self, ctx) -> None:
        defaulted = ctx.frame["misligholdt"] == 1
        keep = ctx.frame.index[defaulted][:20].union(ctx.frame.index[~defaulted][:180])
        ctx.frame = ctx.frame.loc[keep].reset_index(drop=True)
        gate = gate_3_plan(plan(), ctx, BRIEF)
        assert "feasible" in failed(gate)
        assert "q2" in " ".join(problems_of(gate)), "3 features need 30 defaults, there are 20"

        marked = [
            GOOD_QUESTIONS[0],
            {**GOOD_QUESTIONS[1], "feasible": False, "infeasible_reason": "Only 20 defaults."},
            GOOD_QUESTIONS[2],
        ]
        assert gate_3_plan(plan(questions=marked), ctx, BRIEF).outcome == "pass"

    def test_segments_need_two_groups_of_thirty(self, ctx) -> None:
        ctx.frame = ctx.frame.head(80).reset_index(drop=True)
        questions = [
            question("q1", "segment", ["region", "misligholdt"], ["target_by_group"]),
            question("q2", "segment", ["alder", "misligholdt"], ["target_by_group"]),
            question("q3", "data_quality", ["indkomst"], ["missingness"]),
        ]
        gate = gate_3_plan(plan(questions=questions), ctx, BRIEF)
        text = " ".join(problems_of(gate))
        assert "q1" in text, "five regions in 80 rows: no two with 30 rows"
        assert "q2" not in text, "a number can be split into two halves of 40"

    def test_a_plan_with_nothing_feasible_is_flagged(self, ctx) -> None:
        questions = [
            {**q, "feasible": False, "infeasible_reason": "Too few rows."} for q in GOOD_QUESTIONS
        ]
        gate = gate_3_plan(plan(questions=questions), ctx, BRIEF)
        assert gate.outcome == "flag" and "something_to_do" in failed(gate)


def test_a_method_for_another_kind_of_question_is_only_a_note(ctx) -> None:
    questions = [
        *GOOD_QUESTIONS[:2],
        question("q3", "distribution", ["alder", "misligholdt"], ["logistic_regression"]),
    ]
    gate = gate_3_plan(plan(questions=questions), ctx, BRIEF)
    assert gate.outcome == "pass" and failed(gate) == {"methods_fit_question"}
