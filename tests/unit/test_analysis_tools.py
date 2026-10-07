"""The analysis tools: aggregates only, logged as artifacts, and right about known signal.

Run on the benchmark's synthetic loans, where default risk is built to rise with
the loan-to-income ratio and fall with age. So besides the mechanics (validation,
logging, charts, no row-level output, small groups hidden) these tests check
that the tools find what is really there.
"""

from __future__ import annotations

import pandas as pd
import pytest

from benchmark.datasets.synthetic import make_loans
from pipeline.tools import AnalysisContext, ToolError, run_tool, tool_names

KINDS = {
    "id": "id",
    "integer": "numeric",
    "float": "numeric",
    "category": "categorical",
    "date": "date",
    "boolean": "boolean",
}


@pytest.fixture
def ctx(tmp_path):
    data = make_loans(rows=600, seed=3)
    return AnalysisContext(
        run_id="run_tools",
        frame=data.frame.copy(),
        target="misligholdt",
        task_type="binary_classification",
        kinds={c: KINDS[k] for c, k in data.kinds.items()},
        excluded=set(),
        out_dir=tmp_path,
    )


def only(call):
    assert len(call.artifacts) >= 1
    return call.artifacts[0]


class TestRegistry:
    def test_the_planned_tools_exist(self) -> None:
        assert {
            "distribution",
            "target_rate_by_group",
            "compare_by_target",
            "correlation",
            "missingness",
            "baseline_model",
            "logistic_regression",
            "gradient_boosting",
        } <= set(tool_names())

    def test_unknown_tools_and_bad_parameters_are_refused(self, ctx) -> None:
        with pytest.raises(ToolError, match="unknown tool"):
            run_tool(ctx, "magic", {}, question_id="q1")
        with pytest.raises(ToolError, match="bins"):
            run_tool(ctx, "distribution", {"column": "alder", "bins": "many"}, question_id="q1")

    def test_columns_must_exist(self, ctx) -> None:
        with pytest.raises(ToolError, match="kunde_alder"):
            run_tool(ctx, "distribution", {"column": "kunde_alder"}, question_id="q1")

    def test_ids_excluded_columns_and_the_target_are_never_features(self, ctx) -> None:
        ctx.excluded = {"region"}
        for column in ("kunde_id", "region", "misligholdt"):
            with pytest.raises(ToolError, match=column):
                run_tool(ctx, "target_rate_by_group", {"column": column}, question_id="q1")
        with pytest.raises(ToolError, match="region"):
            run_tool(
                ctx, "logistic_regression", {"features": ["alder", "region"]}, question_id="q1"
            )


class TestDescriptive:
    def test_distribution_of_a_number(self, ctx) -> None:
        artifact = only(
            run_tool(ctx, "distribution", {"column": "alder", "bins": 6}, question_id="q1")
        )
        assert sum(b["count"] for b in artifact.data["bins"]) == 600
        assert artifact.data["summary"]["min"] >= 18

    def test_target_rate_by_group_adds_up(self, ctx) -> None:
        artifact = only(
            run_tool(ctx, "target_rate_by_group", {"column": "region"}, question_id="q1")
        )
        groups = artifact.data["groups"]
        assert sum(g["n"] for g in groups) == 600
        for g in groups:
            assert 0 <= g["ci_low"] <= g["rate"] <= g["ci_high"] <= 1

    def test_numbers_are_grouped_into_ranges(self, ctx) -> None:
        artifact = only(
            run_tool(ctx, "target_rate_by_group", {"column": "beloeb", "bins": 4}, question_id="q1")
        )
        rates = [g["rate"] for g in artifact.data["groups"]]
        assert len(rates) == 4
        assert rates[-1] > rates[0], "default should rise with the loan amount"

    def test_small_groups_are_never_shown_on_their_own(self, ctx) -> None:
        ctx.frame.loc[:3, "region"] = "Bornholm"
        artifact = only(
            run_tool(ctx, "target_rate_by_group", {"column": "region"}, question_id="q1")
        )
        names = [g["group"] for g in artifact.data["groups"]]
        assert "Bornholm" not in names
        assert all(g["n"] >= 10 for g in artifact.data["groups"])

    def test_missing_values_get_their_own_group(self, ctx) -> None:
        ctx.frame.loc[:59, "alder"] = None
        artifact = only(
            run_tool(ctx, "target_rate_by_group", {"column": "alder", "bins": 3}, question_id="q1")
        )
        assert any(g["group"] == "(missing)" and g["n"] == 60 for g in artifact.data["groups"])

    def test_compare_by_target(self, ctx) -> None:
        artifact = only(run_tool(ctx, "compare_by_target", {"column": "alder"}, question_id="q1"))
        by_class = {c["target"]: c for c in artifact.data["classes"]}
        assert by_class[1]["mean"] < by_class[0]["mean"], "defaulters should be younger"

    def test_missingness(self, ctx) -> None:
        ctx.frame.loc[:59, "indkomst"] = None
        artifact = only(run_tool(ctx, "missingness", {}, question_id="q1"))
        rows = {r["column"]: r for r in artifact.data["columns"]}
        assert rows["indkomst"]["missing"] == 60


class TestModels:
    def test_the_baseline_is_the_majority_class(self, ctx) -> None:
        artifact = only(run_tool(ctx, "baseline_model", {}, question_id="q1"))
        majority = ctx.frame["misligholdt"].value_counts(normalize=True).max()
        assert artifact.data["metrics"]["accuracy"] == pytest.approx(majority, abs=0.01)
        assert artifact.data["metrics"]["auc"] == pytest.approx(0.5)

    def test_logistic_regression_beats_the_baseline_with_the_right_signs(self, ctx) -> None:
        call = run_tool(ctx, "logistic_regression", {}, question_id="q1")
        metrics = next(a for a in call.artifacts if "metrics" in a.data).data["metrics"]
        assert metrics["auc"] > 0.6
        effects = {
            row["feature"]: row["odds_ratio"]
            for a in call.artifacts
            if "coefficients" in a.data
            for row in a.data["coefficients"]
        }
        assert effects["beloeb"] > 1 and effects["indkomst"] < 1 and effects["alder"] < 1

    def test_gradient_boosting_ranks_the_real_drivers_first(self, ctx) -> None:
        call = run_tool(ctx, "gradient_boosting", {}, question_id="q1")
        importance = next(a for a in call.artifacts if "importance" in a.data).data["importance"]
        top_three = {row["feature"] for row in importance[:3]}
        assert {"beloeb", "indkomst"} <= top_three

    def test_gradient_boosting_stops_early_instead_of_memorising(self, ctx) -> None:
        call = run_tool(ctx, "gradient_boosting", {}, question_id="q1")
        metrics = next(a for a in call.artifacts if "metrics" in a.data).data["metrics"]
        assert metrics["train_test_gap"] < 0.10
        assert max(metrics["rounds_used"]) < 300
        assert "overfitting" not in [f.code for a in call.artifacts for f in a.flags]

    def test_models_run_on_the_smallest_data_they_accept(self, ctx) -> None:
        defaulted = ctx.frame["misligholdt"] == 1
        keep = ctx.frame.index[defaulted][:10].union(ctx.frame.index[~defaulted][:40])
        ctx.frame = ctx.frame.loc[keep].reset_index(drop=True)
        for name in ("baseline_model", "logistic_regression", "gradient_boosting"):
            assert run_tool(ctx, name, {}, question_id="q1").artifacts
        one_defaulter = ctx.frame.index[ctx.frame["misligholdt"] == 1][:1]
        ctx.frame = ctx.frame.drop(index=one_defaulter)
        with pytest.raises(ToolError, match="smaller target group"):
            run_tool(ctx, "logistic_regression", {}, question_id="q1")

    def test_model_metrics_report_fold_spread_and_the_train_gap(self, ctx) -> None:
        call = run_tool(ctx, "logistic_regression", {}, question_id="q1")
        metrics = next(a for a in call.artifacts if "metrics" in a.data).data["metrics"]
        assert metrics["auc_std"] >= 0 and "train_test_gap" in metrics
        assert len(metrics["fold_auc"]) == 5

    def test_a_too_good_model_raises_the_leakage_alarm(self, ctx) -> None:
        ctx.frame["sag_lukket"] = ctx.frame["misligholdt"]
        ctx.kinds["sag_lukket"] = "numeric"
        call = run_tool(ctx, "gradient_boosting", {}, question_id="q1")
        flags = [f.code for a in call.artifacts for f in a.flags]
        assert "suspiciously_good_model" in flags


class TestAuditAndPrivacy:
    def test_every_result_is_an_artifact_tied_to_its_question_and_call(self, ctx) -> None:
        call = run_tool(ctx, "target_rate_by_group", {"column": "region"}, question_id="q2")
        artifact = only(call)
        refs = {(r.kind, r.id) for r in artifact.audit_refs}
        assert ("plan_question", "q2") in refs and ("tool", call.call_id) in refs
        assert artifact.question_id == "q2" and artifact.stage == "analysis"
        assert ctx.catalog.get(artifact.artifact_id) is artifact
        assert ctx.calls[-1] is call

    def test_charts_are_drawn_to_files(self, ctx) -> None:
        call = run_tool(ctx, "target_rate_by_group", {"column": "region"}, question_id="q1")
        charts = [a for a in call.artifacts if a.kind == "chart"]
        assert charts and (ctx.out_dir / charts[0].spec["chart_file"]).exists()

    def test_no_tool_returns_rows(self, ctx) -> None:
        for name, params in [
            ("distribution", {"column": "region"}),
            ("target_rate_by_group", {"column": "alder"}),
            ("compare_by_target", {"column": "beloeb"}),
            ("correlation", {}),
            ("missingness", {}),
            ("logistic_regression", {}),
        ]:
            for artifact in run_tool(ctx, name, params, question_id="q1").artifacts:
                assert not artifact.row_level
                text = str(artifact.data)
                assert not any(f"'{i}'" in text for i in ctx.frame["kunde_id"].head(50))
                assert len(str(artifact.data)) < 20_000

    def test_the_summary_for_the_agent_carries_the_key_numbers(self, ctx) -> None:
        call = run_tool(ctx, "target_rate_by_group", {"column": "region"}, question_id="q1")
        assert "Hovedstaden" in call.summary and "%" in call.summary


def test_the_frame_is_never_changed(ctx) -> None:
    before = ctx.frame.copy()
    for name in ("logistic_regression", "gradient_boosting", "missingness"):
        run_tool(ctx, name, {}, question_id="q1")
    pd.testing.assert_frame_equal(ctx.frame, before)
