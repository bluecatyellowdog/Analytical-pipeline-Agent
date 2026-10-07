"""The analysis tools: aggregates only, logged as artifacts, and right about known signal.

Run on the benchmark's synthetic loans, where default risk is built to rise with
the loan-to-income ratio and fall with age, and the fee has no effect at all.
For a numeric target the tests add an interest rate built to rise 0.03 per
year of age and 0.5 in one region. So besides the mechanics (validation,
logging, charts, no row-level output, small groups hidden) these tests check
that the tools find what is really there, and find nothing where nothing is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.datasets.synthetic import REGIONS, make_loans
from pipeline.tools import AnalysisContext, ToolError, describe_tools, run_tool, tool_names

KINDS = {
    "id": "id",
    "integer": "numeric",
    "float": "numeric",
    "category": "categorical",
    "date": "date",
    "boolean": "boolean",
}


def loans_context(tmp_path, target: str = "misligholdt") -> AnalysisContext:
    data = make_loans(rows=600, seed=3)
    return AnalysisContext(
        run_id="run_tools",
        frame=data.frame.copy(),
        target=target,
        task_type="binary_classification",
        kinds={c: KINDS[k] for c, k in data.kinds.items()},
        excluded=set(),
        out_dir=tmp_path,
    )


@pytest.fixture
def ctx(tmp_path):
    return loans_context(tmp_path)


@pytest.fixture
def rate_ctx(tmp_path):
    """A numeric target with known effects: +0.03 per year of age, +0.5 in one region."""
    ctx = loans_context(tmp_path, target="rente")
    frame = ctx.frame
    top = frame["region"].value_counts().index[0]
    ctx.effect_region = next(r for r in REGIONS if r != top)
    noise = np.random.default_rng(7).normal(0, 0.5, len(frame))
    frame["rente"] = 2.0 + 0.03 * frame["alder"] + 0.5 * (frame["region"] == ctx.effect_region)
    frame["rente"] += noise
    ctx.kinds["rente"] = "numeric"
    ctx.task_type = "regression"
    ctx.excluded = {"misligholdt"}
    return ctx


def only(call):
    assert len(call.artifacts) >= 1
    return call.artifacts[0]


def fit_of(call) -> dict:
    return next(a for a in call.artifacts if "metrics" in a.data).data["metrics"]


def effects_of(call) -> dict:
    artifact = next(a for a in call.artifacts if "effects" in a.data)
    return {row["feature"]: row for row in artifact.data["effects"]}


def flag_codes(call) -> list[str]:
    return [f.code for a in call.artifacts for f in a.flags]


class TestRegistry:
    def test_the_tools_are_descriptive_statistics_and_regressions(self) -> None:
        assert set(tool_names()) == {
            "summary_table",
            "distribution",
            "target_by_group",
            "compare_by_target",
            "correlation",
            "missingness",
            "linear_regression",
            "logistic_regression",
        }

    def test_the_tool_list_for_prompts_is_generated_from_the_registry(self) -> None:
        text = describe_tools()
        for name in tool_names():
            assert f"- {name}:" in text
        assert "features" in text

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
                run_tool(ctx, "target_by_group", {"column": column}, question_id="q1")
        with pytest.raises(ToolError, match="region"):
            run_tool(
                ctx, "logistic_regression", {"features": ["alder", "region"]}, question_id="q1"
            )


class TestDescriptive:
    def test_summary_table_describes_every_kind_of_column(self, ctx) -> None:
        data = only(run_tool(ctx, "summary_table", {}, question_id="q1")).data
        numbers = {r["column"]: r for r in data["numbers"]}
        assert numbers["alder"]["n"] == 600 and numbers["alder"]["min"] >= 18
        assert numbers["alder"]["p25"] <= numbers["alder"]["median"] <= numbers["alder"]["p75"]
        yes_no = {r["column"]: r for r in data["yes_no"]}
        assert yes_no["misligholdt"]["share_yes"] == pytest.approx(ctx.frame["misligholdt"].mean())
        categories = {r["column"]: r for r in data["categories"]}
        assert categories["region"]["values"] == 5
        assert data["dates"][0]["column"] == "oprettet"
        assert "2019-01-01" <= data["dates"][0]["earliest"] <= data["dates"][0]["latest"]

    def test_summary_table_skips_ids_and_hides_values_too_rare_to_show(self, ctx) -> None:
        ctx.frame["postnr"] = [str(8000 + i) for i in range(600)]
        ctx.kinds["postnr"] = "categorical"
        data = only(run_tool(ctx, "summary_table", {}, question_id="q1")).data
        categories = {r["column"]: r for r in data["categories"]}
        assert categories["postnr"]["values"] == 600 and categories["postnr"]["most_common"] is None
        assert "kunde_id" not in str(data)

    def test_distribution_of_a_number(self, ctx) -> None:
        artifact = only(
            run_tool(ctx, "distribution", {"column": "alder", "bins": 6}, question_id="q1")
        )
        assert sum(b["count"] for b in artifact.data["bins"]) == 600
        assert artifact.data["summary"]["min"] >= 18

    def test_target_rate_by_group_adds_up(self, ctx) -> None:
        artifact = only(run_tool(ctx, "target_by_group", {"column": "region"}, question_id="q1"))
        assert artifact.data["measure"] == "rate"
        groups = artifact.data["groups"]
        assert sum(g["n"] for g in groups) == 600
        for g in groups:
            assert 0 <= g["ci_low"] <= g["rate"] <= g["ci_high"] <= 1

    def test_numbers_are_grouped_into_ranges(self, ctx) -> None:
        artifact = only(
            run_tool(ctx, "target_by_group", {"column": "beloeb", "bins": 4}, question_id="q1")
        )
        rates = [g["rate"] for g in artifact.data["groups"]]
        assert len(rates) == 4
        assert rates[-1] > rates[0], "default should rise with the loan amount"

    def test_small_groups_are_never_shown_on_their_own(self, ctx) -> None:
        ctx.frame.loc[:3, "region"] = "Bornholm"
        artifact = only(run_tool(ctx, "target_by_group", {"column": "region"}, question_id="q1"))
        names = [g["group"] for g in artifact.data["groups"]]
        assert "Bornholm" not in names
        assert all(g["n"] >= 10 for g in artifact.data["groups"])

    def test_missing_values_get_their_own_group(self, ctx) -> None:
        ctx.frame.loc[:59, "alder"] = None
        artifact = only(
            run_tool(ctx, "target_by_group", {"column": "alder", "bins": 3}, question_id="q1")
        )
        assert any(g["group"] == "(missing)" and g["n"] == 60 for g in artifact.data["groups"])

    def test_a_numeric_target_is_described_by_its_mean(self, rate_ctx) -> None:
        data = only(
            run_tool(rate_ctx, "target_by_group", {"column": "region"}, question_id="q1")
        ).data
        assert data["measure"] == "mean"
        assert data["overall"] == pytest.approx(rate_ctx.frame["rente"].mean())
        groups = {g["group"]: g for g in data["groups"]}
        others = [g["mean"] for name, g in groups.items() if name != rate_ctx.effect_region]
        assert groups[rate_ctx.effect_region]["mean"] > max(others)
        for g in groups.values():
            assert g["ci_low"] < g["mean"] < g["ci_high"]

    def test_compare_by_target(self, ctx) -> None:
        artifact = only(run_tool(ctx, "compare_by_target", {"column": "alder"}, question_id="q1"))
        by_class = {c["target"]: c for c in artifact.data["classes"]}
        assert by_class[1]["mean"] < by_class[0]["mean"], "defaulters should be younger"

    def test_compare_by_target_needs_a_yes_no_target(self, rate_ctx) -> None:
        with pytest.raises(ToolError, match="yes/no"):
            run_tool(rate_ctx, "compare_by_target", {"column": "alder"}, question_id="q1")

    def test_correlation_includes_a_numeric_target(self, rate_ctx) -> None:
        pairs = only(run_tool(rate_ctx, "correlation", {}, question_id="q1")).data["pairs"]
        pair = next(p for p in pairs if {p["a"], p["b"]} == {"alder", "rente"})
        assert pair["r"] > 0.5

    def test_missingness(self, ctx) -> None:
        ctx.frame.loc[:59, "indkomst"] = None
        artifact = only(run_tool(ctx, "missingness", {}, question_id="q1"))
        rows = {r["column"]: r for r in artifact.data["columns"]}
        assert rows["indkomst"]["missing"] == 60
        assert 0 <= rows["indkomst"]["target_when_missing"] <= 1

    def test_missingness_with_a_numeric_target_compares_means(self, rate_ctx) -> None:
        rate_ctx.frame.loc[:59, "indkomst"] = None
        data = only(run_tool(rate_ctx, "missingness", {}, question_id="q1")).data
        rows = {r["column"]: r for r in data["columns"]}
        assert data["target_measure"] == "mean"
        assert rows["indkomst"]["target_when_missing"] > 2


class TestRegressions:
    def test_logistic_regression_finds_the_planted_effects(self, ctx) -> None:
        call = run_tool(ctx, "logistic_regression", {}, question_id="q1")
        fit = fit_of(call)
        assert fit["rows_used"] == 600 and fit["events"] == ctx.frame["misligholdt"].sum()
        assert fit["lr_test_p"] < 0.001 and fit["auc"] > 0.75
        effects = effects_of(call)
        assert effects["beloeb"]["ci_low"] > 1, "default rises with the loan amount"
        assert effects["indkomst"]["ci_high"] < 1, "and falls with income"
        assert effects["alder"]["odds_ratio"] < 1, "and with age"
        assert effects["gebyr"]["ci_low"] < 1 < effects["gebyr"]["ci_high"], "the fee does nothing"
        assert not flag_codes(call)

    def test_effects_are_per_round_step_and_categories_against_the_most_common(self, ctx) -> None:
        effects = effects_of(run_tool(ctx, "logistic_regression", {}, question_id="q1"))
        assert effects["alder"]["per"] == "per 20"
        assert effects["beloeb"]["per"] == "per 100,000"
        top = ctx.frame["region"].value_counts().index[0]
        regions = [row for row in effects.values() if row["column"] == "region"]
        assert len(regions) == 4 and all(row["per"] == f"vs {top}" for row in regions)

    def test_linear_regression_recovers_a_known_effect(self, rate_ctx) -> None:
        call = run_tool(
            rate_ctx,
            "linear_regression",
            {"features": ["alder", "indkomst", "region"]},
            question_id="q1",
        )
        fit = fit_of(call)
        assert fit["rows_used"] == 600 and fit["f_test_p"] < 0.001
        assert 0.3 < fit["r_squared"] < 0.9
        effects = effects_of(call)
        age = effects["alder"]
        assert age["per"] == "per 20" and age["ci_low"] < 0.6 < age["ci_high"]
        assert age["p_value"] < 0.001
        region = effects[f"region = {rate_ctx.effect_region}"]
        assert region["ci_low"] < 0.5 < region["ci_high"]
        assert not flag_codes(call)

    def test_no_signal_is_a_valid_result(self, tmp_path) -> None:
        ctx = loans_context(tmp_path, target="gebyr")
        ctx.excluded = {"misligholdt"}
        call = run_tool(
            ctx,
            "linear_regression",
            {"features": ["alder", "indkomst", "beloeb"]},
            question_id="q1",
        )
        fit = fit_of(call)
        assert fit["f_test_p"] > 0.05 and fit["r_squared"] < 0.05
        assert not flag_codes(call)

    def test_the_wrong_kind_of_target_is_refused(self, ctx, rate_ctx) -> None:
        with pytest.raises(ToolError, match="use logistic_regression"):
            run_tool(ctx, "linear_regression", {}, question_id="q1")
        with pytest.raises(ToolError, match="use linear_regression"):
            run_tool(rate_ctx, "logistic_regression", {}, question_id="q1")

    def test_a_column_that_gives_the_answer_away_raises_the_leakage_alarm(self, ctx) -> None:
        ctx.frame["sag_lukket"] = ctx.frame["misligholdt"]
        ctx.kinds["sag_lukket"] = "numeric"
        call = run_tool(ctx, "logistic_regression", {}, question_id="q1")
        flags = [f for a in call.artifacts for f in a.flags]
        assert [f.code for f in flags] == ["suspiciously_good_model"]
        assert "sag_lukket" in flags[0].message
        assert not any("effects" in a.data for a in call.artifacts), "no meaningless odds ratios"

    def test_a_near_copy_of_a_numeric_target_raises_the_leakage_alarm(self, rate_ctx) -> None:
        noise = np.random.default_rng(1).normal(0, 0.01, len(rate_ctx.frame))
        rate_ctx.frame["rente_kopi"] = rate_ctx.frame["rente"] + noise
        rate_ctx.kinds["rente_kopi"] = "numeric"
        call = run_tool(rate_ctx, "linear_regression", {}, question_id="q1")
        assert "suspiciously_good_model" in flag_codes(call)

    def test_features_with_the_same_information_are_refused(self, ctx) -> None:
        ctx.frame["alder_maaneder"] = ctx.frame["alder"] * 12
        ctx.kinds["alder_maaneder"] = "numeric"
        with pytest.raises(ToolError, match="same information"):
            run_tool(ctx, "logistic_regression", {}, question_id="q1")

    def test_a_feature_with_one_value_is_refused(self, ctx) -> None:
        ctx.frame["land"] = "DK"
        ctx.kinds["land"] = "categorical"
        with pytest.raises(ToolError, match="land has the same value"):
            run_tool(ctx, "logistic_regression", {}, question_id="q1")

    def test_rows_with_missing_values_are_left_out_and_counted(self, ctx) -> None:
        ctx.frame.loc[:29, "indkomst"] = None
        ctx.frame.loc[590:, "misligholdt"] = None
        fit = fit_of(run_tool(ctx, "logistic_regression", {}, question_id="q1"))
        assert fit["rows_used"] == 560
        assert fit["rows_left_out"] == {"missing misligholdt": 10, "missing indkomst": 30}

    def test_rare_categories_are_pooled_never_shown_alone(self, ctx) -> None:
        ctx.frame.loc[:3, "region"] = "Bornholm"
        call = run_tool(ctx, "logistic_regression", {}, question_id="q1")
        assert "region = Bornholm" not in effects_of(call)
        assert any("region" in note for note in fit_of(call)["notes"])
        effects = next(a for a in call.artifacts if "effects" in a.data)
        assert effects.min_group_n >= 10

    def test_few_rows_per_effect_are_flagged_and_too_few_refused(self, ctx) -> None:
        defaulted = ctx.frame["misligholdt"] == 1
        keep = ctx.frame.index[defaulted][:12].union(ctx.frame.index[~defaulted][:48])
        ctx.frame = ctx.frame.loc[keep].reset_index(drop=True)
        call = run_tool(ctx, "logistic_regression", {}, question_id="q1")
        assert "few_rows_per_effect" in flag_codes(call)
        ctx.frame = ctx.frame.drop(index=ctx.frame.index[ctx.frame["misligholdt"] == 1][:3])
        with pytest.raises(ToolError, match="rarer outcome"):
            run_tool(ctx, "logistic_regression", {}, question_id="q1")


class TestAuditAndPrivacy:
    def test_every_result_is_an_artifact_tied_to_its_question_and_call(self, ctx) -> None:
        call = run_tool(ctx, "target_by_group", {"column": "region"}, question_id="q2")
        artifact = only(call)
        refs = {(r.kind, r.id) for r in artifact.audit_refs}
        assert ("plan_question", "q2") in refs and ("tool", call.call_id) in refs
        assert artifact.question_id == "q2" and artifact.stage == "analysis"
        assert ctx.catalog.get(artifact.artifact_id) is artifact
        assert ctx.calls[-1] is call

    def test_charts_are_drawn_to_files(self, ctx) -> None:
        for name, params in [
            ("target_by_group", {"column": "region"}),
            ("logistic_regression", {}),
        ]:
            charts = [a for a in run_tool(ctx, name, params, question_id="q1").artifacts]
            charts = [a for a in charts if a.kind == "chart"]
            assert charts and (ctx.out_dir / charts[0].spec["chart_file"]).exists()

    def test_no_tool_returns_rows(self, ctx, rate_ctx) -> None:
        calls = [
            (ctx, name, params)
            for name, params in [
                ("summary_table", {}),
                ("distribution", {"column": "region"}),
                ("target_by_group", {"column": "alder"}),
                ("compare_by_target", {"column": "beloeb"}),
                ("correlation", {}),
                ("missingness", {}),
                ("logistic_regression", {}),
            ]
        ] + [(rate_ctx, "linear_regression", {})]
        ids = ctx.frame["kunde_id"].head(50)
        for context, name, params in calls:
            for artifact in run_tool(context, name, params, question_id="q1").artifacts:
                assert not artifact.row_level
                text = str(artifact.data)
                assert not any(f"'{i}'" in text for i in ids)
                assert len(text) < 20_000

    def test_the_summary_for_the_agent_carries_the_key_numbers(self, ctx) -> None:
        call = run_tool(ctx, "target_by_group", {"column": "region"}, question_id="q1")
        assert "Hovedstaden" in call.summary and "%" in call.summary
        call = run_tool(ctx, "logistic_regression", {}, question_id="q1")
        assert "beloeb x" in call.summary and "per 100,000" in call.summary


def test_the_frame_is_never_changed(ctx) -> None:
    before = ctx.frame.copy()
    for name in ("summary_table", "logistic_regression", "missingness", "correlation"):
        run_tool(ctx, name, {}, question_id="q1")
    pd.testing.assert_frame_equal(ctx.frame, before)
