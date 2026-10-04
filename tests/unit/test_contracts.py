"""Contract tests.

Two jobs: every contract has a realistic instance that round-trips through JSON,
and every validator that protects an audit invariant actually bites.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from pipeline.contracts import (
    AnalysisPlan,
    Artifact,
    ArtifactCatalog,
    AuditLog,
    AuditRef,
    ContextBrief,
    Flag,
    Handoff,
    OpRecord,
    PlanQuestion,
    PublishConfig,
    PublishSelection,
    ReviewVerdict,
    load_context_brief,
)
from tests import factories

EXAMPLES: dict[str, BaseModel] = {
    "context_brief": factories.context_brief(),
    "file_info": factories.file_info(),
    "load_result": factories.load_result(),
    "dataset_profile": factories.dataset_profile(),
    "op_record": factories.op_record(),
    "gate_result": factories.gate_result(),
    "handoff": factories.handoff(),
    "review_verdict": factories.review_verdict(),
    "review_verdict_rejected": factories.review_verdict(accepted=False),
    "analysis_plan": factories.analysis_plan(),
    "artifact": factories.artifact(),
    "artifact_catalog": factories.artifact_catalog(),
    "run_manifest": factories.run_manifest(),
    "pii_finding": factories.pii_finding(),
}


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_example_round_trips_through_json(name: str) -> None:
    """Contracts cross process boundaries as JSON; the trip must be lossless."""
    instance = EXAMPLES[name]
    restored = type(instance).model_validate_json(instance.model_dump_json(by_alias=True))
    assert restored == instance


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_unknown_fields_are_rejected(name: str) -> None:
    """A typo at a stage boundary must fail loudly, not arrive as a missing value."""
    instance = EXAMPLES[name]
    payload = instance.model_dump(mode="json", by_alias=True) | {"totally_new_field": 1}
    with pytest.raises(ValidationError):
        type(instance).model_validate(payload)


class TestContextBrief:
    def test_loads_the_example_yaml(self, repo_root) -> None:
        brief = load_context_brief(repo_root / "examples" / "context_brief.yaml")
        assert brief.target == "default_12m"
        assert brief.id_column == "kunde_id"
        assert "koen" in brief.excluded_variables
        assert {c.kind for c in brief.claims} == {
            "column_unique",
            "binary_column",
            "date_range",
            "value_range",
        }

    def test_note_for_finds_a_column_note(self, repo_root) -> None:
        brief = load_context_brief(repo_root / "examples" / "context_brief.yaml")
        assert brief.note_for("beloeb").unit == "DKK"
        assert brief.note_for("not_a_column") is None

    def test_empty_goal_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ContextBrief(description="rows are loans", goal="   ")

    def test_missing_file_raises(self, tmp_path) -> None:
        empty = tmp_path / "brief.yaml"
        empty.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_context_brief(empty)


class TestOpRecord:
    def test_row_accounting_must_balance(self) -> None:
        """rows_before = rows_after + rows_dropped. An op cannot lose a row silently."""
        with pytest.raises(ValidationError, match="row accounting"):
            OpRecord(
                op_name="drop_duplicates",
                code="df = df.drop_duplicates()",
                rows_before=100,
                rows_after=90,
                rows_dropped=3,
            )

    def test_balanced_drop_is_accepted(self) -> None:
        record = OpRecord(
            op_name="drop_duplicates",
            code="df = df.drop_duplicates()",
            rows_before=100,
            rows_after=90,
            rows_dropped=10,
        )
        assert record.op_id.startswith("op_")


class TestAuditLog:
    def test_collects_flags_from_ops_and_gates(self) -> None:
        log = AuditLog(run_id=factories.RUN_ID)
        record = factories.op_record()
        record.flags = [Flag(code="heavy_imputation", message="40% of values imputed.")]
        log.add_op(record)
        log.add_gate(factories.gate_result())

        assert {f.code for f in log.flags} == {"heavy_imputation", "ambiguous_date_order"}
        assert len(log.open_flags) == 2

    def test_ops_touching_reports_declared_targets(self) -> None:
        """The untouched-column check depends on target_columns being the whole truth."""
        log = AuditLog(run_id=factories.RUN_ID)
        log.add_op(factories.op_record())
        assert [o.op_name for o in log.ops_touching("beloeb")] == ["normalize_decimal"]
        assert log.ops_touching("alder") == []

    def test_acknowledged_flags_leave_open_flags(self) -> None:
        log = AuditLog(run_id=factories.RUN_ID)
        gate = factories.gate_result()
        gate.flags[0].acknowledged = True
        log.add_gate(gate)
        assert log.flags and not log.open_flags


class TestHandoff:
    def test_row_accounting_must_balance(self) -> None:
        payload = factories.handoff().model_dump(by_alias=True)
        payload["rows_dropped"] = 99
        with pytest.raises(ValidationError, match="row accounting"):
            Handoff.model_validate(payload)

    def test_row_loss_is_a_share_of_raw_rows(self) -> None:
        handoff = factories.handoff()
        assert handoff.row_loss == pytest.approx(3 / 4_998)
        assert handoff.column("beloeb").dtype == "DOUBLE"

    def test_schema_is_named_schema_on_the_wire(self) -> None:
        """Python cannot use `schema`, but the JSON contract should read naturally."""
        assert "schema" in factories.handoff().model_dump(by_alias=True)


class TestReviewVerdict:
    def test_rejection_requires_a_fix_request(self) -> None:
        with pytest.raises(ValidationError, match="fix request"):
            ReviewVerdict(accepted=False, summary="Something is off.")

    def test_loop_back_rounds_are_capped(self) -> None:
        with pytest.raises(ValidationError):
            ReviewVerdict(accepted=True, summary="fine", round=4)


class TestAnalysisPlan:
    def _question(self, qid: str) -> PlanQuestion:
        return PlanQuestion(
            id=qid,
            question="How does the default rate vary by age?",
            kind="segment",
            columns=["alder"],
            methods=["segment_rates"],
            priority=1,
            success_criteria="Rates per band.",
        )

    def test_question_count_is_capped(self) -> None:
        base = factories.analysis_plan().model_dump()
        base["questions"] = [self._question(f"q{i}").model_dump() for i in range(6)]
        with pytest.raises(ValidationError, match="3-5 questions"):
            AnalysisPlan.model_validate(base)

    def test_too_few_questions_is_rejected(self) -> None:
        base = factories.analysis_plan().model_dump()
        base["questions"] = [self._question("q1").model_dump()]
        with pytest.raises(ValidationError, match="3-5 questions"):
            AnalysisPlan.model_validate(base)

    def test_duplicate_question_ids_are_rejected(self) -> None:
        base = factories.analysis_plan().model_dump()
        base["questions"] = [self._question("q1").model_dump() for _ in range(3)]
        with pytest.raises(ValidationError, match="unique"):
            AnalysisPlan.model_validate(base)

    def test_prediction_task_needs_a_target(self) -> None:
        base = factories.analysis_plan().model_dump()
        base["target"] = None
        with pytest.raises(ValidationError, match="requires a target"):
            AnalysisPlan.model_validate(base)

    def test_target_cannot_be_an_excluded_feature(self) -> None:
        """Otherwise the plan would ask the analysis to both use and not use a column."""
        base = factories.analysis_plan().model_dump()
        base["excluded_features"].append({"column": "default_12m", "reason": "oops"})
        with pytest.raises(ValidationError, match="cannot also be an excluded feature"):
            AnalysisPlan.model_validate(base)

    def test_infeasible_question_needs_a_reason(self) -> None:
        with pytest.raises(ValidationError, match="infeasible without a reason"):
            PlanQuestion(
                id="q1",
                question="Does income predict default?",
                kind="drivers",
                columns=["indkomst"],
                methods=["logistic_regression"],
                priority=1,
                success_criteria="Coefficient with a confidence interval.",
                feasible=False,
            )


class TestArtifact:
    def test_artifact_without_audit_refs_is_rejected(self) -> None:
        """Spec 7: an untraceable artifact must never exist, let alone be published."""
        with pytest.raises(ValidationError, match="no audit references"):
            Artifact(
                run_id=factories.RUN_ID,
                kind="metric",
                title="AUC",
                stage="analysis",
                data={"auc": 0.71},
            )

    def test_catalog_rejects_artifacts_from_another_run(self) -> None:
        catalog = ArtifactCatalog(run_id="run_other")
        with pytest.raises(ValueError, match="belongs to run"):
            catalog.add(factories.artifact())

    def test_catalog_lookup(self) -> None:
        catalog = factories.artifact_catalog()
        found = catalog.artifacts[0]
        assert catalog.get(found.artifact_id) is found
        assert catalog.get("art_missing") is None
        assert catalog.of_kind("chart") == [found]


class TestPublishConfig:
    def test_duplicate_selection_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="selected twice"):
            PublishConfig(
                slug="demo",
                title="Demo",
                version=1,
                run_id=factories.RUN_ID,
                selections=[
                    PublishSelection(artifact_id="art_1", order=1),
                    PublishSelection(artifact_id="art_1", order=2),
                ],
            )

    def test_duplicate_order_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="same order"):
            PublishConfig(
                slug="demo",
                title="Demo",
                version=1,
                run_id=factories.RUN_ID,
                selections=[
                    PublishSelection(artifact_id="art_1", order=1),
                    PublishSelection(artifact_id="art_2", order=1),
                ],
            )

    def test_slug_must_be_url_safe(self) -> None:
        with pytest.raises(ValidationError):
            PublishConfig(slug="Not A Slug", title="Demo", version=1, run_id=factories.RUN_ID)

    def test_selections_are_ordered(self) -> None:
        config = PublishConfig(
            slug="demo",
            title="Demo",
            version=1,
            run_id=factories.RUN_ID,
            selections=[
                PublishSelection(artifact_id="art_2", order=2),
                PublishSelection(artifact_id="art_1", order=1),
            ],
        )
        assert [s.artifact_id for s in config.ordered_selections] == ["art_1", "art_2"]


class TestProfile:
    def test_pii_level_takes_the_strongest_signal(self) -> None:
        profile = factories.dataset_profile()
        column = profile.column("beloeb")
        assert column.pii_level == "none"
        column.pii = [factories.pii_finding()]
        assert column.pii_level == "likely"

    def test_column_lookup_and_names(self) -> None:
        profile = factories.dataset_profile()
        assert profile.column_names == ["kunde_id", "beloeb"]
        assert profile.column("missing") is None


class TestManifest:
    def test_cost_accumulates(self) -> None:
        manifest = factories.run_manifest()
        manifest.cost.add(prompt_tokens=1_000, completion_tokens=100, usd=0.01)
        assert manifest.cost.calls == 2
        assert manifest.cost.prompt_tokens == 9_431
        assert manifest.cost.usd == pytest.approx(0.048)

    def test_run_id_is_generated(self) -> None:
        assert factories.run_manifest().run_id.startswith("run_")


def test_audit_ref_is_required_shape() -> None:
    ref = AuditRef(kind="op", id="op_1")
    assert ref.note is None
