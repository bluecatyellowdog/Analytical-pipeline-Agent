"""Realistic example instances of every contract.

These are the Phase 0 example instances, and they double as test fixtures: if a
contract changes, these stop building and every consumer sees it immediately.
"""

from __future__ import annotations

from pipeline.contracts import (
    AnalysisPlan,
    Artifact,
    ArtifactCatalog,
    AuditRef,
    CheckResult,
    Claim,
    ColumnNote,
    ColumnProfile,
    ColumnRename,
    ColumnSchema,
    ColumnStats,
    ContextBrief,
    DatasetProfile,
    ExcludedFeature,
    FileInfo,
    FixRequest,
    Flag,
    GateResult,
    Handoff,
    LoadResult,
    ModelInfo,
    NumericSummary,
    OpRecord,
    PIIFinding,
    PlanQuestion,
    PublishConfig,
    PublishSelection,
    RejectedLine,
    ReviewVerdict,
    RunManifest,
    SentinelCandidate,
    TypeProposal,
    ValueCount,
)

RUN_ID = "run_000000000001"


def context_brief() -> ContextBrief:
    return ContextBrief(
        description="One row per consumer loan application at a Danish retail bank.",
        goal="Understand which characteristics are associated with default within 12 months.",
        target="default_12m",
        unit_of_observation="one loan application",
        id_column="kunde_id",
        column_notes=[
            ColumnNote(name="beloeb", description="Applied loan amount", unit="DKK"),
        ],
        known_issues=["Applications before 2021 were migrated and have more missing values."],
        focus_areas=["Applicants under 25"],
        excluded_variables=["koen"],
        claims=[Claim(kind="column_unique", column="kunde_id", text="cust_id is unique")],
    )


def file_info() -> FileInfo:
    return FileInfo(
        path="data/raw/laan_2019_2024.csv",
        format="csv",
        size_bytes=482_113,
        file_hash="a" * 64,
        encoding="cp1252",
        delimiter=";",
        detection_notes=["Delimiter ';' chosen: consistent field count over the first 50 lines."],
    )


def load_result() -> LoadResult:
    return LoadResult(
        table="raw.laan_2019_2024",
        file=file_info(),
        n_source_rows=5_000,
        n_loaded=4_998,
        n_rejected=2,
        rejects=[
            RejectedLine(source_line=1_204, reason="6 fields, expected 5", raw="00417;31;...")
        ],
        columns=["kunde_id", "alder", "beloeb", "dato", "email"],
        renames=[
            ColumnRename(position=5, original="", final="column_5", reason="Blank header.")
        ],
        data_hash="b" * 64,
    )


def dataset_profile(table: str = "raw.laan_2019_2024") -> DatasetProfile:
    return DatasetProfile(
        table=table,
        n_rows=4_998,
        n_columns=2,
        n_duplicate_rows=3,
        candidate_keys=["kunde_id"],
        columns=[
            ColumnProfile(
                name="kunde_id",
                position=0,
                raw_dtype="text",
                n_rows=4_998,
                n_missing=0,
                missing_share=0.0,
                n_distinct=4_998,
                distinct_share=1.0,
                top_values=[ValueCount(value="00417", count=1, share=0.0002)],
                type_proposal=TypeProposal(
                    column="kunde_id",
                    proposed="id",
                    coverage=1.0,
                    details={"leading_zeros": True},
                    notes=["Unique with leading zeros; kept as text."],
                ),
            ),
            ColumnProfile(
                name="beloeb",
                position=1,
                raw_dtype="text",
                n_rows=4_998,
                n_missing=412,
                missing_share=0.0824,
                null_tokens={"NA": 412},
                n_distinct=3_211,
                distinct_share=0.6425,
                numeric=NumericSummary(
                    min=5_000.0,
                    q1=48_000.0,
                    median=112_500.0,
                    q3=210_000.0,
                    max=999_000.0,
                    mean=148_233.5,
                    std=121_004.2,
                    n_outliers=37,
                    outlier_bounds=(-438_000.0, 696_000.0),
                ),
                sentinel_candidates=[
                    SentinelCandidate(
                        value="999",
                        count=88,
                        share=0.0176,
                        reason="Common coded-missing value, far from the distribution.",
                    )
                ],
                type_proposal=TypeProposal(
                    column="beloeb",
                    proposed="float",
                    coverage=0.9992,
                    failing_examples=["i alt", "-"],
                    details={"decimal_separator": ",", "thousands_separator": "."},
                ),
                flags=[
                    Flag(
                        code="sentinel_candidate",
                        severity="warn",
                        message="Value 999 appears 88 times and may mean 'unknown'.",
                        column="beloeb",
                        stage="ingestion",
                    )
                ],
            ),
        ],
    )


def op_record() -> OpRecord:
    return OpRecord(
        op_name="normalize_decimal",
        params={"column": "beloeb", "decimal_separator": ",", "thousands_separator": "."},
        code="df['beloeb'] = normalize_decimal(df['beloeb'], dec=',', thousands='.')",
        target_columns=["beloeb"],
        rationale="Profile shows a Danish decimal comma; parsing as-is would misread magnitudes.",
        rows_before=4_998,
        rows_after=4_998,
        cells_changed={"beloeb": 4_586},
        new_nulls={"beloeb": 0},
        before={
            "beloeb": ColumnStats(
                name="beloeb", dtype="text", n_rows=4_998, n_null=412, n_distinct=3_211
            )
        },
        after={
            "beloeb": ColumnStats(
                name="beloeb",
                dtype="float64",
                n_rows=4_998,
                n_null=412,
                n_distinct=3_211,
                min=5_000.0,
                max=999_000.0,
                mean=148_233.5,
            )
        },
        input_hash="b" * 64,
        output_hash="c" * 64,
        duration_ms=41.2,
    )


def gate_result() -> GateResult:
    return GateResult(
        gate="gate_1_ingestion",
        outcome="flag",
        checks=[
            CheckResult(
                name="rows_accounted_for",
                layer="invariant",
                passed=True,
                message="4998 loaded + 2 rejected = 5000 source rows.",
                observed={"n_source_rows": 5_000, "n_loaded": 4_998, "n_rejected": 2},
            ),
            CheckResult(
                name="reject_rate_below_threshold",
                layer="invariant",
                passed=True,
                severity="block",
                message="Reject rate 0.04% is below the 1% block threshold.",
                observed={"reject_rate": 0.0004},
            ),
        ],
        flags=[
            Flag(
                code="ambiguous_date_order",
                severity="warn",
                message="No day above 12 in 'dato': day/month order cannot be determined.",
                column="dato",
                stage="ingestion",
            )
        ],
    )


def handoff() -> Handoff:
    return Handoff(
        run_id=RUN_ID,
        table="clean.laan_2019_2024",
        source_table="raw.laan_2019_2024",
        rows_raw=4_998,
        rows_clean=4_995,
        rows_dropped=3,
        dropped_rows_table="audit.laan_2019_2024__dropped",
        schema=[
            ColumnSchema(
                name="kunde_id", dtype="string", nullable=False, null_rate=0.0, n_distinct=4_995
            ),
            ColumnSchema(
                name="beloeb",
                dtype="float64",
                nullable=True,
                null_rate=0.0824,
                n_distinct=3_209,
                transformations=["op_000000000001"],
            ),
        ],
        dropped_columns={"email": "92% missing and flagged as PII; not needed for the goal."},
        applied_ops=["op_000000000001"],
        profile=dataset_profile("clean.laan_2019_2024"),
        data_hash="d" * 64,
    )


def review_verdict(accepted: bool = True) -> ReviewVerdict:
    if accepted:
        return ReviewVerdict(
            accepted=True,
            summary="Row accounting and replay pass; the 999 sentinel is flagged, not imputed.",
        )
    return ReviewVerdict(
        accepted=False,
        summary="The 999 sentinel in beloeb was left as a real amount and skews the mean.",
        fix_requests=[
            FixRequest(
                column="beloeb",
                issue="999 is a coded missing value but is treated as an amount.",
                suggested_op="replace_sentinel",
                params={"value": 999, "with": None},
                reason="The brief states 999 meant 'unknown' before 2021.",
            )
        ],
    )


def analysis_plan(approved: bool = False) -> AnalysisPlan:
    return AnalysisPlan(
        run_id=RUN_ID,
        objective="Identify characteristics associated with 12-month default.",
        interpretation="Read as a classification problem on default_12m, with age as a focus.",
        assumptions=["Each row is one application and applications are independent."],
        target="default_12m",
        task_type="binary_classification",
        task_type_reason="default_12m has two values and 8.4% positives: binary classification.",
        questions=[
            PlanQuestion(
                id="q1",
                question="How does the default rate vary by applicant age band?",
                kind="segment",
                columns=["alder", "default_12m"],
                methods=["segment_rates"],
                priority=1,
                success_criteria="Default rate per age band with confidence intervals.",
            ),
            PlanQuestion(
                id="q2",
                question="Which features are most associated with default?",
                kind="drivers",
                columns=["alder", "beloeb", "default_12m"],
                methods=["logistic_regression", "shap_summary"],
                priority=2,
                success_criteria="Ranked feature effects agreeing between model and SHAP.",
            ),
            PlanQuestion(
                id="q3",
                question="Can default be predicted better than the base rate?",
                kind="prediction",
                columns=["alder", "beloeb", "default_12m"],
                methods=["dummy_baseline", "logistic_regression", "lightgbm", "cross_validate"],
                priority=3,
                success_criteria="Cross-validated AUC above the dummy baseline with fold variance.",
            ),
        ],
        excluded_features=[
            ExcludedFeature(
                column="koen", reason="Legally sensitive per the brief.", source="brief"
            ),
            ExcludedFeature(
                column="kunde_id", reason="Identifier, no predictive meaning.", source="planner"
            ),
        ],
        success_criteria="Each question answered or explicitly marked infeasible with a reason.",
        limitations=["8.4% of amounts are missing and were not imputed."],
        approved=approved,
        human_edits=["Dropped a fourth question about postnummer: excluded by the brief."]
        if approved
        else [],
    )


def artifact() -> Artifact:
    return Artifact(
        run_id=RUN_ID,
        kind="chart",
        title="Default rate by age band",
        description="Share of applications defaulting within 12 months, by age band.",
        stage="analysis",
        question_id="q1",
        data=[
            {"age_band": "18-24", "n": 612, "default_rate": 0.141},
            {"age_band": "25-34", "n": 1_488, "default_rate": 0.092},
            {"age_band": "35-49", "n": 1_903, "default_rate": 0.071},
            {"age_band": "50+", "n": 992, "default_rate": 0.058},
        ],
        min_group_n=612,
        audit_refs=[
            AuditRef(kind="op", id="op_000000000001", note="Amount normalisation."),
            AuditRef(kind="plan_question", id="q1"),
        ],
        interpretation="Default rates fall with age; the youngest band is about double the oldest.",
    )


def artifact_catalog() -> ArtifactCatalog:
    catalog = ArtifactCatalog(run_id=RUN_ID)
    catalog.add(artifact())
    return catalog


def publish_config(artifact_id: str) -> PublishConfig:
    return PublishConfig(
        slug="danish-consumer-loans",
        title="Consumer loan defaults: an audited analysis",
        version=1,
        run_id=RUN_ID,
        summary="What the pipeline found, and every step it took to get there.",
        selections=[
            PublishSelection(
                artifact_id=artifact_id,
                order=1,
                commentary="Age is the clearest single signal, but see the limitations below.",
            )
        ],
        headline_metrics=[artifact_id],
        acknowledged_flags=["sentinel_candidate"],
        author="chris",
    )


def run_manifest() -> RunManifest:
    manifest = RunManifest(
        pipeline_version="0.1.0",
        python_version="3.12.7",
        platform="Windows-11",
        git_sha="0" * 40,
        context_brief=context_brief(),
        inputs=[file_info()],
        models={
            "cleaning": ModelInfo(provider="anthropic", name="claude-sonnet-5", temperature=0.0),
        },
        prompt_versions={"cleaning": "v1"},
        seeds={"global": 42, "profiling_sample": 42},
        config={"row_loss_block_threshold": 0.20},
        stage_status={"ingestion": "done", "cleaning": "pending"},
    )
    manifest.cost.add(prompt_tokens=8_431, completion_tokens=912, usd=0.038, wall_clock_s=11.4)
    return manifest


def pii_finding() -> PIIFinding:
    return PIIFinding(
        kind="cpr",
        level="likely",
        n_matches=4_912,
        share=0.983,
        detector="regex:cpr",
        matched_column_name=True,
    )
