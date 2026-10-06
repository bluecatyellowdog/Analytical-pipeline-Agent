"""Stage 2 on a real run folder, with every LLM mocked.

ingest -> analyst (writes decisions.yaml) -> the user decides -> pipeline.clean:
the coder plans and runs, Gate 2 checks, the reviewer judges and may send fixes
back. What is pinned down: the files a user is told to open exist and say what
happened, the manifest records every agent and its cost, the review loop ends
the way the design says (accepted, flagged, budget, previous result kept), and
the folder refuses to be cleaned when it should not be.
"""

from __future__ import annotations

import json
import shutil

import duckdb
import pytest
import yaml

from pipeline import clean
from pipeline.agents import analyst
from pipeline.agents.coder import CoderError
from pipeline.agents.decisions import DECISIONS_FILE, DecisionsError
from pipeline.agents.llm import FakeLLM
from pipeline.checks.schema import validate_against_contract
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.handoff import Handoff
from pipeline.ingest import ingest
from pipeline.ingestion.raw import RawStore
from tests.unit.test_coder import DIAGNOSIS, GOOD_STEPS, plan_json, step
from tests.unit.test_decisions import ai_decided

pytestmark = pytest.mark.e2e

PROVIDER = {"provider": "fake"}
SENTINEL = step("replace_sentinel", ["J1"], column="beloeb", values=["999.0"])
CAST_ID = step("cast", ["I1"], column="kunde_id", to="integer")
EMAIL_FIX = step("tokens_to_null", ["R1"], columns=["email"], tokens=["", "NA"])

VERIFIED = [
    {"claim": "No rows were lost.", "evidence": "12 raw = 12 clean"},
    {"claim": "Ages parse as integers.", "evidence": "alder range 19..71"},
]
EMAIL_FINDING = {
    "step": None,
    "column": "email",
    "severity": "error",
    "problem": "Blank and NA emails are still text, not missing.",
    "evidence": "null_tokens {'': 4, 'NA': 1}",
    "fix": {
        "column": "email",
        "issue": "turn blank and NA emails into missing values",
        "suggested_op": "tokens_to_null",
        "params": {"columns": ["email"], "tokens": ["", "NA"]},
        "reason": "they are null tokens",
    },
}


def verdict(findings=(), ambiguities=(), accepted=True) -> str:
    return json.dumps(
        {
            "verdict": {
                "accepted": accepted,
                "summary": "Reviewed.",
                "verified": VERIFIED,
                "findings": list(findings),
                "ambiguities": list(ambiguities),
            }
        }
    )


REJECT = verdict(findings=[EMAIL_FINDING], accepted=False)


def make_run(fixtures_dir, tmp_path, **brief_fields):
    source = tmp_path / "loans.csv"
    shutil.copy(fixtures_dir / "danish_semicolon.csv", source)
    fields = {
        "description": "Loan applications.",
        "goal": "Understand loan size.",
        "target": "beloeb",
        "excluded_variables": ["email"],
        "judgment_calls": "ask",
    }
    fields.update(brief_fields)
    with RawStore.open() as store:
        result = ingest(source, ContextBrief(**fields), store=store)
    return result.write(tmp_path / "run"), source


@pytest.fixture
def run_dir(fixtures_dir, tmp_path):
    """The user decides every judgment call by hand (judgment_calls: ask)."""
    return make_run(fixtures_dir, tmp_path)


def diagnose(folder) -> None:
    analyst.run_on_folder(folder, FakeLLM([DIAGNOSIS.model_dump_json()]), PROVIDER)


def decide(folder, **choices) -> None:
    path = folder / DECISIONS_FILE
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    for call_id, choice in choices.items():
        entries[call_id]["decision"] = choice
    path.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")


def run_clean(folder, coder_replies, review_replies=None, **options):
    reviewing = review_replies is not None
    return clean.clean_folder(
        folder,
        FakeLLM(coder_replies),
        PROVIDER,
        review_client=FakeLLM(review_replies or []),
        reviewing=reviewing,
        **options,
    )


def manifest_of(folder) -> dict:
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


def report_of(folder) -> str:
    return (folder / clean.CLEANING_REPORT).read_text(encoding="utf-8")


class TestCleaning:
    def test_the_whole_flow_writes_clean_data_and_a_report(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        template = yaml.safe_load((folder / DECISIONS_FILE).read_text(encoding="utf-8"))
        assert template["J1"]["decision"] == "undecided"

        decide(folder, J1="Treat 999 as missing")
        outcome = run_clean(folder, [plan_json([*GOOD_STEPS, SENTINEL])])
        assert outcome.status == "not_reviewed"

        for name in (
            clean.CLEANING_PLAN,
            clean.CLEANING_LOG,
            clean.CODER_LOG,
            clean.CLEANING_REPORT,
            clean.CLEAN_PARQUET,
            clean.CLEAN_CSV,
            clean.GATE_2_FILE,
            clean.HANDOFF_FILE,
            clean.SCHEMA_CONTRACT_FILE,
        ):
            assert (folder / name).exists(), name

        parquet = (folder / clean.CLEAN_PARQUET).as_posix()
        rows, missing_amounts = duckdb.sql(
            "SELECT count(*), count(*) FILTER (WHERE beloeb IS NULL) "
            f"FROM read_parquet('{parquet}')"
        ).fetchone()
        assert (rows, missing_amounts) == (12, 3)

        manifest = manifest_of(folder)
        assert manifest["stage_status"]["cleaning"] == "done"
        assert set(manifest["models"]) == {"analyst", "coder"}
        assert manifest["cost"]["calls"] == 2

        report = report_of(folder)
        assert "12 raw -> 12 clean" in report
        assert "**Gate 2: PASS.**" in report or "**Gate 2: FLAG.**" in report
        assert "**Review: NOT_REVIEWED.**" in report
        assert "**J1** decided by you: Treat 999 as missing" in report
        assert "**J2** still open" in report

    def test_the_clean_data_file_matches_its_handoff_contract(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        run_clean(folder, [plan_json()])

        handoff = Handoff.model_validate_json(
            (folder / clean.HANDOFF_FILE).read_text(encoding="utf-8")
        )
        assert handoff.data_file == clean.CLEAN_PARQUET
        assert handoff.column("kunde_id").unique
        assert handoff.profile is not None and handoff.profile.n_rows == 12
        data = (folder / handoff.data_file).as_posix()
        checks = validate_against_contract(
            duckdb.connect(), f"read_parquet('{data}')", handoff.schema_
        )
        assert all(c.passed for c in checks), [c.message for c in checks if not c.passed]

    def test_a_blocked_rerun_leaves_no_stale_clean_data_behind(self, run_dir) -> None:
        """A good run, then a blocked one: the old handoff must not survive to mislead Stage 3."""
        folder, _ = run_dir
        diagnose(folder)
        run_clean(folder, [plan_json()])
        assert (folder / clean.HANDOFF_FILE).exists()

        blocked = plan_json([*GOOD_STEPS, CAST_ID])
        outcome = run_clean(folder, [blocked, blocked, blocked])

        assert outcome.result.blocked and outcome.status == "blocked"
        for name in clean.DELIVERABLES:
            assert not (folder / name).exists(), name
        assert (folder / clean.GATE_2_FILE).exists()
        report = report_of(folder)
        assert "**Gate 2: BLOCK.**" in report and "kunde_id" in report
        assert manifest_of(folder)["stage_status"]["cleaning"] == "blocked"


class TestReview:
    def test_an_accepted_review(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        outcome = run_clean(folder, [plan_json()], [verdict()])

        assert outcome.status == "accepted"
        assert len(outcome.reviews) == 1
        manifest = manifest_of(folder)
        assert manifest["stage_status"]["review"] == "accepted"
        assert {"coder", "reviewer"} <= set(manifest["models"])
        log = json.loads((folder / clean.REVIEW_LOG).read_text(encoding="utf-8"))
        assert log["rounds"][0]["accepted"] is True
        assert "**Review: ACCEPTED.**" in report_of(folder)

    def test_a_fix_goes_back_to_the_coder_and_is_applied(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        coder_replies = [plan_json(), plan_json([*GOOD_STEPS, EMAIL_FIX])]
        outcome = run_clean(folder, coder_replies, [REJECT, verdict()])

        assert outcome.status == "accepted"
        assert len(outcome.reviews) == 2
        assert not outcome.reviews[0].verdict.accepted
        parquet = (folder / clean.CLEAN_PARQUET).as_posix()
        missing_emails = duckdb.sql(
            f"SELECT count(*) FILTER (WHERE email IS NULL) FROM read_parquet('{parquet}')"
        ).fetchone()[0]
        assert missing_emails == 5
        assert "Round 1: sent back with 1 fix request(s)" in report_of(folder)

    def test_a_reviewer_that_keeps_rejecting_ends_flagged(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        coder_replies = [plan_json(), plan_json([*GOOD_STEPS, EMAIL_FIX])] + [
            plan_json([*GOOD_STEPS, EMAIL_FIX])
        ]
        outcome = run_clean(folder, coder_replies, [REJECT, REJECT, REJECT])

        assert outcome.status == "flagged"
        assert len(outcome.reviews) == 3
        report = report_of(folder)
        assert "For you to look at" in report and "Unresolved fix request" in report
        assert (folder / clean.HANDOFF_FILE).exists()

    def test_ambiguities_are_reported_to_the_user(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        question = {
            "column": "beloeb",
            "question": "Does 999 mean unknown in every year?",
            "evidence": "999 appears 3 times",
        }
        outcome = run_clean(folder, [plan_json()], [verdict(ambiguities=[question])])

        assert outcome.status == "accepted"
        assert "Does 999 mean unknown in every year?" in report_of(folder)
        handoff = json.loads((folder / clean.HANDOFF_FILE).read_text(encoding="utf-8"))
        assert any(f["code"] == "review_ambiguity" for f in handoff["flags"])

    def test_the_budget_stops_the_loop_between_rounds(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        outcome = run_clean(folder, [plan_json()], [REJECT], budget_usd=0.0)

        assert outcome.status == "flagged"
        assert len(outcome.reviews) == 1
        assert any("budget" in note for note in outcome.notes)

    def test_fixes_that_break_gate_2_keep_the_previous_result(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        breaking = plan_json([*GOOD_STEPS, EMAIL_FIX, CAST_ID])
        outcome = run_clean(folder, [plan_json(), breaking, breaking, breaking], [REJECT])

        assert outcome.status == "flagged"
        assert not outcome.result.blocked
        assert any("previous result is kept" in note for note in outcome.notes)
        handoff = Handoff.model_validate_json(
            (folder / clean.HANDOFF_FILE).read_text(encoding="utf-8")
        )
        assert handoff.column("kunde_id").dtype == "VARCHAR"


class TestRefusals:
    def test_cleaning_needs_a_diagnosis_first(self, run_dir) -> None:
        folder, _ = run_dir
        with pytest.raises(clean.RunFolderError, match="analyst"):
            run_clean(folder, [])

    def test_a_source_file_that_changed_since_ingestion_is_refused(self, run_dir) -> None:
        folder, source = run_dir
        diagnose(folder)
        with source.open("a", encoding="utf-8") as handle:
            handle.write("99999;40;1.000,00;01-01-2022;\n")
        with pytest.raises(clean.RunFolderError, match="changed"):
            run_clean(folder, [])

    def test_decisions_for_unknown_calls_are_refused(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        (folder / DECISIONS_FILE).write_text("J9:\n  decision: yes\n", encoding="utf-8")
        with pytest.raises(DecisionsError, match="J9"):
            run_clean(folder, [])

    def test_running_the_analyst_again_keeps_the_users_decisions(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        decide(folder, J1="Keep 999")
        diagnose(folder)
        assert "Keep 999" in (folder / DECISIONS_FILE).read_text(encoding="utf-8")

    def test_a_failed_coder_marks_the_stage_blocked_and_logs_every_attempt(self, run_dir) -> None:
        folder, _ = run_dir
        diagnose(folder)
        with pytest.raises(CoderError):
            run_clean(folder, ["no", "no", "no"])

        assert manifest_of(folder)["stage_status"]["cleaning"] == "blocked"
        log = json.loads((folder / clean.CODER_LOG).read_text(encoding="utf-8"))
        assert len(log["attempts"]) == 3


def ai_section(report: str) -> str:
    """The 'Decided by the AI' part of the report, up to the next section."""
    after = report.split("## Decided by the AI: check these")[1]
    ends = [after.find(h) for h in ("## For you to look at", "## Gate 2 checks")]
    return after[: min(i for i in ends if i >= 0)]


class TestAIDecisions:
    """judgment_calls: decide. The AI settles the calls; the report shows every choice."""

    SKIP_J2 = {"id": "J2", "reason": "The decision keeps missing ages as they are."}

    @pytest.fixture
    def deciding(self, fixtures_dir, tmp_path):
        folder, _ = make_run(
            fixtures_dir,
            tmp_path,
            judgment_calls="decide",
            guidance="999 always means unknown.",
        )
        analyst.run_on_folder(folder, FakeLLM([ai_decided().model_dump_json()]), PROVIDER)
        return folder

    def test_the_report_shows_every_ai_decision_least_certain_first(self, deciding) -> None:
        plan = plan_json([*GOOD_STEPS, SENTINEL], skipped=[self.SKIP_J2])
        run_clean(deciding, [plan])
        report = report_of(deciding)

        section = ai_section(report)
        assert section.index("### J2") < section.index("### J1")
        assert "- **Decision:** Treat 999 as missing" in section
        assert "Please check it." in section
        assert "new missing values: beloeb 3" in section
        assert "no change to the data. The decision keeps missing ages as they are." in section
        assert "Impute median: Two ages filled in." in section
        assert f"uv run python -m pipeline.clean {deciding.as_posix()}" in section
        assert "J2 was decided by the AI with low confidence" in report

        handoff = json.loads((deciding / clean.HANDOFF_FILE).read_text(encoding="utf-8"))
        ai_flags = [f for f in handoff["flags"] if f["code"] == "ai_decision"]
        assert {f["details"]["confidence"] for f in ai_flags} == {"high", "low"}

    def test_an_answer_the_user_changes_becomes_theirs(self, deciding) -> None:
        decide(deciding, J1="Keep 999")
        skip_j1 = {"id": "J1", "reason": "The user keeps 999 as an amount."}
        run_clean(deciding, [plan_json(GOOD_STEPS, skipped=[skip_j1, self.SKIP_J2])])

        report = report_of(deciding)
        section = ai_section(report)
        assert "### J1" not in section and "### J2" in section
        assert "**J1** decided by you: Keep 999" in report
        handoff = json.loads((deciding / clean.HANDOFF_FILE).read_text(encoding="utf-8"))
        assert any(f["code"] == "user_decision" for f in handoff["flags"])
