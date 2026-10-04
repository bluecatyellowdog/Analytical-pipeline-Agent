"""The cleaning flow on a real run folder, with the LLM mocked.

ingest -> analyst (writes decisions.yaml) -> the user decides -> coder (plans,
runs, checks, writes clean data and a report). What is pinned down: the files a
user is told to open exist and say what happened, the manifest records every
agent and its cost, and the folder refuses to be cleaned when it should not be.
"""

from __future__ import annotations

import json
import shutil

import duckdb
import pytest
import yaml

from pipeline.agents import analyst, coder
from pipeline.agents.decisions import DECISIONS_FILE, DecisionsError
from pipeline.agents.llm import FakeLLM
from pipeline.checks.schema import validate_against_contract
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.handoff import Handoff
from pipeline.ingest import ingest
from pipeline.ingestion.raw import RawStore
from tests.unit.test_coder import DIAGNOSIS, GOOD_STEPS, plan_json, step

pytestmark = pytest.mark.e2e

PROVIDER = {"provider": "fake"}
SENTINEL = step("replace_sentinel", ["J1"], column="beloeb", values=["999.0"])


@pytest.fixture
def run_dir(fixtures_dir, tmp_path):
    source = tmp_path / "loans.csv"
    shutil.copy(fixtures_dir / "danish_semicolon.csv", source)
    brief = ContextBrief(
        description="Loan applications.",
        goal="Understand loan size.",
        target="beloeb",
        excluded_variables=["email"],
    )
    with RawStore.open() as store:
        result = ingest(source, brief, store=store)
    return result.write(tmp_path / "run"), source


def diagnose(folder) -> None:
    analyst.run_on_folder(folder, FakeLLM([DIAGNOSIS.model_dump_json()]), PROVIDER)


def decide(folder, **choices) -> None:
    path = folder / DECISIONS_FILE
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    for call_id, choice in choices.items():
        entries[call_id]["decision"] = choice
    path.write_text(yaml.safe_dump(entries, allow_unicode=True), encoding="utf-8")


def test_the_whole_flow_writes_clean_data_and_a_report(run_dir) -> None:
    folder, _ = run_dir
    diagnose(folder)
    template = yaml.safe_load((folder / DECISIONS_FILE).read_text(encoding="utf-8"))
    assert template["J1"]["decision"] == "undecided"

    decide(folder, J1="Treat 999 as missing")
    llm = FakeLLM([plan_json([*GOOD_STEPS, SENTINEL])])
    result = coder.run_on_folder(folder, llm, PROVIDER)
    assert result.run.ok

    for name in (
        coder.CLEANING_PLAN,
        coder.CLEANING_LOG,
        coder.CODER_LOG,
        coder.CLEANING_REPORT,
        coder.CLEAN_PARQUET,
        coder.CLEAN_CSV,
        coder.GATE_2_FILE,
        coder.HANDOFF_FILE,
        coder.SCHEMA_CONTRACT_FILE,
    ):
        assert (folder / name).exists(), name

    parquet = (folder / coder.CLEAN_PARQUET).as_posix()
    rows, missing_amounts = duckdb.sql(
        f"SELECT count(*), count(*) FILTER (WHERE beloeb IS NULL) FROM read_parquet('{parquet}')"
    ).fetchone()
    assert (rows, missing_amounts) == (12, 3)

    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_status"]["cleaning"] == "done"
    assert set(manifest["models"]) == {"analyst", "coder"}
    assert manifest["cost"]["calls"] == 2

    report = (folder / coder.CLEANING_REPORT).read_text(encoding="utf-8")
    assert "12 raw -> 12 clean" in report
    assert "**Gate 2: PASS.**" in report or "**Gate 2: FLAG.**" in report
    assert "**J1** decided by you: Treat 999 as missing" in report
    assert "**J2** still open" in report


def test_the_coder_needs_a_diagnosis_first(run_dir) -> None:
    folder, _ = run_dir
    with pytest.raises(coder.RunFolderError, match="analyst"):
        coder.run_on_folder(folder, FakeLLM([]), PROVIDER)


def test_a_source_file_that_changed_since_ingestion_is_refused(run_dir) -> None:
    folder, source = run_dir
    diagnose(folder)
    with source.open("a", encoding="utf-8") as handle:
        handle.write("99999;40;1.000,00;01-01-2022;\n")
    with pytest.raises(coder.RunFolderError, match="changed"):
        coder.run_on_folder(folder, FakeLLM([]), PROVIDER)


def test_decisions_for_unknown_calls_are_refused(run_dir) -> None:
    folder, _ = run_dir
    diagnose(folder)
    (folder / DECISIONS_FILE).write_text("J9:\n  decision: yes\n", encoding="utf-8")
    with pytest.raises(DecisionsError, match="J9"):
        coder.run_on_folder(folder, FakeLLM([]), PROVIDER)


def test_running_the_analyst_again_keeps_the_users_decisions(run_dir) -> None:
    folder, _ = run_dir
    diagnose(folder)
    decide(folder, J1="Keep 999")
    diagnose(folder)
    assert "Keep 999" in (folder / DECISIONS_FILE).read_text(encoding="utf-8")


def test_a_failed_coder_marks_the_stage_blocked_and_logs_every_attempt(run_dir) -> None:
    folder, _ = run_dir
    diagnose(folder)
    with pytest.raises(coder.CoderError):
        coder.run_on_folder(folder, FakeLLM(["no", "no", "no"]), PROVIDER)

    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_status"]["cleaning"] == "blocked"
    log = json.loads((folder / coder.CODER_LOG).read_text(encoding="utf-8"))
    assert len(log["attempts"]) == 3


def test_the_clean_data_file_matches_its_handoff_contract(run_dir) -> None:
    folder, _ = run_dir
    diagnose(folder)
    coder.run_on_folder(folder, FakeLLM([plan_json()]), PROVIDER)

    handoff = Handoff.model_validate_json((folder / coder.HANDOFF_FILE).read_text(encoding="utf-8"))
    assert handoff.data_file == coder.CLEAN_PARQUET
    assert handoff.column("kunde_id").unique
    data = (folder / handoff.data_file).as_posix()
    checks = validate_against_contract(duckdb.connect(), f"read_parquet('{data}')", handoff.schema_)
    assert all(c.passed for c in checks), [c.message for c in checks if not c.passed]


def test_a_blocked_rerun_leaves_no_stale_clean_data_behind(run_dir) -> None:
    """A good run, then a blocked one: the old handoff must not survive to mislead Stage 3."""
    folder, _ = run_dir
    diagnose(folder)
    coder.run_on_folder(folder, FakeLLM([plan_json()]), PROVIDER)
    assert (folder / coder.HANDOFF_FILE).exists()

    cast_id = step("cast", ["I1"], column="kunde_id", to="integer")
    blocked = plan_json([*GOOD_STEPS, cast_id])
    result = coder.run_on_folder(folder, FakeLLM([blocked, blocked, blocked]), PROVIDER)

    assert result.blocked
    for name in (
        coder.HANDOFF_FILE,
        coder.CLEAN_PARQUET,
        coder.CLEAN_CSV,
        coder.SCHEMA_CONTRACT_FILE,
    ):
        assert not (folder / name).exists(), name
    assert (folder / coder.GATE_2_FILE).exists()
    report = (folder / coder.CLEANING_REPORT).read_text(encoding="utf-8")
    assert "**Gate 2: BLOCK.**" in report and "kunde_id" in report
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_status"]["cleaning"] == "blocked"
