"""pipeline.run: the whole workflow in one command, with every LLM mocked.

Ingest, diagnose and clean run back to back into one run folder. What is pinned
down: a clean run leaves the report and the clean data, a Gate 1 block stops
before any LLM call (so it costs nothing), runs/index.md links the cleaning
report, and the closing summary says where to look.
"""

from __future__ import annotations

import json
import shutil

import pytest

from pipeline import clean
from pipeline.agents import analysis, planner
from pipeline.agents.llm import FakeLLM
from pipeline.layout import REPORT_PDF, run_file
from pipeline.run import run_all, summarise
from tests.unit.test_analysis_agent import FIXTURE_ANALYSIS_REPLIES
from tests.unit.test_coder import DIAGNOSIS, plan_json
from tests.unit.test_planner import FIXTURE_PLAN_REPLY

pytestmark = pytest.mark.e2e

VERDICT = json.dumps(
    {
        "verdict": {
            "accepted": True,
            "summary": "Checked.",
            "verified": [
                {"claim": "No rows were lost.", "evidence": "12 raw = 12 clean"},
                {"claim": "Ages parse as integers.", "evidence": "alder range 19..71"},
            ],
        }
    }
)
BRIEF = """\
description: Loan applications.
goal: Understand loan size.
target: beloeb
excluded_variables: [email]
judgment_calls: ask
"""


@pytest.fixture
def workspace(fixtures_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copy(fixtures_dir / "danish_semicolon.csv", tmp_path / "loans.csv")
    shutil.copy(fixtures_dir / "too_many_fields.csv", tmp_path / "broken.csv")
    (tmp_path / "brief.yaml").write_text(BRIEF, encoding="utf-8")
    return tmp_path


def test_one_command_runs_all_three_steps(workspace) -> None:
    llm = FakeLLM(
        [DIAGNOSIS.model_dump_json(), plan_json(), FIXTURE_PLAN_REPLY, *FIXTURE_ANALYSIS_REPLIES]
    )
    summary = run_all("loans.csv", "brief.yaml", llm, clients={"reviewer": FakeLLM([VERDICT])})

    assert summary.stopped_at is None
    folder = summary.run_dir
    for name in (
        "ingestion_report.md",
        "diagnosis.md",
        "decisions.yaml",
        clean.CLEANING_REPORT,
        clean.CLEAN_CSV,
        clean.HANDOFF_FILE,
        planner.PLAN_YAML,
        planner.GATE_3_FILE,
        analysis.ANALYSIS_REPORT,
        analysis.ARTIFACTS_JSON,
        analysis.GATE_4_FILE,
    ):
        assert run_file(folder, name).exists(), name
    manifest = json.loads(run_file(folder, "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_status"] == {
        "ingestion": "done",
        "diagnosis": "done",
        "cleaning": "done",
        "review": "accepted",
        "planning": "done",
        "analysis": "done",
    }
    assert summary.planning.plan.approved_by == "auto"


def test_each_agent_runs_on_its_own_model(workspace) -> None:
    """settings.yaml gives agents their own models; the manifest records which one ran."""
    clients = {
        "diagnostic": FakeLLM([DIAGNOSIS.model_dump_json()], model="m-diagnostic"),
        "coder": FakeLLM([plan_json()], model="m-coder"),
        "reviewer": FakeLLM([VERDICT], model="m-reviewer"),
        "planner": FakeLLM([FIXTURE_PLAN_REPLY], model="m-planner"),
        "analysis": FakeLLM(list(FIXTURE_ANALYSIS_REPLIES), model="m-analysis"),
    }
    unused = FakeLLM([])
    summary = run_all("loans.csv", "brief.yaml", unused, clients=clients)

    assert summary.stopped_at is None
    assert unused.calls == []
    manifest = json.loads(run_file(summary.run_dir, "manifest.json").read_text(encoding="utf-8"))
    assert {agent: info["name"] for agent, info in manifest["models"].items()} == {
        agent: f"m-{agent}" for agent in clients
    }


def test_the_index_links_the_report(workspace) -> None:
    llm = FakeLLM(
        [DIAGNOSIS.model_dump_json(), plan_json(), FIXTURE_PLAN_REPLY, *FIXTURE_ANALYSIS_REPLIES]
    )
    summary = run_all("loans.csv", "brief.yaml", llm, clients={"reviewer": FakeLLM([VERDICT])})
    index = (workspace / "runs" / "index.md").read_text(encoding="utf-8")
    assert f"{summary.run_dir.name}/{REPORT_PDF}" in index


def test_a_gate_1_block_stops_before_any_llm_call(workspace) -> None:
    llm = FakeLLM([])
    summary = run_all("broken.csv", "brief.yaml", llm)

    assert summary.stopped_at == "ingest"
    assert llm.calls == []
    assert not run_file(summary.run_dir, "diagnosis.json").exists()
    assert "Gate 1: BLOCK" in summarise(summary)


def test_the_summary_says_where_to_look(workspace) -> None:
    llm = FakeLLM(
        [DIAGNOSIS.model_dump_json(), plan_json(), FIXTURE_PLAN_REPLY, *FIXTURE_ANALYSIS_REPLIES]
    )
    summary = run_all("loans.csv", "brief.yaml", llm, clients={"reviewer": FakeLLM([VERDICT])})
    text = summarise(summary)

    for expected in (
        "Gate 1: PASS",
        "Gate 2:",
        "Review: ACCEPTED",
        "4. Plan",
        "Gate 3: PASS",
        "5. Analyse",
        "Gate 4: PASS",
        "Clean data:",
        "To change the cleaning, edit",
        "To change the analysis, edit",
    ):
        assert expected in text, expected
    assert f"Start here: {run_file(summary.run_dir, REPORT_PDF).resolve()}" in text
