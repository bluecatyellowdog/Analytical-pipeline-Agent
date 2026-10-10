"""The reports of a run: REPORT.pdf with the essentials, detail documents behind it, a tidy folder.

Built from the run folder after a mocked run. What is pinned down: the folder
layout (REPORT.pdf on top, then details/, edit/, data/, logs/), each section of
REPORT.pdf including the self-critical wrap-up, each detail document, that no
personal value and no rejected line's raw text reaches a PDF, and that a folder
from before the layout is moved into it.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest
from pypdf import PdfReader

from pipeline.agents.llm import FakeLLM
from pipeline.layout import (
    ANALYSIS_PDF,
    AUDIT_PDF,
    DATA_QUALITY_PDF,
    DECISIONS_PDF,
    REPORT_PDF,
    run_file,
    upgrade_layout,
)
from pipeline.reports import build_reports
from pipeline.run import run_all, summarise
from tests.unit.test_analysis_agent import FIXTURE_ANALYSIS_REPLIES
from tests.unit.test_coder import DIAGNOSIS, GOOD_STEPS, plan_json, step
from tests.unit.test_planner import FIXTURE_PLAN_REPLY

pytestmark = pytest.mark.e2e

VERDICT = json.dumps(
    {
        "verdict": {
            "accepted": True,
            "summary": "Checked the Danish amounts and the dates.",
            "verified": [
                {"claim": "No rows were lost.", "evidence": "12 raw = 12 clean"},
                {"claim": "Ages parse as integers.", "evidence": "alder range 19..71"},
            ],
            "ambiguities": [
                {
                    "column": "beloeb",
                    "question": "Does 999 mean unknown in every year?",
                    "evidence": "999 appears 3 times",
                }
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
CODE = "def transform(df):\n    df['beloeb_tusinder'] = df['beloeb'] / 1000\n    return df\n"
CUSTOM = step("custom_code", ["I3"], code=CODE, adds=["beloeb_tusinder"])


@pytest.fixture
def workspace(fixtures_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copy(fixtures_dir / "danish_semicolon.csv", tmp_path / "loans.csv")
    shutil.copy(fixtures_dir / "too_many_fields.csv", tmp_path / "broken.csv")
    (tmp_path / "brief.yaml").write_text(BRIEF, encoding="utf-8")
    return tmp_path


def pdf_text(path) -> str:
    text = "\n".join(page.extract_text() for page in PdfReader(path).pages)
    return re.sub(r"\s+", " ", text)


@pytest.fixture
def full_run(workspace):
    llm = FakeLLM(
        [
            DIAGNOSIS.model_dump_json(),
            plan_json([*GOOD_STEPS, CUSTOM]),
            FIXTURE_PLAN_REPLY,
            *FIXTURE_ANALYSIS_REPLIES,
        ]
    )
    return run_all("loans.csv", "brief.yaml", llm, clients={"reviewer": FakeLLM([VERDICT])})


def report(run) -> str:
    return pdf_text(run_file(run.run_dir, REPORT_PDF))


class TestTheFolder:
    def test_the_top_holds_the_report_and_four_folders(self, full_run) -> None:
        top = sorted(p.name for p in full_run.run_dir.iterdir())
        assert top == [REPORT_PDF, "data", "details", "edit", "logs"]

    def test_each_file_is_where_a_person_looks_for_it(self, full_run) -> None:
        folder = full_run.run_dir
        for path in (
            "details/1_analysis.pdf",
            "details/2_data_quality.pdf",
            "details/2_data_columns.xlsx",
            "details/3_decisions.pdf",
            "details/4_audit.pdf",
            "edit/decisions.yaml",
            "edit/analysis_plan.yaml",
            "data/clean_data.csv",
            "data/clean_data.parquet",
            "logs/manifest.json",
            "logs/analysis.json",
        ):
            assert (folder / path).exists(), path

    def test_the_summary_points_to_the_report(self, full_run) -> None:
        text = summarise(full_run)
        assert f"Start here: {run_file(full_run.run_dir, REPORT_PDF).resolve()}" in text
        assert "edit" in text and "decisions.yaml" in text


class TestTheReport:
    def test_it_is_short_and_has_every_section(self, full_run) -> None:
        path = run_file(full_run.run_dir, REPORT_PDF)
        assert len(PdfReader(path).pages) <= 4
        text = report(full_run)
        for expected in (
            "loans: what the data says",
            "In brief",
            "Your question: Understand loan size.",
            "1 What we found",
            "2 How good the data is",
            "3 Decisions made for you",
            "4 Wrap-up: what might be wrong, and what we recommend",
            "More detail",
        ):
            assert expected in text, expected

    def test_the_answer_and_the_status_come_first(self, full_run) -> None:
        first_page = re.sub(
            r"\s+", " ", PdfReader(run_file(full_run.run_dir, REPORT_PDF)).pages[0].extract_text()
        )
        assert "Loan amounts vary widely." in first_page
        assert "Status: Gate 2 raised a warning" in first_page

    def test_data_quality_and_decisions_in_short(self, full_run) -> None:
        text = report(full_run)
        assert "12 in the file, 12 read" in text
        assert "Does 999 mean unknown?" in text and "nobody" in text

    def test_the_wrap_up_is_self_critical_and_says_what_to_do(self, full_run) -> None:
        text = report(full_run)
        assert "J1 is undecided" in text and "Decide J1 in edit/decisions.yaml" in text
        assert "The reviewer asked (beloeb): Does 999 mean unknown in every year?" in text
        assert "one unusual amount can move every summary" in text
        assert "Collect more loans" in text

    def test_it_links_to_every_detail_document(self, full_run) -> None:
        text = report(full_run)
        for name in (ANALYSIS_PDF, DATA_QUALITY_PDF, DECISIONS_PDF, AUDIT_PDF):
            assert f"details/{name}" in text, name


class TestTheDetails:
    def read(self, run, name) -> str:
        return pdf_text(run_file(run.run_dir, name))

    def test_the_analysis_with_its_evidence_and_plan(self, full_run) -> None:
        text = self.read(full_run, ANALYSIS_PDF)
        for expected in (
            "Loan amounts vary widely. (moderate)",
            "T1 summary_table: 12 rows.",
            "not possible with this data",
            "Gate 4 checks",
            "The plan behind the analysis",
            "email (brief): Excluded by the brief.",
        ):
            assert expected in text, expected

    def test_data_quality_with_every_change_and_the_agents_code(self, full_run) -> None:
        text = self.read(full_run, DATA_QUALITY_PDF)
        for expected in (
            "The data as received",
            "What was wrong",
            "What was changed",
            "Checks on the clean data",
            "The review",
            "normalize_decimal",
            "def transform(df):",
            "df['beloeb_tusinder'] = df['beloeb'] / 1000",
        ):
            assert expected in text, expected

    def test_decisions_with_options_and_how_to_change_them(self, full_run) -> None:
        text = self.read(full_run, DECISIONS_PDF)
        for expected in (
            "J1 (beloeb): Does 999 mean unknown?",
            "Option: Treat 999 as missing",
            "Open: nobody has decided it",
            "Measured:",
            "How to change a decision",
        ):
            assert expected in text, expected

    def test_the_audit_lists_the_folders(self, full_run) -> None:
        text = self.read(full_run, AUDIT_PDF)
        assert "Models" in text and "logs" in text and "manifest.json" in text

    def test_no_personal_value_reaches_any_pdf(self, full_run) -> None:
        for path in full_run.run_dir.rglob("*.pdf"):
            text = pdf_text(path)
            assert "@example.dk" not in text, path.name
            assert "00417" not in text, path.name


class TestOtherRuns:
    def test_a_blocked_file_still_gets_a_report_without_the_raw_lines(self, workspace) -> None:
        summary = run_all("broken.csv", "brief.yaml", FakeLLM([]))
        text = report(summary)
        assert "Gate 1 blocked" in text and "The loading step stopped" in text
        quality = pdf_text(run_file(summary.run_dir, DATA_QUALITY_PDF))
        assert "4 fields, expected 3" in quality
        assert "2,Beta,200,EXTRA" not in quality

    def test_the_reports_can_be_rebuilt_from_the_folder(self, full_run) -> None:
        path = run_file(full_run.run_dir, REPORT_PDF)
        path.unlink()
        assert build_reports(full_run.run_dir) == path and path.exists()

    def test_a_folder_from_before_the_layout_is_moved_into_it(self, full_run) -> None:
        folder = full_run.run_dir
        for path in [p for p in folder.rglob("*") if p.is_file()]:
            if path.parent != folder:
                path.replace(folder / path.name)
        (folder / "run_report.pdf").write_bytes(b"%PDF old")
        moved = upgrade_layout(folder)
        assert any("decisions.yaml -> edit/decisions.yaml" in m for m in moved)
        assert not (folder / "run_report.pdf").exists()
        assert (folder / "logs" / "manifest.json").exists()
        assert build_reports(folder).exists()
