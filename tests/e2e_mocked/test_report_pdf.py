"""The PDF run report: a summary on top, then everything that happened, and nothing personal.

Built from the run folder after a mocked run. The PDF is the file people share,
so besides checking that each part is there (summary, dirty data, changes, the
agent's own code, checks, review), these tests check that no personal value and
no rejected line's raw text ends up in it.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest
from pypdf import PdfReader

from pipeline.agents.llm import FakeLLM
from pipeline.report_pdf import PDF_FILENAME, build_pdf
from pipeline.run import run_all, summarise
from tests.unit.test_coder import DIAGNOSIS, GOOD_STEPS, plan_json, step

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
    llm = FakeLLM([DIAGNOSIS.model_dump_json(), plan_json([*GOOD_STEPS, CUSTOM])])
    summary = run_all("loans.csv", "brief.yaml", llm, review_client=FakeLLM([VERDICT]))
    return summary


class TestAFullRun:
    def test_the_run_writes_a_pdf_and_says_where(self, full_run) -> None:
        path = full_run.run_dir / PDF_FILENAME
        assert path.read_bytes().startswith(b"%PDF")
        assert str(path.resolve()) in summarise(full_run)

    def test_the_summary_comes_first(self, full_run) -> None:
        first_page = PdfReader(full_run.run_dir / PDF_FILENAME).pages[0].extract_text()
        for expected in ("Summary", "Gate 1", "Gate 2", "Review", "12"):
            assert expected in first_page, expected

    def test_every_part_of_the_story_is_there(self, full_run) -> None:
        text = pdf_text(full_run.run_dir / PDF_FILENAME)
        for expected in (
            "The data as received",
            "What was wrong",
            "What was changed",
            "Checks on the clean data",
            "Review",
            "Audit",
            "Does 999 mean unknown?",
            "normalize_decimal",
            "Does 999 mean unknown in every year?",
        ):
            assert expected in text, expected

    def test_the_agents_own_code_is_shown_in_full(self, full_run) -> None:
        text = pdf_text(full_run.run_dir / PDF_FILENAME)
        assert "def transform(df):" in text
        assert "df['beloeb_tusinder'] = df['beloeb'] / 1000" in text
        assert "New column(s): beloeb_tusinder (DOUBLE)" in text

    def test_no_personal_value_is_in_the_pdf(self, full_run) -> None:
        text = pdf_text(full_run.run_dir / PDF_FILENAME)
        assert "@example.dk" not in text
        assert "00417" not in text


class TestABlockedRun:
    def test_a_blocked_file_still_gets_a_pdf_without_the_raw_lines(self, workspace) -> None:
        summary = run_all("broken.csv", "brief.yaml", FakeLLM([]))
        text = pdf_text(summary.run_dir / PDF_FILENAME)
        assert "BLOCK" in text
        assert "4 fields, expected 3" in text
        assert "2,Beta,200,EXTRA" not in text

    def test_the_pdf_can_be_rebuilt_from_the_folder(self, full_run) -> None:
        path = full_run.run_dir / PDF_FILENAME
        path.unlink()
        assert build_pdf(full_run.run_dir) == path
        assert path.exists()
