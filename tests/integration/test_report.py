"""The ingestion report and the saved rejected lines."""

from __future__ import annotations

import csv

import pytest

from pipeline.contracts.ingestion import RejectedLine
from pipeline.ingest import ingest, main
from pipeline.ingestion.raw import RawStore
from pipeline.layout import run_file
from pipeline.report import MAX_REJECTS_IN_REPORT, _cell, write_rejected_csv

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


def read_rejects(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class TestReport:
    def test_states_how_much_was_read_and_how_much_was_not(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "too_many_fields.csv", store=store).report
        assert "| Loaded | 3 | 60.00% |" in report
        assert "| Rejected | 2 | 40.00% |" in report
        assert "Loaded + rejected = 5 of 5 source rows." in report

    def test_lists_each_rejected_line_with_its_reason_and_text(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "too_many_fields.csv", store=store).report
        assert "| 3 | 4 fields, expected 3 | 2,Beta,200,EXTRA\\n |" in report
        assert "| 5 | 2 fields, expected 3 | 4,Delta\\n |" in report

    def test_leads_with_the_gate_verdict(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "too_many_fields.csv", store=store).report
        assert report.splitlines()[2].startswith("**Gate 1: BLOCK.**")

    def test_a_clean_file_says_nothing_was_rejected(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "danish_semicolon.csv", store=store).report
        assert "No lines were rejected. Every data row in the file was loaded." in report
        assert "| Rejected | 0 | 0.00% |" in report

    def test_reports_how_the_file_was_interpreted(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "danish_semicolon.csv", store=store).report
        assert "encoding `utf-8`" in report
        assert "delimiter `;`" in report
        assert "Delimiter ';' chosen" in report

    def test_reports_renamed_headers(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "duplicate_blank_headers.csv", store=store).report
        assert "Headers that had to be renamed" in report

    def test_shows_values_that_did_not_fit_a_columns_type(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "ambiguous_dates.csv", store=store).report
        assert "## Column by column" in report
        assert "(ambiguous)" in report

    def test_the_preview_masks_personal_and_identifier_values(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "danish_semicolon.csv", store=store).report
        assert "## First 5 loaded row(s)" in report
        for personal in ("anna.hansen", "@example.dk", "00417"):
            assert personal not in report, f"{personal!r} leaked into the report"
        assert "aaaa.aaaaaa@aaaaaaa.aa" in report

    def test_the_preview_maps_rows_back_to_the_source_file(self, fixtures_dir, store) -> None:
        report = ingest(fixtures_dir / "danish_semicolon.csv", store=store).report
        assert "| Line in file |" in report
        assert "| 2 |" in report

    def test_reports_failed_brief_claims(self, fixtures_dir, store) -> None:
        from pipeline.contracts.context_brief import ContextBrief

        brief = ContextBrief(description="x", goal="y", target="default_12m")
        report = ingest(fixtures_dir / "danish_semicolon.csv", brief, store=store).report
        assert "**does not hold**" in report
        assert "default_12m" in report

    def test_a_long_reject_list_is_truncated_in_the_report_but_not_in_the_file(
        self, tmp_path, store
    ) -> None:
        bad = "\n".join(f"{i},only_two" for i in range(MAX_REJECTS_IN_REPORT + 10))
        path = tmp_path / "many_bad.csv"
        path.write_text(f"id,navn,beloeb\n{bad}\n", encoding="utf-8")

        result = ingest(path, store=store)
        assert result.load_result.n_rejected == MAX_REJECTS_IN_REPORT + 10
        assert "...and 10 more in `rejected_lines.csv`." in result.report

        out = result.write(tmp_path / "out")
        assert len(read_rejects(run_file(out, "rejected_lines.csv"))) == MAX_REJECTS_IN_REPORT + 10


class TestRejectedFile:
    def test_saves_every_rejected_line_in_full(self, fixtures_dir, store, tmp_path) -> None:
        out = ingest(fixtures_dir / "too_many_fields.csv", store=store).write(tmp_path)
        rows = read_rejects(run_file(out, "rejected_lines.csv"))
        assert rows == [
            {"source_line": "3", "reason": "4 fields, expected 3", "raw": "2,Beta,200,EXTRA\n"},
            {"source_line": "5", "reason": "2 fields, expected 3", "raw": "4,Delta\n"},
        ]

    def test_source_line_points_at_the_original_file(self, fixtures_dir, store, tmp_path) -> None:
        """The reason the file exists: find the bad line in the source."""
        source = (fixtures_dir / "too_many_fields.csv").read_text(encoding="utf-8").splitlines()
        out = ingest(fixtures_dir / "too_many_fields.csv", store=store).write(tmp_path)
        for row in read_rejects(run_file(out, "rejected_lines.csv")):
            assert source[int(row["source_line"]) - 1] == row["raw"].rstrip("\n")

    def test_is_written_even_when_nothing_was_rejected(self, fixtures_dir, store, tmp_path) -> None:
        out = ingest(fixtures_dir / "danish_semicolon.csv", store=store).write(tmp_path)
        assert read_rejects(run_file(out, "rejected_lines.csv")) == []
        header = run_file(out, "rejected_lines.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        assert header == "source_line,reason,raw"

    def test_keeps_danish_characters_and_a_bom_for_excel(self, tmp_path) -> None:
        target = tmp_path / "r.csv"
        path = write_rejected_csv(
            [RejectedLine(source_line=7, reason="æøå", raw="blåbær")],
            target,
        )
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
        assert read_rejects(path)[0]["raw"] == "blåbær"

    def test_survives_commas_quotes_and_newlines_in_the_raw_text(self, tmp_path) -> None:
        raw = 'a,"b ""quoted""",c\nsecond line'
        path = write_rejected_csv(
            [RejectedLine(source_line=1, reason="r", raw=raw)], tmp_path / "r.csv"
        )
        assert read_rejects(path)[0]["raw"] == raw


class TestMarkdownSafety:
    def test_pipes_and_newlines_cannot_break_a_table_row(self) -> None:
        cell = _cell("a|b\nc\r")
        assert "\n" not in cell and "\r" not in cell
        assert cell == "a\\|b\\nc\\r"

    def test_long_values_are_truncated_with_an_ellipsis(self) -> None:
        assert _cell("x" * 500, 10) == "xxxxxxxxx…"

    def test_none_is_an_empty_cell(self) -> None:
        assert _cell(None) == ""


class TestCli:
    def test_prints_the_read_summary_and_where_the_files_are(
        self, fixtures_dir, tmp_path, capsys
    ) -> None:
        code = main([str(fixtures_dir / "too_many_fields.csv"), "--out", str(tmp_path / "run")])
        output = capsys.readouterr().out
        assert code == 1
        assert "Read 3 of 5 data rows; 2 rejected (40.00%)." in output
        assert "REPORT.pdf" in output
        assert "rejected_lines.csv" in output

    def test_does_not_point_at_a_rejects_file_when_there_were_none(
        self, fixtures_dir, tmp_path, capsys
    ) -> None:
        main([str(fixtures_dir / "danish_semicolon.csv"), "--out", str(tmp_path / "run")])
        output = capsys.readouterr().out
        assert "Read 12 of 12 data rows; 0 rejected (0.00%)." in output
        assert "Rejected lines:" not in output
