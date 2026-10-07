"""The benchmark runner, with a stand-in pipeline so nothing calls an LLM.

Pins down that every scenario can be built with its answer key, that the
runner hands the broken file and brief to the pipeline and scores what comes
back, and that the summary says how each scenario went.
"""

from __future__ import annotations

import json

import duckdb
import pandas as pd
import pytest
import yaml

from benchmark.datasets.synthetic import make_loans
from benchmark.run_benchmark import RunInfo, build_case, run_benchmark
from benchmark.scenarios import SCENARIOS

pytestmark = pytest.mark.e2e


def perfect_pipeline(data):
    """A stand-in that 'cleans' perfectly: it writes the clean data back out."""

    def run(source, brief, out):
        out.mkdir(parents=True)
        frame = data.frame.copy()
        frame.insert(0, "_source_line", range(2, len(frame) + 2))
        key = json.loads((source.parent / "answer_key.json").read_text(encoding="utf-8"))
        for injection in key["injections"]:
            if injection["expected"] == "missing":
                for row in injection["rows"]:
                    frame.loc[frame["_source_line"] == row + 2, injection["column"]] = None
        path = (out / "clean_data.parquet").as_posix()
        duckdb.from_df(frame).write_parquet(path)
        (out / "diagnosis.json").write_text(
            json.dumps({"summary": "s", "issues": [], "judgment_calls": [], "columns": []}),
            encoding="utf-8",
        )
        return RunInfo(run_dir=out, stopped_at=None, cost_usd=0.0, gate_2="pass", review="accepted")

    return run


class TestScenarios:
    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_every_scenario_builds_with_its_answer_key(self, name, tmp_path) -> None:
        data = make_loans(rows=80, seed=1)
        case = build_case(SCENARIOS[name], data, seed=3, case_dir=tmp_path)

        assert case.source.exists() and case.brief.exists()
        key = json.loads((tmp_path / "answer_key.json").read_text(encoding="utf-8"))
        assert {i["kind"] for i in key["injections"]} == {p.kind for p in SCENARIOS[name].plants}
        brief = yaml.safe_load(case.brief.read_text(encoding="utf-8"))
        assert brief["target"] == "misligholdt" and brief["judgment_calls"] == "decide"

    def test_documented_codes_reach_the_brief(self, tmp_path) -> None:
        data = make_loans(rows=80, seed=1)
        case = build_case(SCENARIOS["danish_export"], data, seed=3, case_dir=tmp_path)
        brief = yaml.safe_load(case.brief.read_text(encoding="utf-8"))
        notes = {n["name"]: n for n in brief["column_notes"]}
        assert "999" in notes["indkomst"]["known_issues"]

    def test_the_broken_file_uses_the_scenarios_delimiter(self, tmp_path) -> None:
        data = make_loans(rows=80, seed=1)
        case = build_case(SCENARIOS["danish_export"], data, seed=3, case_dir=tmp_path)
        header = case.source.read_text(encoding="utf-8").splitlines()[0]
        assert header.count(";") == len(data.frame.columns) - 1


class TestRunner:
    def test_a_perfect_pipeline_scores_perfectly(self, tmp_path) -> None:
        data = make_loans(rows=80, seed=1)
        results = run_benchmark(
            ["danish_export", "manual_entry"],
            data=data,
            seed=3,
            out_dir=tmp_path,
            pipeline=perfect_pipeline(data),
        )
        assert [r.scenario for r in results] == ["danish_export", "manual_entry"]
        for result in results:
            assert result.score.repair_rate == 1.0
            assert result.score.damaged_cells == 0

    def test_the_summary_reports_each_scenario(self, tmp_path) -> None:
        data = make_loans(rows=80, seed=1)
        run_benchmark(
            ["danish_export"], data=data, seed=3, out_dir=tmp_path, pipeline=perfect_pipeline(data)
        )
        summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
        assert "danish_export" in summary and "100%" in summary
        assert "decimal_comma" in summary
        scores = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
        assert scores["cases"][0]["repair_rate"] == 1.0

    def test_a_pipeline_that_stops_early_scores_zero(self, tmp_path) -> None:
        data = make_loans(rows=80, seed=1)

        def stops(source, brief, out):
            out.mkdir(parents=True)
            return RunInfo(run_dir=out, stopped_at="ingest", cost_usd=0.0)

        result = run_benchmark(
            ["manual_entry"], data=data, seed=3, out_dir=tmp_path, pipeline=stops
        )[0]
        assert result.score.repair_rate == 0.0
        assert result.stopped_at == "ingest"


def test_scores_are_read_from_parquet_with_real_types(tmp_path) -> None:
    from benchmark.run_benchmark import load_clean_rows

    frame = pd.DataFrame({"_source_line": [2], "alder": [31], "region": ["Nord"]})
    path = tmp_path / "clean_data.parquet"
    duckdb.from_df(frame).write_parquet(path.as_posix())
    assert load_clean_rows(path) == [{"_source_line": 2, "alder": 31, "region": "Nord"}]
