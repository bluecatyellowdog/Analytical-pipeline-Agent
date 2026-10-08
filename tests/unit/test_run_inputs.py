"""What `pipeline.run` is given: a brief that names its data file, or a data file and a brief."""

from __future__ import annotations

import pytest

from pipeline.contracts.context_brief import load_context_brief
from pipeline.run import RunInputError, resolve_inputs

BRIEF = "description: Loans.\ngoal: Understand default.\n"


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_a_brief_may_name_its_data_file(tmp_path) -> None:
    brief = write(tmp_path / "briefs" / "loans.yaml", BRIEF + "data_file: data/raw/loans.csv\n")
    assert load_context_brief(brief).data_file == "data/raw/loans.csv"


def test_a_brief_alone_is_enough_when_it_names_the_data(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "data" / "raw" / "loans.csv", "id,x\n1,2\n")
    write(tmp_path / "briefs" / "loans.yaml", BRIEF + "data_file: data/raw/loans.csv\n")
    source, brief = resolve_inputs("briefs/loans.yaml", None)
    assert source.as_posix() == "data/raw/loans.csv" and brief.as_posix() == "briefs/loans.yaml"


def test_the_old_way_still_works(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "loans.csv", "id,x\n1,2\n")
    write(tmp_path / "brief.yaml", BRIEF)
    source, brief = resolve_inputs("loans.csv", "brief.yaml")
    assert source.name == "loans.csv" and brief.name == "brief.yaml"
    assert resolve_inputs("loans.csv", None) == (source, None)


def test_a_brief_without_a_data_file_says_what_to_add(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "briefs" / "loans.yaml", BRIEF)
    with pytest.raises(RunInputError, match="data_file:"):
        resolve_inputs("briefs/loans.yaml", None)


def test_a_missing_data_file_is_named(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "briefs" / "loans.yaml", BRIEF + "data_file: data/raw/nowhere.csv\n")
    with pytest.raises(RunInputError, match="data/raw/nowhere.csv"):
        resolve_inputs("briefs/loans.yaml", None)
