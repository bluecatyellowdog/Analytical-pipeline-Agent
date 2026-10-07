"""The benchmark's scorer, checked against outcomes whose right score is obvious.

A perfect cleaning must score 100% fixed and no damage; doing nothing must score
0% fixed and still no damage (leaving text as text is not harm); breaking a good
cell or dropping a good row must show up as damage. Detection and false alarms
are read from the diagnosis.
"""

from __future__ import annotations

import pytest

from benchmark.datasets.synthetic import make_loans
from benchmark.injectors import Corrupted, inject
from benchmark.metrics import score_case
from pipeline.contracts.diagnosis import Diagnosis


@pytest.fixture(scope="module")
def setup():
    data = make_loans(rows=60, seed=2)
    corrupted = Corrupted.from_clean(data)
    inject(corrupted, data, "decimal_comma", seed=1, column="beloeb")
    inject(corrupted, data, "null_tokens", seed=2, column="alder", share=0.1)
    inject(corrupted, data, "sentinel", seed=3, column="indkomst", code="999", share=0.1)
    inject(corrupted, data, "whitespace_case", seed=4, column="region", share=0.2)
    inject(corrupted, data, "outliers", seed=5, column="gebyr", share=0.05)
    inject(corrupted, data, "duplicates", seed=6, share=0.1)
    return data, corrupted


def perfect_rows(data, corrupted) -> list[dict]:
    """What an ideal cleaning returns: the clean data, missing where only missing is right."""
    missing = {
        (row, i.column) for i in corrupted.injections if i.expected == "missing" for row in i.rows
    }
    rows = []
    for position in range(len(data.frame)):
        values = data.frame.iloc[position].to_dict()
        for column in list(values):
            if (position, column) in missing:
                values[column] = None
        rows.append({"_source_line": position + 2, **values})
    return rows


def untouched_rows(corrupted) -> list[dict]:
    return [
        {"_source_line": position + 2, **corrupted.table.iloc[position].to_dict()}
        for position in range(len(corrupted.table))
    ]


def diagnosis(*issues, calls=()) -> Diagnosis:
    return Diagnosis.model_validate(
        {
            "summary": "s",
            "issues": [
                {"id": f"I{n}", "evidence": "e", "fix": "f", **issue}
                for n, issue in enumerate(issues, start=1)
            ],
            "judgment_calls": [
                {
                    "id": f"J{n}",
                    "options": ["a", "b"],
                    "recommendation": "r",
                    "why_it_matters": "w",
                    **call,
                }
                for n, call in enumerate(calls, start=1)
            ],
            "columns": [],
        }
    )


class TestFixing:
    def test_a_perfect_cleaning_fixes_everything_and_damages_nothing(self, setup) -> None:
        data, corrupted = setup
        score = score_case(data, corrupted, perfect_rows(data, corrupted), diagnosis())
        fixable = [s for s in score.injections if s.expected != "flag"]
        assert all(s.fixed == s.cells for s in fixable), [
            (s.kind, s.fixed, s.cells) for s in fixable
        ]
        assert score.repair_rate == 1.0
        assert score.damaged_cells == 0 and score.wrongly_dropped_rows == 0

    def test_doing_nothing_fixes_nothing_but_is_not_damage(self, setup) -> None:
        data, corrupted = setup
        score = score_case(data, corrupted, untouched_rows(corrupted), diagnosis())
        assert score.repair_rate == 0.0
        assert score.damaged_cells == 0

    def test_breaking_a_good_cell_or_dropping_a_good_row_is_damage(self, setup) -> None:
        data, corrupted = setup
        rows = perfect_rows(data, corrupted)
        touched = {row for i in corrupted.injections if i.column == "gebyr" for row in i.rows}
        target = next(r for r in rows if r["_source_line"] - 2 not in touched)
        target["gebyr"] = target["gebyr"] + 1
        victim = next(r for r in rows if r["_source_line"] == 12)
        rows.remove(victim)
        score = score_case(data, corrupted, rows, diagnosis())
        assert score.damaged_cells >= 1
        assert score.wrongly_dropped_rows == 1
        assert any("gebyr" in example for example in score.damaged_examples)

    def test_values_are_compared_by_meaning_not_by_spelling(self, setup) -> None:
        data, corrupted = setup
        rows = perfect_rows(data, corrupted)
        for row in rows:
            row["region"] = row["region"].lower()
            row["misligholdt"] = bool(row["misligholdt"])
            row["alder"] = float(row["alder"]) if row["alder"] is not None else None
        score = score_case(data, corrupted, rows, diagnosis())
        assert score.damaged_cells == 0

    def test_no_clean_data_means_nothing_fixed(self, setup) -> None:
        data, corrupted = setup
        score = score_case(data, corrupted, None, diagnosis())
        assert score.repair_rate == 0.0 and not score.clean_available


class TestFinding:
    def test_problems_count_as_found_when_the_diagnosis_names_them(self, setup) -> None:
        data, corrupted = setup
        found = diagnosis(
            {"column": "beloeb", "kind": "decimal_format", "suggested_op": "normalize_decimal"},
            {"column": "alder", "kind": "missing_tokens", "suggested_op": "tokens_to_null"},
            {"column": None, "kind": "duplicates", "suggested_op": "drop_duplicates"},
            calls=[{"column": "indkomst", "question": "Does 999 mean unknown?"}],
        )
        score = score_case(data, corrupted, perfect_rows(data, corrupted), found)
        by_kind = {s.kind: s.found for s in score.injections}
        assert by_kind["decimal_comma"] and by_kind["null_tokens"] and by_kind["duplicates"]
        assert by_kind["sentinel"]
        assert not by_kind["whitespace_case"] and not by_kind["outliers"]

    def test_problems_raised_where_none_were_planted_are_false_alarms(self, setup) -> None:
        data, corrupted = setup
        noisy = diagnosis(
            {"column": "kunde_id", "kind": "duplicates", "suggested_op": "drop_duplicates"},
            {"column": "oprettet", "kind": "outliers", "suggested_op": None},
            {"column": "beloeb", "kind": "type", "suggested_op": "cast"},
        )
        score = score_case(data, corrupted, perfect_rows(data, corrupted), noisy)
        assert any("oprettet" in alarm for alarm in score.false_alarms)
        assert not any("beloeb" in alarm for alarm in score.false_alarms)


def test_an_outlier_counts_as_found_when_the_ai_quotes_a_planted_value(setup) -> None:
    """From the first real run: the AI called the fees 'impossible' without saying 'outlier'."""
    data, corrupted = setup
    planted = next(i for i in corrupted.injections if i.kind == "outliers")
    value = int(float(corrupted.table.loc[planted.rows[0], "gebyr"]))
    question = (
        f"The fee maximum of {value:,} DKK exceeds even the largest loan. "
        f"How should these fees be handled?"
    )
    found = diagnosis(calls=[{"column": "gebyr", "question": question}])
    score = score_case(data, corrupted, perfect_rows(data, corrupted), found)
    assert next(s for s in score.injections if s.kind == "outliers").found
    assert score.false_alarms == []
