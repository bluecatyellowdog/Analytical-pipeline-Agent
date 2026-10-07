"""The benchmark's clean base data and its breakers.

The scores are only as trustworthy as the answer key, so these tests pin down
that the base data is clean, that each breaker changes exactly the cells its
answer key lists and nothing else, and that everything is reproducible.
"""

from __future__ import annotations

import datetime as dt

import pytest

from benchmark.datasets.synthetic import make_loans
from benchmark.injectors import Corrupted, inject


@pytest.fixture(scope="module")
def loans():
    return make_loans(rows=200, seed=1)


def fresh(loans) -> Corrupted:
    return Corrupted.from_clean(loans)


class TestBaseData:
    def test_it_is_reproducible(self) -> None:
        assert make_loans(rows=50, seed=3).frame.equals(make_loans(rows=50, seed=3).frame)

    def test_it_is_clean(self, loans) -> None:
        frame = loans.frame
        assert frame["kunde_id"].is_unique
        assert frame["kunde_id"].str.fullmatch(r"\d{5}").all()
        assert not frame.isna().any().any()
        assert frame["alder"].between(18, 80).all()
        assert set(frame["misligholdt"]) == {0, 1}
        assert isinstance(frame["oprettet"].iloc[0], dt.date)
        assert loans.kinds["beloeb"] == "float" and loans.target == "misligholdt"

    def test_danish_letters_are_in_the_categories(self, loans) -> None:
        assert any("æ" in r or "ø" in r for r in loans.frame["region"].unique())

    def test_its_text_form_is_plain(self, loans) -> None:
        text = fresh(loans).table
        assert text.loc[0, "beloeb"].count(".") == 1
        assert "," not in "".join(text["beloeb"])


class TestBreakers:
    def changed_cells(self, before, after) -> set[tuple[int, str]]:
        return {
            (row, column)
            for column in before.columns
            for row in range(len(before))
            if before.loc[row, column] != after.loc[row, column]
        }

    @pytest.mark.parametrize(
        ("kind", "params"),
        [
            ("null_tokens", {"column": "alder", "share": 0.1}),
            ("decimal_comma", {"column": "beloeb"}),
            ("date_format", {"column": "oprettet", "fmt": "%d-%m-%Y"}),
            ("sentinel", {"column": "indkomst", "code": "999", "share": 0.05}),
            ("whitespace_case", {"column": "region", "share": 0.2}),
            ("currency_suffix", {"column": "gebyr"}),
            ("outliers", {"column": "indkomst", "share": 0.02}),
        ],
    )
    def test_a_breaker_changes_exactly_what_its_answer_key_says(self, loans, kind, params):
        corrupted = fresh(loans)
        before = corrupted.table.copy()
        inject(corrupted, loans, kind, seed=7, **params)

        injection = corrupted.injections[-1]
        assert injection.kind == kind and injection.column == params["column"]
        assert self.changed_cells(before, corrupted.table) == {
            (row, params["column"]) for row in injection.rows
        }

    def test_answers_say_what_the_clean_value_should_be(self, loans) -> None:
        corrupted = fresh(loans)
        inject(corrupted, loans, "null_tokens", seed=1, column="alder", share=0.1)
        inject(corrupted, loans, "decimal_comma", seed=1, column="beloeb")
        inject(corrupted, loans, "outliers", seed=1, column="indkomst", share=0.02)
        expected = {i.kind: i.expected for i in corrupted.injections}
        assert expected == {
            "null_tokens": "missing",
            "decimal_comma": "original",
            "outliers": "flag",
        }

    def test_danish_decimals_look_danish(self, loans) -> None:
        corrupted = fresh(loans)
        inject(corrupted, loans, "decimal_comma", seed=1, column="beloeb")
        value = loans.frame.loc[0, "beloeb"]
        whole, cents = f"{value:.2f}".split(".")
        assert corrupted.table.loc[0, "beloeb"] == f"{int(whole):,}".replace(",", ".") + "," + cents

    def test_duplicates_are_appended_copies_of_existing_rows(self, loans) -> None:
        corrupted = fresh(loans)
        inject(corrupted, loans, "duplicates", seed=1, share=0.05)
        injection = corrupted.injections[-1]
        assert injection.expected == "removed" and injection.column is None
        assert len(corrupted.table) == 200 + len(injection.rows)
        for row in injection.rows:
            source = corrupted.origin[row]
            assert corrupted.table.loc[row].tolist() == corrupted.table.loc[source].tolist()

    def test_a_leaky_column_copies_the_target_almost_exactly(self, loans) -> None:
        corrupted = fresh(loans)
        inject(corrupted, loans, "leaky_column", seed=1, name="sag_lukket")
        leak = corrupted.table["sag_lukket"].astype(int)
        target = loans.frame["misligholdt"]
        assert (leak == target).mean() > 0.9
        assert corrupted.injections[-1].expected == "flag"

    def test_a_documented_code_adds_a_note_for_the_brief(self, loans) -> None:
        corrupted = fresh(loans)
        inject(corrupted, loans, "sentinel", seed=1, column="indkomst", code="999", share=0.05)
        assert any("999" in note["known_issues"] for note in corrupted.column_notes)

    def test_breakers_never_hit_the_same_cell_twice(self, loans) -> None:
        corrupted = fresh(loans)
        inject(corrupted, loans, "null_tokens", seed=1, column="indkomst", share=0.3)
        inject(corrupted, loans, "sentinel", seed=1, column="indkomst", code="999", share=0.3)
        first, second = (set(i.rows) for i in corrupted.injections)
        assert not first & second

    def test_the_same_seed_breaks_the_same_cells(self, loans) -> None:
        a, b = fresh(loans), fresh(loans)
        for corrupted in (a, b):
            inject(corrupted, loans, "null_tokens", seed=5, column="alder", share=0.1)
        assert a.table.equals(b.table)


def test_duplicates_must_come_last(loans) -> None:
    corrupted = fresh(loans)
    inject(corrupted, loans, "duplicates", seed=1, share=0.05)
    with pytest.raises(ValueError, match="last"):
        inject(corrupted, loans, "null_tokens", seed=1, column="alder", share=0.1)
