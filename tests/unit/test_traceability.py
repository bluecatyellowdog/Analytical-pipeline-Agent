"""The number check: every number in AI-written text must be one the tools logged.

Numbers are read with the precision they were written in, so a rounded value
still matches (74% matches a logged 0.742), while a number the AI worked out
itself (a difference, a ratio) matches nothing and is reported.
"""

from __future__ import annotations

import pytest

from pipeline.checks.traceability import numbers_in, untraced, values_of


class TestReading:
    def test_numbers_are_read_with_their_kind_and_precision(self) -> None:
        text = "Women 74.2% (n=314), odds x2.99, 3x more, p<0.001, 1,234 rows, effect -0.12."
        found = [(n.raw, n.kind, n.value, n.decimals) for n in numbers_in(text)]
        assert found == [
            ("74.2%", "percent", pytest.approx(0.742), 1),
            ("314", "number", 314.0, 0),
            ("x2.99", "multiplier", 2.99, 2),
            ("3x", "multiplier", 3.0, 0),
            ("0.001", "threshold", 0.001, 3),
            ("1,234", "number", 1234.0, 0),
            ("-0.12", "number", -0.12, 2),
        ]

    def test_ids_ordinals_and_names_hold_no_numbers(self) -> None:
        text = "q1 and T4 cite art_04d1265806f9 and F2; the 2nd row; R2 and COVID19."
        assert numbers_in(text) == []

    def test_ranges_and_dates_are_read_as_separate_numbers(self) -> None:
        assert [n.value for n in numbers_in("ages 18-29 from 2019-01-02")] == [18, 29, 2019, 1, 2]


LOGGED = {
    "rate": 0.74203,
    "n": 314,
    "odds_ratio": 2.9912,
    "effect": -0.1234,
    "group": "5,308-86,851",
    "per": "per 100,000",
    "nested": [{"p_value": 0.00002, "flag": True}],
}


class TestMatching:
    def test_numbers_inside_logged_labels_count(self) -> None:
        values = values_of(LOGGED)
        assert {5308, 86851, 100000} <= set(values)
        assert 1.0 not in values, "True is not a number"

    def test_rounded_numbers_match(self) -> None:
        text = (
            "74.2% of 314, about 74% overall, odds x2.99 (x3.0 rounded), an effect of -0.12, "
            "lower by 0.123, in the 5,308-86,851 group, per 100,000"
        )
        assert untraced(text, values_of(LOGGED)) == []

    def test_numbers_the_ai_worked_out_itself_are_reported(self) -> None:
        text = "a 55 point gap, 3.9 times higher, 74.5% and 315 people"
        assert untraced(text, values_of(LOGGED)) == ["55", "3.9", "74.5%", "315"]

    def test_small_counts_conventions_and_context_are_allowed(self) -> None:
        text = "The 3 classes, a 95% interval, p<0.05 and p<0.001, the voyage of 1912."
        assert untraced(text, [], allowed=[1912]) == []
        assert untraced(text, []) == ["1912"]

    def test_a_rate_written_without_a_percent_sign_still_matches(self) -> None:
        assert untraced("74.2 in every hundred", values_of({"rate": 0.742})) == []
