"""Type inference (spec 4.3).

These tests are mostly about what the pipeline must *refuse* to decide: an
ambiguous day/month order, a decimal separator with no evidence, an ID that
would survive a cast only by losing its leading zero.
"""

from __future__ import annotations

import datetime as dt

import pytest

from pipeline.ingestion.types import (
    NULL_TOKENS,
    detect_number_format,
    has_id_name,
    is_missing,
    parse_number,
    propose_type,
    sentinel_candidates,
)


def proposal(column: str, values, **kwargs):
    return propose_type(column, values, **kwargs)[0]


def flags_for(column: str, values, **kwargs):
    return {f.code for f in propose_type(column, values, **kwargs)[1]}


class TestMissingValues:
    @pytest.mark.parametrize("token", sorted(NULL_TOKENS))
    def test_null_tokens_count_as_missing(self, token: str) -> None:
        assert is_missing(token)

    def test_case_and_whitespace_do_not_matter(self) -> None:
        assert is_missing("  N/A  ") and is_missing("NULL") and is_missing(None)

    def test_a_real_value_is_not_missing(self) -> None:
        assert not is_missing("0") and not is_missing("false")


class TestNumberFormat:
    def test_danish_amount_sets_the_decimal_comma(self) -> None:
        fmt = detect_number_format(["1.234,50", "999", "7.800,00"])
        assert fmt.decimal == "," and fmt.thousands == "."

    def test_english_amount_sets_the_decimal_dot(self) -> None:
        fmt = detect_number_format(["1,234.50", "7,800.00"])
        assert fmt.decimal == "." and fmt.thousands == ","

    def test_a_lone_separator_with_three_digits_is_not_evidence(self) -> None:
        """'1.234' is either one thousand or a precise decimal; the data cannot say."""
        fmt = detect_number_format(["1.234", "5.678"])
        assert fmt.evidence == 0

    def test_two_digits_after_a_comma_is_evidence(self) -> None:
        fmt = detect_number_format(["12,5", "3,75"])
        assert fmt.decimal == "," and fmt.evidence == 2

    def test_parse_number_respects_the_detected_format(self) -> None:
        fmt = detect_number_format(["1.234,50"])
        assert parse_number("1.234,50", fmt) == 1234.50
        assert parse_number("999", fmt) == 999.0

    def test_currency_symbols_are_stripped(self) -> None:
        fmt = detect_number_format(["1.234,50 kr"])
        assert parse_number("1.234,50 kr", fmt) == 1234.50

    def test_unparseable_value_returns_none_rather_than_guessing(self) -> None:
        fmt = detect_number_format(["1,5"])
        assert parse_number("i alt", fmt) is None


class TestProposals:
    def test_danish_amounts_are_float_with_a_comma(self) -> None:
        p = proposal("beloeb", ["12.345,50", "999", "7.800,00", "1.234,50"])
        assert p.proposed == "float"
        assert p.coverage == 1.0
        assert p.details["decimal_separator"] == ","
        assert p.details["thousands_separator"] == "."

    def test_leading_zero_id_stays_text(self) -> None:
        """The '007' problem: casting would destroy the identifier."""
        p = proposal("kunde_id", ["00417", "00418", "00419"])
        assert p.proposed == "id"
        assert p.details["leading_zeros"] is True

    def test_plain_integers_named_id_stay_text(self) -> None:
        p = proposal("id", ["1", "2", "3"])
        assert p.proposed == "id"

    def test_a_non_unique_numeric_column_is_an_integer(self) -> None:
        p = proposal("alder", ["31", "54", "31", "23"])
        assert p.proposed == "integer"

    def test_unique_numbers_without_id_evidence_stay_numeric(self) -> None:
        p = proposal("antal", ["10", "20", "30"])
        assert p.proposed == "integer"

    def test_booleans_in_danish_and_english(self) -> None:
        assert proposal("aktiv", ["ja", "nej", "ja"]).proposed == "boolean"
        assert proposal("flag", ["true", "false"]).proposed == "boolean"
        assert proposal("valgt", ["1", "0", "1", "0"]).proposed == "boolean"

    def test_more_than_two_distinct_values_is_not_boolean(self) -> None:
        assert proposal("x", ["1", "0", "2"]).proposed != "boolean"

    def test_unambiguous_dates_get_a_format(self) -> None:
        p = proposal("dato", ["03-04-2021", "15-07-2021", "28-02-2020", "30-01-2019"])
        assert p.proposed == "date"
        assert p.details["date_format"] == "%d-%m-%Y"
        assert not p.ambiguous

    def test_ambiguous_day_month_order_is_flagged_not_guessed(self) -> None:
        """Spec 5.3: 03/04 is 3 April or 4 March and the data cannot settle it."""
        values = ["03/04/2021", "11/12/2021", "01/02/2022"]
        p = proposal("dato", values)
        assert p.proposed == "date"
        assert p.ambiguous
        assert set(p.alternatives) == {"%d/%m/%Y", "%m/%d/%Y"}
        assert "ambiguous_date_order" in flags_for("dato", values)

    def test_a_day_above_12_resolves_the_order(self) -> None:
        values = ["03/04/2021", "25/12/2021", "01/02/2022"]
        p = proposal("dato", values)
        assert not p.ambiguous
        assert p.details["date_format"] == "%d/%m/%Y"

    def test_mixed_date_formats_are_reported_with_coverage(self) -> None:
        values = ["03-04-2021", "15-07-2021", "28-02-2020", "2019-01-30", "17-06-2023"]
        p = proposal("dato", values)
        assert p.coverage == pytest.approx(0.8)
        assert "mixed_date_formats" in flags_for("dato", values)
        assert p.failing_examples == ["2019-01-30"]

    def test_compact_dates_are_not_mistaken_for_numbers(self) -> None:
        """'20210403' parses as an integer; the column name decides."""
        assert proposal("dato", ["20210403", "20211115", "20200228"]).proposed == "date"
        assert proposal("antal", ["20210403", "20211115", "20200228"]).proposed == "integer"

    def test_low_cardinality_text_is_categorical(self) -> None:
        p = proposal("region", ["nord", "syd", "nord", "syd", "nord", "syd", "nord", "syd"])
        assert p.proposed == "categorical"

    def test_high_cardinality_text_stays_text(self) -> None:
        p = proposal("kommentar", ["kunden ringede", "sagen lukket", "intet nyt"])
        assert p.proposed == "text"

    def test_an_all_missing_column_gets_no_type(self) -> None:
        p = proposal("tom", ["", "NA", None, "null"])
        assert p.proposed == "empty"
        assert p.coverage == 0.0

    def test_code_like_values_are_identifiers_only_after_real_types_fail(self) -> None:
        """'A1' is an identifier; '03-04-2021' is a date that merely looks code-like."""
        assert proposal("kode", ["A1", "B7", "C3", "D9"]).proposed == "id"
        assert proposal("navn", ["Alpha", "Beta", "Gamma"]).proposed == "text"

    def test_a_column_with_no_confident_type_is_ambiguous(self) -> None:
        values = ["1", "2", "3", "x", "y", "z", "1", "2", "3", "x"]
        p = proposal("blandet", values)
        assert p.proposed == "ambiguous"
        assert p.ambiguous
        assert p.failing_examples
        assert "ambiguous_type" in flags_for("blandet", values)

    def test_partial_coverage_above_the_threshold_is_still_flagged(self) -> None:
        """95% is good enough to propose, not good enough to cast silently."""
        values = [str(i) for i in range(39)] + ["ukendt"]
        p = proposal("alder", values)
        assert p.proposed == "integer"
        assert p.coverage == pytest.approx(0.975)
        assert p.failing_examples == ["ukendt"]
        assert "partial_type_coverage" in flags_for("alder", values)

    def test_missing_values_are_excluded_from_coverage(self) -> None:
        p = proposal("alder", ["31", "NA", "54", "", "23"])
        assert p.proposed == "integer"
        assert p.coverage == 1.0


class TestDeclaredTypes:
    """Parquet carries its own types; they are restated, not re-inferred."""

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            ("BOOLEAN", "boolean"),
            ("INTEGER", "integer"),
            ("BIGINT", "integer"),
            ("DECIMAL(8,2)", "float"),
            ("DOUBLE", "float"),
            ("DATE", "date"),
            ("TIMESTAMP", "datetime"),
        ],
    )
    def test_declared_types_are_restated(self, declared: str, expected: str) -> None:
        p = proposal("x", [1, 2], declared_type=declared)
        assert p.proposed == expected
        assert p.coverage == 1.0
        assert p.details["declared_type"] == declared

    def test_varchar_still_goes_through_the_heuristics(self) -> None:
        """A Parquet string column can still be an identifier worth protecting."""
        p = proposal("kunde_id", ["001", "002"], declared_type="VARCHAR")
        assert p.proposed == "id"

    def test_native_python_values_are_handled(self) -> None:
        p = proposal("dato", [dt.date(2021, 4, 3), dt.date(2021, 7, 15)], declared_type="DATE")
        assert p.proposed == "date"


class TestSentinels:
    def test_coded_missing_values_are_counted(self) -> None:
        counts = sentinel_candidates(["999", "1200", "999", "999", "-1"])
        assert counts == {"999": 3, "-1": 1}

    def test_sentinel_dates_are_counted(self) -> None:
        assert sentinel_candidates(["1900-01-01", "2021-04-03"]) == {"1900-01-01": 1}

    def test_missing_values_are_not_sentinels(self) -> None:
        assert sentinel_candidates(["NA", "", None]) == {}


class TestNameHints:
    @pytest.mark.parametrize("name", ["id", "kunde_id", "sagsnr", "kundenummer", "postkode"])
    def test_identifier_names(self, name: str) -> None:
        assert has_id_name(name)

    @pytest.mark.parametrize("name", ["alder", "beloeb", "antal", "navn"])
    def test_non_identifier_names(self, name: str) -> None:
        assert not has_id_name(name)
