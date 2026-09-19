"""Profiling and PII detection against the fixture set."""

from __future__ import annotations

import pytest
import yaml

from pipeline.ingestion.loaders import load
from pipeline.ingestion.pii import detect_pii, is_valid_cpr, mask_shape
from pipeline.ingestion.profiler import profile_column, profile_table
from pipeline.ingestion.raw import RawStore


@pytest.fixture(scope="session")
def expected(fixtures_dir) -> dict:
    return yaml.safe_load((fixtures_dir / "expected.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def store():
    with RawStore.open() as store:
        yield store


def profile_fixture(fixtures_dir, store, name: str):
    _, table = load(fixtures_dir / name, store)
    return profile_table(store, table)


class TestProfileAgainstExpectations:
    @pytest.mark.parametrize(
        "name",
        [
            "danish_semicolon.csv",
            "bom_excel_export.csv",
            "latin1_aeoeaa.csv",
            "ambiguous_dates.csv",
            "two_sheets.xlsx",
            "typed.parquet",
        ],
    )
    def test_proposed_types_match_expectations(self, fixtures_dir, store, expected, name) -> None:
        profile = profile_fixture(fixtures_dir, store, name)
        for column, spec in expected[name].get("types", {}).items():
            actual = profile.column(column)
            assert actual is not None, f"{column} missing from profile"
            assert actual.type_proposal.proposed == spec["proposed"], column
            assert actual.type_proposal.coverage == pytest.approx(spec["coverage"]), column
            if "ambiguous" in spec:
                assert actual.type_proposal.ambiguous == spec["ambiguous"], column
            for key, value in spec.get("details", {}).items():
                assert actual.type_proposal.details.get(key) == value, f"{column}.{key}"
            for key, value in spec.get("alternatives", {}).items():
                assert actual.type_proposal.alternatives.get(key) == pytest.approx(value), column

    @pytest.mark.parametrize(
        "name",
        ["danish_semicolon.csv", "bom_excel_export.csv", "cpr_numbers.csv", "typed.parquet"],
    )
    def test_column_counts_match_expectations(self, fixtures_dir, store, expected, name) -> None:
        profile = profile_fixture(fixtures_dir, store, name)
        for column, spec in expected[name].get("columns_detail", {}).items():
            actual = profile.column(column)
            for key, value in spec.items():
                assert getattr(actual, key) == value, f"{name}:{column}.{key}"

    def test_candidate_keys_match_expectations(self, fixtures_dir, store, expected) -> None:
        profile = profile_fixture(fixtures_dir, store, "danish_semicolon.csv")
        assert profile.candidate_keys == expected["danish_semicolon.csv"]["candidate_keys"]

    def test_sentinels_are_flagged_not_treated_as_missing(
        self, fixtures_dir, store, expected
    ) -> None:
        """999 stays a value; only the brief or the cleaning agent may reinterpret it."""
        profile = profile_fixture(fixtures_dir, store, "danish_semicolon.csv")
        beloeb = profile.column("beloeb")
        spec = expected["danish_semicolon.csv"]["sentinels"]["beloeb"][0]

        assert [(s.value, s.count) for s in beloeb.sentinel_candidates] == [
            (spec["value"], spec["count"])
        ]
        assert beloeb.n_missing == 0, "a sentinel is not a missing value"
        assert "sentinel_candidate" in {f.code for f in beloeb.flags}

    def test_ambiguous_dates_are_flagged_on_the_column(self, fixtures_dir, store) -> None:
        profile = profile_fixture(fixtures_dir, store, "ambiguous_dates.csv")
        assert "ambiguous_date_order" in {f.code for f in profile.column("dato").flags}


class TestColumnProfile:
    def test_null_tokens_are_counted_separately(self) -> None:
        column = profile_column("alder", ["31", "NA", "", "54", None], position=0)
        assert column.n_missing == 3
        assert column.null_tokens == {"NA": 1, "": 1, "<null>": 1}
        assert column.n_blank == 1

    def test_distinct_counts_exclude_missing_values(self) -> None:
        column = profile_column("x", ["a", "a", "b", "NA", ""], position=0)
        assert column.n_distinct == 2

    def test_numeric_summary_uses_the_detected_number_format(self) -> None:
        """Without the decimal comma, 1.234,50 would be read as 1.23450."""
        column = profile_column("beloeb", ["1.234,50", "2.000,00", "3.500,25"], position=0)
        assert column.numeric is not None
        assert column.numeric.min == 1234.50
        assert column.numeric.max == 3500.25
        assert column.numeric.mean == pytest.approx(2244.9166, abs=1e-3)

    def test_far_outliers_are_counted_and_flagged(self) -> None:
        values = [str(v) for v in [10, 11, 12, 13, 14, 15, 16, 17, 18, 5000]]
        column = profile_column("x", values, position=0)
        assert column.numeric.n_outliers == 1
        assert "far_outliers" in {f.code for f in column.flags}

    def test_quantiles_handle_a_single_value(self) -> None:
        column = profile_column("x", ["42"], position=0)
        assert column.numeric.median == 42.0
        assert column.numeric.std is None

    def test_text_columns_get_a_length_summary(self) -> None:
        column = profile_column("navn", ["ab", "abcd"], position=0)
        assert column.length.min == 2 and column.length.max == 4

    def test_high_missing_share_is_flagged(self) -> None:
        column = profile_column("x", ["1", "NA", "NA", "NA"], position=0)
        assert "high_missing_share" in {f.code for f in column.flags}

    def test_an_all_missing_column_does_not_crash(self) -> None:
        column = profile_column("tom", ["", "NA", None], position=0)
        assert column.n_distinct == 0
        assert column.type_proposal.proposed == "empty"
        assert column.top_values == []


class TestDuplicatesAndKeys:
    def test_duplicate_rows_are_counted(self, store) -> None:
        table = store.create_text_table(
            "t", ["a", "b"], [["1", "x"], ["1", "x"], ["2", "y"]], [1, 2, 3]
        )
        assert profile_table(store, table).n_duplicate_rows == 1

    def test_candidate_key_needs_uniqueness_and_no_missing(self, store) -> None:
        table = store.create_text_table(
            "t", ["unique_col", "has_null"], [["1", "a"], ["2", None], ["3", "c"]], [1, 2, 3]
        )
        profile = profile_table(store, table)
        assert profile.candidate_keys == ["unique_col"]

    def test_no_candidate_key_is_reported(self, store) -> None:
        table = store.create_text_table("t", ["a"], [["1"], ["1"]], [1, 2])
        assert "no_candidate_key" in {f.code for f in profile_table(store, table).flags}


class TestPII:
    def test_emails_are_detected_and_masked_in_the_profile(self, fixtures_dir, store) -> None:
        profile = profile_fixture(fixtures_dir, store, "danish_semicolon.csv")
        email = profile.column("email")
        assert email.pii_level == "likely"
        assert [f.kind for f in email.pii][0] == "email"
        assert all("@example.dk" not in v.value for v in email.top_values), (
            "PII reached the profile"
        )

    def test_cpr_matches_expectations(self, fixtures_dir, store, expected) -> None:
        profile = profile_fixture(fixtures_dir, store, "cpr_numbers.csv")
        spec = expected["cpr_numbers.csv"]["pii"]["cpr"]
        finding = next(f for f in profile.column("cpr").pii if f.kind == "cpr")
        assert finding.level == spec["level"]
        assert finding.n_matches == spec["n_matches"]
        assert finding.matched_column_name == spec["matched_column_name"]

    @pytest.mark.parametrize("value", ["010190-1234", "311299-5678", "290204-1111", "150385-9999"])
    def test_valid_cpr_numbers(self, value: str) -> None:
        assert is_valid_cpr(value)

    @pytest.mark.parametrize("value", ["320199-1234", "001390-4321", "123456-7890", "abc"])
    def test_cpr_lookalikes_are_rejected(self, value: str) -> None:
        """The first six digits must be a real date, or every 10-digit code matches."""
        assert not is_valid_cpr(value)

    def test_leap_day_is_valid_in_a_leap_year(self) -> None:
        assert is_valid_cpr("290204-1111")

    def test_amount_columns_are_not_mistaken_for_phone_numbers(self) -> None:
        """An 8-digit number is only a phone number when something else says so."""
        assert detect_pii("beloeb", ["12345678", "87654321"]) == []
        assert detect_pii("telefon", ["12345678", "87654321"])

    def test_a_formatted_phone_number_is_caught_without_a_name_hint(self) -> None:
        findings = detect_pii("kontakt", ["+45 12 34 56 78", "+45 87 65 43 21"])
        assert [f.kind for f in findings] == ["phone"]

    def test_column_name_alone_raises_free_text_names(self) -> None:
        findings = detect_pii("kunde_navn", ["Søren Østergård", "Åse Bæk"])
        assert [f.kind for f in findings] == ["name"]
        assert findings[0].level == "possible"

    def test_masking_keeps_shape_and_drops_content(self) -> None:
        assert mask_shape("anna.hansen@example.dk") == "aaaa.aaaaaa@aaaaaaa.aa"
        assert mask_shape("010190-1234") == "999999-9999"
        assert mask_shape("Søren") == "Aaaaa"

    def test_a_clean_column_has_no_findings(self) -> None:
        assert detect_pii("alder", ["31", "54", "23"]) == []
