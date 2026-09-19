"""Loading the fixture set from spec 4.5.

Expectations live in tests/fixtures/expected.yaml, so this file checks
behaviour against a written specification rather than against itself.
"""

from __future__ import annotations

import pytest
import yaml

from pipeline.ingestion.detect import detect, detect_delimiter, detect_encoding
from pipeline.ingestion.loaders import load, normalize_headers, parse_csv
from pipeline.ingestion.raw import RawStore

FIXTURES = [
    "danish_semicolon.csv",
    "bom_excel_export.csv",
    "latin1_aeoeaa.csv",
    "quoted_newline.csv",
    "duplicate_blank_headers.csv",
    "too_many_fields.csv",
    "ambiguous_dates.csv",
    "cpr_numbers.csv",
    "two_sheets.xlsx",
    "typed.parquet",
]


@pytest.fixture(scope="session")
def expected(fixtures_dir) -> dict:
    return yaml.safe_load((fixtures_dir / "expected.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def store() -> RawStore:
    with RawStore.open() as store:
        yield store


def load_fixture(fixtures_dir, store: RawStore, name: str):
    return load(fixtures_dir / name, store)


class TestDetection:
    @pytest.mark.parametrize("name", FIXTURES)
    def test_detection_matches_expectations(self, fixtures_dir, expected, name) -> None:
        spec = expected[name]
        info = detect(fixtures_dir / name)
        assert info.format == spec["format"]
        if "encoding" in spec:
            assert info.encoding == spec["encoding"]
        if "delimiter" in spec:
            assert info.delimiter == spec["delimiter"]
        if "sheets" in spec:
            assert info.available_sheets == spec["sheets"]
            assert info.sheet == spec["default_sheet"]
        assert len(info.file_hash) == 64
        assert info.detection_notes, "every detection decision must be recorded"

    def test_bom_is_not_part_of_the_first_column_name(self, fixtures_dir, store) -> None:
        """The classic Excel export bug: a first column called '\\ufeffid'."""
        result, _ = load_fixture(fixtures_dir, store, "bom_excel_export.csv")
        assert result.columns[0] == "id"

    def test_utf8_is_preferred_over_the_cp1252_fallback(self) -> None:
        encoding, notes, flags = detect_encoding("navn;by\nSøren;København\n".encode())
        assert encoding == "utf-8"
        assert not flags and notes

    def test_non_utf8_falls_back_and_is_flagged(self) -> None:
        """cp1252 decodes anything, so choosing it is a guess and must be visible."""
        encoding, _, flags = detect_encoding("Søren;København\n".encode("cp1252"))
        assert encoding == "cp1252"
        assert [f.code for f in flags] == ["encoding_guessed"]

    def test_delimiter_prefers_consistency_over_field_count(self) -> None:
        """Danish amounts contain commas; only ';' splits every line the same way."""
        text = "a;b;c\n1.234,50;x;y\n2.000,00;z;w\n"
        delimiter, notes, _ = detect_delimiter(text)
        assert delimiter == ";"
        assert notes

    def test_single_column_file_is_flagged(self) -> None:
        delimiter, _, flags = detect_delimiter("just_one_column\nvalue\nother\n")
        assert delimiter == ","
        assert [f.code for f in flags] == ["delimiter_not_detected"]

    def test_unsupported_extension_is_an_error(self, tmp_path) -> None:
        path = tmp_path / "data.docx"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="unsupported file type"):
            detect(path)


class TestLoading:
    @pytest.mark.parametrize("name", FIXTURES)
    def test_row_accounting_matches_expectations(self, fixtures_dir, store, expected, name):
        """Spec Gate 1: loaded + rejected = rows in source, on every fixture."""
        spec = expected[name]
        result, _ = load_fixture(fixtures_dir, store, name)
        assert result.n_loaded == spec["n_loaded"]
        assert result.n_rejected == spec["n_rejected"]
        assert result.n_source_rows == spec["n_source_rows"]
        assert result.n_loaded + result.n_rejected == result.n_source_rows

    @pytest.mark.parametrize("name", FIXTURES)
    def test_columns_match_expectations(self, fixtures_dir, store, expected, name) -> None:
        result, _ = load_fixture(fixtures_dir, store, name)
        assert result.columns == expected[name]["columns"]

    @pytest.mark.parametrize("name", FIXTURES)
    def test_stored_rows_match_the_load_result(self, fixtures_dir, store, expected, name) -> None:
        result, table = load_fixture(fixtures_dir, store, name)
        assert store.row_count(table.name) == result.n_loaded
        assert store.columns_of(table.name) == result.columns

    def test_danish_export_keeps_every_value_literal(self, fixtures_dir, store) -> None:
        """Nothing is converted at load: leading zeros, decimal commas and NA survive."""
        _, table = load_fixture(fixtures_dir, store, "danish_semicolon.csv")
        assert store.column_values(table.name, "kunde_id")[:3] == ["00417", "00418", "00419"]
        assert store.column_values(table.name, "beloeb")[:2] == ["12.345,50", "999"]
        assert store.column_values(table.name, "alder")[1] == "NA"

    def test_latin1_values_round_trip(self, fixtures_dir, store, expected) -> None:
        _, table = load_fixture(fixtures_dir, store, "latin1_aeoeaa.csv")
        spec = expected["latin1_aeoeaa.csv"]["values"]
        assert store.column_values(table.name, "navn") == spec["navn"]
        assert store.column_values(table.name, "by") == spec["by"]

    def test_quoted_newline_is_one_record(self, fixtures_dir, store, expected) -> None:
        """Rows follow records, not physical lines."""
        result, table = load_fixture(fixtures_dir, store, "quoted_newline.csv")
        spec = expected["quoted_newline.csv"]
        assert result.n_loaded == 3
        assert store.column_values(table.name, "kommentar") == spec["values"]["kommentar"]

    def test_source_lines_point_at_the_start_of_each_record(
        self, fixtures_dir, store, expected
    ) -> None:
        """A reject is useless if it cannot be found in the file."""
        _, table = load_fixture(fixtures_dir, store, "quoted_newline.csv")
        stored = store.connection.execute(
            f"SELECT _source_line FROM {table.name} ORDER BY _row_id"
        ).fetchall()
        assert [row[0] for row in stored] == expected["quoted_newline.csv"]["source_lines"]

    def test_duplicate_and_blank_headers_are_renamed_and_reported(
        self, fixtures_dir, store, expected
    ) -> None:
        result, _ = load_fixture(fixtures_dir, store, "duplicate_blank_headers.csv")
        spec = expected["duplicate_blank_headers.csv"]
        assert result.columns == spec["columns"]
        assert [
            {"position": r.position, "original": r.original, "final": r.final}
            for r in result.renames
        ] == spec["renames"]
        assert all(r.reason for r in result.renames)
        assert "headers_renamed" in {f.code for f in result.flags}

    def test_malformed_lines_are_rejected_with_a_reason(
        self, fixtures_dir, store, expected
    ) -> None:
        """Spec 4.1: unreadable lines go to the rejects table, they are not lost."""
        result, table = load_fixture(fixtures_dir, store, "too_many_fields.csv")
        spec = expected["too_many_fields.csv"]["rejects"]
        assert [{"source_line": r.source_line, "reason": r.reason} for r in result.rejects] == spec
        assert all(r.raw for r in result.rejects)

        stored = store.connection.execute(f"SELECT * FROM {table.rejects_table}").fetchall()
        assert len(stored) == len(spec)

    def test_excel_text_ids_are_not_turned_into_numbers(
        self, fixtures_dir, store, expected
    ) -> None:
        _, table = load_fixture(fixtures_dir, store, "two_sheets.xlsx")
        expected_ids = expected["two_sheets.xlsx"]["values"]["kunde_id"]
        assert store.column_values(table.name, "kunde_id") == expected_ids

    def test_excel_second_sheet_can_be_requested(self, fixtures_dir, store) -> None:
        result, _ = load(fixtures_dir / "two_sheets.xlsx", store, sheet="konti")
        assert result.columns == ["konto_id", "kunde_id", "type"]
        assert result.n_loaded == 2

    def test_multiple_sheets_are_flagged_when_none_was_chosen(self, fixtures_dir, store) -> None:
        result, _ = load_fixture(fixtures_dir, store, "two_sheets.xlsx")
        assert "multiple_sheets" in {f.code for f in result.flags}

    def test_parquet_keeps_its_own_types(self, fixtures_dir, store, expected) -> None:
        """Spec 4.2: Parquet is the documented exception to loading as text."""
        _, table = load_fixture(fixtures_dir, store, "typed.parquet")
        assert table.typed
        assert table.column_types["kunde_id"] == "VARCHAR"
        assert table.column_types["alder"] == "INTEGER"
        assert table.column_types["beloeb"].startswith("DECIMAL")
        assert table.column_types["dato"] == "DATE"
        assert table.column_types["aktiv"] == "BOOLEAN"
        assert (
            store.column_values(table.name, "kunde_id")
            == (expected["typed.parquet"]["values"]["kunde_id"])
        )
        assert store.column_values(table.name, "alder") == [31, 54, 23, None]


class TestRawLayer:
    def test_data_hash_is_stable_across_loads(self, fixtures_dir) -> None:
        """Same bytes in, same hash out: provenance depends on it."""
        with RawStore.open() as first, RawStore.open() as second:
            a, _ = load_fixture(fixtures_dir, first, "danish_semicolon.csv")
            b, _ = load_fixture(fixtures_dir, second, "danish_semicolon.csv")
        assert a.data_hash == b.data_hash

    def test_data_hash_changes_when_a_value_changes(self, fixtures_dir, store) -> None:
        _, table = load_fixture(fixtures_dir, store, "danish_semicolon.csv")
        store.connection.execute(f"UPDATE {table.name} SET alder = '32' WHERE _row_id = 1")
        assert store.data_hash(table.name, table.columns) != table.data_hash

    def test_verify_catches_a_modified_raw_table(self, fixtures_dir, store) -> None:
        """The raw layer is immutable; if that stops being true, nothing else holds."""
        _, table = load_fixture(fixtures_dir, store, "danish_semicolon.csv")
        store.verify(table)
        store.connection.execute(f"DELETE FROM {table.name} WHERE _row_id = 1")
        with pytest.raises(RuntimeError, match="changed since it was loaded"):
            store.verify(table)

    def test_reserved_column_names_are_rejected(self, store) -> None:
        with pytest.raises(ValueError, match="reserved"):
            store.create_text_table("t", ["_row_id"], [["1"]], [1])

    def test_column_hash_ignores_other_columns(self, fixtures_dir, store) -> None:
        """The untouched-column check depends on this being true."""
        _, table = load_fixture(fixtures_dir, store, "danish_semicolon.csv")
        before = store.column_hash(table.name, "kunde_id")
        store.connection.execute(f"UPDATE {table.name} SET alder = '99' WHERE _row_id = 1")
        assert store.column_hash(table.name, "kunde_id") == before


class TestHeaderNormalization:
    def test_blank_headers_use_their_position(self) -> None:
        names, renames = normalize_headers(["id", "", ""])
        assert names == ["id", "column_2", "column_3"]
        assert [r.position for r in renames] == [2, 3]

    def test_duplicates_are_suffixed_in_order(self) -> None:
        names, _ = normalize_headers(["navn", "navn", "navn"])
        assert names == ["navn", "navn_2", "navn_3"]

    def test_a_suffix_collision_is_resolved(self) -> None:
        """'navn_2' already exists, so the duplicate cannot silently overwrite it."""
        names, _ = normalize_headers(["navn", "navn_2", "navn"])
        assert names == ["navn", "navn_2", "navn_3"]
        assert len(set(names)) == 3

    def test_reserved_names_are_moved_aside(self) -> None:
        names, renames = normalize_headers(["_row_id", "x"])
        assert names == ["_row_id_source", "x"]
        assert "reserved" in renames[0].reason

    def test_whitespace_is_stripped(self) -> None:
        names, _ = normalize_headers(["  id  ", "\tnavn\n"])
        assert names == ["id", "navn"]


class TestCsvParsing:
    def test_empty_file_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            parse_csv("", ",")

    def test_blank_lines_are_skipped_not_counted(self) -> None:
        parsed = parse_csv("a,b\n1,2\n\n3,4\n", ",")
        assert parsed.n_source_rows == 2
        assert "Skipped 1 blank line(s)." in parsed.notes

    def test_empty_field_is_kept_as_empty_text(self) -> None:
        """An absent value and a blank value are different things; both survive."""
        parsed = parse_csv("a,b\n1,\n", ",")
        assert parsed.rows == [["1", ""]]

    def test_empty_columns_are_reported(self) -> None:
        parsed = parse_csv("a,b\n1,\n2,\n", ",")
        assert parsed.empty_columns() == ["b"]
