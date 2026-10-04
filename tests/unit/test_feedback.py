"""Masking error messages before they go back to an LLM."""

from __future__ import annotations

from pipeline.agents.feedback import MAX_FEEDBACK_CHARS, mask_feedback


def test_a_quoted_data_value_becomes_its_shape() -> None:
    text = "ValueError: could not convert string to float: '12.345,50'"
    masked = mask_feedback(text)
    assert "12.345,50" not in masked
    assert "'99.999,99'" in masked


def test_column_names_and_ids_stay_readable() -> None:
    text = "column(s) ['alder'] not in table clean.t__v1 (have: ['kunde_id', 'alder'])"
    assert mask_feedback(text, keep=["alder", "kunde_id"]) == text


def test_what_the_model_wrote_itself_stays_readable() -> None:
    reply = '{"steps": [{"op": "magic_fix", "params": {}}]}'
    text = "unknown op 'magic_fix'; available: [...]"
    assert "'magic_fix'" in mask_feedback(text, known=reply)


def test_short_strings_do_not_count_as_the_models_own() -> None:
    assert mask_feedback("bad value 'ab'", known="ab cd") == "bad value 'aa'"


def test_long_unquoted_numbers_are_masked() -> None:
    masked = mask_feedback("row with 0101901234 failed (3 rows)")
    assert "0101901234" not in masked
    assert "9999999999" in masked
    assert "(3 rows)" in masked


def test_double_quotes_are_masked_too() -> None:
    assert 'string "Anna Hansen"' not in mask_feedback('Could not convert string "Anna Hansen"')


def test_long_feedback_is_cut() -> None:
    assert len(mask_feedback("x" * 5000)) == MAX_FEEDBACK_CHARS
