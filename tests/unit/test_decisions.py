"""decisions.yaml: the AI's choices pre-filled, the user's edits recognised as theirs.

In decide mode the analyst settles every judgment call; the file shows each
choice with who made it, how sure it was and why. Editing a decision makes it
the user's; writing `undecided` leaves that data alone. Nothing the user wrote
is ever overwritten.
"""

from __future__ import annotations

import yaml

from pipeline.agents.decisions import load_decisions, write_template
from pipeline.contracts.diagnosis import Diagnosis
from tests.unit.test_coder import DIAGNOSIS


def ai_decided() -> Diagnosis:
    data = DIAGNOSIS.model_dump(mode="json")
    data["judgment_calls"][0]["ai_decision"] = {
        "choice": "Treat 999 as missing",
        "confidence": "high",
        "reasoning": "The brief says 999 means unknown.",
        "expected_effect": "Three amounts become missing.",
        "least_destructive": False,
        "alternatives": [{"option": "Keep 999", "effect": "Amounts of 999 stay."}],
    }
    data["judgment_calls"][1]["ai_decision"] = {
        "choice": "Leave missing",
        "confidence": "low",
        "reasoning": "The brief does not say how to treat missing ages.",
        "expected_effect": "Nothing changes.",
        "least_destructive": True,
        "alternatives": [{"option": "Impute median", "effect": "Two ages filled in."}],
    }
    return Diagnosis.model_validate(data)


def entries(path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class TestTemplate:
    def test_ai_decisions_are_prefilled_with_who_and_why(self, tmp_path) -> None:
        path = tmp_path / "decisions.yaml"
        assert write_template(path, ai_decided())
        j1, j2 = entries(path)["J1"], entries(path)["J2"]
        assert j1["decision"] == "Treat 999 as missing"
        assert j1["ai_choice"] == "Treat 999 as missing"
        assert j1["confidence"] == "high" and "999 means unknown" in j1["why"]
        assert j2["confidence"] == "low"
        assert "change one" in path.read_text(encoding="utf-8")

    def test_without_ai_decisions_every_call_waits(self, tmp_path) -> None:
        path = tmp_path / "decisions.yaml"
        write_template(path, DIAGNOSIS)
        assert entries(path)["J1"]["decision"] == "undecided"
        assert "ai_choice" not in entries(path)["J1"]

    def test_an_untouched_old_template_is_replaced(self, tmp_path) -> None:
        path = tmp_path / "decisions.yaml"
        write_template(path, DIAGNOSIS)
        assert write_template(path, ai_decided())
        assert entries(path)["J1"]["ai_choice"] == "Treat 999 as missing"

    def test_an_answer_the_user_changed_is_never_overwritten(self, tmp_path) -> None:
        path = tmp_path / "decisions.yaml"
        write_template(path, ai_decided())
        text = path.read_text(encoding="utf-8").replace(
            "decision: Treat 999 as missing", "decision: Keep 999"
        )
        path.write_text(text, encoding="utf-8")
        write_template(path, ai_decided())
        assert entries(path)["J1"]["decision"] == "Keep 999"
        assert entries(path)["J2"]["decision"] == "Leave missing"


class TestLoading:
    def test_unchanged_choices_stay_the_ais(self, tmp_path) -> None:
        path = tmp_path / "decisions.yaml"
        write_template(path, ai_decided())
        decisions = {d.id: d for d in load_decisions(path, ai_decided())}
        assert decisions["J1"].decided and decisions["J1"].decided_by == "ai"
        assert decisions["J2"].decided_by == "ai"

    def test_an_edited_choice_becomes_the_users(self, tmp_path) -> None:
        path = tmp_path / "decisions.yaml"
        write_template(path, ai_decided())
        text = path.read_text(encoding="utf-8").replace(
            "decision: Treat 999 as missing", "decision: Keep 999"
        )
        path.write_text(text, encoding="utf-8")
        decisions = {d.id: d for d in load_decisions(path, ai_decided())}
        assert decisions["J1"].choice == "Keep 999"
        assert decisions["J1"].decided_by == "user"

    def test_undecided_leaves_the_data_alone(self, tmp_path) -> None:
        path = tmp_path / "decisions.yaml"
        write_template(path, ai_decided())
        text = path.read_text(encoding="utf-8").replace(
            "decision: Leave missing", "decision: undecided"
        )
        path.write_text(text, encoding="utf-8")
        decisions = {d.id: d for d in load_decisions(path, ai_decided())}
        assert not decisions["J2"].decided

    def test_without_a_file_the_ais_choices_apply(self, tmp_path) -> None:
        decisions = load_decisions(tmp_path / "missing.yaml", ai_decided())
        assert all(d.decided and d.decided_by == "ai" for d in decisions)
