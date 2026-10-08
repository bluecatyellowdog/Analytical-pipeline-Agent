"""The decisions on judgment calls, as a file the user can read and change in the run folder.

When the brief says `judgment_calls: decide` (the default), the diagnostic agent settles
each call from the brief and the file is pre-filled with its choices: `ai_choice`
records what the AI chose, `decision` is what will be carried out. Changing a
`decision` makes it the user's; writing `undecided` leaves that data untouched.
With `judgment_calls: ask`, every call starts as `undecided` and waits for the user.

The file is the user's: re-running the diagnosis refreshes the AI's choices but
never an answer the user changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pipeline.contracts.cleaning import UNDECIDED, Decision
from pipeline.contracts.diagnosis import Diagnosis, JudgmentCall

DECISIONS_FILE = "decisions.yaml"

HEADER_ASK = """\
# Your decisions on the judgment calls from the diagnosis.
#
# For each one, replace `undecided` with your choice: copy one of the options,
# or write your own in plain words. Calls left `undecided` are not touched by
# the cleaning; they stay flagged in the report.
#
# Then run:  uv run python -m pipeline.clean <this run folder>

"""

HEADER_DECIDE = """\
# Decisions on the judgment calls from the diagnosis.
#
# The AI decided each call from your brief (judgment_calls: decide). `ai_choice`
# is what it chose; `decision` is what the cleaning carries out. Its reasons, how
# sure it was and the alternatives are in cleaning_report.md.
#
# To change one, replace its `decision:` with another option or your own words,
# or with `undecided` to leave that data untouched. Then run:
#   uv run python -m pipeline.clean <this run folder>

"""


class DecisionsError(ValueError):
    """The decisions file does not fit the diagnosis it answers."""


def write_template(path: Path, diagnosis: Diagnosis) -> bool:
    """Write the file, keeping every answer the user changed. True when it was written."""
    if not diagnosis.judgment_calls:
        return False
    changed = _user_answers(path)
    body = {call.id: _entry(call, changed.get(call.id)) for call in diagnosis.judgment_calls}
    any_ai = any(call.ai_decision for call in diagnosis.judgment_calls)
    text = yaml.safe_dump(body, allow_unicode=True, sort_keys=False, width=100)
    path.write_text((HEADER_DECIDE if any_ai else HEADER_ASK) + text, encoding="utf-8")
    return True


def load_decisions(path: Path, diagnosis: Diagnosis) -> list[Decision]:
    """Every judgment call with its decision and who made it."""
    known = {call.id for call in diagnosis.judgment_calls}
    raw = _read(path)
    unknown = sorted(str(k) for k in raw if str(k) not in known)
    if unknown:
        raise DecisionsError(
            f"{path.name} answers {unknown}, which are not judgment calls in the diagnosis "
            f"(known: {sorted(known)})"
        )

    decisions = []
    for call in diagnosis.judgment_calls:
        ai = call.ai_decision
        if call.id not in raw:
            choice = ai.choice if ai else UNDECIDED
            decisions.append(Decision(id=call.id, choice=choice, decided_by="ai" if ai else "user"))
            continue
        text = _decision_text(raw[call.id])
        by = "ai" if ai is not None and text == ai.choice else "user"
        decisions.append(Decision(id=call.id, choice=text, decided_by=by))
    return decisions


def _entry(call: JudgmentCall, user_answer: str | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "column": call.column,
        "question": call.question,
        "options": list(call.options),
    }
    ai = call.ai_decision
    if ai is not None:
        entry["ai_choice"] = ai.choice
        entry["confidence"] = ai.confidence
        entry["why"] = ai.reasoning
        default = ai.choice
    else:
        entry["recommendation"] = call.recommendation
        default = UNDECIDED
    entry["decision"] = user_answer if user_answer is not None else default
    return entry


def _user_answers(path: Path) -> dict[str, str]:
    """Answers the user changed in an existing file: anything but the AI's choice or undecided."""
    answers = {}
    for call_id, entry in _read(path).items():
        text = _decision_text(entry)
        ai_choice = entry.get("ai_choice") if isinstance(entry, dict) else None
        if text != (ai_choice if ai_choice is not None else UNDECIDED):
            answers[str(call_id)] = text
    return answers


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise DecisionsError(f"{path.name} should map judgment-call ids to entries")
    return {str(k): v for k, v in raw.items()}


def _decision_text(entry: Any) -> str:
    choice = entry.get("decision") if isinstance(entry, dict) else entry
    text = str(choice).strip() if choice is not None else ""
    return text or UNDECIDED
