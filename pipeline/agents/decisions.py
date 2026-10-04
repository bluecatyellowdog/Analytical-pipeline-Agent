"""The user's decisions on judgment calls, as a file they edit in the run folder.

The analyst writes `decisions.yaml` with every judgment call set to `undecided`.
The user replaces that with a choice (one of the options, or their own words).
The coder implements decided calls and leaves undecided ones untouched, so a
judgment call is never settled by an LLM on its own.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from pipeline.contracts.cleaning import UNDECIDED, Decision
from pipeline.contracts.diagnosis import Diagnosis

DECISIONS_FILE = "decisions.yaml"

HEADER = """\
# Your decisions on the analyst's judgment calls.
#
# For each one, replace `undecided` with your choice: copy one of the options,
# or write your own in plain words. Calls left `undecided` are not touched by
# the cleaning; they stay flagged in the report.
#
# Then run:  uv run python -m pipeline.clean <this run folder>

"""


class DecisionsError(ValueError):
    """The decisions file does not fit the diagnosis it answers."""


def write_template(path: Path, diagnosis: Diagnosis) -> bool:
    """Write the file for the user to fill in. Never overwrites their answers."""
    if path.exists() or not diagnosis.judgment_calls:
        return False
    body = {
        call.id: {
            "column": call.column,
            "question": call.question,
            "options": list(call.options),
            "recommendation": call.recommendation,
            "decision": UNDECIDED,
        }
        for call in diagnosis.judgment_calls
    }
    text = yaml.safe_dump(body, allow_unicode=True, sort_keys=False, width=100)
    path.write_text(HEADER + text, encoding="utf-8")
    return True


def load_decisions(path: Path, diagnosis: Diagnosis) -> list[Decision]:
    """Every judgment call with the user's choice, `undecided` where there is none."""
    known = {call.id for call in diagnosis.judgment_calls}
    raw = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise DecisionsError(f"{path.name} should map judgment-call ids to entries")

    unknown = sorted(str(k) for k in raw if str(k) not in known)
    if unknown:
        raise DecisionsError(
            f"{path.name} answers {unknown}, which are not judgment calls in the diagnosis "
            f"(known: {sorted(known)})"
        )

    decisions = []
    for call in diagnosis.judgment_calls:
        entry = raw.get(call.id)
        choice = entry.get("decision") if isinstance(entry, dict) else entry
        text = str(choice).strip() if choice is not None else ""
        decisions.append(Decision(id=call.id, choice=text or UNDECIDED))
    return decisions
