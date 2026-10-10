"""The plain report's content, built once and drawn twice: REPORT.pdf and REPORT.html.

Both formats show exactly what is here, so they cannot disagree. Everything comes
from the run folder: the numbers from the logs, the traffic lights from the gates
and the decisions by fixed rules below, and the text the AI wrote, marked as such.

    Your question            from the brief
    The answer               the AI's summary, in a sentence or two
    What we found            at most four findings, the strongest first, each with its chart
    What you could do        at most three of the AI's recommendations
    How much can you trust   data, checks, decisions: green, yellow or red
    Keep in mind             the first three points of the wrap-up, from the logs first
    Want more?               the deeper documents and what each holds
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pipeline.layout import (
    ANALYSIS_PDF,
    AUDIT_PDF,
    DATA_COLUMNS_XLSX,
    DATA_QUALITY_PDF,
    DECISIONS_PDF,
    DETAILS,
    TECHNICAL_PDF,
)
from pipeline.reports.common import Run, chart_path, decision_rank
from pipeline.reports.technical import STRENGTH_ORDER, missing_columns, wrap_up_points

MAX_FINDINGS = 4
MAX_ACTIONS = 3
MAX_CAUTIONS = 3
ANSWER_SENTENCES = 2
MISSING_WORTH_A_LOOK = 0.10

Colour = Literal["green", "yellow", "red"]

STRENGTH_WORDS = {
    "strong": "Clear evidence",
    "moderate": "Fair evidence",
    "weak": "Weak evidence",
    "none": "No clear evidence",
}
GATE_NAMES = {1: "reading the file", 2: "cleaning", 3: "analysis plan", 4: "analysis numbers"}
STOPPED = {
    "ingestion": "Stopped: the file could not be read faithfully, so nothing was cleaned or "
    "analysed.",
    "diagnosis": "Stopped: the AI could not diagnose the data, so nothing was cleaned or analysed.",
    "cleaning": "Stopped: the cleaned data did not pass its checks, so there was nothing safe to "
    "analyse.",
    "planning": "Stopped: no analysis plan passed its check, so the analysis did not run.",
    "analysis": "Stopped: the analysis did not pass its number check, so it is not shown.",
}
DOCUMENTS = (
    (TECHNICAL_PDF, "The same run with the exact figures, for a technical reader"),
    (ANALYSIS_PDF, "Every finding with all its charts and the numbers behind them"),
    (DATA_QUALITY_PDF, "The data as received, what was wrong, and what was changed"),
    (DATA_COLUMNS_XLSX, "Every column of the data, in Excel"),
    (DECISIONS_PDF, "Every decision: the options, why, and how to change it"),
    (AUDIT_PDF, "Which AI models ran, what it cost, and every file"),
)


@dataclass
class KeyFinding:
    id: str
    sentence: str
    strength: str
    detail: str
    number: str | None = None
    caveats: list[str] = field(default_factory=list)
    charts: list[Path] = field(default_factory=list)

    @property
    def evidence(self) -> str:
        return STRENGTH_WORDS.get(self.strength, self.strength)


@dataclass
class Light:
    colour: Colour
    topic: str
    text: str
    where: str

    @property
    def word(self) -> str:
        return {"green": "Good", "yellow": "Check", "red": "Problem"}[self.colour]


@dataclass
class DecisionRow:
    id: str
    question: str
    decided: str
    by: str
    sure: str


@dataclass
class MainSummary:
    title: str
    file_name: str
    rows: int
    question: str | None
    answer: str | None
    stopped: str | None
    findings: list[KeyFinding]
    findings_not_shown: int
    actions: list[str]
    lights: list[Light]
    cautions: list[str]
    all_points: list[str]
    decisions: list[DecisionRow]
    documents: list[tuple[str, str]]


def build_summary(run: Run) -> MainSummary:
    brief = run.manifest.get("context_brief") or {}
    up = (run.analysis or {}).get("write_up") or {}
    findings = sorted(up.get("findings", []), key=lambda f: STRENGTH_ORDER.get(f["strength"], 9))
    points = wrap_up_points(run)
    doubts = [p.doubt for p in points if p.doubt]
    return MainSummary(
        title=brief.get("title") or f"What the data says: {_plain_name(run.file_name)}",
        file_name=run.file_name,
        rows=run.load.get("n_loaded", 0),
        question=brief.get("goal"),
        answer=_first_sentences(up["summary"]) if up.get("summary") else None,
        stopped=_stopped(run),
        findings=[_key_finding(run, f) for f in findings[:MAX_FINDINGS]],
        findings_not_shown=max(0, len(findings) - MAX_FINDINGS),
        actions=up.get("recommendations", [])[:MAX_ACTIONS],
        lights=[_data_light(run), _checks_light(run), _decisions_light(run)],
        cautions=doubts[:MAX_CAUTIONS],
        all_points=doubts,
        decisions=_decision_rows(run),
        documents=[(f"{DETAILS}/{name}", what) for name, what in DOCUMENTS],
    )


# -- the parts ------------------------------------------------------------------


def _plain_name(file_name: str) -> str:
    return re.sub(r"[_\-]+", " ", Path(file_name).stem).strip()


def _first_sentences(text: str, count: int = ANSWER_SENTENCES) -> str:
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return " ".join(sentences[:count])


def _stopped(run: Run) -> str | None:
    for stage, status in run.stages.items():
        if status == "blocked":
            return STOPPED.get(stage, f"Stopped at the {stage} step.")
    if not run.analysis:
        return "The analysis has not run yet."
    return None


def _key_finding(run: Run, finding: dict) -> KeyFinding:
    calls = {c["call_id"]: c for c in (run.analysis or {}).get("calls", [])}
    charts: list[Path] = []
    for call_id in finding.get("evidence", []):
        for artifact_id in calls.get(call_id, {}).get("artifact_ids", []):
            path = chart_path(run, run.artifacts.get(artifact_id, {}))
            if path is not None and path not in charts:
                charts.append(path)
    return KeyFinding(
        id=finding["id"],
        sentence=finding["headline"],
        strength=finding["strength"],
        detail=finding.get("detail", ""),
        caveats=finding.get("caveats", []),
        charts=charts,
    )


def _data_light(run: Run) -> Light:
    where = f"{DETAILS}/{DATA_QUALITY_PDF}"
    if run.stages.get("ingestion") == "blocked":
        return Light("red", "Data", "The file could not be read faithfully.", where)
    if run.stages.get("cleaning") == "blocked":
        return Light("red", "Data", "The cleaned data did not pass its checks.", where)
    rows = run.load.get("n_loaded", 0)
    rejected = run.load.get("n_rejected", 0)
    dropped = sum(r.get("rows_dropped", 0) for r in run.records)
    parts = [f"{rows:,} rows"]
    if rejected:
        parts.append(f"{rejected:,} line(s) could not be read")
    if dropped:
        parts.append(f"{dropped:,} removed while cleaning")
    missing = missing_columns(run)
    if not missing:
        parts.append("every value is present" if run.handoff else "not cleaned yet")
    elif len(missing) == 1:
        name, rate = missing[0]
        parts.append(f"{name} is missing for {rate:.1%} of rows")
    else:
        name, rate = missing[0]
        parts.append(f"{len(missing)} columns have gaps, most in {name} ({rate:.1%})")
    worth_a_look = rejected or dropped or any(r >= MISSING_WORTH_A_LOOK for _, r in missing)
    return Light("yellow" if worth_a_look else "green", "Data", "; ".join(parts) + ".", where)


def _checks_light(run: Run) -> Light:
    where = f"{DETAILS}/{TECHNICAL_PDF}"
    gates = {n: g for n, g in enumerate((run.gate_1, run.gate_2, run.gate_3, run.gate_4), 1) if g}
    blocked = [n for n, g in gates.items() if g.get("outcome") == "block"]
    flagged = [n for n, g in gates.items() if g.get("outcome") == "flag"]
    if blocked:
        names = ", ".join(GATE_NAMES[n] for n in blocked)
        return Light("red", "Checks", f"The automatic check on {names} stopped the run.", where)
    review = run.stages.get("review")
    open_review = review == "flagged"
    if flagged or open_review:
        parts = []
        if flagged:
            names = ", ".join(GATE_NAMES[n] for n in flagged)
            parts.append(f"the automatic check on {names} raised a warning")
        if open_review:
            parts.append("the reviewer left points open")
        text = "; ".join(parts)
        return Light("yellow", "Checks", text[0].upper() + text[1:] + ".", where)
    text = f"All {len(gates)} automatic checks passed"
    if review == "accepted":
        text += ", and the reviewer accepted the cleaning"
    return Light("green", "Checks", text + ".", where)


def _decisions_light(run: Run) -> Light:
    where = f"{DETAILS}/{DECISIONS_PDF}"
    calls = run.diagnosis.judgment_calls if run.diagnosis else []
    if not calls:
        return Light("green", "Decisions", "No judgment calls were needed.", where)
    open_ids = [c.id for c in calls if not _decided(run, c.id)]
    low_ids = [
        c.id
        for c in calls
        if _decided(run, c.id)
        and run.decisions[c.id].decided_by == "ai"
        and c.ai_decision is not None
        and c.ai_decision.confidence == "low"
    ]
    if open_ids:
        text = (
            f"{len(open_ids)} of {len(calls)} not decided ({', '.join(open_ids)}), so that data "
            f"was left as it was."
        )
        return Light("yellow", "Decisions", text, where)
    by_ai = sum(1 for c in calls if run.decisions[c.id].decided_by == "ai")
    if low_ids:
        text = (
            f"{len(calls)} made, {by_ai} by the AI; {len(low_ids)} with low confidence "
            f"({', '.join(low_ids)}): worth a check."
        )
        return Light("yellow", "Decisions", text, where)
    who = "by the AI with good confidence" if by_ai else "by you"
    return Light("green", "Decisions", f"{len(calls)} made, {who}.", where)


def _decided(run: Run, call_id: str) -> bool:
    decision = run.decisions.get(call_id)
    return decision is not None and decision.decided


def _decision_rows(run: Run) -> list[DecisionRow]:
    calls = run.diagnosis.judgment_calls if run.diagnosis else []
    rows = []
    for call in sorted(calls, key=lambda c: decision_rank(run, c)):
        decision = run.decisions.get(call.id)
        if decision is None or not decision.decided:
            rows.append(DecisionRow(call.id, call.question, "not decided", "nobody", "-"))
        elif decision.decided_by == "user":
            rows.append(DecisionRow(call.id, call.question, decision.choice, "you", "-"))
        else:
            sure = call.ai_decision.confidence if call.ai_decision else "-"
            rows.append(DecisionRow(call.id, call.question, decision.choice, "the AI", sure))
    return rows
