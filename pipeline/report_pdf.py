"""The run report as one PDF: a summary on the first page, then everything that happened.

    uv run python -m pipeline.report_pdf runs/<run folder>

Built only from the files in the run folder, so it can be rebuilt for any run
and says nothing the logs do not. Sections follow the workflow: the data as
received (and what was dirty about it), what was wrong (issues and judgment
calls), what was changed (every step, with the agent's own code in full), the
checks on the clean data, the review, and an audit appendix. Text written by
an AI is marked as such; every number comes from a log.

A PDF gets shared, so it holds no personal data: values come from the masked
profile, and rejected lines appear only as line number and reason (their raw
text stays in rejected_lines.csv on the user's machine).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from pipeline.agents.decisions import DECISIONS_FILE, load_decisions
from pipeline.contracts.diagnosis import Diagnosis

PDF_FILENAME = "run_report.pdf"
MAX_REJECTS = 50
MAX_QUERY_ANSWER_LINES = 6

PAGE_WIDTH = A4[0] - 36 * mm
NEWLINE = chr(10)
OUTCOME_COLOURS = {"PASS": "#1a7f37", "FLAG": "#9a6700", "BLOCK": "#cf222e"}
_ASCII = {
    "→": "->",
    "←": "<-",
    "≤": "<=",
    "≥": ">=",
    "✓": "ok",
    "✗": "x",
    "…": "...",
    "–": "-",
    "—": "-",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    " ": " ",
}

_BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=9, leading=12)
STYLES = {
    "title": ParagraphStyle(
        "title", parent=_BASE, fontName="Helvetica-Bold", fontSize=18, leading=22
    ),
    "subtitle": ParagraphStyle("subtitle", parent=_BASE, textColor=colors.HexColor("#57606a")),
    "h1": ParagraphStyle(
        "h1",
        parent=_BASE,
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    ),
    "h2": ParagraphStyle(
        "h2",
        parent=_BASE,
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        spaceBefore=8,
        spaceAfter=3,
    ),
    "body": ParagraphStyle("body", parent=_BASE, spaceAfter=4),
    "ai": ParagraphStyle(
        "ai",
        parent=_BASE,
        spaceAfter=4,
        leftIndent=6,
        borderPadding=(2, 4, 2, 4),
        backColor=colors.HexColor("#f6f8fa"),
    ),
    "note": ParagraphStyle(
        "note", parent=_BASE, fontSize=7.5, leading=10, textColor=colors.HexColor("#57606a")
    ),
    "bullet": ParagraphStyle("bullet", parent=_BASE, leftIndent=10, bulletIndent=2, spaceAfter=2),
    "cell": ParagraphStyle("cell", parent=_BASE, fontSize=7.5, leading=9.5),
    "cellb": ParagraphStyle(
        "cellb", parent=_BASE, fontName="Helvetica-Bold", fontSize=7.5, leading=9.5
    ),
    "code": ParagraphStyle(
        "code",
        parent=_BASE,
        fontName="Courier",
        fontSize=7.5,
        leading=9.5,
        leftIndent=4,
        backColor=colors.HexColor("#f6f8fa"),
        borderPadding=4,
        spaceBefore=2,
        spaceAfter=6,
    ),
}


# -- reading the run folder -------------------------------------------------


def _load(run_dir: Path, name: str) -> Any:
    path = run_dir / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class Run:
    """Everything the run folder holds, read once."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.manifest = _load(run_dir, "manifest.json") or {}
        self.load = _load(run_dir, "load_result.json") or {}
        self.gate_1 = _load(run_dir, "gate_1.json")
        self.claims = _load(run_dir, "claims.json") or []
        self.profile = _load(run_dir, "profile_llm.json") or {"columns": []}
        diagnosis = _load(run_dir, "diagnosis.json")
        self.diagnosis = Diagnosis.model_validate(diagnosis) if diagnosis else None
        self.decisions = (
            {d.id: d for d in load_decisions(run_dir / DECISIONS_FILE, self.diagnosis)}
            if self.diagnosis
            else {}
        )
        self.plan = _load(run_dir, "cleaning_plan.json")
        self.log = _load(run_dir, "cleaning_log.json")
        self.gate_2 = _load(run_dir, "gate_2.json")
        self.handoff = _load(run_dir, "handoff.json")
        self.review = _load(run_dir, "review_log.json")
        self.analyst_log = _load(run_dir, "analyst_log.json")
        self.coder_log = _load(run_dir, "coder_log.json")

    @property
    def file_name(self) -> str:
        path = (self.load.get("file") or {}).get("path", "")
        return Path(path).name or self.dir.name

    @property
    def stages(self) -> dict[str, str]:
        return self.manifest.get("stage_status", {})

    @property
    def records(self) -> list[dict[str, Any]]:
        return (self.log or {}).get("records", [])

    def steps(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        if not self.plan:
            return []
        return list(zip(self.plan.get("steps", []), self.records, strict=False))

    def costs(self) -> dict[str, float]:
        def total(attempts):
            return sum(a.get("cost_usd") or 0.0 for a in attempts or [])

        review = 0.0
        if self.review:
            review = sum(total(r.get("attempts")) for r in self.review.get("rounds", []))
            review += total(self.review.get("failed_attempts"))
        return {
            "analyst": total((self.analyst_log or {}).get("attempts")),
            "coder": total((self.coder_log or {}).get("attempts")),
            "reviewer": review,
        }


# -- text helpers -----------------------------------------------------------


def _safe(value: Any) -> str:
    """Text the PDF's built-in fonts can draw: cp1252, with common symbols spelled out."""
    text = "" if value is None else str(value)
    for symbol, replacement in _ASCII.items():
        text = text.replace(symbol, replacement)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def P(text: Any, style: str = "body") -> Paragraph:  # noqa: N802 - reads like a tag
    return Paragraph(escape(_safe(text)).replace(NEWLINE, "<br/>"), STYLES[style])


def M(markup: str, style: str = "body") -> Paragraph:  # noqa: N802
    """A paragraph whose caller already escaped the text and added simple markup."""
    return Paragraph(markup, STYLES[style])


def E(text: Any) -> str:  # noqa: N802
    return escape(_safe(text))


def bullets(items: list[str], style: str = "bullet") -> list[Paragraph]:
    return [Paragraph(escape(_safe(item)), STYLES[style], bulletText="•") for item in items]


def outcome(word: str) -> str:
    word = (word or "").upper()
    colour = OUTCOME_COLOURS.get(word, "#24292f")
    return f'<font color="{colour}"><b>{E(word)}</b></font>'


def table(rows: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    cells = [
        [
            cell
            if isinstance(cell, Paragraph)
            else P(cell, "cellb" if header and i == 0 else "cell")
            for cell in row
        ]
        for i, row in enumerate(rows)
    ]
    result = Table(cells, colWidths=[w * PAGE_WIDTH for w in widths], repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaeef2")))
    result.setStyle(TableStyle(style))
    return result


def code_block(text: str) -> Preformatted:
    return Preformatted(_safe(text.rstrip()), STYLES["code"], maxLineLength=100, newLineChars="")


def ai_text(text: Any) -> Paragraph:
    """Text an AI wrote, marked so a reader can tell it from measured facts."""
    return M(f'<font size="7" color="#57606a">AI-written:</font> {E(text)}', "ai")


# -- the sections -----------------------------------------------------------


def _attention(run: Run) -> list[str]:
    """What a person should look at first, gathered from every step."""
    items: list[str] = []
    stages = run.stages
    if stages.get("ingestion") == "blocked":
        items.append(
            "Gate 1 blocked the file, so nothing after ingest ran. See 'The data as received'."
        )
    if stages.get("cleaning") == "blocked":
        items.append(
            "Cleaning is blocked: no clean data was written. See 'Checks on the clean data'."
        )
    for claim in run.claims:
        if claim.get("status") == "fail":
            text = claim.get("claim", {}).get("text") or claim.get("claim", {}).get("kind")
            items.append(
                f"Your brief said '{text}', but the data disagrees: {claim.get('message')}"
            )
    if run.review:
        items += run.review.get("notes", [])
    if run.diagnosis:
        for call in run.diagnosis.judgment_calls:
            decision = run.decisions.get(call.id)
            ai = call.ai_decision
            if ai and decision and decision.decided_by == "ai" and ai.confidence == "low":
                items.append(
                    f"{call.id} ({call.column}) was decided by the AI with low confidence: "
                    f"'{ai.choice}'. {call.question}"
                )
    for check in (run.gate_2 or {}).get("checks", []):
        if not check.get("passed") and check.get("severity") == "warn":
            items.append(f"Gate 2 flagged {check['name']}: {check['message']}")
    if run.review and run.review.get("rounds"):
        final = run.review["rounds"][-1]
        for ambiguity in final["outcome"].get("ambiguities", []):
            where = f" ({ambiguity['column']})" if ambiguity.get("column") else ""
            items.append(f"Question from the reviewer{where}: {ambiguity['question']}")
        for finding in final["outcome"].get("findings", []):
            if not finding.get("fix"):
                items.append(
                    f"Review point ({finding.get('column') or 'table'}): {finding['problem']}"
                )
        if not final["accepted"]:
            for fix in final["verdict"].get("fix_requests", []):
                items.append(
                    f"Unresolved fix request ({fix.get('column') or 'table'}): {fix['issue']}"
                )
        if final.get("weak"):
            items.append(
                "The reviewer could not verify enough with numbers, so its verdict carries "
                "little weight."
            )
    return items


def _summary(run: Run) -> list:
    load = run.load
    rows_raw = load.get("n_source_rows", 0)
    stages = run.stages
    log = run.log or {}
    rows_clean = log.get("rows_clean")
    records = run.records
    type_changes = sum(
        1
        for record in records
        for column in record.get("target_columns", [])
        if column in record.get("before", {})
        and column in record.get("after", {})
        and record["before"][column]["dtype"] != record["after"][column]["dtype"]
    )
    steps = run.plan.get("steps", []) if run.plan else []
    n_custom = sum(1 for s in steps if s.get("op") == "custom_code")

    gate_1 = (run.gate_1 or {}).get("outcome", "n/a")
    gate_2 = (run.gate_2 or {}).get("outcome")
    review_status = stages.get("review")
    result_rows = [
        ["Step", "Result", "What it means"],
        [
            P("1. Ingest (Gate 1)", "cell"),
            M(outcome(gate_1), "cell"),
            P(_gate_1_meaning(gate_1), "cell"),
        ],
    ]
    if run.diagnosis:
        result_rows.append(
            [
                P("2. Diagnose", "cell"),
                M("<b>done</b>", "cell"),
                P(
                    f"{len(run.diagnosis.issues)} issue(s) found, "
                    f"{len(run.diagnosis.judgment_calls)} judgment call(s).",
                    "cell",
                ),
            ]
        )
    if gate_2:
        result_rows.append(
            [
                P("3. Clean: Gate 2", "cell"),
                M(outcome(gate_2), "cell"),
                P(_gate_2_meaning(gate_2), "cell"),
            ]
        )
    if review_status:
        rounds = len((run.review or {}).get("rounds", []))
        result_rows.append(
            [
                P("3. Clean: Review", "cell"),
                M(f"<b>{E(review_status.upper())}</b>", "cell"),
                P(f"{rounds} review round(s). " + _review_meaning(review_status), "cell"),
            ]
        )

    numbers: list[list[Any]] = []
    numbers.append(
        [
            "Rows",
            f"{rows_raw:,} in the file, {load.get('n_loaded', 0):,} loaded, "
            f"{load.get('n_rejected', 0):,} rejected"
            + (f", {rows_clean:,} after cleaning" if rows_clean is not None else ""),
        ]
    )
    numbers.append(["Columns", str(len(load.get("columns", [])))])
    if records:
        changed = sum(sum(r.get("cells_changed", {}).values()) for r in records)
        nulls = sum(sum(r.get("new_nulls", {}).values()) for r in records)
        dropped = sum(r.get("rows_dropped", 0) for r in records)
        numbers.append(
            ["Cleaning steps", f"{len(steps)} ({n_custom} with new code written by the AI)"]
        )
        numbers.append(
            [
                "Values changed",
                f"{changed:,} cell(s); {type_changes} column(s) converted "
                f"to a new type; {nulls:,} value(s) turned into missing; {dropped:,} row(s) "
                f"dropped",
            ]
        )
    if run.diagnosis:
        decided = run.decisions.values()
        by_ai = sum(1 for d in decided if d.decided and d.decided_by == "ai")
        by_you = sum(1 for d in decided if d.decided and d.decided_by == "user")
        open_ = sum(1 for d in decided if not d.decided)
        numbers.append(
            ["Judgment calls", f"{by_ai} decided by the AI, {by_you} by you, {open_} open"]
        )
    costs = run.costs()
    total = sum(costs.values())
    numbers.append(
        [
            "LLM cost",
            f"${total:.2f} for this run (analyst ${costs['analyst']:.2f}, "
            f"coder ${costs['coder']:.2f}, reviewer ${costs['reviewer']:.2f})",
        ]
    )

    story = [
        P("Data cleaning report", "title"),
        P(
            f"{run.file_name}  -  run {run.manifest.get('run_id', '')}  -  "
            f"{datetime.now():%Y-%m-%d %H:%M}",
            "subtitle",
        ),
        Spacer(1, 8),
        P("Summary", "h1"),
        table(result_rows, [0.2, 0.12, 0.68]),
        Spacer(1, 6),
        table(numbers, [0.2, 0.8], header=False),
        P("Look at these first", "h2"),
    ]
    attention = _attention(run)
    story += bullets(attention) if attention else [P("Nothing needs your attention.")]
    files = []
    if (run.dir / "clean_data.csv").exists():
        files.append("clean_data.csv / clean_data.parquet: the cleaned data")
    files += [
        "decisions.yaml: the decisions on judgment calls; change one and run the cleaning again",
        "cleaning_report.md and the *_log.json files: the full technical audit trail",
    ]
    story += [P("Files in the run folder", "h2"), *bullets(files)]
    return story


def _gate_1_meaning(word: str) -> str:
    return {
        "pass": "The file was read completely and faithfully.",
        "flag": "The file was read; some things deserve a look.",
        "block": "The file was not read faithfully, so the run stopped here.",
    }.get(word, "")


def _gate_2_meaning(word: str) -> str:
    return {
        "pass": "The clean data matches the diagnosis and every check passed.",
        "flag": "The clean data is usable; some checks deserve a look.",
        "block": "The result may not be used; no clean data was written.",
    }.get(word, "")


def _review_meaning(status: str) -> str:
    return {
        "accepted": "The reviewer checked the result and asked for no further fixes.",
        "flagged": "Some review points are unresolved; see 'Look at these first'.",
        "not_reviewed": "The reviewer was switched off.",
    }.get(status, "")


def _received(run: Run) -> list:
    load, info = run.load, run.load.get("file", {})
    story = [PageBreak(), P("1. The data as received", "h1")]
    story.append(
        P(
            "The file is loaded exactly as it is, every value as text. Nothing is changed "
            "at this step; it only describes what is there.",
            "note",
        )
    )
    facts = [
        ["File", run.file_name],
        [
            "Format",
            ", ".join(
                x
                for x in [
                    info.get("format"),
                    f"encoding {info.get('encoding')}" if info.get("encoding") else "",
                    f"delimiter '{info.get('delimiter')}'" if info.get("delimiter") else "",
                    f"sheet {info.get('sheet')}" if info.get("sheet") else "",
                ]
                if x
            ),
        ],
        [
            "Rows",
            f"{load.get('n_source_rows', 0):,} in the file, {load.get('n_loaded', 0):,} loaded, "
            f"{load.get('n_rejected', 0):,} rejected",
        ],
        [
            "Fingerprint",
            f"file {str(info.get('file_hash', ''))[:16]}..., data "
            f"{str(load.get('data_hash', ''))[:16]}...",
        ],
    ]
    story.append(table(facts, [0.2, 0.8], header=False))
    notes = info.get("detection_notes", [])
    if notes:
        story += [P("How the file was interpreted", "h2"), *bullets(notes)]

    rejects = load.get("rejects", [])
    story.append(P("Lines that could not be read", "h2"))
    if rejects:
        rows = [["Line in file", "Reason"]]
        rows += [[str(r["source_line"]), r["reason"]] for r in rejects[:MAX_REJECTS]]
        story.append(table(rows, [0.2, 0.8]))
        story.append(
            P(
                f"{len(rejects)} line(s) in total. Their full text is in rejected_lines.csv in "
                f"the run folder; it is left out here because it may contain personal data.",
                "note",
            )
        )
    else:
        story.append(P("None: every line was read."))

    renames = load.get("renames", [])
    if renames:
        rows = [["Position", "In the file", "Now called", "Why"]]
        rows += [
            [str(r["position"]), repr(r["original"]), r["final"], r["reason"]] for r in renames
        ]
        story += [P("Headers that had to be renamed", "h2"), table(rows, [0.12, 0.25, 0.25, 0.38])]

    columns = run.profile.get("columns", [])
    if columns:
        rows = [
            ["Column", "Stored as", "Read as", "Fits", "Missing", "Distinct", "What stands out"]
        ]
        for c in columns:
            notes_ = []
            if c.get("pii"):
                notes_.append(f"personal data ({', '.join(c['pii'].get('kinds', []))}), masked")
            elif c.get("masked"):
                notes_.append("identifier, masked")
            if c.get("null_tokens"):
                notes_.append("missing written as " + ", ".join(repr(k) for k in c["null_tokens"]))
            if c.get("sentinel_candidates"):
                notes_.append("possible codes for missing: " + ", ".join(c["sentinel_candidates"]))
            if c.get("unparsed_examples"):
                notes_.append("values that do not fit: " + ", ".join(c["unparsed_examples"]))
            if c.get("ambiguous"):
                notes_.append("type is ambiguous")
            if c.get("far_outliers"):
                notes_.append(f"{c['far_outliers']} far outlier(s)")
            if c.get("examples"):
                notes_.append("e.g. " + ", ".join(str(e) for e in c["examples"]))
            rows.append(
                [
                    c["name"],
                    c.get("stored_as", "text"),
                    c.get("type", ""),
                    f"{c.get('coverage', 0):.0%}",
                    c.get("missing", ""),
                    f"{c.get('distinct', 0):,}",
                    "; ".join(notes_),
                ]
            )
        story += [
            P("Column by column (dirty data spotted at load)", "h2"),
            table(rows, [0.12, 0.08, 0.1, 0.07, 0.09, 0.09, 0.45]),
        ]

    if run.claims:
        label = {"pass": "holds", "fail": "does NOT hold", "not_checkable": "could not be checked"}
        story += [
            P("What your brief claimed", "h2"),
            *bullets(
                [
                    f"{(c.get('claim', {}).get('text') or c.get('claim', {}).get('kind'))}: "
                    f"{label.get(c.get('status'), c.get('status'))}. {c.get('message', '')}"
                    for c in run.claims
                ]
            ),
        ]
    if run.gate_1:
        story += [P("Gate 1 checks", "h2"), _checks_table(run.gate_1)]
    return story


def _checks_table(gate: dict[str, Any]) -> Table:
    rows = [["Check", "Result", "Details"]]
    for check in gate.get("checks", []):
        word = "pass" if check["passed"] else ("FLAG" if check["severity"] == "warn" else "BLOCK")
        rows.append(
            [
                check["name"],
                M(outcome(word) if word != "pass" else "pass", "cell"),
                check.get("message", ""),
            ]
        )
    return table(rows, [0.3, 0.1, 0.6])


def _wrong(run: Run) -> list:
    diagnosis = run.diagnosis
    if diagnosis is None:
        return []
    story = [PageBreak(), P("2. What was wrong", "h1"), ai_text(diagnosis.summary)]
    rows = [["ID", "Column", "Kind", "What was found", "What should be done"]]
    for issue in diagnosis.issues:
        fix = issue.fix + (f" (op: {issue.suggested_op})" if issue.suggested_op else "")
        rows.append([issue.id, issue.column or "table", issue.kind, issue.evidence, fix])
    story += [
        P("Issues: clear errors with a clear fix", "h2"),
        table(rows, [0.05, 0.12, 0.12, 0.33, 0.38]),
    ]

    if diagnosis.judgment_calls:
        story.append(P("Judgment calls: questions the data cannot answer by itself", "h2"))
    for call in diagnosis.judgment_calls:
        decision = run.decisions.get(call.id)
        ai = call.ai_decision
        block = [
            M(f"<b>{E(call.id)}</b> ({E(call.column or 'table')}): {E(call.question)}", "body")
        ]
        block += bullets([f"Option: {option}" for option in call.options])
        if decision is not None and decision.decided and decision.decided_by == "user":
            block.append(M(f"<b>Decided by you:</b> {E(decision.choice)}"))
        elif decision is not None and decision.decided and ai is not None:
            caution = (
                " The brief does not settle this, so the least destructive option was taken."
                if ai.least_destructive
                else ""
            )
            block.append(
                M(
                    f"<b>Decided by the AI:</b> {E(ai.choice)} "
                    f"(confidence: {E(ai.confidence)}).{E(caution)}"
                )
            )
            block.append(ai_text(f"Why: {ai.reasoning} Expected effect: {ai.expected_effect}"))
            if ai.alternatives:
                block += bullets([f"Alternative: {a.option}. {a.effect}" for a in ai.alternatives])
        else:
            block.append(
                M("<b>Open:</b> nobody has decided it, so the data it concerns was left as it is.")
            )
        block.append(Spacer(1, 4))
        story.append(KeepTogether(block))

    rows = [["Column", "Role", "Clean type", "Notes"]]
    rows += [[c.name, c.role, c.target_type, c.notes or ""] for c in diagnosis.columns]
    story += [P("What each column should become", "h2"), table(rows, [0.2, 0.15, 0.15, 0.5])]
    return story


def _changed(run: Run) -> list:
    if not run.plan or not run.log:
        return []
    log = run.log
    story = [PageBreak(), P("3. What was changed", "h1"), ai_text(run.plan.get("summary", ""))]
    story.append(
        P(
            f"Rows: {log.get('rows_raw', 0):,} before, {log.get('rows_clean', 0):,} after. "
            f"Every step "
            f"below ran on a copy, starting from the original data, and every change it made was "
            f"measured; the original data is never modified.",
            "body",
        )
    )
    for number, (step, record) in enumerate(run.steps(), start=1):
        custom = step.get("op") == "custom_code"
        title = f"Step {number}: {step.get('op')}" + (
            " (new code written by the AI)" if custom else ""
        )
        block = [
            P(f"{title}, for {', '.join(step.get('addresses', []))}", "h2"),
            ai_text(step.get("rationale", "")),
        ]
        facts = []
        cells = record.get("cells_changed", {})
        facts.append(
            "Changed: "
            + ("; ".join(f"{c}: {n:,} cell(s)" for c, n in cells.items()) or "no values")
        )
        types = [
            f"{c}: {record['before'][c]['dtype']} -> {record['after'][c]['dtype']}"
            for c in record.get("target_columns", [])
            if c in record.get("before", {})
            and c in record.get("after", {})
            and record["before"][c]["dtype"] != record["after"][c]["dtype"]
        ]
        if types:
            facts.append("New type: " + "; ".join(types))
        before, after = record.get("before", {}), record.get("after", {})
        added = [c for c in after if c not in before]
        removed = [c for c in before if c not in after]
        if added:
            facts.append("New column(s): " + ", ".join(f"{c} ({after[c]['dtype']})" for c in added))
        if removed:
            facts.append("Removed column(s): " + ", ".join(removed))
        nulls = record.get("new_nulls", {})
        if nulls:
            facts.append(
                "Turned into missing: " + "; ".join(f"{c}: {n:,}" for c, n in nulls.items())
            )
        if record.get("rows_dropped"):
            facts.append(
                f"Rows dropped: {record['rows_dropped']:,} (each listed with its reason in "
                f"dropped_rows.csv)"
            )
        facts += [f"Flag: {flag.get('message')}" for flag in record.get("flags", [])]
        block += bullets(facts)
        params = step.get("params", {})
        if custom:
            declared = {k: v for k, v in params.items() if k != "code" and v not in (None, [], "")}
            block.append(
                P(
                    "The code, exactly as it ran (in a locked-down sandbox, allowed to change "
                    f"only what it declared: {json.dumps(declared, ensure_ascii=False)}):",
                    "note",
                )
            )
            block.append(code_block(params.get("code", "")))
        else:
            block.append(
                P(
                    f"Parameters: {json.dumps(record.get('params', params), ensure_ascii=False)}",
                    "note",
                )
            )
        story.append(KeepTogether(block))

    skipped = run.plan.get("skipped", [])
    if skipped:
        story += [
            P("Not done, with reasons", "h2"),
            *bullets([f"{s['id']}: {s['reason']}" for s in skipped]),
        ]
    return story


def _checks(run: Run) -> list:
    if not run.gate_2:
        return []
    story = [
        PageBreak(),
        P("4. Checks on the clean data", "h1"),
        M(
            f"Gate 2: {outcome(run.gate_2.get('outcome'))}. "
            f"{E(_gate_2_meaning(run.gate_2.get('outcome')))}"
        ),
        _checks_table(run.gate_2),
    ]
    if run.handoff:
        rows = [["Column", "Type", "Expected", "Role", "Missing", "Distinct", "Note"]]
        for c in run.handoff.get("schema", []):
            role = (c.get("role") or "added") + (", key" if c.get("unique") else "")
            rows.append(
                [
                    c["name"],
                    c["dtype"],
                    c.get("expected_type") or "",
                    role,
                    f"{c.get('null_rate', 0):.1%}",
                    f"{c.get('n_distinct', 0):,}",
                    c.get("note") or "",
                ]
            )
        story += [
            P("The clean data, column by column (the schema contract)", "h2"),
            table(rows, [0.15, 0.11, 0.1, 0.14, 0.09, 0.09, 0.32]),
        ]
    return story


def _review(run: Run) -> list:
    review = run.review
    if not review:
        return []
    story = [
        PageBreak(),
        P("5. Review", "h1"),
        P(
            "A second AI checked whether the cleaning was right, using its own counting queries "
            "on the data (never rows, and personal values only as their shape).",
            "note",
        ),
    ]
    story += bullets(review.get("notes", []))
    for round_ in review.get("rounds", []):
        verdict, out = round_["verdict"], round_["outcome"]
        state = (
            "accepted"
            if verdict["accepted"]
            else f"sent back with {len(verdict.get('fix_requests', []))} fix request(s)"
        )
        story += [P(f"Round {round_['round']}: {state}", "h2"), ai_text(out.get("summary", ""))]
        if out.get("verified"):
            story += [
                P("What it verified, with the numbers that show it:", "body"),
                *bullets([f"{v['claim']} ({v['evidence']})" for v in out["verified"]]),
            ]
        if out.get("findings"):
            items = []
            for f in out["findings"]:
                where = f"step {f['step']}, " if f.get("step") else ""
                fix = f" Fix asked: {f['fix']['issue']}." if f.get("fix") else ""
                items.append(
                    f"[{f['severity']}] {where}{f.get('column') or 'table'}: {f['problem']} "
                    f"({f['evidence']}).{fix}"
                )
            story += [P("What it found:", "body"), *bullets(items)]
        if out.get("ambiguities"):
            story += [
                P("Questions for you:", "body"),
                *bullets([a["question"] for a in out["ambiguities"]]),
            ]
        if round_.get("queries"):
            rows = [["Query", "What it wanted to know", "Answer"]]
            for q in round_["queries"]:
                answer = "\n".join(q["answer"].splitlines()[:MAX_QUERY_ANSWER_LINES])
                rows.append([q["query"].get("id", ""), q["query"].get("purpose", ""), answer])
            story += [P("Queries it ran", "h2"), table(rows, [0.08, 0.37, 0.55])]
        if round_.get("dropped"):
            story += [
                P("Claims discarded for citing numbers it was never shown:", "body"),
                *bullets(round_["dropped"]),
            ]
    return story


def _audit(run: Run) -> list:
    manifest = run.manifest
    costs = run.costs()
    rows = [["Agent", "Model", "Prompt version", "Calls", "Cost"]]
    calls = {
        "analyst": len((run.analyst_log or {}).get("attempts", [])),
        "coder": len((run.coder_log or {}).get("attempts", [])),
        "reviewer": sum(len(r.get("attempts", [])) for r in (run.review or {}).get("rounds", [])),
    }
    for agent in ("analyst", "coder", "reviewer"):
        model = manifest.get("models", {}).get(agent, {}).get("name", "-")
        version = manifest.get("prompt_versions", {}).get(agent, "-")
        rows.append([agent, model, version, str(calls[agent]), f"${costs[agent]:.4f}"])
    facts = [
        ["Run", manifest.get("run_id", "")],
        [
            "Pipeline",
            f"version {manifest.get('pipeline_version', '')}, git "
            f"{str(manifest.get('git_sha') or 'unknown')[:10]}"
            + (" (uncommitted changes)" if manifest.get("git_dirty") else ""),
        ],
        ["Steps", ", ".join(f"{k}: {v}" for k, v in run.stages.items())],
        [
            "All LLM cost on this folder",
            f"${manifest.get('cost', {}).get('usd', 0):.4f} over "
            f"{manifest.get('cost', {}).get('calls', 0)} call(s), "
            f"including earlier runs",
        ],
    ]
    files = sorted(p.name for p in run.dir.iterdir() if p.is_file())
    return [
        PageBreak(),
        P("6. Audit", "h1"),
        table(facts, [0.25, 0.75], header=False),
        P("Models", "h2"),
        table(rows, [0.14, 0.36, 0.18, 0.12, 0.2]),
        P("Files in the run folder", "h2"),
        P(", ".join(files), "note"),
        Spacer(1, 6),
        P(
            "Every number in this report comes from the run's logs. Text marked 'AI-written' was "
            "written by a language model and is shown as it wrote it. The original data was never "
            "modified, and no AI saw a row or a personal value.",
            "note",
        ),
    ]


# -- building ---------------------------------------------------------------


def build_pdf(run_dir: str | Path) -> Path:
    """Write run_report.pdf into the run folder and return its path."""
    run_dir = Path(run_dir)
    run = Run(run_dir)
    target = run_dir / PDF_FILENAME
    story = [
        *_summary(run),
        *_received(run),
        *_wrong(run),
        *_changed(run),
        *_checks(run),
        *_review(run),
        *_audit(run),
    ]
    footer = _safe(f"Data cleaning report - {run.file_name}")

    def on_page(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#57606a"))
        canvas.drawString(18 * mm, 10 * mm, footer)
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title=f"Data cleaning report: {run.file_name}",
        author="Auditable data science pipeline",
    )
    document.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return target


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not (Path(args[0]) / "manifest.json").exists():
        print("usage: python -m pipeline.report_pdf runs/<run folder>")
        return 1
    print(f"Wrote {build_pdf(args[0]).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
