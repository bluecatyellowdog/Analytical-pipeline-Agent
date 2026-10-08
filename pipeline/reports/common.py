"""What every report shares: the styles, small text helpers, and the run folder read once.

Every document is built only from the files in the run folder, so it can be
rebuilt for any run and says nothing the logs do not. A PDF gets shared, so it
holds no personal data. Text written by an AI is marked as such.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
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
from pipeline.layout import CHARTS, run_file

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


def load_json(run_dir: Path, name: str) -> Any:
    path = run_file(run_dir, name)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class Run:
    """Everything the run folder holds, read once."""

    def __init__(self, run_dir: Path) -> None:
        self.dir = run_dir
        self.manifest = load_json(run_dir, "manifest.json") or {}
        self.load = load_json(run_dir, "load_result.json") or {}
        self.gate_1 = load_json(run_dir, "gate_1.json")
        self.claims = load_json(run_dir, "claims.json") or []
        self.profile = load_json(run_dir, "profile_llm.json") or {"columns": []}
        diagnosis = load_json(run_dir, "diagnosis.json")
        self.diagnosis = Diagnosis.model_validate(diagnosis) if diagnosis else None
        self.decisions = (
            {d.id: d for d in load_decisions(run_file(run_dir, DECISIONS_FILE), self.diagnosis)}
            if self.diagnosis
            else {}
        )
        self.plan = load_json(run_dir, "cleaning_plan.json")
        self.log = load_json(run_dir, "cleaning_log.json")
        self.gate_2 = load_json(run_dir, "gate_2.json")
        self.handoff = load_json(run_dir, "handoff.json")
        self.review = load_json(run_dir, "review_log.json")
        self.diagnostic_log = load_json(run_dir, "diagnostic_log.json") or load_json(
            run_dir,
            "analyst_log.json",  # the name before the rename
        )
        self.coder_log = load_json(run_dir, "coder_log.json")
        self.analysis = load_json(run_dir, "analysis.json")
        self.analysis_plan = (self.analysis or {}).get("plan") or load_json(
            run_dir, "analysis_plan_ai.json"
        )
        self.gate_3 = load_json(run_dir, "gate_3.json")
        self.planner_log = load_json(run_dir, "planner_log.json")
        self.gate_4 = load_json(run_dir, "gate_4.json")
        self.analysis_log = load_json(run_dir, "analysis_log.json")
        catalog = load_json(run_dir, "artifacts.json") or {}
        self.artifacts = {a["artifact_id"]: a for a in catalog.get("artifacts", [])}

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
            "diagnostic": total((self.diagnostic_log or {}).get("attempts")),
            "coder": total((self.coder_log or {}).get("attempts")),
            "reviewer": review,
            "planner": total((self.planner_log or {}).get("attempts")),
            "analysis": total((self.analysis_log or {}).get("attempts")),
        }

    @property
    def title(self) -> str:
        return "Data cleaning and analysis report" if self.analysis else "Data cleaning report"


# -- text helpers -----------------------------------------------------------


def safe(value: Any) -> str:
    """Text the PDF's built-in fonts can draw: cp1252, with common symbols spelled out."""
    text = "" if value is None else str(value)
    for symbol, replacement in _ASCII.items():
        text = text.replace(symbol, replacement)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def P(text: Any, style: str = "body") -> Paragraph:  # noqa: N802 - reads like a tag
    return Paragraph(escape(safe(text)).replace(NEWLINE, "<br/>"), STYLES[style])


def M(markup: str, style: str = "body") -> Paragraph:  # noqa: N802
    """A paragraph whose caller already escaped the text and added simple markup."""
    return Paragraph(markup, STYLES[style])


def E(text: Any) -> str:  # noqa: N802
    return escape(safe(text))


def bullets(items: list[str], style: str = "bullet") -> list[Paragraph]:
    return [Paragraph(escape(safe(item)), STYLES[style], bulletText="•") for item in items]


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
    return Preformatted(safe(text.rstrip()), STYLES["code"], maxLineLength=100, newLineChars="")


def ai_text(text: Any) -> Paragraph:
    """Text an AI wrote, marked so a reader can tell it from measured facts."""
    return M(f'<font size="7" color="#57606a">AI-written:</font> {E(text)}', "ai")


# -- shared pieces ------------------------------------------------------------


def checks_table(gate: dict[str, Any]) -> Table:
    rows = [["Check", "Result", "Details"]]
    for check in gate.get("checks", []):
        failed = {"warn": "FLAG", "info": "note"}.get(check["severity"], "BLOCK")
        word = "pass" if check["passed"] else failed
        rows.append(
            [
                check["name"],
                M(outcome(word) if word != "pass" else "pass", "cell"),
                check.get("message", ""),
            ]
        )
    return table(rows, [0.3, 0.1, 0.6])


def gate_1_meaning(word: str) -> str:
    return {
        "pass": "The file was read completely and faithfully.",
        "flag": "The file was read; some things deserve a look.",
        "block": "The file was not read faithfully, so the run stopped here.",
    }.get(word, "")


def gate_2_meaning(word: str) -> str:
    return {
        "pass": "The clean data matches the diagnosis and every check passed.",
        "flag": "The clean data is usable; some checks deserve a look.",
        "block": "The result may not be used; no clean data was written.",
    }.get(word, "")


def review_meaning(status: str) -> str:
    return {
        "accepted": "The reviewer checked the result and asked for no further fixes.",
        "flagged": "Some review points are unresolved; see 'Look at these first'.",
        "not_reviewed": "The reviewer was switched off.",
    }.get(status, "")


STATUS_WORDS = {
    "answered": "answered",
    "partly": "partly answered",
    "not_answered": "not answered",
    "infeasible": "not possible with this data",
}


CHART_WIDTH = 150 * mm


def chart(run: Run, artifact: dict[str, Any], width: float = CHART_WIDTH) -> Image | None:
    chart_file = artifact.get("spec", {}).get("chart_file")
    if not chart_file:
        return None
    path = run.dir / chart_file
    if not path.exists():  # a run from before the current layout kept its charts elsewhere
        path = run.dir / CHARTS / Path(chart_file).name
    if not path.exists():
        return None
    pixels_wide, pixels_high = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * pixels_high / pixels_wide)


def tagged_bullets(items: list[tuple[str, bool]]) -> list[Paragraph]:
    """Bullets where text an AI wrote carries a small 'AI' tag."""
    tag = '<font size="6.5" color="#57606a">AI</font> '
    return [
        Paragraph((tag if by_ai else "") + E(text), STYLES["bullet"], bulletText="•")
        for text, by_ai in items
    ]


CONFIDENCE_RANK = {"low": 1, "medium": 2, "high": 3}


def decision_rank(run: Run, call: Any) -> int:
    """Open calls first, then the AI's least certain decisions, then yours."""
    decision = run.decisions.get(call.id)
    if decision is None or not decision.decided:
        return 0
    if decision.decided_by == "user":
        return 4
    confidence = call.ai_decision.confidence if call.ai_decision else "high"
    return CONFIDENCE_RANK.get(confidence, 3)


def measured_effect(run: Run, call_id: str, column: str | None = None) -> str:
    """What the cleaning steps that carried out a decision changed, as measured. A step that
    handled several things at once is counted only in the decision's own column."""
    changed = nulls = dropped = 0
    columns: set[str] = set()
    for step, record in run.steps():
        if call_id in step.get("addresses", []):
            cells = record.get("cells_changed", {})
            new_nulls = record.get("new_nulls", {})
            if column is not None:
                cells = {c: n for c, n in cells.items() if c == column}
                new_nulls = {c: n for c, n in new_nulls.items() if c == column}
            changed += sum(cells.values())
            columns |= {c for c, n in cells.items() if n}
            nulls += sum(new_nulls.values())
            dropped += record.get("rows_dropped", 0)
    if not (changed or nulls or dropped):
        return "Measured: nothing in the data was changed for it."
    parts = []
    if changed:
        where = f" in {', '.join(sorted(columns))}" if columns else ""
        parts.append(f"{changed:,} value(s) changed{where}")
    if nulls:
        parts.append(f"{nulls:,} value(s) turned into missing")
    if dropped:
        parts.append(f"{dropped:,} row(s) removed")
    return "Measured: " + "; ".join(parts) + "."


def link(href: str, text: str) -> str:
    """A clickable link to another file of the run (works in most PDF viewers)."""
    return f'<a href="{escape(href)}" color="#0969da">{E(text)}</a>'


def title_block(run: Run, title: str, intro: str) -> list:
    """The top of a detail document: what it is, and the way back to REPORT.pdf."""
    run_id = run.manifest.get("run_id", "")
    return [
        P(title, "title"),
        M(
            f"{E(run.file_name)} - run {E(run_id)} - part of {link('../REPORT.pdf', 'REPORT.pdf')}",
            "subtitle",
        ),
        Spacer(1, 4),
        P(intro, "note"),
        Spacer(1, 6),
    ]


def strip_break(story: list) -> list:
    """A document's first section needs no page break before it."""
    while story and isinstance(story[0], PageBreak):
        story = story[1:]
    return story


def write_pdf(path: Path, title: str, story: list) -> Path:
    """Build one PDF with the run's footer on every page."""
    path.parent.mkdir(parents=True, exist_ok=True)
    footer = safe(title)

    def on_page(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#57606a"))
        canvas.drawString(18 * mm, 10 * mm, footer)
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {document.page}")
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="Auditable data science pipeline",
    )
    document.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return path
