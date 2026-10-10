"""REPORT.pdf: the answer in plain words, in at most two pages, always in the same order.

The content is the MainSummary (summary.py); this file only draws it. REPORT.html
draws the same content for a browser.
"""

from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, Spacer, Table, TableStyle

from pipeline.reports.common import (
    PAGE_WIDTH,
    STYLES,
    E,
    M,
    P,
    bullets,
    chart_image,
    link,
)
from pipeline.reports.summary import KeyFinding, Light, MainSummary

FINDING_CHART_WIDTH = 0.44 * PAGE_WIDTH
LIGHT_COLOURS = {
    "green": ("#dafbe1", "#1a7f37"),
    "yellow": ("#fff8c5", "#9a6700"),
    "red": ("#ffebe9", "#cf222e"),
}
GREY = "#57606a"

_STYLES = {
    "big": ParagraphStyle("big", parent=STYLES["body"], fontSize=12, leading=16, spaceAfter=4),
    "question": ParagraphStyle(
        "question",
        parent=STYLES["body"],
        fontSize=10,
        leading=13,
        backColor=colors.HexColor("#f6f8fa"),
        borderPadding=(5, 6, 5, 6),
        spaceBefore=4,
        spaceAfter=8,
    ),
    "number": ParagraphStyle(
        "number",
        parent=STYLES["body"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#0969da"),
    ),
    "finding": ParagraphStyle("finding", parent=STYLES["body"], fontSize=10.5, leading=14),
}


def plain_document(summary: MainSummary) -> list:
    return [
        P(summary.title, "title"),
        P(f"{summary.file_name}  -  {summary.rows:,} rows", "subtitle"),
        Spacer(1, 6),
        *_question(summary),
        *_answer(summary),
        *_found(summary),
        *_actions(summary),
        *_trust(summary),
        *_keep_in_mind(summary),
        *_more(summary),
    ]


def _question(summary: MainSummary) -> list:
    if not summary.question:
        return []
    return [Paragraph(f"<b>Your question</b><br/>{E(summary.question)}", _STYLES["question"])]


def _ai_note(text: str) -> Paragraph:
    return M(f'<font color="{GREY}">{E(text)}</font>', "note")


def _answer(summary: MainSummary) -> list:
    story = [P("The answer", "h1")]
    if summary.stopped:
        story.append(Paragraph(E(summary.stopped), _STYLES["big"]))
    if summary.answer:
        story += [
            Paragraph(E(summary.answer), _STYLES["big"]),
            _ai_note("Written by the AI from the analysis; every number was checked against it."),
        ]
    return story


def _found(summary: MainSummary) -> list:
    story = [P("What we found", "h1")]
    if not summary.findings:
        return [*story, P("Nothing yet: the analysis did not run.")]
    for finding in summary.findings:
        story.append(KeepTogether([_finding_row(finding), Spacer(1, 6)]))
    if summary.findings_not_shown:
        story.append(
            _ai_note(f"{summary.findings_not_shown} more finding(s) in details/1_analysis.pdf.")
        )
    return story


def _finding_row(finding: KeyFinding) -> Table:
    text = []
    if finding.number:
        text.append(Paragraph(E(finding.number), _STYLES["number"]))
    text += [
        Paragraph(E(finding.sentence), _STYLES["finding"]),
        M(
            f'<font color="{GREY}">{E(finding.evidence)}. More: {finding.id} in '
            f"{link('details/1_analysis.pdf', 'details/1_analysis.pdf')}</font>",
            "note",
        ),
    ]
    if not finding.charts:
        return _plain_table([[text]], [PAGE_WIDTH])
    picture = chart_image(finding.charts[0], FINDING_CHART_WIDTH)
    gap = 4 * mm
    return _plain_table(
        [[picture, text]], [FINDING_CHART_WIDTH + gap, PAGE_WIDTH - FINDING_CHART_WIDTH - gap]
    )


def _plain_table(rows: list, widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths)
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def _actions(summary: MainSummary) -> list:
    if not summary.actions:
        return []
    return [
        P("What you could do", "h1"),
        *bullets(summary.actions),
        _ai_note("Suggested by the AI from the findings."),
    ]


def _trust(summary: MainSummary) -> list:
    rows = [_light_row(light) for light in summary.lights]
    table = Table(rows, colWidths=[0.11 * PAGE_WIDTH, 0.13 * PAGE_WIDTH, 0.76 * PAGE_WIDTH])
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#d0d7de")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i, light in enumerate(summary.lights):
        background, _ = LIGHT_COLOURS[light.colour]
        style.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(background)))
    table.setStyle(TableStyle(style))
    return [P("How much can you trust this?", "h1"), table]


def _light_row(light: Light) -> list:
    _, ink = LIGHT_COLOURS[light.colour]
    return [
        M(f'<font color="{ink}"><b>{E(light.word)}</b></font>', "body"),
        M(f"<b>{E(light.topic)}</b>", "body"),
        M(f'{E(light.text)} <font color="{GREY}">More: {link(light.where, light.where)}</font>'),
    ]


def _keep_in_mind(summary: MainSummary) -> list:
    if not summary.cautions:
        return []
    story = [P("Keep in mind", "h1"), *bullets(summary.cautions)]
    rest = len(summary.all_points) - len(summary.cautions)
    if rest > 0:
        story.append(_ai_note(f"{rest} more point(s) in details/0_technical_summary.pdf."))
    return story


def _more(summary: MainSummary) -> list:
    rows = [[M(link(path, path), "cell"), P(what, "cell")] for path, what in summary.documents]
    table = Table(rows, colWidths=[0.36 * PAGE_WIDTH, 0.64 * PAGE_WIDTH])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -2), 0.25, colors.HexColor("#d0d7de")),
            ]
        )
    )
    return [
        P("Want more?", "h1"),
        table,
        Spacer(1, 4),
        _ai_note(
            "To change a cleaning decision, edit edit/decisions.yaml; to change the questions, "
            "edit edit/analysis_plan.yaml. Then run again."
        ),
    ]
