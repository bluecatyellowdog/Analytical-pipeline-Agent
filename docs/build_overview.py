# ruff: noqa: E501 - most of this file is prose for the PDF
"""Build docs/pipeline_overview.pdf: a flowchart of the pipeline, then what each part does.

uv run python docs/build_overview.py docs/pipeline_overview.pdf
"""

import sys
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path(sys.argv[1])

YOU = colors.HexColor("#d8f0dc")
YOU_EDGE = colors.HexColor("#2f8a3e")
AI = colors.HexColor("#dbe7fb")
AI_EDGE = colors.HexColor("#2f5fb3")
CODE = colors.HexColor("#eceff2")
CODE_EDGE = colors.HexColor("#57606a")
GATE = colors.HexColor("#fdebd3")
GATE_EDGE = colors.HexColor("#c36a00")
INK = colors.HexColor("#24292f")
GREY = colors.HexColor("#57606a")

WIDTH, HEIGHT = 520, 690
MAIN_X, MAIN_W = 150, 220  # the main column
SIDE_X, SIDE_W = 400, 118  # things beside a step
BOX_H, GATE_H = 34, 26


def box(d, x, y, w, h, fill, edge, title, sub="", tag=""):
    d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=edge, strokeWidth=1))
    if tag:
        d.add(
            String(
                x + w - 4,
                y + h - 9,
                tag,
                fontName="Helvetica-Bold",
                fontSize=6,
                fillColor=edge,
                textAnchor="end",
            )
        )
    lines = [title] + ([sub] if sub else [])
    top = y + h / 2 + (5 if sub else -3)
    d.add(
        String(
            x + w / 2,
            top,
            title,
            fontName="Helvetica-Bold",
            fontSize=8.5,
            fillColor=INK,
            textAnchor="middle",
        )
    )
    if sub:
        d.add(
            String(
                x + w / 2,
                top - 11,
                sub,
                fontName="Helvetica",
                fontSize=7,
                fillColor=GREY,
                textAnchor="middle",
            )
        )
    return lines


def side(d, y, h, fill, edge, title, sub, tag):
    box(d, SIDE_X, y, SIDE_W, h, fill, edge, title, sub)


def arrow(d, x1, y1, x2, y2, colour=INK, dashed=False, head=True):
    line = Line(x1, y1, x2, y2, strokeColor=colour, strokeWidth=1)
    if dashed:
        line.strokeDashArray = [3, 2]
    d.add(line)
    if head:
        if x1 == x2:  # vertical
            sign = -1 if y2 < y1 else 1
            pts = [x2, y2, x2 - 3, y2 - 6 * sign, x2 + 3, y2 - 6 * sign]
        else:
            sign = -1 if x2 < x1 else 1
            pts = [x2, y2, x2 - 6 * sign, y2 - 3, x2 - 6 * sign, y2 + 3]
        d.add(Polygon(pts, fillColor=colour, strokeColor=colour))


def label(d, x, y, text, size=6.5, colour=GREY, anchor="start"):
    d.add(
        String(
            x,
            y,
            text,
            fontName="Helvetica-Oblique",
            fontSize=size,
            fillColor=colour,
            textAnchor=anchor,
        )
    )


def loop_back(d, from_y, to_y, text, x=MAIN_X - 18, colour=GATE_EDGE):
    """A retry loop on the left: from a gate back up to the step it checks."""
    d.add(Line(MAIN_X, from_y, x, from_y, strokeColor=colour, strokeWidth=1))
    d.add(Line(x, from_y, x, to_y, strokeColor=colour, strokeWidth=1))
    arrow(d, x, to_y, MAIN_X, to_y, colour)
    label(d, x - 4, (from_y + to_y) / 2, text, anchor="end", colour=colour)


def flowchart() -> Drawing:
    d = Drawing(WIDTH, HEIGHT)
    step = 52
    rows = {}
    y = HEIGHT - 40
    nodes = [
        (
            "you",
            "Your data file + a short brief",
            "what the data is, what you want to learn",
            YOU,
            YOU_EDGE,
            "YOU",
        ),
        (
            "ingest",
            "1. Ingest",
            "load as text, profile, check the brief",
            CODE,
            CODE_EDGE,
            "CODE",
        ),
        ("g1", "Gate 1: was the file read faithfully?", "", GATE, GATE_EDGE, "GATE"),
        ("diag", "2. Diagnostic agent", "what is wrong + the judgment calls", AI, AI_EDGE, "AI"),
        (
            "coder",
            "3. Coder agent",
            "writes the cleaning plan (own code if needed)",
            AI,
            AI_EDGE,
            "AI",
        ),
        ("g2", "Gate 2: is the clean data as promised?", "", GATE, GATE_EDGE, "GATE"),
        (
            "review",
            "4. Reviewer agent",
            "checks the cleaning with its own counts",
            AI,
            AI_EDGE,
            "AI",
        ),
        ("plan", "5. Planner agent", "3-5 analysis questions + tools", AI, AI_EDGE, "AI"),
        ("g3", "Gate 3: is the plan valid, safe, feasible?", "", GATE, GATE_EDGE, "GATE"),
        (
            "analysis",
            "6. Analysis agent",
            "picks tools, follows up, writes findings",
            AI,
            AI_EDGE,
            "AI",
        ),
        (
            "g4",
            "Gate 4: every number traced?",
            "every question handled, nothing overclaimed",
            GATE,
            GATE_EDGE,
            "GATE",
        ),
        ("report", "7. Reports", "REPORT.pdf + detail documents", CODE, CODE_EDGE, "CODE"),
    ]
    for key, title, sub, fill, edge, tag in nodes:
        h = GATE_H if key.startswith("g") and not sub else BOX_H
        rows[key] = (y - h, h)
        box(d, MAIN_X, y - h, MAIN_W, h, fill, edge, title, sub, tag)
        y -= step if not key.startswith("g") else step - 6

    keys = [k for k, *_ in nodes]
    for upper, lower in zip(keys, keys[1:], strict=False):
        uy, _ = rows[upper]
        ly, lh = rows[lower]
        arrow(d, MAIN_X + MAIN_W / 2, uy, MAIN_X + MAIN_W / 2, ly + lh)

    def mid(key):
        y0, h = rows[key]
        return y0 + h / 2

    def top(key):
        y0, h = rows[key]
        return y0 + h

    # stops: a gate that blocks ends the run
    for key in ("g1", "g2"):
        label(
            d, MAIN_X + MAIN_W + 4, mid(key) - 2, "BLOCK = stop, report says why", colour=GATE_EDGE
        )

    # retry loops
    loop_back(d, mid("g3"), mid("plan"), "problems go back (max 2x)")
    loop_back(d, mid("g4"), mid("analysis"), "problems go back (max 2x)")
    loop_back(d, mid("review"), mid("coder"), "fixes go back (max 2x)", x=MAIN_X - 32)

    # things beside the steps
    side(
        d,
        rows["diag"][0],
        BOX_H,
        YOU,
        YOU_EDGE,
        "edit/decisions.yaml",
        "you can change any call",
        "YOU",
    )
    arrow(d, MAIN_X + MAIN_W, mid("diag"), SIDE_X, mid("diag"), YOU_EDGE)
    side(
        d,
        rows["coder"][0] - 4,
        BOX_H + 4,
        CODE,
        CODE_EDGE,
        "Executor + sandbox",
        "runs & measures each step",
        "CODE",
    )
    arrow(d, MAIN_X + MAIN_W, mid("coder"), SIDE_X, mid("coder"))
    side(
        d,
        rows["plan"][0],
        BOX_H,
        YOU,
        YOU_EDGE,
        "edit/analysis_plan.yaml",
        "you can change the plan",
        "YOU",
    )
    arrow(d, MAIN_X + MAIN_W, mid("plan"), SIDE_X, mid("plan"), YOU_EDGE)
    side(
        d,
        rows["analysis"][0] - 6,
        BOX_H + 12,
        CODE,
        CODE_EDGE,
        "11 analysis tools",
        "rates, tables, trends, regressions",
        "CODE",
    )
    arrow(d, MAIN_X + MAIN_W, mid("analysis") + 4, SIDE_X, mid("analysis") + 4)
    arrow(d, SIDE_X, mid("analysis") - 4, MAIN_X + MAIN_W, mid("analysis") - 4)
    label(d, MAIN_X + MAIN_W + 4, mid("analysis") + 8, "calls", size=6)
    label(d, MAIN_X + MAIN_W + 4, mid("analysis") - 12, "results", size=6)
    label(d, SIDE_X + SIDE_W / 2, rows["analysis"][0] - 16, "up to 3 rounds", anchor="middle")

    # you read, and may change and re-run
    report_bottom = rows["report"][0]
    label(
        d,
        MAIN_X + MAIN_W / 2,
        report_bottom - 12,
        "You read REPORT.pdf. To change the result: edit decisions.yaml or analysis_plan.yaml,",
        size=7,
        colour=YOU_EDGE,
        anchor="middle",
    )
    label(
        d,
        MAIN_X + MAIN_W / 2,
        report_bottom - 22,
        "then run that step again. The original file is never touched, so nothing is lost.",
        size=7,
        colour=YOU_EDGE,
        anchor="middle",
    )

    # what the AI sees
    label(
        d,
        4,
        30,
        "The AI never sees rows: only summaries and totals.",
        size=7.5,
        colour=AI_EDGE,
    )
    label(
        d,
        4,
        18,
        "Every change and every number is logged; the original file is never modified.",
        size=7.5,
        colour=CODE_EDGE,
    )

    # legend
    lx = 4
    for fill, edge, text in (
        (YOU, YOU_EDGE, "you"),
        (AI, AI_EDGE, "AI agent"),
        (CODE, CODE_EDGE, "plain code"),
        (GATE, GATE_EDGE, "check (gate)"),
    ):
        d.add(Rect(lx, HEIGHT - 14, 10, 8, fillColor=fill, strokeColor=edge))
        d.add(String(lx + 13, HEIGHT - 13, text, fontName="Helvetica", fontSize=7, fillColor=INK))
        lx += 70
    return d


BASE = ParagraphStyle("base", fontName="Helvetica", fontSize=9.5, leading=13, textColor=INK)
S = {
    "title": ParagraphStyle("t", parent=BASE, fontName="Helvetica-Bold", fontSize=18, leading=22),
    "sub": ParagraphStyle("s", parent=BASE, textColor=GREY, fontSize=9),
    "h1": ParagraphStyle(
        "h1",
        parent=BASE,
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6,
    ),
    "h2": ParagraphStyle(
        "h2",
        parent=BASE,
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=3,
    ),
    "body": ParagraphStyle("b", parent=BASE, spaceAfter=4),
    "cell": ParagraphStyle("c", parent=BASE, fontSize=8, leading=10.5),
    "cellb": ParagraphStyle("cb", parent=BASE, fontSize=8, leading=10.5, fontName="Helvetica-Bold"),
}


def p(text, style="body"):
    return Paragraph(text, S[style])


def grid(rows, widths, colour=CODE):
    cells = [[p(c, "cellb" if i == 0 else "cell") for c in row] for i, row in enumerate(rows)]
    t = Table(cells, colWidths=[w * 174 * mm for w in widths], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
                ("BACKGROUND", (0, 0), (-1, 0), colour),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return t


def step(number, name, kind, purpose, gets, does, gives, safeguards):
    colour = CODE_EDGE if kind == "code" else AI_EDGE
    return KeepTogether(
        [
            Paragraph(
                f'{number}. {name} <font size="8" color="{colour.hexval()}">[{kind}]</font>',
                S["h2"],
            ),
            grid(
                [
                    ["Purpose", purpose],
                    ["Gets", gets],
                    ["Does", does],
                    ["Gives", gives],
                    ["Safeguards", safeguards],
                ],
                [0.16, 0.84],
                colour=colors.white,
            ),
            Spacer(1, 4),
        ]
    )


story = [
    p("The pipeline at a glance", "title"),
    p("Auditable agentic data science pipeline - how a data file becomes a checked report", "sub"),
    Spacer(1, 8),
    flowchart(),
    PageBreak(),
    p("The idea in short", "h1"),
    p(
        "You give the pipeline a data file and a short brief: what the data is and what you want to "
        "learn. It loads and checks the file, cleans it, plans an analysis, runs it, and writes a "
        "report. <b>The AI decides; plain code does the work.</b> Five AI agents make the judgment "
        "calls (what is wrong, how to fix it, what to ask, what the results mean), while tested code "
        "loads, changes, measures and calculates. After each AI step a <b>gate</b>, a set of fixed "
        "checks in plain code, decides whether the result may go on."
    ),
    p(
        "Three rules hold everywhere: the AI never sees rows, only summaries "
        "and totals; every change and every number is logged, and the original file is never "
        "modified; and nothing is decided silently - every choice is in the report with its reasons, "
        "and you can change it and run again."
    ),
    p("Gates: what the outcomes mean", "h2"),
    grid(
        [
            ["Outcome", "Meaning"],
            ["PASS", "Everything checked out; the run continues."],
            ["FLAG", "The run continues, and the issue is listed in the report's wrap-up."],
            [
                "BLOCK",
                "The run stops at this step. For an AI step, the problems first go back to the "
                "agent to fix (at most twice).",
            ],
        ],
        [0.15, 0.85],
        colour=GATE,
    ),
    p("The steps", "h1"),
    step(
        1,
        "Ingest",
        "code",
        "Read the file exactly as it is, and describe it.",
        "Your data file (CSV, Excel, Parquet, JSON) and the brief.",
        "Loads every value as text, so nothing is silently converted. Profiles each column "
        "(type, missing values, odd values), and checks the "
        "claims in your brief (e.g. 'the ID is unique').",
        "A masked summary - the only view of the data the AI ever gets - and a fingerprint of the "
        "original data.",
        "<b>Gate 1</b>: every row accounted for, under 1% of lines unreadable, unique column "
        "names, every type recognised or flagged. No AI is involved, so a blocked file costs "
        "nothing.",
    ),
    step(
        2,
        "Diagnostic agent",
        "AI",
        "Find what is wrong with the data, and the questions the data cannot answer by itself.",
        "The masked summary and the brief.",
        "Lists <b>issues</b> (clear errors with a clear fix, e.g. 'NA' meaning missing, Danish "
        "decimal commas) and <b>judgment calls</b> (choices either way could be right, e.g. "
        "'does 999 mean unknown?'). It decides each call from your brief, taking the option that "
        "changes the least when the brief does not settle it.",
        "The diagnosis, and <b>edit/decisions.yaml</b> where you can change any decision.",
        "Fixed checks: every column covered, the target never filled in, excluded columns kept "
        "out, every needed type conversion planned. Problems go back to the agent, at most twice.",
    ),
    step(
        3,
        "Coder agent + executor",
        "AI + code",
        "Turn the diagnosis into cleaning steps, and carry them out safely.",
        "The masked summary, the diagnosis and the decisions.",
        "Writes a cleaning plan from 15 ready-made operations, or its own Python when none fits. "
        "The <b>executor</b> (code) runs each step on a copy, starting from the original data, "
        "and measures exactly what changed; AI-written code runs in a locked sandbox and may only "
        "change what it declared.",
        "The clean data, a log of every change, and a schema contract describing the clean data.",
        "<b>Gate 2</b>: every column has the promised type, the ID is still unique, the target "
        "was never filled in, rows lost under 5% pass (5-20% flag, over 20% block), and replaying "
        "the plan from the original gives the identical result.",
    ),
    step(
        4,
        "Reviewer agent",
        "AI",
        "A second opinion: is the cleaning actually right?",
        "The brief, the diagnosis, every step with its measured effect, and Gate 2's results.",
        "Asks its own counting questions of the data (only totals come back, never rows), then "
        "accepts, sends specific fixes back to the coder (at most twice), and lists questions "
        "only you can answer.",
        "A verdict, and its questions for you in the report.",
        "A claim citing a number it was never shown is thrown out; it needs two claims verified "
        "with numbers to accept. Optional (--no-review).",
    ),
    step(
        5,
        "Planner agent",
        "AI",
        "Decide what is worth finding out.",
        "Your brief (above all the goal), a totals-only summary of the clean data, the list of "
        "tools, and columns that predict the outcome suspiciously well.",
        "Writes 3-5 questions tied to your goal, each with the columns and tools to answer it; "
        "says how it read your brief; leaves out columns with reasons; marks questions this data "
        "cannot answer, with the reason.",
        "The plan, in <b>edit/analysis_plan.yaml</b>. It runs straight on; edit it and re-run "
        "to change it, and your version wins.",
        "<b>Gate 3</b>: every column and tool exists, the target is yours, no ID, free-text or "
        "excluded is analysed, a column that gives the answer away is left out, and every "
        "question has enough rows or is marked not possible.",
    ),
    step(
        6,
        "Analysis agent + tools",
        "AI + code",
        "Answer the plan's questions, and follow what is interesting.",
        "The brief, the plan, the data summary and the cleaning limitations.",
        "In up to three rounds it asks for tool calls, reads the results, and follows up on "
        "what stands out (e.g. 'Cherbourg looks good - is that the port or its passengers?'). "
        "The tools (code) compute everything. Then it writes an answer per question, findings "
        "with the tool results they rest on, its own doubts, and recommendations.",
        "The write-up, and every tool result and finding saved with references to how it was made.",
        "<b>Gate 4</b>: every plan question answered or marked not possible, every finding rests "
        "on calls that ran, every number in the text is one a tool reported (rounding allowed, "
        "numbers the AI worked out itself are not), and a model no better than no model at all "
        "cannot back a 'strong' finding.",
    ),
    step(
        7,
        "Reports",
        "code",
        "Tell the whole story in a form that is easy to read.",
        "Everything in the run folder.",
        "Builds REPORT.pdf (3-4 pages: in brief, what we found, how good the data is, the "
        "decisions made for you, and a self-critical wrap-up) and four detail documents behind "
        "it.",
        "REPORT.pdf plus details/1_analysis, 2_data_quality, 3_decisions and 4_audit.",
        "Every number comes from the logs; AI-written text is marked.",
    ),
    p("The agents side by side", "h1"),
    grid(
        [
            ["Agent", "Decides", "Writes code?", "Checked by"],
            ["Diagnostic", "What is wrong; the judgment calls", "No", "Diagnosis checks"],
            ["Coder", "How to clean it", "Sometimes, in a sandbox", "Executor + Gate 2"],
            [
                "Reviewer",
                "Whether the cleaning is right",
                "No (fixed counting queries)",
                "Its own verified numbers",
            ],
            ["Planner", "What to find out", "No", "Gate 3"],
            ["Analysis", "Which tools to run; what it means", "No (picks tools)", "Gate 4"],
        ],
        [0.17, 0.35, 0.25, 0.23],
        colour=AI,
    ),
    KeepTogether(
        [
            p("The analysis tools", "h1"),
            p(
                "The analysis agent cannot calculate anything itself; it chooses from these tools, and "
                "every result is aggregated (groups under 10 rows are hidden)."
            ),
            grid(
                [
                    ["Tool", "Answers"],
                    ["summary_table", "The basic numbers for every column"],
                    ["distribution", "How one column's values are spread"],
                    [
                        "missingness",
                        "Where values are missing, and whether that relates to the outcome",
                    ],
                    ["correlation", "Which numbers move together"],
                    ["target_by_group", "The outcome per group or range, with 95% intervals"],
                    [
                        "target_by_two_groups",
                        "The outcome per combination of two columns, e.g. sex within class",
                    ],
                    [
                        "target_by_ratio",
                        "The outcome by one number relative to another, e.g. loan / income",
                    ],
                    ["trend_over_time", "The outcome per year, quarter or month"],
                    ["compare_by_target", "How a number differs between the two outcomes"],
                    ["linear_regression", "Effects on a number outcome, the others held fixed"],
                    ["logistic_regression", "Effects on a yes/no outcome, the others held fixed"],
                    [
                        "subset / cuts",
                        "Any tool on one subgroup only; your own ranges, e.g. under 16",
                    ],
                ],
                [0.3, 0.7],
            ),
        ]
    ),
    p("What a run leaves behind", "h1"),
    grid(
        [
            ["Where", "What"],
            ["REPORT.pdf", "Start here: the essentials in 3-4 pages"],
            ["details/", "One document per section, the Excel column overview, the charts"],
            ["edit/", "decisions.yaml and analysis_plan.yaml: change these, then re-run"],
            ["data/", "The clean data, the removed rows and the unreadable lines"],
            ["logs/", "The machine-readable audit trail: every AI attempt, step, check and result"],
        ],
        [0.2, 0.8],
    ),
]

document = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=18 * mm,
    rightMargin=18 * mm,
    topMargin=14 * mm,
    bottomMargin=14 * mm,
    title="The pipeline at a glance",
)
document.build(story)
print(f"Wrote {OUT}")
