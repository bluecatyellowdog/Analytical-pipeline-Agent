"""details/0_technical_summary.pdf: the whole run in a few pages, for a technical reader.

REPORT.pdf and REPORT.html say the same in plain words; this keeps the exact
answers, effect sizes and every point of the wrap-up.

    In brief                  the question, the answer in a few lines, the status
    1  What we found          each plan question: its answer, its findings, one chart
    2  How good the data is   rows, problems fixed, what is still missing, the checks
    3  Decisions made for you every judgment call: what was decided, how sure
    4  Wrap-up                what might be wrong, and what we recommend
    More detail               where each section continues

Built only from the run folder. Every number comes from a log; text written by
an AI is marked, and was checked by its gate before it got here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Spacer

from pipeline.layout import (
    ANALYSIS_PDF,
    AUDIT_PDF,
    DATA_COLUMNS_XLSX,
    DATA_QUALITY_PDF,
    DECISIONS_PDF,
    REPORT_PDF,
    run_file,
)
from pipeline.reports.common import (
    STATUS_WORDS,
    E,
    M,
    P,
    Run,
    bullets,
    chart,
    decision_rank,
    link,
    table,
    tagged_bullets,
)

STRENGTH_ORDER = {"strong": 0, "moderate": 1, "weak": 2, "none": 3}
MAIN_CHART_WIDTH = 115 * mm
MAX_POINTS = 10
MISSING_WORTH_SAYING = 0.10

ISSUE_WORDS = {
    "type": "values stored as the wrong type",
    "missing_tokens": "codes for missing values",
    "sentinel": "coded values like 999",
    "decimal_format": "number formats",
    "date_format": "date formats",
    "whitespace": "stray spaces",
    "case": "inconsistent capitals",
    "categories": "inconsistent categories",
    "duplicates": "duplicate rows",
    "outliers": "extreme values",
    "invalid_values": "impossible values",
    "identifier": "identifier problems",
    "personal_data": "personal data",
}
STEP_NAMES = {
    "ingestion": "loading",
    "diagnosis": "diagnosis",
    "cleaning": "cleaning",
    "review": "review",
    "planning": "planning",
    "analysis": "analysis",
}
STOP_ADVICE = {
    "ingestion": "Fix the problems listed in details/2_data_quality.pdf (section 1) and run again.",
    "diagnosis": "See logs/diagnostic_log.json for what went wrong, then run again.",
    "cleaning": "See details/2_data_quality.pdf (section 4); change a decision in "
    "edit/decisions.yaml if needed and clean again.",
    "planning": "See logs/gate_3.json; edit edit/analysis_plan.yaml, or delete it, and run "
    "the analysis again.",
    "analysis": "See logs/gate_4.json and logs/analysis_log.json, then run the analysis again.",
}
GATE_ADVICE = {
    1: "See details/2_data_quality.pdf, section 1.",
    2: "See details/2_data_quality.pdf, section 4.",
    3: "See the plan in details/1_analysis.pdf.",
    4: "See details/1_analysis.pdf.",
}


@dataclass
class Point:
    """One line of the wrap-up: something that might be wrong, and what to do about it."""

    doubt: str | None
    advice: str | None = None
    ai: bool = False


def technical_document(run: Run) -> list:
    points = wrap_up_points(run)
    return [
        *_header(run),
        *_in_brief(run, points),
        *_found(run),
        *_quality(run),
        *_decisions(run),
        *_wrap_up(points),
        *_more(run),
    ]


def technical_title(run: Run) -> str:
    return f"Technical summary - {Path(run.file_name).stem}"


# -- the parts ------------------------------------------------------------------


def _more_line(name: str, what: str) -> M:
    """This document lives in details/, next to the ones it points to."""
    return M(f"More: {link(name, name)} - {E(what)}", "note")


def _header(run: Run) -> list:
    rows = run.load.get("n_loaded", 0)
    run_id = run.manifest.get("run_id", "")
    return [
        P(technical_title(run), "title"),
        M(
            f"{E(run.file_name)}  -  {rows:,} rows  -  run {E(run_id)}  -  "
            f"{datetime.now():%Y-%m-%d %H:%M}  -  the plain version is "
            f"{link(f'../{REPORT_PDF}', REPORT_PDF)}",
            "subtitle",
        ),
        Spacer(1, 6),
    ]


def _why_no_analysis(run: Run) -> str:
    stages = run.stages
    if stages.get("analysis") == "blocked":
        return "The analysis ran, but its write-up did not pass Gate 4; section 4 says what to do."
    if stages.get("planning") == "blocked":
        return "No analysis plan passed Gate 3, so the analysis did not run."
    if stages.get("cleaning") == "blocked":
        return "Gate 2 held the clean data back, so there was nothing to analyse."
    if stages.get("ingestion") == "blocked":
        return "The file was not read faithfully (Gate 1), so the run stopped before cleaning."
    return (
        "The analysis has not run yet. To run it: "
        f"uv run python -m pipeline.analyze {run.dir.as_posix()}"
    )


def _status(run: Run, points: list[Point]) -> str:
    gates = [
        (n, g) for n, g in ((1, run.gate_1), (2, run.gate_2), (3, run.gate_3), (4, run.gate_4)) if g
    ]
    stopped = [STEP_NAMES.get(s, s) for s, v in run.stages.items() if v == "blocked"]
    blocked = [str(n) for n, g in gates if g.get("outcome") == "block"]
    flagged = [str(n) for n, g in gates if g.get("outcome") == "flag"]
    parts = []
    if stopped:
        parts.append(f"stopped at the {stopped[0]} step")
    if blocked:
        parts.append(f"Gate {', '.join(blocked)} blocked")
    if flagged:
        parts.append(f"Gate {', '.join(flagged)} raised a warning")
    if not parts:
        parts.append(f"all {len(gates)} checks passed")
    doubts = sum(1 for p in points if p.doubt)
    if doubts:
        parts.append(f"{doubts} thing(s) worth a look in section 4")
    return "; ".join(parts) + "."


def _in_brief(run: Run, points: list[Point]) -> list:
    brief = run.manifest.get("context_brief") or {}
    story = [P("In brief", "h1")]
    if brief.get("goal"):
        story.append(M(f"<b>Your question:</b> {E(brief['goal'])}"))
    up = (run.analysis or {}).get("write_up")
    if up and up.get("findings"):
        top = sorted(up["findings"], key=lambda f: STRENGTH_ORDER.get(f["strength"], 9))[:5]
        story.append(
            M(
                '<b>The answer, in short</b> <font size="7" color="#57606a">(written by the AI; '
                "every number checked against the tool results)</font>"
            )
        )
        story += bullets([f["headline"] for f in top])
    else:
        story.append(P(_why_no_analysis(run)))
    story.append(M(f"<b>Status:</b> {E(_status(run, points))}"))
    return story


def _found(run: Run) -> list:
    story = [P("1  What we found", "h1")]
    if not run.analysis:
        return [*story, P(_why_no_analysis(run))]
    up, plan = run.analysis["write_up"], run.analysis["plan"]
    calls = {c["call_id"]: c for c in run.analysis["calls"]}
    findings = {f["id"]: f for f in up["findings"]}
    answers = {a["question_id"]: a for a in up["answers"]}
    story.append(
        P(
            "Answers and findings are written by the AI from the tool results; Gate 4 checked "
            "that every number in them is one a tool reported.",
            "note",
        )
    )
    used: set[str] = set()
    charted: set[str] = set()
    for question in sorted(plan["questions"], key=lambda q: q.get("priority", 5)):
        answer = answers.get(question["id"])
        if answer is None:
            continue
        block = [
            M(f"<b>{E(question['question'])}</b>"),
            M(f"<i>{E(STATUS_WORDS[answer['status']])}.</i> {E(answer['answer'])}"),
        ]
        cited = [findings[i] for i in answer.get("finding_ids", []) if i in findings]
        used |= {f["id"] for f in cited}
        block += bullets([f"{f['headline']} ({f['strength']})" for f in cited])
        if answer.get("reason"):
            block.append(P(f"Why not fully: {answer['reason']}", "note"))
        image = _first_chart(run, cited, calls, charted)
        if image is not None:
            block.append(image)
        block.append(Spacer(1, 4))
        story.append(KeepTogether(block))
    extra = [f for f in up["findings"] if f["id"] not in used]
    if extra:
        story += [
            M("<b>Also worth knowing</b>"),
            *bullets([f"{f['headline']} ({f['strength']})" for f in extra]),
        ]
    story.append(_more_line(ANALYSIS_PDF, "every finding with all its charts and tool results"))
    return story


def _first_chart(run: Run, cited: list[dict], calls: dict, charted: set[str]):
    for finding in cited:
        for call_id in finding["evidence"]:
            call = calls.get(call_id)
            if call is None or call_id in charted:
                continue
            for artifact_id in call.get("artifact_ids", []):
                image = chart(run, run.artifacts.get(artifact_id, {}), MAIN_CHART_WIDTH)
                if image is not None:
                    charted.add(call_id)
                    return image
    return None


def _quality(run: Run) -> list:
    load, log = run.load, run.log or {}
    records = run.records
    rows_clean = log.get("rows_clean")
    dropped = sum(r.get("rows_dropped", 0) for r in records)
    rejected = load.get("n_rejected", 0)
    facts = [
        [
            "Rows",
            f"{load.get('n_source_rows', 0):,} in the file, {load.get('n_loaded', 0):,} read, "
            f"{rejected:,} could not be read"
            + (
                f"; {rows_clean:,} after cleaning ({dropped:,} removed)"
                if rows_clean is not None
                else ""
            ),
        ]
    ]
    if run.diagnosis:
        kinds = Counter(i.kind for i in run.diagnosis.issues)
        facts.append(
            [
                "Problems found",
                f"{len(run.diagnosis.issues)}"
                + (
                    ": "
                    + ", ".join(f"{ISSUE_WORDS.get(k, k)} ({n})" for k, n in kinds.most_common())
                    if kinds
                    else ""
                ),
            ]
        )
    if records:
        changed = sum(sum(r.get("cells_changed", {}).values()) for r in records)
        nulls = sum(sum(r.get("new_nulls", {}).values()) for r in records)
        converted = sum(
            1
            for r in records
            for c in r.get("target_columns", [])
            if c in r.get("before", {})
            and c in r.get("after", {})
            and r["before"][c]["dtype"] != r["after"][c]["dtype"]
        )
        facts.append(
            [
                "Values changed",
                f"{changed:,} value(s) changed; {converted} column(s) converted to the right "
                f"type; {nulls:,} value(s) turned into missing",
            ]
        )
    missing = missing_columns(run)
    if run.handoff:
        facts.append(
            [
                "Still missing",
                ", ".join(f"{name} {rate:.1%}" for name, rate in missing[:5])
                or "nothing: every value is present",
            ]
        )
    checks = [
        f"loading (Gate 1): {(run.gate_1 or {}).get('outcome', 'not run')}",
        f"cleaning (Gate 2): {(run.gate_2 or {}).get('outcome', 'not run')}",
        f"review: {run.stages.get('review', 'not run')}",
    ]
    facts.append(["Checks", "; ".join(checks)])

    meaning = []
    total = rows_clean or 0
    for name, rate in missing:
        if rate >= MISSING_WORTH_SAYING and total:
            known = round(total * (1 - rate))
            meaning.append(
                f"Results that use {name} rest on the {known:,} rows where it is known "
                f"(of {total:,})."
            )
    if dropped:
        meaning.append(f"The {dropped:,} row(s) removed while cleaning are in no result.")
    if rejected:
        meaning.append(f"The {rejected:,} line(s) that could not be read are in no result.")
    story = [
        P("2  How good the data is", "h1"),
        table(facts, [0.2, 0.8], header=False),
        M("<b>What this means for the findings</b>"),
        *(bullets(meaning) or [P("Nothing in the data itself limits the findings.")]),
        _more_line(DATA_QUALITY_PDF, "the data as received, every problem, change and check"),
        M(f"Column by column in Excel: {link(DATA_COLUMNS_XLSX, DATA_COLUMNS_XLSX)}", "note"),
    ]
    return story


def missing_columns(run: Run) -> list[tuple[str, float]]:
    """Columns of the clean data with missing values, most missing first."""
    if not run.handoff:
        return []
    columns = [
        (c["name"], c.get("null_rate", 0.0))
        for c in run.handoff.get("schema", [])
        if c.get("null_rate", 0.0) > 0 and c.get("role") not in ("identifier", "excluded")
    ]
    return sorted(columns, key=lambda c: -c[1])


def _decisions(run: Run) -> list:
    story = [P("3  Decisions made for you", "h1")]
    calls = run.diagnosis.judgment_calls if run.diagnosis else []
    if not calls:
        return [*story, P("None: the data raised no judgment calls.")]
    rows = [["", "Question", "Decided", "By", "How sure"]]
    for call in sorted(calls, key=lambda c: decision_rank(run, c)):
        decision = run.decisions.get(call.id)
        ai = call.ai_decision
        if decision is None or not decision.decided:
            decided, by, sure = "left as it is (open)", "nobody", "-"
        elif decision.decided_by == "user":
            decided, by, sure = decision.choice, "you", "-"
        else:
            decided, by = decision.choice, "the AI"
            sure = ai.confidence if ai else "-"
            if sure == "low":
                sure = "low: check it"
        rows.append([call.id, call.question, decided, by, sure])
    story += [
        table(rows, [0.06, 0.44, 0.26, 0.1, 0.14]),
        P(
            "To change one: edit edit/decisions.yaml, then run the cleaning and the analysis "
            "again.",
            "body",
        ),
        _more_line(DECISIONS_PDF, "every decision with its reasons, alternatives and effect"),
    ]
    return story


def _wrap_up(points: list[Point]) -> list:
    doubts = [(p.doubt, p.ai) for p in points if p.doubt]
    advice = list(dict.fromkeys((p.advice, p.ai) for p in points if p.advice))
    story = [
        P("4  Wrap-up: what might be wrong, and what we recommend", "h1"),
        P(
            "Gathered from the checks, the decisions and the review, plus the analysis's own "
            "doubts (marked AI). It is what to keep in mind, and where to look.",
            "note",
        ),
        P("What might be wrong", "h2"),
        *(tagged_bullets(doubts[:MAX_POINTS]) or [P("Nothing stood out.")]),
    ]
    if len(doubts) > MAX_POINTS:
        story.append(P(f"...and {len(doubts) - MAX_POINTS} more in the detail documents.", "note"))
    story += [
        P("What we recommend", "h2"),
        *(tagged_bullets(advice[:MAX_POINTS]) or [P("Nothing to do.")]),
    ]
    return story


def _more(run: Run) -> list:
    documents = [
        (ANALYSIS_PDF, "every finding with all its charts and tool results; the plan"),
        (DATA_QUALITY_PDF, "the data as received, every problem, every change, the review"),
        (DATA_COLUMNS_XLSX, "the data column by column, in Excel"),
        (DECISIONS_PDF, "every decision with its reasons, alternatives and measured effect"),
        (AUDIT_PDF, "models, costs, prompt versions and every file"),
    ]
    rows = [[M(link(name, f"details/{name}"), "cell"), what] for name, what in documents]
    data = ["data/clean_data.csv and .parquet: the cleaned data"]
    if run_file(run.dir, "dropped_rows.csv").exists():
        data.append("data/dropped_rows.csv: the rows removed while cleaning, with reasons")
    if run.load.get("n_rejected", 0):
        data.append("data/rejected_lines.csv: the lines that could not be read")
    return [
        P("More detail", "h1"),
        table(rows, [0.32, 0.68], header=False),
        M("<b>To change the result</b>"),
        *bullets(
            [
                "edit/decisions.yaml: the cleaning decisions, then run pipeline.clean",
                "edit/analysis_plan.yaml: the analysis questions, then run pipeline.analyze",
            ]
        ),
        M("<b>The data</b>"),
        *bullets(data),
        P("logs/ holds the machine-readable audit trail.", "note"),
    ]


# -- the wrap-up ------------------------------------------------------------------


def wrap_up_points(run: Run) -> list[Point]:
    """What might be wrong and what to do, from the logs first and the AI's doubts last."""
    points: list[Point] = []
    for stage, status in run.stages.items():
        if status == "blocked":
            points.append(
                Point(
                    f"The {STEP_NAMES.get(stage, stage)} step stopped, so the steps after it "
                    f"did not run.",
                    STOP_ADVICE.get(stage),
                )
            )
    for claim in run.claims:
        if claim.get("status") == "fail":
            text = claim.get("claim", {}).get("text") or claim.get("claim", {}).get("kind")
            points.append(
                Point(
                    f"Your brief says '{text}', but the data disagrees: {claim.get('message')}",
                    "Check the brief and the data source; correct whichever is wrong.",
                )
            )
    rejected = run.load.get("n_rejected", 0)
    if rejected:
        points.append(
            Point(
                f"{rejected:,} line(s) of the file could not be read, so they are in no result.",
                "See data/rejected_lines.csv for each line and why.",
            )
        )
    if run.diagnosis:
        for call in sorted(run.diagnosis.judgment_calls, key=lambda c: decision_rank(run, c)):
            decision = run.decisions.get(call.id)
            ai = call.ai_decision
            if decision is None or not decision.decided:
                points.append(
                    Point(
                        f"{call.id} is undecided, so its data was left as it is: {call.question}",
                        f"Decide {call.id} in edit/decisions.yaml, then clean and analyse again.",
                    )
                )
            elif decision.decided_by == "ai" and ai is not None and ai.confidence == "low":
                points.append(
                    Point(
                        f"{call.id} was decided by the AI with low confidence: '{ai.choice}'. "
                        f"{call.question}",
                        f"Check {call.id}; if it is wrong, change it in edit/decisions.yaml, "
                        f"then clean and analyse again.",
                    )
                )
    dropped = sum(r.get("rows_dropped", 0) for r in run.records)
    if dropped:
        points.append(
            Point(
                f"{dropped:,} row(s) were removed while cleaning.",
                "See data/dropped_rows.csv: every removed row with its reason.",
            )
        )
    points += _review_points(run)
    for number, gate in ((1, run.gate_1), (2, run.gate_2), (3, run.gate_3), (4, run.gate_4)):
        for check in (gate or {}).get("checks", []):
            if check.get("passed") or check.get("severity") != "warn":
                continue
            for problem in check.get("observed", {}).get("problems") or [check.get("message")]:
                points.append(Point(f"Gate {number}: {problem}", GATE_ADVICE[number]))
    for name, rate in missing_columns(run):
        if rate >= MISSING_WORTH_SAYING:
            points.append(
                Point(
                    f"{name} is missing for {rate:.1%} of rows, so results that use it rest on "
                    f"fewer rows, and the missing ones may differ from the rest."
                )
            )
    up = (run.analysis or {}).get("write_up") or {}
    for answer in up.get("answers", []):
        if answer["status"] in ("partly", "not_answered") and answer.get("reason"):
            points.append(
                Point(
                    f"{answer['question_id']} was {STATUS_WORDS[answer['status']]}: "
                    f"{answer['reason']}",
                    ai=True,
                )
            )
    for finding in up.get("findings", []):
        if finding["strength"] == "weak":
            points.append(Point(f"'{finding['headline']}' rests on weak evidence.", ai=True))
    points += [Point(text, ai=True) for text in up.get("doubts", [])]
    points += [Point(None, text, ai=True) for text in up.get("recommendations", [])]
    points += [
        Point(None, f"Worth asking next: {q}", ai=True) for q in up.get("next_questions", [])
    ]
    return points


def _review_points(run: Run) -> list[Point]:
    review = run.review or {}
    points = [Point(note) for note in review.get("notes", [])]
    if not review.get("rounds"):
        return points
    final = review["rounds"][-1]
    advice = "Answer it in the brief (guidance or column notes) and run again."
    for ambiguity in final["outcome"].get("ambiguities", []):
        where = f" ({ambiguity['column']})" if ambiguity.get("column") else ""
        points.append(Point(f"The reviewer asked{where}: {ambiguity['question']}", advice, ai=True))
    for finding in final["outcome"].get("findings", []):
        if not finding.get("fix"):
            points.append(
                Point(
                    f"The reviewer noted ({finding.get('column') or 'table'}): "
                    f"{finding['problem']}",
                    ai=True,
                )
            )
    if not final["accepted"]:
        for fix in final["verdict"].get("fix_requests", []):
            points.append(
                Point(
                    f"Still open from the review ({fix.get('column') or 'table'}): {fix['issue']}",
                    "See details/2_data_quality.pdf, section 5.",
                    ai=True,
                )
            )
    if final.get("weak"):
        points.append(
            Point(
                "The reviewer could not verify enough with numbers, so its approval carries "
                "little weight."
            )
        )
    return points
