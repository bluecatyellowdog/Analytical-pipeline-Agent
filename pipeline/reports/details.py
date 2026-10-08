"""The detail documents: one per section of REPORT.pdf, for whoever wants to dig deeper.

details/1_analysis.pdf      every finding with its charts and tool results, and the plan
details/2_data_quality.pdf  the data as received, what was wrong, every change, the checks,
                            the review
details/3_decisions.pdf     every judgment call: the options, the decision, why, what it changed
details/4_audit.pdf         models, costs, prompt versions and the files in the run folder
"""

from __future__ import annotations

import json
from typing import Any

from reportlab.platypus import KeepTogether, PageBreak, Spacer

from pipeline.layout import CHARTS, DATA, DECISIONS_PDF, DETAILS, EDIT, LOGS
from pipeline.reports.common import (
    MAX_QUERY_ANSWER_LINES,
    MAX_REJECTS,
    STATUS_WORDS,
    E,
    M,
    P,
    Run,
    ai_text,
    bullets,
    chart,
    checks_table,
    code_block,
    decision_rank,
    gate_2_meaning,
    measured_effect,
    outcome,
    strip_break,
    table,
    title_block,
)

OLD_NAMES = {"diagnostic": "analyst"}
"""Agent names before a rename, so older runs still show their models."""


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
        story += [P("Gate 1 checks", "h2"), checks_table(run.gate_1)]
    return story


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
        story.append(
            P(
                f"There were also {len(diagnosis.judgment_calls)} judgment call(s), questions the "
                f"data cannot answer by itself. They are in {DECISIONS_PDF}.",
                "body",
            )
        )

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
            f"{E(gate_2_meaning(run.gate_2.get('outcome')))}"
        ),
        checks_table(run.gate_2),
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
        P("5. The review", "h1"),
        P(
            "A second AI checked whether the cleaning was right, using its own counting queries "
            "on the data (never rows).",
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


def _finding(run: Run, finding: dict[str, Any], calls: dict, shown: set[str]) -> list:
    """One finding: the AI's words, then the tool results and charts it rests on."""
    story = [
        M(f"<b>{E(finding['headline'])}</b> ({E(finding['strength'])})"),
        ai_text(finding["detail"]),
    ]
    story += bullets([f"Caveat: {caveat}" for caveat in finding.get("caveats", [])])
    for call_id in finding["evidence"]:
        call = calls.get(call_id)
        if call is None:
            continue
        story.append(P(f"{call_id} {call['tool']}: {call['summary']}", "note"))
        if call_id in shown:
            continue
        shown.add(call_id)
        for artifact_id in call.get("artifact_ids", []):
            image = chart(run, run.artifacts.get(artifact_id, {}))
            if image is not None:
                story.append(image)
    return story


def _findings(run: Run) -> list:
    """What the analysis found, question by question, with its evidence beside it."""
    if not run.analysis:
        return []
    up, plan = run.analysis["write_up"], run.analysis["plan"]
    calls = {c["call_id"]: c for c in run.analysis["calls"]}
    findings = {f["id"]: f for f in up["findings"]}
    story = [
        P(
            "Written by the AI from the tool results, which are shown under each finding (the "
            "lines starting T1, T2, ...). Gate 4 checked that every number in the AI's text is "
            "one a tool reported.",
            "note",
        ),
        ai_text(up["summary"]),
    ]
    answers = {a["question_id"]: a for a in up["answers"]}
    shown: set[str] = set()
    used: set[str] = set()
    for question in sorted(plan["questions"], key=lambda q: q.get("priority", 5)):
        answer = answers.get(question["id"])
        if answer is None:
            continue
        story += [
            P(f"{question['id']}. {question['question']}", "h2"),
            M(f"<b>{E(STATUS_WORDS[answer['status']])}.</b> {E(answer['answer'])}"),
        ]
        if answer.get("reason"):
            story.append(P(f"Why: {answer['reason']}", "note"))
        for finding_id in answer.get("finding_ids", []):
            used.add(finding_id)
            story += _finding(run, findings[finding_id], calls, shown)
    extra = [f for f in up["findings"] if f["id"] not in used]
    if extra:
        story.append(P("Also worth knowing", "h2"))
        for finding in extra:
            story += _finding(run, finding, calls, shown)
    if up.get("limitations"):
        story += [P("Limitations", "h2"), *bullets(up["limitations"])]
    if up.get("next_questions"):
        story += [P("Worth looking at next", "h2"), *bullets(up["next_questions"])]
    if run.gate_4:
        story += [P("Gate 4 checks", "h2"), checks_table(run.gate_4)]
    return story


def _plan(run: Run) -> list:
    plan = run.analysis_plan
    if not plan and not run.gate_3:
        return []
    edited = (plan or {}).get("approved_by") == "user"
    story = [
        PageBreak(),
        P("The plan behind the analysis", "h1"),
        P(
            "Written by the AI from your brief and a summary of the clean data (never rows), "
            + (
                "then edited by you and checked by Gate 3."
                if edited
                else "checked by Gate 3 and approved automatically. To change it, edit "
                "edit/analysis_plan.yaml and run the analysis again: your version wins."
            ),
            "note",
        ),
    ]
    if edited:
        story += [P("What you changed:", "body"), *bullets(plan.get("human_edits", []))]
    if plan:
        story.append(ai_text(plan.get("objective", "")))
        if plan.get("interpretation"):
            story += [P("How the brief was read:", "body"), ai_text(plan["interpretation"])]
        rows = [["#", "Question", "Tools", "Status"]]
        for q in sorted(plan.get("questions", []), key=lambda q: q.get("priority", 5)):
            status = (
                "planned"
                if q.get("feasible", True)
                else f"not possible: {q.get('infeasible_reason')}"
            )
            rows.append([q["id"], q["question"], ", ".join(q.get("methods", [])) or "-", status])
        story += [P("Questions", "h2"), table(rows, [0.06, 0.5, 0.2, 0.24])]
        excluded = plan.get("excluded_features", [])
        if excluded:
            story += [
                P("Left out of the analysis", "h2"),
                *bullets([f"{e['column']} ({e['source']}): {e['reason']}" for e in excluded]),
            ]
        if plan.get("limitations"):
            story += [P("Limitations", "h2"), *bullets(plan["limitations"])]
    if run.gate_3:
        story += [P("Gate 3 checks", "h2"), checks_table(run.gate_3)]
    return story


def _audit(run: Run) -> list:
    manifest = run.manifest
    costs = run.costs()
    rows = [["Agent", "Model", "Prompt version", "Calls", "Cost"]]
    calls = {
        "diagnostic": len((run.diagnostic_log or {}).get("attempts", [])),
        "coder": len((run.coder_log or {}).get("attempts", [])),
        "reviewer": sum(len(r.get("attempts", [])) for r in (run.review or {}).get("rounds", [])),
        "planner": len((run.planner_log or {}).get("attempts", [])),
        "analysis": len((run.analysis_log or {}).get("attempts", [])),
    }
    for agent in ("diagnostic", "coder", "reviewer", "planner", "analysis"):
        models = manifest.get("models", {})
        model = (models.get(agent) or models.get(OLD_NAMES.get(agent, ""), {})).get("name", "-")
        versions = manifest.get("prompt_versions", {})
        version = versions.get(agent) or versions.get(OLD_NAMES.get(agent, ""), "-")
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
    folders = [["Folder", "Files"]]
    for folder in (run.dir, *(run.dir / name for name in (DETAILS, EDIT, DATA, LOGS, CHARTS))):
        if folder.is_dir():
            names = sorted(p.name for p in folder.iterdir() if p.is_file())
            label = folder.relative_to(run.dir).as_posix() if folder != run.dir else "(top)"
            folders.append([label, ", ".join(names) or "-"])
    return [
        table(facts, [0.25, 0.75], header=False),
        P("Models", "h2"),
        table(rows, [0.14, 0.36, 0.18, 0.12, 0.2]),
        P("Files in the run folder", "h2"),
        table(folders, [0.18, 0.82]),
        Spacer(1, 6),
        P(
            "Every number in this report comes from the run's logs. Text marked 'AI-written' was "
            "written by a language model and is shown as it wrote it. The original data was never "
            "modified, and no AI saw a row.",
            "note",
        ),
    ]


def _decisions(run: Run) -> list:
    """Every judgment call: the options, the decision and by whom, why, and what it changed."""
    diagnosis = run.diagnosis
    if diagnosis is None or not diagnosis.judgment_calls:
        return [P("There were no judgment calls in this run.")]
    story = []
    for call in sorted(diagnosis.judgment_calls, key=lambda c: decision_rank(run, c)):
        decision = run.decisions.get(call.id)
        ai = call.ai_decision
        block = [P(f"{call.id} ({call.column or 'table'}): {call.question}", "h2")]
        block += bullets([f"Option: {option}" for option in call.options])
        if decision is not None and decision.decided and decision.decided_by == "user":
            block.append(M(f"<b>Decided by you:</b> {E(decision.choice)}"))
        elif decision is not None and decision.decided and ai is not None:
            caution = (
                " The brief does not settle this, so the option that changes the least was taken."
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
        block.append(P(measured_effect(run, call.id, call.column), "body"))
        block.append(Spacer(1, 4))
        story.append(KeepTogether(block))
    folder = run.dir.as_posix()
    story += [
        P("How to change a decision", "h2"),
        *bullets(
            [
                "Open edit/decisions.yaml and replace the decision's 'decision:' line with another "
                "option, your own words, or 'undecided' (which leaves that data untouched).",
                f"Clean again: uv run python -m pipeline.clean {folder}",
                f"Then update the findings: uv run python -m pipeline.analyze {folder}",
                "The cleaning always starts again from the original file, so nothing is lost.",
            ]
        ),
    ]
    return story


# -- the documents ----------------------------------------------------------------


def analysis_document(run: Run) -> list:
    body = _findings(run) + _plan(run)
    return title_block(
        run,
        "What the analysis found",
        "Every finding with the tool results and charts behind it, then the plan the analysis "
        "followed. The short version is in REPORT.pdf, section 1.",
    ) + (strip_break(body) or [P("The analysis has not run for this data.")])


def data_quality_document(run: Run) -> list:
    body = _received(run) + _wrong(run) + _changed(run) + _checks(run) + _review(run)
    return title_block(
        run,
        "How good the data is",
        "The data as it arrived, what was wrong with it, every change made (with any code the AI "
        "wrote), the checks on the clean data and the review. The short version is in "
        "REPORT.pdf, section 2.",
    ) + strip_break(body)


def decisions_document(run: Run) -> list:
    return title_block(
        run,
        "Decisions made for you",
        "A judgment call is a cleaning question the data cannot answer by itself, such as "
        "whether 999 means 'unknown'. Each one below shows the options, what was decided and by "
        "whom, why, and what it changed in the data. Least certain first. The short version is "
        "in REPORT.pdf, section 3.",
    ) + _decisions(run)


def audit_document(run: Run) -> list:
    return title_block(
        run,
        "Audit",
        "Which models ran, at what cost, with which prompt versions, and every file in the run "
        "folder. The machine-readable logs are in logs/.",
    ) + strip_break(_audit(run))
