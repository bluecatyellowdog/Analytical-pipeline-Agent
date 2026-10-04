"""The reviewer agent: judges whether a cleaning is right, with the numbers to show it.

Gate 2 proves a cleaning is accounted for. The reviewer looks for what is wrong
anyway: a decision carried out with the wrong scope, dates parsed with day and
month swapped, categories merged that should not be, problems left in the clean
data. It reads an evidence pack (the plan, what each step actually changed,
Gate 2, the schema contract, masked profiles of raw and clean data), may ask
aggregate questions through `pipeline/tools/queries.py`, and replies with a
verdict.

The verdict is checked before it counts. Every claim must quote a number that
appears in the evidence it was shown; claims that do not are dropped, which
catches invented evidence (though not a wrong reading of a real number). An
accept needs at least two verified checks. `accepted` is derived from whether
any finding carries a fix, never taken from the model. Ambiguities are reported
to the user, never decided.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pipeline.agents.coder import decisions_block
from pipeline.agents.common import (
    MAX_RETRIES,
    Attempt,
    ask,
    brief_block,
    compact_validation_error,
    extract_json_object,
    short_hash,
    total_cost,
)
from pipeline.agents.llm import LLMClient, Message
from pipeline.contracts.audit import ColumnStats, GateResult, OpRecord
from pipeline.contracts.cleaning import CleaningPlan, Decision
from pipeline.contracts.common import Flag
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.diagnosis import Diagnosis
from pipeline.contracts.handoff import ColumnSchema
from pipeline.contracts.plan import ReviewVerdict
from pipeline.contracts.review import ReviewOutcome, ReviewReply
from pipeline.ingestion.pii import mask_shape
from pipeline.ops import op_names
from pipeline.tools.queries import QueryContext, QueryError, run_query

PROMPT_PATH = Path(__file__).parent / "prompts" / "reviewer.md"
MAX_QUERY_ROUNDS = 2
MIN_VERIFICATIONS = 2

_NUMBER = re.compile(r"\d+(?:[.,:/-]\d+)*")


class ReviewerError(RuntimeError):
    """No usable verdict within the allowed retries."""

    def __init__(self, message: str, attempts: list[Attempt], queries: list[QueryRecord]):
        super().__init__(message)
        self.attempts = attempts
        self.queries = queries


@dataclass
class QueryRecord:
    round: int
    query: dict[str, Any]
    answer: str


@dataclass
class ReviewInputs:
    brief: ContextBrief
    diagnosis: Diagnosis
    decisions: list[Decision]
    plan: CleaningPlan
    records: list[OpRecord]
    gate: GateResult
    contract: list[ColumnSchema]
    profile_raw: dict[str, Any]
    profile_clean: dict[str, Any]
    masked: frozenset[str] = frozenset()


@dataclass
class ReviewResult:
    round: int
    outcome: ReviewOutcome
    """The verdict as checked: unsupported claims removed, `accepted` derived."""
    verdict: ReviewVerdict
    """The official contract Stage 3 reads."""
    queries: list[QueryRecord]
    dropped: list[str]
    """Claims removed from the final verdict, and why."""
    attempts: list[Attempt]
    prompt_version: str
    weak: bool = False
    """True when it could not verify enough to stand behind an accept."""
    notes: list[str] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return total_cost(self.attempts)


# -- prompt -----------------------------------------------------------------


def system_prompt() -> str:
    schema = json.dumps(ReviewReply.model_json_schema(), indent=1)
    return PROMPT_PATH.read_text(encoding="utf-8").replace("{schema}", schema)


def prompt_version() -> str:
    return short_hash(system_prompt())


def masked_from_profile(profile_llm: dict[str, Any]) -> frozenset[str]:
    """Columns the compact profile shows only as shapes: personal data and identifiers."""
    return frozenset(
        entry["name"]
        for entry in profile_llm.get("columns", [])
        if entry.get("pii") or entry.get("masked")
    )


def build_messages(inputs: ReviewInputs) -> list[Message]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": evidence_pack(inputs)},
    ]


def evidence_pack(inputs: ReviewInputs) -> str:
    """Everything the reviewer may see. Built from logs and masked profiles, never rows."""
    diagnosis = json.dumps(inputs.diagnosis.model_dump(mode="json"), indent=1, ensure_ascii=False)
    gate = inputs.gate
    gate_lines = [
        f"- {'pass' if c.passed else 'FAIL (' + c.severity + ')'} {c.name}: {c.message}"
        for c in gate.checks
    ]
    contract_lines = [
        f"- {c.name}: {c.dtype} (expected {c.expected_type or 'none'}, role {c.role or 'added'}"
        + (", key" if c.unique else "")
        + f"), missing {c.null_rate:.1%}, distinct {c.n_distinct}"
        + (f"; {c.note}" if c.note else "")
        for c in inputs.contract
    ]
    skipped = [f"- {s.id}: {s.reason}" for s in inputs.plan.skipped] or ["- none"]
    sections = [
        f"Context brief:\n{brief_block(inputs.brief)}",
        _fenced("diagnosis", diagnosis),
        decisions_block(inputs.diagnosis, inputs.decisions),
        f"Cleaning plan: {inputs.plan.summary}",
        _fenced("steps", _steps_text(inputs)),
        "Skipped by the coder, with reasons:\n" + "\n".join(skipped),
        f"Gate 2: {gate.outcome.upper()}\n" + "\n".join(gate_lines),
        "Schema contract of the clean data:\n" + "\n".join(contract_lines),
        _fenced("profile_raw", json.dumps(inputs.profile_raw, indent=1, ensure_ascii=False)),
        _fenced("profile_clean", json.dumps(inputs.profile_clean, indent=1, ensure_ascii=False)),
        "Ask queries if you need more evidence, or give your verdict.",
    ]
    return "\n\n".join(sections)


def _steps_text(inputs: ReviewInputs) -> str:
    lines: list[str] = []
    steps = zip(inputs.plan.steps, inputs.records, strict=True)
    for number, (step, record) in enumerate(steps, start=1):
        lines.append(f"Step {number} ({step.op}) for {', '.join(step.addresses)}: {step.rationale}")
        if step.op == "custom_code":
            declared = {k: v for k, v in step.params.items() if k != "code" and v not in (None, [])}
            lines.append(f"  declared: {json.dumps(declared, ensure_ascii=False)}")
            lines += ["  code:"] + [f"    {line}" for line in step.params["code"].splitlines()]
        else:
            lines.append(f"  params: {json.dumps(record.params, ensure_ascii=False)}")
        changed = ", ".join(f"{c} {n} cell(s)" for c, n in record.cells_changed.items()) or "none"
        nulls = ", ".join(f"{c} {n}" for c, n in record.new_nulls.items()) or "none"
        lines.append(
            f"  changed: {changed}; new missing: {nulls}; rows {record.rows_before} -> "
            f"{record.rows_after} ({record.rows_dropped} dropped)"
        )
        shown = list(record.target_columns) + [
            c for c in record.after if c not in record.before and c not in record.target_columns
        ]
        for column in shown:
            before, after = record.before.get(column), record.after.get(column)
            lines.append("  " + _stats_line(column, before, after, column in inputs.masked))
    return "\n".join(lines)


def _stats_line(
    column: str, before: ColumnStats | None, after: ColumnStats | None, masked: bool
) -> str:
    if after is None:
        return f"{column}: removed"
    if before is None:
        return f"{column}: added as {after.dtype}; " + _stats(after, masked)
    return (
        f"{column}: {before.dtype} -> {after.dtype}; missing {before.n_null} -> {after.n_null}; "
        f"distinct {before.n_distinct} -> {after.n_distinct}; "
        f"min {_value(before.min, masked)} -> {_value(after.min, masked)}; "
        f"max {_value(before.max, masked)} -> {_value(after.max, masked)}"
        + (f"; mean {_value(after.mean, False)}" if after.mean is not None else "")
    )


def _stats(stats: ColumnStats, masked: bool) -> str:
    return (
        f"missing {stats.n_null}, distinct {stats.n_distinct}, min {_value(stats.min, masked)}, "
        f"max {_value(stats.max, masked)}"
    )


def _value(value: Any, masked: bool) -> str:
    if value is None:
        return "none"
    if masked:
        return "'" + mask_shape(value) + "'"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else format(value, ".10g")
    if isinstance(value, str):
        return "'" + value + "'"
    return str(value)


def _fenced(tag: str, text: str) -> str:
    """Data-derived text, with '<' escaped so nothing in it can close the block."""
    return f"<{tag}>\n{text.replace('<', chr(92) + 'u003c')}\n</{tag}>"


# -- checking the verdict ---------------------------------------------------


def parse_reply(text: str) -> ReviewReply:
    try:
        return ReviewReply.model_validate_json(extract_json_object(text))
    except ValidationError as error:
        raise ValueError(compact_validation_error(error)) from error


def check_outcome(
    outcome: ReviewOutcome, *, corpus: str, n_steps: int, columns: set[str]
) -> tuple[ReviewOutcome, list[str], list[str]]:
    """The verdict with unsupported or invalid claims removed, what to send back, what went."""
    known = _numbers(corpus)
    problems: list[str] = []
    dropped: list[str] = []

    def supported(evidence: str) -> bool:
        return bool(_numbers(evidence) & known)

    verified = []
    for item in outcome.verified:
        if supported(item.evidence):
            verified.append(item)
            continue
        reason = f"verified '{item.claim}' quotes no number you were shown ({item.evidence})"
        problems.append(reason + "; quote numbers from the evidence or query them")
        dropped.append(reason)

    findings = []
    known_ops = set(op_names())
    for finding in outcome.findings:
        faults = []
        if finding.step is not None and not 1 <= finding.step <= n_steps:
            faults.append(f"step {finding.step} does not exist (the plan has {n_steps})")
        for column in {finding.column, finding.fix.column if finding.fix else None} - {None}:
            if column not in columns:
                faults.append(f"column {column!r} does not exist")
        if finding.fix and finding.fix.suggested_op and finding.fix.suggested_op not in known_ops:
            faults.append(f"op {finding.fix.suggested_op!r} does not exist")
        if not supported(finding.evidence):
            faults.append(f"its evidence quotes no number you were shown ({finding.evidence})")
        if faults:
            reason = f"finding '{finding.problem}': " + "; ".join(faults)
            problems.append(reason)
            dropped.append(reason)
        else:
            findings.append(finding)

    ambiguities = []
    for ambiguity in outcome.ambiguities:
        if ambiguity.column is not None and ambiguity.column not in columns:
            reason = f"ambiguity '{ambiguity.question}': column {ambiguity.column!r} does not exist"
            problems.append(reason)
            dropped.append(reason)
        else:
            ambiguities.append(ambiguity)

    if len(verified) < MIN_VERIFICATIONS:
        problems.append(
            f"verify at least {MIN_VERIFICATIONS} things with evidence quoting numbers you were "
            f"shown (you have {len(verified)}); query the data if you need more"
        )

    checked = ReviewOutcome(
        accepted=not any(f.fix for f in findings),
        summary=outcome.summary,
        verified=verified,
        findings=findings,
        ambiguities=ambiguities,
    )
    return checked, problems, dropped


def to_verdict(outcome: ReviewOutcome, round_number: int, *, weak: bool) -> ReviewVerdict:
    """The official verdict: findings and ambiguities as flags, fixes as fix requests."""
    flags = [
        Flag(
            code="review_finding",
            severity="warn",
            message=f"{finding.problem} (evidence: {finding.evidence})",
            column=finding.column,
            stage="review",
            details={"severity": finding.severity, "step": finding.step, "fix": bool(finding.fix)},
        )
        for finding in outcome.findings
    ]
    flags += [
        Flag(
            code="review_ambiguity",
            severity="warn",
            message=f"{ambiguity.question} (evidence: {ambiguity.evidence})",
            column=ambiguity.column,
            stage="review",
        )
        for ambiguity in outcome.ambiguities
    ]
    if weak:
        flags.append(
            Flag(
                code="review_unverified",
                severity="warn",
                message=f"The reviewer verified fewer than {MIN_VERIFICATIONS} things with "
                f"numbers from the evidence, so its verdict carries little weight.",
                stage="review",
            )
        )
    fixes = [finding.fix for finding in outcome.findings if finding.fix]
    return ReviewVerdict(
        accepted=not fixes,
        round=round_number,
        summary=outcome.summary,
        findings=flags,
        fix_requests=fixes,
    )


# -- the loop ---------------------------------------------------------------


def review(
    client: LLMClient,
    inputs: ReviewInputs,
    query_ctx: QueryContext,
    *,
    round_number: int = 1,
    max_retries: int = MAX_RETRIES,
    max_query_rounds: int = MAX_QUERY_ROUNDS,
) -> ReviewResult:
    """Ask; answer its queries; check its verdict; send problems back at most twice."""
    version = prompt_version()
    messages = build_messages(inputs)
    corpus = messages[-1]["content"]
    columns = (
        {c["name"] for c in inputs.profile_raw.get("columns", [])}
        | {c.name for c in inputs.contract}
        | {c.name for c in inputs.diagnosis.columns}
    )
    attempts: list[Attempt] = []
    queries: list[QueryRecord] = []
    query_rounds = retries = 0

    while True:
        response, attempt = ask(client, messages, len(attempts) + 1)
        attempts.append(attempt)
        try:
            reply = parse_reply(response.text)
        except ValueError as error:
            reply = None
            attempt.problems = [str(error)]

        if reply is not None and reply.queries is not None:
            if query_rounds < max_query_rounds:
                query_rounds += 1
                answers = []
                for query in reply.queries:
                    try:
                        answer = run_query(query_ctx, query)
                    except QueryError as error:
                        answer = f"{query.id} refused: {error}"
                    queries.append(QueryRecord(query_rounds, query.model_dump(mode="json"), answer))
                    answers.append(answer)
                left = max_query_rounds - query_rounds
                attempt.feedback = (
                    _fenced("query_results", "\n\n".join(answers))
                    + "\n"
                    + (
                        f"You may ask {left} more round(s) of queries, or give your verdict."
                        if left
                        else "No more queries: give your verdict now."
                    )
                )
                corpus += "\n" + attempt.feedback
                messages = [
                    *messages,
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": attempt.feedback},
                ]
                continue
            attempt.problems = ["the query budget is used up; give your verdict now"]

        elif reply is not None:
            checked, attempt.problems, dropped = check_outcome(
                reply.verdict, corpus=corpus, n_steps=len(inputs.plan.steps), columns=columns
            )
            if not attempt.problems or retries >= max_retries:
                weak = len(checked.verified) < MIN_VERIFICATIONS
                return ReviewResult(
                    round=round_number,
                    outcome=checked,
                    verdict=to_verdict(checked, round_number, weak=weak),
                    queries=queries,
                    dropped=dropped,
                    attempts=attempts,
                    prompt_version=version,
                    weak=weak,
                )

        if retries >= max_retries:
            raise ReviewerError(
                f"no usable review after {len(attempts)} attempt(s); "
                f"last problems: {attempt.problems}",
                attempts,
                queries,
            )
        retries += 1
        attempt.feedback = (
            "The review cannot be used yet:\n"
            + "\n".join(f"- {p}" for p in attempt.problems)
            + "\nReturn the corrected reply as one JSON object."
        )
        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": attempt.feedback},
        ]


def _numbers(text: str) -> set[str]:
    return {_normalise(token) for token in _NUMBER.findall(text)}


def _normalise(token: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", token):
        token = token.replace(",", "")
    if re.fullmatch(r"\d+\.\d+", token):
        token = token.rstrip("0").rstrip(".")
    return token


__all__ = [
    "QueryRecord",
    "ReviewInputs",
    "ReviewResult",
    "ReviewerError",
    "build_messages",
    "check_outcome",
    "evidence_pack",
    "masked_from_profile",
    "review",
    "to_verdict",
]
