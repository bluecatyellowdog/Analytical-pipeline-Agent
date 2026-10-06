"""The coder agent: diagnosis + the user's decisions in, an executed, checked cleaning plan out.

The LLM writes a plan; it never touches data. The plan is checked against the
diagnosis before anything runs (every issue handled or skipped with a reason,
no undecided judgment call acted on, the target never imputed or dropped), then
executed from raw by `pipeline/ops/plan.py`, where every step is measured,
logged and finally reconciled and replayed. If a step is refused, the error goes
back to the model with data values masked, at most twice. A plan that only drew
warnings (e.g. a step that changed nothing) gets a second chance but is kept as
a fallback, so a worse retry cannot lose a working plan.

The run folder, the review loop and the outputs live in `pipeline/clean.py`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from pipeline.agents.common import (
    MAX_RETRIES,
    Attempt,
    ask,
    brief_block,
    compact_validation_error,
    extract_json_object,
    profile_block,
    short_hash,
    total_cost,
)
from pipeline.agents.feedback import mask_feedback
from pipeline.agents.llm import LLMClient, Message
from pipeline.checks.gates import gate_2_cleaning
from pipeline.contracts.audit import GateResult
from pipeline.contracts.cleaning import CleaningPlan, Decision
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.diagnosis import Diagnosis
from pipeline.contracts.handoff import ColumnSchema
from pipeline.contracts.plan import FixRequest
from pipeline.ingestion.raw import RawStore, RawTable
from pipeline.ops import REGISTRY, op_names
from pipeline.ops.plan import PlanRun, execute_plan, validate_steps

PROMPT_PATH = Path(__file__).parent / "prompts" / "coder.md"


class CoderError(RuntimeError):
    """No plan ran cleanly within the allowed retries."""

    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass
class CoderResult:
    plan: CleaningPlan
    run: PlanRun
    attempts: list[Attempt]
    prompt_version: str
    gate: GateResult
    contract: list[ColumnSchema]

    @property
    def cost_usd(self) -> float:
        return total_cost(self.attempts)

    @property
    def blocked(self) -> bool:
        return self.gate.outcome == "block"


# -- prompt -----------------------------------------------------------------


def describe_ops() -> str:
    """Every op with its parameters, generated from the param models so it cannot drift."""
    lines: list[str] = []
    for name in op_names():
        spec = REGISTRY[name]
        lines.append(f"- {name}: {(spec.doc.strip().splitlines() or [''])[0]}")
        schema = spec.params_model.model_json_schema()
        required = set(schema.get("required", []))
        for field_name, field_schema in schema.get("properties", {}).items():
            bits = [_type_of(field_schema)]
            if field_name in required:
                bits.append("required")
            elif "default" in field_schema:
                bits.append(f"default {json.dumps(field_schema['default'], ensure_ascii=False)}")
            description = field_schema.get("description")
            lines.append(
                f"    {field_name}: {', '.join(bits)}" + (f". {description}" if description else "")
            )
    return "\n".join(lines)


def system_prompt() -> str:
    schema = json.dumps(CleaningPlan.model_json_schema(), indent=1)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{ops}", describe_ops()).replace("{schema}", schema)


def prompt_version() -> str:
    """Changes whenever the prompt, the op library or the plan schema changes."""
    return short_hash(system_prompt())


def decisions_block(diagnosis: Diagnosis, decisions: list[Decision]) -> str:
    by_id = {d.id: d for d in decisions}
    decided: list[str] = []
    undecided: list[str] = []
    for call in diagnosis.judgment_calls:
        where = f" ({call.column})" if call.column else ""
        decision = by_id.get(call.id)
        if decision is not None and decision.decided:
            who = "by the AI from the brief" if decision.decided_by == "ai" else "by the user"
            decided.append(
                f"- {call.id}{where}: {call.question} -> decided {who}: {decision.choice}"
            )
        else:
            undecided.append(f"- {call.id}{where}: {call.question}")
    return (
        "Decisions on the judgment calls.\n"
        "Decided (carry these out as decided):\n"
        + ("\n".join(decided) or "- none")
        + "\nUndecided (leave alone; do not address these ids):\n"
        + ("\n".join(undecided) or "- none")
    )


def fix_ids(fix_requests: Sequence[FixRequest]) -> list[str]:
    """The ids a plan uses to address the reviewer's fix requests: R1, R2, ..."""
    return [f"R{number}" for number in range(1, len(fix_requests) + 1)]


def fixes_block(fix_requests: Sequence[FixRequest], previous_plan: CleaningPlan | None) -> str:
    """The reviewer's requests, and the plan they are about."""
    lines = [
        "The reviewer checked the result of your previous plan and asks for these fixes. "
        'Address each by its id in "addresses", or list it in "skipped" with the reason:'
    ]
    for fix_id, fix in zip(fix_ids(fix_requests), fix_requests, strict=True):
        where = f" ({fix.column})" if fix.column else ""
        suggested = (
            f" Suggested: {fix.suggested_op} {json.dumps(fix.params, ensure_ascii=False)}."
            if fix.suggested_op
            else ""
        )
        line = f"- {fix_id}{where}: {fix.issue}.{suggested} Why: {fix.reason}"
        lines.append(line.replace("<", "\\u003c"))
    if previous_plan is not None:
        previous = previous_plan.model_dump_json(indent=1).replace("<", "\\u003c")
        lines += [
            "Your previous plan, to start from (keep what was right):",
            f"<previous_plan>\n{previous}\n</previous_plan>",
        ]
    return "\n".join(lines)


def build_messages(
    profile_llm: dict[str, Any],
    brief: ContextBrief,
    diagnosis: Diagnosis,
    decisions: list[Decision],
    *,
    fix_requests: Sequence[FixRequest] = (),
    previous_plan: CleaningPlan | None = None,
) -> list[Message]:
    """The whole prompt: brief, diagnosis, decisions, any fix requests, the masked profile."""
    diagnosis_text = json.dumps(
        diagnosis.model_dump(mode="json"), indent=1, ensure_ascii=False
    ).replace("<", "\\u003c")
    fixes = f"{fixes_block(fix_requests, previous_plan)}\n\n" if fix_requests else ""
    user = (
        f"Context brief:\n{brief_block(brief)}\n"
        f"<diagnosis>\n{diagnosis_text}\n</diagnosis>\n\n"
        f"{decisions_block(diagnosis, decisions)}\n\n"
        f"{fixes}"
        f"{profile_block(profile_llm)}\n\n"
        "Return the cleaning plan as one JSON object."
    )
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def parse_reply(text: str) -> CleaningPlan:
    """The JSON object in the reply, as a CleaningPlan. Raises ValueError if there is none."""
    try:
        return CleaningPlan.model_validate_json(extract_json_object(text))
    except ValidationError as error:
        raise ValueError(compact_validation_error(error)) from error


# -- checks -----------------------------------------------------------------


def check_plan(
    plan: CleaningPlan,
    diagnosis: Diagnosis,
    decisions: list[Decision],
    brief: ContextBrief,
    *,
    fix_requests: Sequence[FixRequest] = (),
) -> list[str]:
    """Everything wrong with a plan before it runs. Empty means it may run."""
    problems = validate_steps(plan)

    issue_ids = [i.id for i in diagnosis.issues] + fix_ids(fix_requests)
    call_ids = [j.id for j in diagnosis.judgment_calls]
    decided = {d.id for d in decisions if d.decided}
    addressed: set[str] = set()

    for number, step in enumerate(plan.steps, start=1):
        for item in step.addresses:
            if item in call_ids and item not in decided:
                problems.append(
                    f"step {number} addresses {item}, which is undecided; "
                    f"leave undecided judgment calls alone"
                )
            elif item not in issue_ids and item not in decided:
                problems.append(
                    f"step {number} addresses {item!r}, which is not an issue, a decided "
                    f"judgment call or a fix request"
                )
            else:
                addressed.add(item)

    skipped = {s.id for s in plan.skipped}
    for item in sorted(skipped - set(issue_ids) - set(call_ids)):
        problems.append(f"skipped lists {item!r}, which is not in the diagnosis")

    required = [i.id for i in diagnosis.issues if i.suggested_op] + [
        j for j in call_ids if j in decided
    ]
    required += fix_ids(fix_requests)
    for item in required:
        if item not in addressed and item not in skipped:
            problems.append(
                f"{item} is neither addressed by a step nor listed in skipped with a reason"
            )

    target = brief.target
    if target:
        for number, step in enumerate(plan.steps, start=1):
            column = step.params.get("column")
            if step.op == "impute" and column == target:
                problems.append(
                    f"step {number} imputes the target {target!r}; the target is never imputed"
                )
            dropped = column == target if step.op == "drop_column" else False
            if step.op == "custom_code" and target in (step.params.get("drops") or []):
                dropped = True
            if dropped:
                problems.append(
                    f"step {number} drops the target {target!r}; the target is never dropped"
                )
    return problems


# -- the loop ---------------------------------------------------------------


def write_and_run(
    profile_llm: dict[str, Any],
    brief: ContextBrief,
    diagnosis: Diagnosis,
    decisions: list[Decision],
    client: LLMClient,
    *,
    store: RawStore,
    raw: RawTable,
    run_id: str,
    max_retries: int = MAX_RETRIES,
    fix_requests: Sequence[FixRequest] = (),
    previous_plan: CleaningPlan | None = None,
) -> CoderResult:
    """Ask for a plan, check it, run it, gate it; send problems back until one is usable.

    A result Gate 2 blocks goes back to the model like a refused step. If no
    attempt ends fully clean, the best one is kept: one that only drew warnings,
    otherwise one Gate 2 blocked, so the report can say exactly why. Only when no
    plan ran at all is there nothing to report but the attempts.
    """
    version = prompt_version()
    messages = build_messages(
        profile_llm,
        brief,
        diagnosis,
        decisions,
        fix_requests=fix_requests,
        previous_plan=previous_plan,
    )
    context = {
        "store": store,
        "raw": raw,
        "run_id": run_id,
        "diagnosis": diagnosis,
        "brief": brief,
        "profile_llm": profile_llm,
    }
    keep = (
        set(raw.columns)
        | set(op_names())
        | {i.id for i in diagnosis.issues}
        | {j.id for j in diagnosis.judgment_calls}
        | set(fix_ids(fix_requests))
    )
    attempts: list[Attempt] = []
    with_warnings: CleaningPlan | None = None
    gate_blocked: CleaningPlan | None = None
    last = max_retries + 1

    for number in range(1, last + 1):
        response, attempt = ask(client, messages, number)
        attempts.append(attempt)
        run: PlanRun | None = None

        try:
            plan = parse_reply(response.text)
        except ValueError as error:
            attempt.problems = [str(error)]
        else:
            attempt.problems = check_plan(
                plan, diagnosis, decisions, brief, fix_requests=fix_requests
            )
            if not attempt.problems:
                run, gate, contract = run_and_gate(plan, number, **context)
                attempt.warnings = list(run.warnings)
                attempt.problems = list(run.problems) + _gate_problems(gate)
                if gate is not None and gate.outcome != "block":
                    if not run.warnings or number == last:
                        return CoderResult(plan, run, attempts, version, gate, contract)
                    with_warnings = plan
                elif gate is not None:
                    gate_blocked = plan

        if number == last:
            break
        attempt.feedback = _feedback(attempt, run, response.text, keep)
        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": attempt.feedback},
        ]

    for candidate in (with_warnings, gate_blocked):
        if candidate is not None:
            run, gate, contract = run_and_gate(candidate, len(attempts), **context)
            if gate is not None:
                return CoderResult(candidate, run, attempts, version, gate, contract)

    raise CoderError(
        f"no cleaning plan ran after {len(attempts)} attempts; "
        f"last problems: {attempts[-1].problems}",
        attempts,
    )


def run_and_gate(
    plan: CleaningPlan,
    attempt: int,
    *,
    store: RawStore,
    raw: RawTable,
    run_id: str,
    diagnosis: Diagnosis,
    brief: ContextBrief,
    profile_llm: dict[str, Any],
) -> tuple[PlanRun, GateResult | None, list[ColumnSchema]]:
    """Execute a plan from raw and, if every step ran, put the result through Gate 2."""
    protected = [brief.target] if brief.target and brief.target in raw.columns else []
    run = execute_plan(store, raw, plan, run_id=run_id, protected=protected)
    if not run.ok:
        return run, None, []
    null_tokens = {c["name"]: list(c.get("null_tokens") or {}) for c in profile_llm["columns"]}
    gate, contract = gate_2_cleaning(
        store,
        raw,
        clean_table=run.clean_table,
        records=run.records,
        reconciliation=run.checks,
        plan=plan,
        diagnosis=diagnosis,
        brief=brief,
        null_tokens=null_tokens,
        attempt=attempt,
    )
    return run, gate, contract


def rerun(result: CoderResult, **context: Any) -> CoderResult:
    """Restore an earlier result: run its plan again from raw. No LLM call; same outcome."""
    run, gate, contract = run_and_gate(result.plan, len(result.attempts), **context)
    if gate is None:
        raise CoderError(f"re-running an earlier plan failed: {run.problems}", result.attempts)
    return CoderResult(result.plan, run, result.attempts, result.prompt_version, gate, contract)


def _gate_problems(gate: GateResult | None) -> list[str]:
    if gate is None:
        return []
    return [
        f"Gate 2 check {c.name} failed: {c.message}"
        for c in gate.checks
        if not c.passed and c.severity == "block"
    ]


def _feedback(attempt: Attempt, run: PlanRun | None, reply: str, keep: set[str]) -> str:
    """What goes back to the model: problems and warnings, with data values masked."""
    lines: list[str] = []
    if run is not None and run.failed_step and run.failed_step > 1:
        lines.append(f"Steps 1-{run.failed_step - 1} ran without problems.")
    if attempt.problems:
        lines.append("The plan cannot be used yet:")
        lines += [f"- {mask_feedback(p, known=reply, keep=keep)}" for p in attempt.problems]
    if attempt.warnings:
        lines.append("The plan ran, but:")
        lines += [f"- {mask_feedback(w, known=reply, keep=keep)}" for w in attempt.warnings]
    lines.append(
        "Return the complete corrected plan as one JSON object. It is run again from the raw data."
    )
    return "\n".join(lines)


def _type_of(schema: dict[str, Any]) -> str:
    if "enum" in schema:
        return "one of " + " | ".join(json.dumps(v, ensure_ascii=False) for v in schema["enum"])
    if "anyOf" in schema:
        return " or ".join(_type_of(option) for option in schema["anyOf"])
    kind = schema.get("type")
    if kind == "array":
        return f"list of {_type_of(schema.get('items', {}))}"
    if kind == "object":
        values = schema.get("additionalProperties")
        return f"mapping of text to {_type_of(values)}" if isinstance(values, dict) else "object"
    names = {"string": "text", "integer": "integer", "number": "number", "boolean": "true/false"}
    return names.get(kind, "null" if kind == "null" else "any value")
