"""The planner agent: brief + clean data summary in, an analysis plan that passed Gate 3 out.

The AI decides what is worth finding out; Gate 3 decides whether the plan can
run. The prompt holds the brief, a summary of the clean data built from
aggregates only (no rows, no value of an identifier or personal column), the
tool list generated from the registry, the columns a check found to give the
answer away, and the limitations cleaning recorded. A plan that fails Gate 3
goes back with every problem, at most twice.

A plan that passes is approved automatically and the run goes straight on
(D034). The user changes it by editing analysis_plan.yaml; an edited plan is
never overwritten by a new one.

    uv run python -m pipeline.agents.planner runs/<run folder>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pipeline.agents.common import (
    MAX_RETRIES,
    Attempt,
    ask,
    brief_block,
    compact_validation_error,
    extract_json_object,
    fenced,
    load_manifest,
    record_in_manifest,
    save_manifest,
    short_hash,
    total_cost,
    write_attempt_log,
)
from pipeline.agents.llm import LLMClient, LLMConfigError, Message, connect
from pipeline.checks.gates import problems_of
from pipeline.checks.plan_gate import gate_3_plan, leak_suspects
from pipeline.contracts.audit import GateResult
from pipeline.contracts.common import utcnow
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.plan import AnalysisPlan
from pipeline.tools import describe_tools
from pipeline.tools.context import (
    AnalysisInputError,
    describe_data,
    load_context,
    read_handoff,
)
from pipeline.tools.registry import AnalysisContext

PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.md"
PLAN_YAML = "analysis_plan.yaml"
PLAN_AI_JSON = "analysis_plan_ai.json"
PLANNER_LOG = "planner_log.json"
GATE_3_FILE = "gate_3.json"
MAX_LIMITATIONS = 15

SET_BY_CODE = ("run_id", "created_at", "approved", "approved_at", "approved_by", "human_edits")
"""Plan fields the code fills in, never the model: who approved what, and when."""

YAML_HEADER = """\
# The analysis plan, written by the AI from your brief and approved automatically.
# To change it: edit this file and run the analysis again. Your version wins, and
# every change you make is recorded in the report. Delete the file to have the AI
# write a new plan. Tools you can use in "methods" are listed in README.md.
"""


class PlannerError(RuntimeError):
    """No plan that passes Gate 3 after the allowed retries."""

    def __init__(self, message: str, attempts: list[Attempt], gate: GateResult | None) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.gate = gate


@dataclass
class PlannerResult:
    plan: AnalysisPlan
    gate: GateResult
    attempts: list[Attempt]
    prompt_version: str
    leak_suspects: dict[str, float] = field(default_factory=dict)

    @property
    def cost_usd(self) -> float:
        return total_cost(self.attempts)


def system_prompt() -> str:
    schema = AnalysisPlan.model_json_schema()
    for name in SET_BY_CODE:
        schema["properties"].pop(name, None)
    schema["required"] = [r for r in schema.get("required", []) if r not in SET_BY_CODE]
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{tools}", describe_tools()).replace(
        "{schema}", json.dumps(schema, indent=1)
    )


def prompt_version() -> str:
    """Changes whenever the prompt, the tools or the plan schema changes."""
    return short_hash(system_prompt())


def build_messages(
    ctx: AnalysisContext,
    brief: ContextBrief,
    *,
    suspects: dict[str, float],
    limitations: list[str],
) -> list[Message]:
    """The whole prompt. Built only from the brief and aggregates of the clean data."""
    leaks = (
        "These columns predict the target almost perfectly on their own, so they probably "
        "give the answer away; exclude them with source 'leakage':\n"
        + "\n".join(
            f"- {column} (strength {strength:.2f})" for column, strength in suspects.items()
        )
        if suspects
        else "No column predicts the target almost perfectly on its own."
    )
    noted = "\n".join(f"- {item}" for item in limitations[:MAX_LIMITATIONS]) or "None recorded."
    user = (
        f"Context brief:\n{brief_block(brief)}\n"
        f"{fenced('data', describe_data(ctx))}\n\n"
        f"{leaks}\n\n"
        f"Limitations recorded while cleaning:\n{fenced('limitations', noted)}\n\n"
        "Return the plan as one JSON object."
    )
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def parse_reply(text: str, run_id: str) -> AnalysisPlan:
    """The JSON object in the reply, as a plan. Fields only the code sets are ignored."""
    raw = json.loads(extract_json_object(text))
    if not isinstance(raw, dict):
        raise ValueError("the reply must be one JSON object")
    for name in SET_BY_CODE:
        raw.pop(name, None)
    try:
        return AnalysisPlan.model_validate({**raw, "run_id": run_id})
    except ValidationError as error:
        raise ValueError(compact_validation_error(error)) from error


def plan_analysis(
    ctx: AnalysisContext,
    brief: ContextBrief,
    client: LLMClient,
    *,
    limitations: list[str] | None = None,
    max_retries: int = MAX_RETRIES,
) -> PlannerResult:
    """Ask, run Gate 3, and send problems back until the plan passes or retries run out."""
    suspects = leak_suspects(ctx)
    messages = build_messages(ctx, brief, suspects=suspects, limitations=limitations or [])
    attempts: list[Attempt] = []
    gate: GateResult | None = None

    for number in range(1, max_retries + 2):
        response, attempt = ask(client, messages, number)
        attempts.append(attempt)
        try:
            plan = parse_reply(response.text, ctx.run_id)
            gate = gate_3_plan(plan, ctx, brief)
            if gate.blocked:
                attempt.problems = problems_of(gate)
            else:
                attempt.warnings = problems_of(gate)
        except ValueError as error:
            attempt.problems = [str(error)]

        if not attempt.problems:
            approved = AnalysisPlan.model_validate(
                plan.model_dump()
                | {"approved": True, "approved_at": utcnow(), "approved_by": "auto"}
            )
            return PlannerResult(approved, gate, attempts, prompt_version(), suspects)

        attempt.feedback = (
            "The plan cannot be used yet (Gate 3):\n"
            + "\n".join(f"- {p}" for p in attempt.problems)
            + "\nReturn the complete corrected JSON object."
        )
        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": attempt.feedback},
        ]

    raise PlannerError(
        f"no plan passed Gate 3 after {len(attempts)} attempts; "
        f"last problems: {attempts[-1].problems}",
        attempts,
        gate,
    )


# -- the plan on disk -----------------------------------------------------------


def plan_yaml(plan: AnalysisPlan) -> str:
    """The plan as the user reads and edits it: no approval fields, which the code sets."""
    body = plan.model_dump(mode="json", exclude=set(SET_BY_CODE))
    return YAML_HEADER + yaml.safe_dump(body, allow_unicode=True, sort_keys=False, width=100)


def user_edited(run_dir: Path) -> bool:
    """Whether analysis_plan.yaml differs from the plan the AI wrote."""
    edited, original = run_dir / PLAN_YAML, run_dir / PLAN_AI_JSON
    if not edited.exists() or not original.exists():
        return edited.exists()
    ai = AnalysisPlan.model_validate_json(original.read_text(encoding="utf-8"))
    mine = yaml.safe_load(edited.read_text(encoding="utf-8")) or {}
    return mine != ai.model_dump(mode="json", exclude=set(SET_BY_CODE))


def plan_in_force(run_dir: Path) -> AnalysisPlan | None:
    """The plan the analysis runs: your edit of analysis_plan.yaml when there is one, else
    the AI's. An edited plan is approved by the user, with every change listed."""
    ai_path, yaml_path = run_dir / PLAN_AI_JSON, run_dir / PLAN_YAML
    ai = (
        AnalysisPlan.model_validate_json(ai_path.read_text(encoding="utf-8"))
        if ai_path.exists()
        else None
    )
    if not user_edited(run_dir):
        return ai
    mine = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(mine, dict):
        raise AnalysisInputError(f"{PLAN_YAML} must hold the plan's fields, as the AI wrote them")
    try:
        return AnalysisPlan.model_validate(
            {k: v for k, v in mine.items() if k not in SET_BY_CODE}
            | {
                "run_id": ai.run_id if ai else load_manifest(run_dir).run_id,
                "approved": True,
                "approved_at": utcnow(),
                "approved_by": "user",
                "human_edits": describe_edits(ai, mine),
            }
        )
    except ValidationError as error:
        raise AnalysisInputError(
            f"{PLAN_YAML} is not a valid plan: {compact_validation_error(error)}"
        ) from error


def describe_edits(ai: AnalysisPlan | None, mine: dict[str, Any]) -> list[str]:
    """What the user changed, field by field and question by question."""
    if ai is None:
        return ["The whole plan was written by the user."]
    original = ai.model_dump(mode="json", exclude=set(SET_BY_CODE))
    edits = [
        f"Changed {name}."
        for name in original
        if name != "questions" and mine.get(name, original[name]) != original[name]
    ]
    before = {q["id"]: q for q in original["questions"]}
    after = {q.get("id"): q for q in mine.get("questions", []) if isinstance(q, dict)}
    for qid, question in after.items():
        if qid not in before:
            edits.append(f"Added question {qid}: {question.get('question', '')}")
        elif question != before[qid]:
            fields = sorted(
                k for k in set(question) | set(before[qid]) if question.get(k) != before[qid].get(k)
            )
            edits.append(f"Changed question {qid} ({', '.join(fields)}).")
    edits += [
        f"Removed question {qid}: {q['question']}" for qid, q in before.items() if qid not in after
    ]
    return edits or ["Saved the file without changing the plan."]


def _limitations(flags: list[Any]) -> list[str]:
    seen: list[str] = []
    for flag in flags:
        if flag.message not in seen:
            seen.append(flag.message)
    return seen


def run_on_folder(run_dir: Path, client: LLMClient, provider: dict[str, str]) -> PlannerResult:
    """Plan the analysis of a cleaned run folder and write the plan next to it."""
    if user_edited(run_dir):
        raise AnalysisInputError(
            f"{PLAN_YAML} has your edits, so no new plan is written over it. Delete it to have "
            f"the AI plan again."
        )
    handoff = read_handoff(run_dir)
    ctx = load_context(run_dir, handoff=handoff)
    manifest = load_manifest(run_dir)

    try:
        result = plan_analysis(
            ctx, manifest.context_brief, client, limitations=_limitations(handoff.flags)
        )
    except PlannerError as error:
        write_attempt_log(run_dir / PLANNER_LOG, error.attempts, prompt_version())
        if error.gate is not None:
            (run_dir / GATE_3_FILE).write_text(error.gate.model_dump_json(indent=2), "utf-8")
        record_in_manifest(manifest, "planner", provider, error.attempts, prompt_version())
        manifest.stage_status["planning"] = "blocked"
        save_manifest(run_dir, manifest)
        raise

    (run_dir / PLAN_AI_JSON).write_text(result.plan.model_dump_json(indent=2), encoding="utf-8")
    (run_dir / PLAN_YAML).write_text(plan_yaml(result.plan), encoding="utf-8")
    (run_dir / GATE_3_FILE).write_text(result.gate.model_dump_json(indent=2), encoding="utf-8")
    write_attempt_log(run_dir / PLANNER_LOG, result.attempts, result.prompt_version)
    record_in_manifest(manifest, "planner", provider, result.attempts, result.prompt_version)
    manifest.stage_status["planning"] = "done"
    save_manifest(run_dir, manifest)
    return result


def render_text(result: PlannerResult) -> str:
    """The plan in a few lines, for the terminal."""
    plan = result.plan
    lines = [f"Objective: {plan.objective}"]
    if plan.interpretation:
        lines.append(f"How the brief was read: {plan.interpretation}")
    for q in sorted(plan.questions, key=lambda q: q.priority):
        status = (
            f"[{', '.join(q.methods)}]" if q.feasible else f"[not possible: {q.infeasible_reason}]"
        )
        lines.append(f"  {q.id}. {q.question} {status}")
    if plan.excluded_features:
        lines.append(
            "Left out: " + "; ".join(f"{e.column} ({e.reason})" for e in plan.excluded_features)
        )
    lines.append(f"Gate 3: {result.gate.outcome.upper()}, approved automatically.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.agents.planner",
        description="Plan the analysis of a cleaned run folder. Runs no analysis.",
    )
    parser.add_argument("run_dir", help="A cleaned run folder, e.g. runs/loans_...")
    args = parser.parse_args(argv)

    try:
        client = connect()
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1
    run_dir = Path(args.run_dir)
    print(f"Asking {client.config.model} for an analysis plan...")
    try:
        result = run_on_folder(run_dir, client, client.config.public())
    except AnalysisInputError as error:
        print(f"Cannot plan {run_dir}: {error}")
        return 1
    except PlannerError as error:
        print(f"\n{error}\nEvery attempt is logged in {(run_dir / PLANNER_LOG).resolve()}")
        return 1
    print()
    print(render_text(result))
    print(f"\nWrote {(run_dir / PLAN_YAML).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
