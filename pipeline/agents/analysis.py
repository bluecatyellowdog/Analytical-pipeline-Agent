"""The analysis agent: runs the plan with the tools, follows what is interesting, and
writes up what it found, checked by Gate 4.

The LLM decides, the tools compute. In each round the agent asks for tool
calls; the code validates and runs them (aggregates only, each result logged
as an artifact) and shows the results back under ids T1, T2, ... After at most
MAX_ROUNDS rounds it writes: an answer per plan question, findings that cite
the calls behind them, limitations and what to look at next. Gate 4 checks the
write-up against the logged results (every question handled, every number
traced, no overclaiming); problems go back at most twice. Each finding then
becomes an artifact that points to its plan question and its calls.

The plan it runs is the user's edit of analysis_plan.yaml when there is one,
checked by Gate 3 as the user's, else the AI's approved plan.

    uv run python -m pipeline.analyze runs/<run folder>
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

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
)
from pipeline.agents.llm import LLMClient, Message
from pipeline.agents.planner import GATE_3_FILE, SET_BY_CODE, plan_in_force
from pipeline.checks.analysis_gate import gate_4_analysis
from pipeline.checks.gates import problems_of
from pipeline.checks.plan_gate import gate_3_plan
from pipeline.contracts.analysis import (
    MAX_CALLS_PER_ROUND,
    AgentMove,
    AnalysisOutcome,
    CallRecord,
    RunRound,
    WriteUp,
)
from pipeline.contracts.artifact import Artifact, ArtifactCatalog, AuditRef
from pipeline.contracts.audit import GateResult
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.plan import AnalysisPlan
from pipeline.tools import ToolError, describe_tools, run_tool
from pipeline.tools.context import AnalysisInputError, describe_data, load_context, read_handoff
from pipeline.tools.registry import AnalysisContext

PROMPT_PATH = Path(__file__).parent / "prompts" / "analysis.md"
ANALYSIS_JSON = "analysis.json"
ARTIFACTS_JSON = "artifacts.json"
ANALYSIS_LOG = "analysis_log.json"
GATE_4_FILE = "gate_4.json"
ANALYSIS_REPORT = "analysis_report.md"
CHARTS_DIR = "analysis"
OUTPUTS = (ANALYSIS_JSON, ARTIFACTS_JSON, ANALYSIS_LOG, GATE_4_FILE, ANALYSIS_REPORT)
MAX_ROUNDS = 3
MAX_CALLS = 20
MOVE: TypeAdapter = TypeAdapter(AgentMove)


class AnalysisError(RuntimeError):
    """No write-up passed Gate 4 after the allowed retries."""

    def __init__(
        self,
        message: str,
        attempts: list[Attempt],
        calls: list[CallRecord],
        gate: GateResult | None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.calls = calls
        self.gate = gate


@dataclass
class AnalysisResult:
    outcome: AnalysisOutcome
    gate: GateResult
    attempts: list[Attempt]
    prompt_version: str
    catalog: ArtifactCatalog

    @property
    def cost_usd(self) -> float:
        return total_cost(self.attempts)


def system_prompt() -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    values = {
        "{tools}": describe_tools(),
        "{schema}": json.dumps(MOVE.json_schema(), indent=1),
        "{max_rounds}": str(MAX_ROUNDS),
        "{max_calls}": str(MAX_CALLS),
        "{max_calls_per_round}": str(MAX_CALLS_PER_ROUND),
    }
    for key, value in values.items():
        template = template.replace(key, value)
    return template


def prompt_version() -> str:
    """Changes whenever the prompt, the tools or the move schema changes."""
    return short_hash(system_prompt())


def plan_text(plan: AnalysisPlan) -> str:
    body = plan.model_dump(mode="json", exclude=set(SET_BY_CODE))
    return yaml.safe_dump(body, allow_unicode=True, sort_keys=False, width=100)


def build_messages(ctx: AnalysisContext, plan: AnalysisPlan, brief: ContextBrief) -> list[Message]:
    """The whole opening prompt. Built from the brief, the plan and aggregates of the data."""
    user = (
        f"Context brief:\n{brief_block(brief)}\n"
        f"The approved plan:\n{plan_text(plan)}\n"
        f"{fenced('data', describe_data(ctx))}\n\n"
        "Start with a round of tool calls for the plan's questions."
    )
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def parse_move(text: str) -> RunRound | WriteUp:
    """The JSON object in a reply, as a round of calls or a write-up."""
    try:
        return MOVE.validate_json(extract_json_object(text))
    except ValidationError as error:
        raise ValueError(compact_validation_error(error)) from error


# -- running tool calls ---------------------------------------------------------


def run_round(
    ctx: AnalysisContext,
    move: RunRound,
    plan: AnalysisPlan,
    done: list[CallRecord],
    round_number: int,
    max_calls: int = MAX_CALLS,
) -> list[CallRecord]:
    """Run the calls of one round, as many as the budget allows. A refused call is kept."""
    questions = {q.id for q in plan.questions}
    records: list[CallRecord] = []
    for request in move.calls[: max(0, max_calls - len(done))]:
        question = request.question if request.question in questions else None
        base = {
            "call_id": f"T{len(done) + len(records) + 1}",
            "round": round_number,
            "tool": request.tool,
            "params": request.params,
            "question_id": question,
            "why": request.why,
        }
        try:
            call = run_tool(ctx, request.tool, request.params, question_id=question)
        except ToolError as error:
            records.append(CallRecord(**base, error=str(error)))
            continue
        except Exception as error:  # noqa: BLE001 - one broken call must not end the analysis
            records.append(
                CallRecord(
                    **base,
                    error=f"the tool failed ({type(error).__name__}); try other parameters",
                    internal_error=f"{type(error).__name__}: {error}",
                )
            )
            continue
        records.append(
            CallRecord(
                **base | {"params": call.params},
                summary=call.summary,
                artifact_ids=[a.artifact_id for a in call.artifacts],
                tool_call_id=call.call_id,
                seconds=call.seconds,
            )
        )
    return records


def results_block(ctx: AnalysisContext, records: list[CallRecord]) -> str:
    """The results of a round as the agent sees them: aggregates, warnings and refusals."""
    parts = []
    for record in records:
        params = {k: v for k, v in record.params.items() if v is not None}
        head = f"{record.call_id} {record.tool} {json.dumps(params, ensure_ascii=False)}"
        head += f" for {record.question_id}" if record.question_id else ""
        if not record.ok:
            parts.append(f"{head}: refused: {record.error}")
            continue
        warnings = [
            f"Warning: {flag.message}"
            for artifact_id in record.artifact_ids
            if (artifact := ctx.catalog.get(artifact_id)) is not None
            for flag in artifact.flags
        ]
        parts.append("\n".join([f"{head}:", record.summary, *warnings]))
    return fenced("results", "\n\n".join(parts))


def _next_step(rounds: int, max_rounds: int, used: int, max_calls: int) -> str:
    if rounds >= max_rounds or used >= max_calls:
        return "That was the last round: write the write-up now."
    return (
        f"Rounds left: {max_rounds - rounds}; calls left: {max_calls - used}. Follow up on "
        f"what is interesting, or write the write-up."
    )


# -- the loop -------------------------------------------------------------------


def analyse(
    ctx: AnalysisContext,
    plan: AnalysisPlan,
    brief: ContextBrief,
    client: LLMClient,
    *,
    max_rounds: int = MAX_ROUNDS,
    max_calls: int = MAX_CALLS,
    max_retries: int = MAX_RETRIES,
) -> AnalysisResult:
    """Rounds of tool calls, then a write-up that must pass Gate 4."""
    messages = build_messages(ctx, plan, brief)
    context = f"{brief_block(brief)}\n{plan_text(plan)}"
    attempts: list[Attempt] = []
    calls: list[CallRecord] = []
    rounds = failures = 0
    gate: GateResult | None = None

    while True:
        response, attempt = ask(client, messages, len(attempts) + 1)
        attempts.append(attempt)
        try:
            move = parse_move(response.text)
        except ValueError as error:
            move, attempt.problems = None, [str(error)]

        if isinstance(move, RunRound) and rounds < max_rounds and len(calls) < max_calls:
            rounds += 1
            records = run_round(ctx, move, plan, calls, rounds, max_calls)
            calls += records
            attempt.feedback = (
                f"{results_block(ctx, records)}\n"
                f"{_next_step(rounds, max_rounds, len(calls), max_calls)}"
            )
        else:
            if isinstance(move, RunRound):
                attempt.problems = ["No tool rounds are left: write the write-up now."]
            elif isinstance(move, WriteUp):
                gate = gate_4_analysis(move, plan, calls, ctx.catalog, context=context)
                if not gate.blocked:
                    attempt.warnings = problems_of(gate)
                    outcome = _outcome(ctx, plan, move, calls)
                    return AnalysisResult(outcome, gate, attempts, prompt_version(), ctx.catalog)
                attempt.problems = problems_of(gate)
            failures += 1
            if failures > max_retries:
                raise AnalysisError(
                    f"no write-up passed Gate 4 after {failures} tries; "
                    f"last problems: {attempt.problems}",
                    attempts,
                    calls,
                    gate,
                )
            attempt.feedback = (
                "The write-up cannot be used yet:\n"
                + "\n".join(f"- {p}" for p in attempt.problems)
                + "\nReturn the complete corrected write-up as one JSON object."
            )
        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": attempt.feedback},
        ]


def _outcome(
    ctx: AnalysisContext, plan: AnalysisPlan, write_up: WriteUp, calls: list[CallRecord]
) -> AnalysisOutcome:
    """The record of the analysis, with every finding added to the catalog as an artifact."""
    by_id = {c.call_id: c for c in calls}
    finding_artifacts = {}
    for finding in write_up.findings:
        cited = [by_id[i] for i in finding.evidence]
        refs = (
            [AuditRef(kind="plan_question", id=finding.question_id)] if finding.question_id else []
        )
        refs += [
            AuditRef(kind="tool", id=c.tool_call_id or c.call_id, note=c.call_id) for c in cited
        ]
        evidence = [a for c in cited for a in c.artifact_ids]
        artifact = Artifact(
            run_id=ctx.run_id,
            kind="finding",
            title=finding.headline,
            stage="analysis",
            question_id=finding.question_id,
            data={
                "finding": finding.id,
                "strength": finding.strength,
                "evidence": evidence,
                "caveats": finding.caveats,
            },
            audit_refs=refs,
            interpretation=finding.detail,
            flags=[flag for a in evidence if (art := ctx.catalog.get(a)) for flag in art.flags],
        )
        ctx.catalog.add(artifact)
        finding_artifacts[finding.id] = artifact.artifact_id
    return AnalysisOutcome(
        run_id=ctx.run_id,
        plan=plan,
        write_up=write_up,
        calls=calls,
        finding_artifacts=finding_artifacts,
    )


# -- the run folder ---------------------------------------------------------------


def run_on_folder(run_dir: Path, client: LLMClient, provider: dict[str, str]) -> AnalysisResult:
    """Analyse a planned run folder and write the results next to it."""
    plan = plan_in_force(run_dir)
    if plan is None:
        raise AnalysisInputError(
            "there is no analysis plan yet; `python -m pipeline.analyze` plans and analyses"
        )
    manifest = load_manifest(run_dir)
    brief = manifest.context_brief
    ctx = load_context(run_dir, handoff=read_handoff(run_dir))
    if plan.approved_by == "user":
        gate_3 = gate_3_plan(plan, ctx, brief, edited_by_user=True)
        (run_dir / GATE_3_FILE).write_text(gate_3.model_dump_json(indent=2), encoding="utf-8")
        if gate_3.blocked:
            raise AnalysisInputError(
                "your edited analysis_plan.yaml cannot run (Gate 3): "
                + "; ".join(problems_of(gate_3))
            )
    ctx.target, ctx.task_type = plan.target, plan.task_type
    ctx.excluded |= plan.excluded_columns

    for name in OUTPUTS:
        (run_dir / name).unlink(missing_ok=True)
    shutil.rmtree(run_dir / CHARTS_DIR, ignore_errors=True)

    try:
        result = analyse(ctx, plan, brief, client)
    except AnalysisError as error:
        _write_log(run_dir, error.attempts, error.calls)
        if error.gate is not None:
            (run_dir / GATE_4_FILE).write_text(error.gate.model_dump_json(indent=2), "utf-8")
        record_in_manifest(manifest, "analysis", provider, error.attempts, prompt_version())
        manifest.stage_status["analysis"] = "blocked"
        save_manifest(run_dir, manifest)
        raise

    (run_dir / ANALYSIS_JSON).write_text(result.outcome.model_dump_json(indent=2), "utf-8")
    (run_dir / ARTIFACTS_JSON).write_text(result.catalog.model_dump_json(indent=2), "utf-8")
    (run_dir / GATE_4_FILE).write_text(result.gate.model_dump_json(indent=2), "utf-8")
    (run_dir / ANALYSIS_REPORT).write_text(render_markdown(result), encoding="utf-8")
    _write_log(run_dir, result.attempts, result.outcome.calls)
    record_in_manifest(manifest, "analysis", provider, result.attempts, result.prompt_version)
    manifest.stage_status["analysis"] = "done"
    save_manifest(run_dir, manifest)
    return result


def _write_log(run_dir: Path, attempts: list[Attempt], calls: list[CallRecord]) -> None:
    log = {
        "prompt_version": prompt_version(),
        "attempts": [asdict(a) for a in attempts],
        "calls": [c.model_dump(mode="json") for c in calls],
    }
    (run_dir / ANALYSIS_LOG).write_text(json.dumps(log, indent=2, ensure_ascii=False), "utf-8")


# -- reading it ---------------------------------------------------------------------


STATUS = {
    "answered": "answered",
    "partly": "partly answered",
    "not_answered": "not answered",
    "infeasible": "not possible with this data",
}


def render_markdown(result: AnalysisResult) -> str:
    """The write-up with its evidence, for the run folder."""
    outcome, catalog = result.outcome, result.catalog
    up, plan = outcome.write_up, outcome.plan
    lines = [
        "# What the analysis found",
        "",
        "_AI-written from the tool results below; Gate 4 checked that every number in it is "
        "one the tools reported._",
        "",
        up.summary,
        "",
    ]
    answers = {a.question_id: a for a in up.answers}
    shown: set[str] = set()
    for question in sorted(plan.questions, key=lambda q: q.priority):
        answer = answers.get(question.id)
        if answer is None:
            continue
        lines += [
            f"## {question.id}. {question.question}",
            "",
            f"**{STATUS[answer.status]}.** {answer.answer}",
        ]
        if answer.reason:
            lines.append(f"_Why: {answer.reason}_")
        for finding_id in answer.finding_ids:
            lines += _finding_lines(up.finding(finding_id), outcome, catalog)
            shown.add(finding_id)
        lines.append("")
    extra = [f for f in up.findings if f.id not in shown]
    if extra:
        lines += ["## Also worth knowing", ""]
        for finding in extra:
            lines += _finding_lines(finding, outcome, catalog)
        lines.append("")
    if up.limitations:
        lines += ["## Limitations", "", *(f"- {item}" for item in up.limitations), ""]
    if up.next_questions:
        lines += ["## Worth looking at next", "", *(f"- {item}" for item in up.next_questions), ""]
    lines += ["## Tool calls", "", "| Id | Tool | For | Result |", "| --- | --- | --- | --- |"]
    for call in outcome.calls:
        result_text = (
            call.summary.splitlines()[0] if call.ok and call.summary else f"refused: {call.error}"
        )
        lines.append(
            f"| {call.call_id} | {call.tool} | {call.question_id or '-'} | "
            f"{result_text.replace('|', '/')} |"
        )
    lines += ["", f"Gate 4: {result.gate.outcome.upper()}"]
    lines += [f"- {problem}" for problem in problems_of(result.gate)]
    return "\n".join(lines) + "\n"


def _finding_lines(finding, outcome: AnalysisOutcome, catalog: ArtifactCatalog) -> list[str]:
    if finding is None:
        return []
    lines = [
        "",
        f"- **{finding.headline}** ({finding.strength}) {finding.detail} "
        f"_Evidence: {', '.join(finding.evidence)}._",
    ]
    lines += [f"  - Caveat: {caveat}" for caveat in finding.caveats]
    for call_id in finding.evidence:
        call = outcome.call(call_id)
        for artifact_id in call.artifact_ids if call else []:
            artifact = catalog.get(artifact_id)
            if artifact is not None and artifact.spec.get("chart_file"):
                lines.append(f"  - ![{artifact.title}]({artifact.spec['chart_file']})")
    return lines


def render_text(result: AnalysisResult) -> str:
    """The write-up in a few lines, for the terminal."""
    up = result.outcome.write_up
    lines = [up.summary, ""]
    for finding in up.findings:
        lines.append(f"- {finding.headline} ({finding.strength})")
    lines.append(f"\nGate 4: {result.gate.outcome.upper()}")
    return "\n".join(lines)
