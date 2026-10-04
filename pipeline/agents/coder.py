"""The coder agent: diagnosis + the user's decisions in, an executed, checked cleaning plan out.

The LLM writes a plan; it never touches data. The plan is checked against the
diagnosis before anything runs (every issue handled or skipped with a reason,
no undecided judgment call acted on, the target never imputed or dropped), then
executed from raw by `pipeline/ops/plan.py`, where every step is measured,
logged and finally reconciled and replayed. If a step is refused, the error goes
back to the model with data values masked, at most twice. A plan that only drew
warnings (e.g. a step that changed nothing) gets a second chance but is kept as
a fallback, so a worse retry cannot lose a working plan.

    uv run python -m pipeline.agents.coder runs/<run folder>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pipeline.agents.analyst import DIAGNOSIS_JSON
from pipeline.agents.common import (
    MAX_RETRIES,
    Attempt,
    ask,
    brief_block,
    compact_validation_error,
    extract_json_object,
    load_manifest,
    profile_block,
    record_in_manifest,
    save_manifest,
    short_hash,
    total_cost,
    write_attempt_log,
)
from pipeline.agents.decisions import DECISIONS_FILE, DecisionsError, load_decisions
from pipeline.agents.feedback import mask_feedback
from pipeline.agents.llm import LLMClient, LLMConfigError, Message, connect
from pipeline.checks.gates import gate_2_cleaning
from pipeline.contracts.audit import GateResult
from pipeline.contracts.cleaning import CleaningPlan, Decision
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.diagnosis import Diagnosis
from pipeline.contracts.handoff import ColumnSchema, Handoff
from pipeline.ingestion.loaders import load
from pipeline.ingestion.raw import ROW_ID, RawStore, RawTable, quote_ident
from pipeline.ops import REGISTRY, op_names
from pipeline.ops.plan import PlanRun, execute_plan, validate_steps

PROMPT_PATH = Path(__file__).parent / "prompts" / "coder.md"
CLEANING_PLAN = "cleaning_plan.json"
CLEANING_LOG = "cleaning_log.json"
CODER_LOG = "coder_log.json"
CLEANING_REPORT = "cleaning_report.md"
CLEAN_PARQUET = "clean_data.parquet"
CLEAN_CSV = "clean_data.csv"
DROPPED_CSV = "dropped_rows.csv"
GATE_2_FILE = "gate_2.json"
HANDOFF_FILE = "handoff.json"
SCHEMA_CONTRACT_FILE = "schema_contract.yaml"
DELIVERABLES = (CLEAN_PARQUET, CLEAN_CSV, DROPPED_CSV, HANDOFF_FILE, SCHEMA_CONTRACT_FILE)
"""Written only for a result Gate 2 let through; cleared first, so none is ever stale."""


class CoderError(RuntimeError):
    """No plan ran cleanly within the allowed retries."""

    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


class RunFolderError(ValueError):
    """The run folder is not ready to be cleaned, with what to do about it."""


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
            decided.append(f"- {call.id}{where}: {call.question} -> decided: {decision.choice}")
        else:
            undecided.append(f"- {call.id}{where}: {call.question}")
    return (
        "Decisions on the judgment calls.\n"
        "Decided (carry these out as decided):\n"
        + ("\n".join(decided) or "- none")
        + "\nUndecided (leave alone; do not address these ids):\n"
        + ("\n".join(undecided) or "- none")
    )


def build_messages(
    profile_llm: dict[str, Any],
    brief: ContextBrief,
    diagnosis: Diagnosis,
    decisions: list[Decision],
) -> list[Message]:
    """The whole prompt: brief, diagnosis, decisions and the masked profile. No rows."""
    diagnosis_text = json.dumps(
        diagnosis.model_dump(mode="json"), indent=1, ensure_ascii=False
    ).replace("<", "\\u003c")
    user = (
        f"Context brief:\n{brief_block(brief)}\n"
        f"<diagnosis>\n{diagnosis_text}\n</diagnosis>\n\n"
        f"{decisions_block(diagnosis, decisions)}\n\n"
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
) -> list[str]:
    """Everything wrong with a plan before it runs. Empty means it may run."""
    problems = validate_steps(plan)

    issue_ids = [i.id for i in diagnosis.issues]
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
                    f"step {number} addresses {item!r}, which is not an issue or a decided "
                    f"judgment call"
                )
            else:
                addressed.add(item)

    skipped = {s.id for s in plan.skipped}
    for item in sorted(skipped - set(issue_ids) - set(call_ids)):
        problems.append(f"skipped lists {item!r}, which is not in the diagnosis")

    required = [i.id for i in diagnosis.issues if i.suggested_op] + [
        j for j in call_ids if j in decided
    ]
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
) -> CoderResult:
    """Ask for a plan, check it, run it, gate it; send problems back until one is usable.

    A result Gate 2 blocks goes back to the model like a refused step. If no
    attempt ends fully clean, the best one is kept: one that only drew warnings,
    otherwise one Gate 2 blocked, so the report can say exactly why. Only when no
    plan ran at all is there nothing to report but the attempts.
    """
    version = prompt_version()
    messages = build_messages(profile_llm, brief, diagnosis, decisions)
    protected = [brief.target] if brief.target and brief.target in raw.columns else []
    null_tokens = {c["name"]: list(c.get("null_tokens") or {}) for c in profile_llm["columns"]}
    keep = (
        set(raw.columns)
        | set(op_names())
        | {i.id for i in diagnosis.issues}
        | {j.id for j in diagnosis.judgment_calls}
    )
    attempts: list[Attempt] = []
    with_warnings: CleaningPlan | None = None
    gate_blocked: CleaningPlan | None = None
    last = max_retries + 1

    def run_and_gate(
        plan: CleaningPlan, number: int
    ) -> tuple[PlanRun, GateResult | None, list[ColumnSchema]]:
        run = execute_plan(store, raw, plan, run_id=run_id, protected=protected)
        if not run.ok:
            return run, None, []
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
            attempt=number,
        )
        return run, gate, contract

    for number in range(1, last + 1):
        response, attempt = ask(client, messages, number)
        attempts.append(attempt)
        run: PlanRun | None = None

        try:
            plan = parse_reply(response.text)
        except ValueError as error:
            attempt.problems = [str(error)]
        else:
            attempt.problems = check_plan(plan, diagnosis, decisions, brief)
            if not attempt.problems:
                run, gate, contract = run_and_gate(plan, number)
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
            run, gate, contract = run_and_gate(candidate, len(attempts))
            if gate is not None:
                return CoderResult(candidate, run, attempts, version, gate, contract)

    raise CoderError(
        f"no cleaning plan ran after {len(attempts)} attempts; "
        f"last problems: {attempts[-1].problems}",
        attempts,
    )


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


# -- the run folder ---------------------------------------------------------


def run_on_folder(run_dir: Path, client: LLMClient, provider: dict[str, str]) -> CoderResult:
    """Clean an ingested, diagnosed run folder and write everything next to it."""
    manifest = load_manifest(run_dir)
    if manifest.stage_status.get("ingestion") == "blocked":
        raise RunFolderError(
            "Gate 1 blocked this load, so it may not be cleaned. Fix the problems in "
            "ingestion_report.md and ingest the file again."
        )
    if not (run_dir / DIAGNOSIS_JSON).exists():
        raise RunFolderError(
            "there is no diagnosis yet. Run the analyst first:\n"
            f"  uv run python -m pipeline.agents.analyst {run_dir.as_posix()}"
        )
    if not manifest.inputs:
        raise RunFolderError("the manifest records no input file")

    diagnosis = Diagnosis.model_validate_json(
        (run_dir / DIAGNOSIS_JSON).read_text(encoding="utf-8")
    )
    decisions = load_decisions(run_dir / DECISIONS_FILE, diagnosis)
    profile_llm = json.loads((run_dir / "profile_llm.json").read_text(encoding="utf-8"))

    info = manifest.inputs[0]
    source = Path(info.path)
    if not source.exists():
        raise RunFolderError(
            f"the source file {info.path} was not found. Run the command from the folder "
            f"you ingested from."
        )

    with RawStore.open() as store:
        table = profile_llm["table"].split(".", 1)[-1]
        load_result, raw = load(source, store, table=table, sheet=info.sheet)
        if load_result.file.file_hash != info.file_hash:
            raise RunFolderError(
                f"{source.name} has changed since it was ingested (hash "
                f"{info.file_hash[:12]} then, {load_result.file.file_hash[:12]} now). "
                f"Ingest it again so the profile and diagnosis describe the current file."
            )

        try:
            result = write_and_run(
                profile_llm,
                manifest.context_brief,
                diagnosis,
                decisions,
                client,
                store=store,
                raw=raw,
                run_id=manifest.run_id,
            )
        except CoderError as error:
            write_attempt_log(run_dir / CODER_LOG, error.attempts, prompt_version())
            record_in_manifest(manifest, "coder", provider, error.attempts, prompt_version())
            manifest.stage_status["cleaning"] = "blocked"
            save_manifest(run_dir, manifest)
            raise

        _write_outputs(run_dir, store, raw, result, diagnosis, decisions, run_id=manifest.run_id)

    record_in_manifest(manifest, "coder", provider, result.attempts, result.prompt_version)
    manifest.stage_status["cleaning"] = "blocked" if result.blocked else "done"
    save_manifest(run_dir, manifest)
    return result


def _write_outputs(
    run_dir: Path,
    store: RawStore,
    raw: RawTable,
    result: CoderResult,
    diagnosis: Diagnosis,
    decisions: list[Decision],
    *,
    run_id: str,
) -> None:
    """The logs and the report always; clean data, handoff and contract only past Gate 2."""
    for name in DELIVERABLES:
        (run_dir / name).unlink(missing_ok=True)

    run, gate = result.run, result.gate
    rows_clean = store.row_count(run.clean_table)
    (run_dir / CLEANING_PLAN).write_text(result.plan.model_dump_json(indent=2), encoding="utf-8")
    log = {
        "raw_table": raw.name,
        "clean_table": run.clean_table,
        "rows_raw": raw.n_rows,
        "rows_clean": rows_clean,
        "warnings": run.warnings,
        "records": [r.model_dump(mode="json") for r in run.records],
        "gate_2": gate.model_dump(mode="json"),
    }
    (run_dir / CLEANING_LOG).write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_attempt_log(run_dir / CODER_LOG, result.attempts, result.prompt_version)
    (run_dir / GATE_2_FILE).write_text(gate.model_dump_json(indent=2), encoding="utf-8")

    if run.dropped_table:
        _copy(
            store,
            f"SELECT * EXCLUDE ({ROW_ID}) FROM {_qualified(run.dropped_table)} ORDER BY {ROW_ID}",
            run_dir / DROPPED_CSV,
            "HEADER, DELIMITER ','",
        )
    if not result.blocked:
        clean = f"SELECT * EXCLUDE ({ROW_ID}) FROM {_qualified(run.clean_table)} ORDER BY {ROW_ID}"
        _copy(store, clean, run_dir / CLEAN_PARQUET, "FORMAT parquet")
        _copy(store, clean, run_dir / CLEAN_CSV, "HEADER, DELIMITER ','")
        handoff = _handoff(store, raw, result, run_id=run_id, rows_clean=rows_clean)
        (run_dir / HANDOFF_FILE).write_text(
            handoff.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
        )
        (run_dir / SCHEMA_CONTRACT_FILE).write_text(
            _contract_yaml(handoff, gate.outcome), encoding="utf-8"
        )

    report = render_report(result, diagnosis, decisions, rows_raw=raw.n_rows, rows_clean=rows_clean)
    (run_dir / CLEANING_REPORT).write_text(report, encoding="utf-8")


def _handoff(
    store: RawStore, raw: RawTable, result: CoderResult, *, run_id: str, rows_clean: int
) -> Handoff:
    """What Stage 3 reads instead of the data: the contract, the accounting, the flags."""
    run = result.run
    dropped_columns: dict[str, str] = {}
    for step, record in zip(result.plan.steps, run.records, strict=True):
        if record.op_name == "drop_column":
            dropped_columns[record.params["column"]] = record.params["reason"]
        elif record.op_name == "custom_code":
            for column in record.params.get("drops") or []:
                dropped_columns[column] = step.rationale
    return Handoff(
        run_id=run_id,
        table=run.clean_table,
        data_file=CLEAN_PARQUET,
        source_table=raw.name,
        rows_raw=raw.n_rows,
        rows_clean=rows_clean,
        rows_dropped=sum(r.rows_dropped for r in run.records),
        dropped_rows_table=run.dropped_table,
        schema=result.contract,
        dropped_columns=dropped_columns,
        applied_ops=[r.op_id for r in run.records],
        flags=list(result.gate.flags),
        data_hash=store.data_hash(run.clean_table),
    )


def _contract_yaml(handoff: Handoff, outcome: str) -> str:
    """The schema contract in a form a person reads; handoff.json holds the same."""
    columns = {}
    for column in handoff.schema_:
        entry = {
            "type": column.dtype,
            "expected": column.expected_type,
            "role": column.role,
            "unique": column.unique,
            "missing_allowed": column.nullable,
            "missing_share": round(column.null_rate, 4),
            "distinct": column.n_distinct,
            "renamed_from": column.source_column,
            "note": column.note,
        }
        columns[column.name] = {k: v for k, v in entry.items() if v is not None}
    header = (
        f"# Schema contract for {handoff.table}, verified by Gate 2 ({outcome.upper()}).\n"
        f"# Later stages re-check {handoff.data_file} against it before using the data.\n\n"
    )
    body = {"table": handoff.table, "data_file": handoff.data_file, "rows": handoff.rows_clean}
    body["columns"] = columns
    return header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100)


def _copy(store: RawStore, query: str, path: Path, options: str) -> None:
    target = path.as_posix().replace("'", "''")
    store.connection.execute(f"COPY ({query}) TO '{target}' ({options})")


def _qualified(table: str) -> str:
    schema, _, name = table.rpartition(".")
    return f"{schema}.{quote_ident(name)}"


# -- the report -------------------------------------------------------------


def render_report(
    result: CoderResult,
    diagnosis: Diagnosis,
    decisions: list[Decision],
    *,
    rows_raw: int,
    rows_clean: int,
) -> str:
    """The cleaning report. Every number comes from the execution log, never from the LLM."""
    run, plan, gate = result.run, result.plan, result.gate
    n_custom = sum(1 for s in plan.steps if s.op == "custom_code")
    dropped = sum(r.rows_dropped for r in run.records)
    lines = [
        f"# Cleaning report: {run.clean_table}",
        "",
        f"**Gate 2: {gate.outcome.upper()}.** {GATE_VERDICT[gate.outcome]}",
        "",
        f"All {len(plan.steps)} step(s) ran. "
        + (f"{n_custom} used custom code, shown in full below. " if n_custom else "")
        + "Nothing outside these steps changed the data, and the raw data is untouched.",
        "",
        f"- **Rows:** {rows_raw:,} raw -> {rows_clean:,} clean, {dropped:,} dropped"
        + (f" (each stored with its reason in `{DROPPED_CSV}`)" if dropped else ""),
        f"- **Columns:** {', '.join(run.columns)}",
        f"- **Plan:** {plan.summary}",
        "",
        "## Gate 2 checks",
        "",
    ]
    order = {False: 0, True: 1}
    for check in sorted(gate.checks, key=lambda c: (order[c.passed], c.severity != "block")):
        mark = "pass" if check.passed else f"**FAIL ({check.severity})**"
        lines.append(f"- {mark} `{check.name}`: {check.message}")
    lines += ["", "## Steps", ""]
    for number, (step, record) in enumerate(zip(plan.steps, run.records, strict=True), start=1):
        custom = step.op == "custom_code"
        lines.append(
            f"### {number}. `{step.op}`"
            + (" (custom code, not a library op)" if custom else "")
            + f" for {', '.join(step.addresses)}"
        )
        lines += ["", step.rationale, ""]
        changes = [f"{column}: {count:,} cell(s)" for column, count in record.cells_changed.items()]
        types = [
            f"{column}: {record.before[column].dtype} -> {record.after[column].dtype}"
            for column in record.target_columns
            if column in record.before
            and column in record.after
            and record.before[column].dtype != record.after[column].dtype
        ]
        nulls = [f"{column}: {count:,}" for column, count in record.new_nulls.items()]
        lines.append(f"- Changed: {'; '.join(changes) or 'no values'}")
        if types:
            lines.append(f"- Type: {'; '.join(types)}")
        if nulls:
            lines.append(f"- New missing values: {'; '.join(nulls)}")
        if record.rows_dropped:
            lines.append(f"- Rows dropped: {record.rows_dropped:,}")
        lines += [f"- Flag: {flag.message}" for flag in record.flags]
        if custom:
            declared = {
                k: v for k, v in step.params.items() if k != "code" and v not in (None, [], "")
            }
            lines += [
                f"- Declared: {json.dumps(declared, ensure_ascii=False)}",
                "",
                "```python",
                step.params["code"].strip(),
                "```",
            ]
        else:
            lines.append(f"- Parameters: `{json.dumps(record.params, ensure_ascii=False)}`")
        lines.append("")

    lines += _contract_table(result.contract)

    if plan.skipped:
        lines += ["## Not done, with reasons", ""]
        lines += [f"- **{s.id}**: {s.reason}" for s in plan.skipped]
        lines.append("")

    by_id = {d.id: d for d in decisions}
    if diagnosis.judgment_calls:
        lines += ["## Judgment calls", ""]
        for call in diagnosis.judgment_calls:
            decision = by_id.get(call.id)
            if decision is not None and decision.decided:
                lines.append(f"- **{call.id}** decided by you: {decision.choice}")
            else:
                lines.append(
                    f"- **{call.id}** still open, data left as it is: {call.question} "
                    f"(decide it in `{DECISIONS_FILE}` and run the coder again)"
                )
        lines.append("")

    if run.warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in run.warnings]
        lines.append("")

    cost = (
        f", ${result.cost_usd:.4f}" if any(a.cost_usd is not None for a in result.attempts) else ""
    )
    lines += [
        f"_Plan written by {result.attempts[-1].model} (prompt {result.prompt_version}, "
        f"{len(result.attempts)} attempt(s){cost}) from the masked profile and the diagnosis. "
        f"Every number in this report comes from the execution log._",
    ]
    return "\n".join(lines) + "\n"


GATE_VERDICT = {
    "pass": "The clean data matches the diagnosis and every check passed.",
    "flag": "The clean data is usable; the flagged checks below deserve a look.",
    "block": (
        "This result may not be used, so no clean data, handoff or schema contract was "
        "written. The failing checks are listed below."
    ),
}


def _contract_table(contract: list[ColumnSchema]) -> list[str]:
    lines = [
        "## Schema contract",
        "",
        "What each clean column is, checked against the diagnosis. "
        f"Saved as `{SCHEMA_CONTRACT_FILE}` and in `{HANDOFF_FILE}` when Gate 2 lets it through.",
        "",
        "| Column | Type | Expected | Role | Missing | Distinct | Note |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for column in contract:
        role = (column.role or "added") + (", key" if column.unique else "")
        lines.append(
            f"| {column.name} | {column.dtype} | {column.expected_type or ''} | {role} | "
            f"{column.null_rate:.1%} | {column.n_distinct:,} | "
            f"{(column.note or '').replace('|', '/')} |"
        )
    return [*lines, ""]


# -- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.agents.coder",
        description="Write and run a cleaning plan for a diagnosed run folder.",
    )
    parser.add_argument("run_dir", help="A run folder the analyst has diagnosed")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)

    if not (run_dir / "manifest.json").exists():
        print(f"{run_dir} is not a run folder (no manifest.json).")
        return 1
    try:
        client = connect()
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1

    print(f"Asking {client.config.model} for a cleaning plan...")
    try:
        result = run_on_folder(run_dir, client, client.config.public())
    except (RunFolderError, DecisionsError) as error:
        print(f"Cannot clean yet: {error}")
        return 1
    except CoderError as error:
        print(f"\n{error}\nEvery attempt is logged in {(run_dir / CODER_LOG).resolve()}")
        return 1

    run = result.run
    n_custom = sum(1 for s in result.plan.steps if s.op == "custom_code")
    cost = f", ${result.cost_usd:.4f}" if result.cost_usd else ""
    print(
        f"\nGate 2: {result.gate.outcome.upper()}. {len(result.plan.steps)} step(s) ran "
        f"({n_custom} custom code), {len(result.attempts)} attempt(s){cost}."
    )
    for warning in run.warnings:
        print(f"Warning: {warning}")
    print(f"Report:     {(run_dir / CLEANING_REPORT).resolve()}")
    if result.blocked:
        print("No clean data was written; the report says why.")
        return 1
    print(f"Clean data: {(run_dir / CLEAN_CSV).resolve()}")
    print(f"Handoff:    {(run_dir / HANDOFF_FILE).resolve()}")
    return 0


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


if __name__ == "__main__":
    sys.exit(main())
