"""Stage 2 end to end: the coder cleans, Gate 2 checks, the reviewer judges.

    uv run python -m pipeline.clean runs/<run folder> [--no-review] [--budget 0.50]

The coder writes and runs a plan (`pipeline/agents/coder.py`); Gate 2 decides
whether the result may be used at all; the reviewer (`pipeline/agents/reviewer.py`)
judges whether it is right, querying the data when it needs to, and its fix
requests go back to the coder, at most twice. What the reviewer cannot get fixed,
and every ambiguity it reports, is flagged for the user, never decided. A budget
stops the loop between rounds; a round in progress finishes.

Into the run folder go the logs and the report, always; the clean data,
handoff.json and the schema contract only when Gate 2 does not block. Those are
deleted first on every run, so a blocked re-run never leaves an old one behind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from pipeline.agents.analyst import DIAGNOSIS_JSON
from pipeline.agents.coder import CoderError, CoderResult, rerun, write_and_run
from pipeline.agents.coder import prompt_version as coder_prompt_version
from pipeline.agents.common import (
    Attempt,
    load_manifest,
    record_in_manifest,
    save_manifest,
    total_cost,
    write_attempt_log,
)
from pipeline.agents.decisions import DECISIONS_FILE, DecisionsError, load_decisions
from pipeline.agents.llm import LLMClient, LLMConfigError, connect
from pipeline.agents.reviewer import (
    ReviewerError,
    ReviewInputs,
    ReviewResult,
    masked_from_profile,
    review,
)
from pipeline.agents.reviewer import prompt_version as reviewer_prompt_version
from pipeline.contracts.cleaning import Decision
from pipeline.contracts.common import Flag
from pipeline.contracts.diagnosis import Diagnosis
from pipeline.contracts.handoff import ColumnSchema, Handoff
from pipeline.contracts.manifest import RunManifest
from pipeline.contracts.profile import DatasetProfile
from pipeline.ingestion.compact import compact_profile
from pipeline.ingestion.loaders import load
from pipeline.ingestion.profiler import profile_table
from pipeline.ingestion.raw import ROW_ID, RawStore, RawTable, quote_ident
from pipeline.layout import is_run_folder, run_file
from pipeline.tools.queries import QueryContext

CLEANING_PLAN = "cleaning_plan.json"
CLEANING_LOG = "cleaning_log.json"
CODER_LOG = "coder_log.json"
REVIEW_LOG = "review_log.json"
CLEANING_REPORT = "cleaning_report.md"
CLEAN_PARQUET = "clean_data.parquet"
CLEAN_CSV = "clean_data.csv"
DROPPED_CSV = "dropped_rows.csv"
GATE_2_FILE = "gate_2.json"
HANDOFF_FILE = "handoff.json"
SCHEMA_CONTRACT_FILE = "schema_contract.yaml"
DELIVERABLES = (CLEAN_PARQUET, CLEAN_CSV, DROPPED_CSV, HANDOFF_FILE, SCHEMA_CONTRACT_FILE)
"""Written only for a result Gate 2 let through; deleted first, so none is ever stale."""

MAX_LOOP_BACKS = 2
"""Spec Stage 3: the review sends fixes back to cleaning at most twice."""
DEFAULT_BUDGET_USD = 0.50
"""Spec 6: the target cost per dataset. Checked between rounds."""


class RunFolderError(ValueError):
    """The run folder is not ready to be cleaned, with what to do about it."""


@dataclass
class FolderInputs:
    manifest: RunManifest
    diagnosis: Diagnosis
    decisions: list[Decision]
    profile_llm: dict[str, Any]
    source: Path
    sheet: str | None
    table: str


@dataclass
class CleaningOutcome:
    result: CoderResult
    status: str
    """'accepted', 'flagged', 'not_reviewed' or 'blocked'."""
    reviews: list[ReviewResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    coder_attempts: list[Attempt] = field(default_factory=list)
    review_failures: list[Attempt] = field(default_factory=list)
    clean_profile: DatasetProfile | None = None

    @property
    def review_attempts(self) -> list[Attempt]:
        return [a for r in self.reviews for a in r.attempts] + self.review_failures

    @property
    def cost_usd(self) -> float:
        return total_cost(self.coder_attempts) + total_cost(self.review_attempts)


# -- the stage --------------------------------------------------------------


def load_inputs(run_dir: Path) -> FolderInputs:
    """Everything cleaning needs from the folder, or a clear reason it cannot start."""
    manifest = load_manifest(run_dir)
    if manifest.stage_status.get("ingestion") == "blocked":
        raise RunFolderError(
            "Gate 1 blocked this load, so it may not be cleaned. Fix the problems in "
            "ingestion_report.md and ingest the file again."
        )
    if not run_file(run_dir, DIAGNOSIS_JSON).exists():
        raise RunFolderError(
            "there is no diagnosis yet. Run the analyst first:\n"
            f"  uv run python -m pipeline.agents.analyst {run_dir.as_posix()}"
        )
    if not manifest.inputs:
        raise RunFolderError("the manifest records no input file")

    diagnosis = Diagnosis.model_validate_json(
        run_file(run_dir, DIAGNOSIS_JSON).read_text(encoding="utf-8")
    )
    profile_llm = json.loads(run_file(run_dir, "profile_llm.json").read_text(encoding="utf-8"))
    info = manifest.inputs[0]
    source = Path(info.path)
    if not source.exists():
        raise RunFolderError(
            f"the source file {info.path} was not found. Run the command from the folder "
            f"you ingested from."
        )
    return FolderInputs(
        manifest=manifest,
        diagnosis=diagnosis,
        decisions=load_decisions(run_file(run_dir, DECISIONS_FILE), diagnosis),
        profile_llm=profile_llm,
        source=source,
        sheet=info.sheet,
        table=profile_llm["table"].split(".", 1)[-1],
    )


def clean_folder(
    run_dir: Path,
    client: LLMClient,
    provider: dict[str, str],
    *,
    review_client: LLMClient | None = None,
    review_provider: dict[str, str] | None = None,
    reviewing: bool = True,
    budget_usd: float = DEFAULT_BUDGET_USD,
) -> CleaningOutcome:
    """Run Stage 2 on a diagnosed run folder and write everything next to it."""
    inputs = load_inputs(run_dir)
    manifest = inputs.manifest
    brief = manifest.context_brief

    with RawStore.open() as store:
        raw = _reload(store, inputs)
        context = {
            "store": store,
            "raw": raw,
            "run_id": manifest.run_id,
            "diagnosis": inputs.diagnosis,
            "brief": brief,
            "profile_llm": inputs.profile_llm,
        }
        try:
            result = write_and_run(
                inputs.profile_llm,
                brief,
                inputs.diagnosis,
                inputs.decisions,
                client,
                store=store,
                raw=raw,
                run_id=manifest.run_id,
            )
        except CoderError as error:
            write_attempt_log(run_file(run_dir, CODER_LOG), error.attempts, coder_prompt_version())
            record_in_manifest(manifest, "coder", provider, error.attempts, coder_prompt_version())
            manifest.stage_status["cleaning"] = "blocked"
            save_manifest(run_dir, manifest)
            raise

        outcome = CleaningOutcome(
            result=result,
            status="blocked" if result.blocked else "not_reviewed",
            coder_attempts=list(result.attempts),
        )
        if not result.blocked and reviewing:
            _review_loop(outcome, inputs, client, review_client or client, context, budget_usd)
        if not outcome.result.blocked:
            outcome.clean_profile = _profile_clean(store, outcome.result.run.clean_table)
        _write_outputs(run_dir, store, raw, outcome, inputs)

    record_in_manifest(manifest, "coder", provider, outcome.coder_attempts, coder_prompt_version())
    if outcome.review_attempts:
        record_in_manifest(
            manifest,
            "reviewer",
            review_provider or provider,
            outcome.review_attempts,
            reviewer_prompt_version(),
        )
    manifest.stage_status["cleaning"] = "blocked" if outcome.result.blocked else "done"
    if reviewing and not outcome.result.blocked:
        manifest.stage_status["review"] = outcome.status
    save_manifest(run_dir, manifest)
    return outcome


def _review_loop(
    outcome: CleaningOutcome,
    inputs: FolderInputs,
    coder_client: LLMClient,
    review_client: LLMClient,
    context: dict[str, Any],
    budget_usd: float,
) -> None:
    """Review, send fixes back, review again; at most MAX_LOOP_BACKS times."""
    brief = inputs.manifest.context_brief
    for round_number in range(1, MAX_LOOP_BACKS + 2):
        if round_number > 1 and outcome.cost_usd >= budget_usd:
            outcome.status = "flagged"
            what = f"review round {round_number}"
            outcome.notes.append(_budget_note(budget_usd, outcome.cost_usd, what))
            return
        current = outcome.result
        review_inputs, query_ctx = _review_setup(context, current, inputs)
        try:
            reviewed = review(review_client, review_inputs, query_ctx, round_number=round_number)
        except ReviewerError as error:
            outcome.review_failures += error.attempts
            outcome.status = "flagged"
            outcome.notes.append(
                f"The reviewer gave no usable verdict in round {round_number}; the cleaning "
                f"stands as Gate 2 left it, unreviewed beyond that point."
            )
            return
        outcome.reviews.append(reviewed)

        if reviewed.verdict.accepted:
            outcome.status = "flagged" if reviewed.weak else "accepted"
            return
        if round_number > MAX_LOOP_BACKS:
            outcome.status = "flagged"
            outcome.notes.append(
                f"After {MAX_LOOP_BACKS} rounds of fixes the reviewer still asks for "
                f"{len(reviewed.verdict.fix_requests)} fix(es); they are listed for you below."
            )
            return
        if outcome.cost_usd >= budget_usd:
            outcome.status = "flagged"
            outcome.notes.append(
                _budget_note(budget_usd, outcome.cost_usd, "applying the reviewer's fixes")
            )
            return

        try:
            fixed = write_and_run(
                inputs.profile_llm,
                brief,
                inputs.diagnosis,
                inputs.decisions,
                coder_client,
                store=context["store"],
                raw=context["raw"],
                run_id=context["run_id"],
                fix_requests=reviewed.verdict.fix_requests,
                previous_plan=current.plan,
            )
        except CoderError as error:
            outcome.coder_attempts += error.attempts
            outcome.result = rerun(current, **context)
            outcome.status = "flagged"
            outcome.notes.append(
                "The coder could not turn the reviewer's fixes into a plan that runs; "
                "the previous result is kept."
            )
            return
        outcome.coder_attempts += fixed.attempts
        if fixed.blocked:
            outcome.result = rerun(current, **context)
            outcome.status = "flagged"
            outcome.notes.append(
                "Applying the reviewer's fixes made Gate 2 block; the previous result is kept."
            )
            return
        outcome.result = fixed


def _review_setup(
    context: dict[str, Any], result: CoderResult, inputs: FolderInputs
) -> tuple[ReviewInputs, QueryContext]:
    store, raw = context["store"], context["raw"]
    clean_table = result.run.clean_table
    profile_clean = compact_profile(_profile_clean(store, clean_table))
    masked = set(masked_from_profile(inputs.profile_llm) | masked_from_profile(profile_clean))
    masked |= {c.name for c in result.contract if c.source_column in masked}
    review_inputs = ReviewInputs(
        brief=inputs.manifest.context_brief,
        diagnosis=inputs.diagnosis,
        decisions=inputs.decisions,
        plan=result.plan,
        records=result.run.records,
        gate=result.gate,
        contract=result.contract,
        profile_raw=inputs.profile_llm,
        profile_clean=profile_clean,
        masked=frozenset(masked),
    )
    query_ctx = QueryContext(
        store=store,
        raw_table=raw.name,
        clean_table=clean_table,
        masked=frozenset(masked),
        clean_to_raw={c.name: c.source_column for c in result.contract if c.source_column},
    )
    return review_inputs, query_ctx


def _reload(store: RawStore, inputs: FolderInputs) -> RawTable:
    """Load the source again, refusing it if it is not the file that was ingested."""
    expected = inputs.manifest.inputs[0].file_hash
    load_result, raw = load(inputs.source, store, table=inputs.table, sheet=inputs.sheet)
    if load_result.file.file_hash != expected:
        raise RunFolderError(
            f"{inputs.source.name} has changed since it was ingested (hash {expected[:12]} "
            f"then, {load_result.file.file_hash[:12]} now). Ingest it again so the profile "
            f"and diagnosis describe the current file."
        )
    return raw


def _profile_clean(store: RawStore, table: str) -> DatasetProfile:
    clean = RawTable(
        name=table,
        columns=store.columns_of(table),
        column_types=store.column_types_of(table),
        typed=True,
        n_rows=store.row_count(table),
        data_hash=store.data_hash(table),
    )
    return profile_table(store, clean)


def _budget_note(budget: float, spent: float, what: str) -> str:
    return (
        f"Stopped before {what}: this run has spent ${spent:.4f} of its ${budget:.2f} budget. "
        f"Run again with a higher --budget to continue."
    )


# -- outputs ----------------------------------------------------------------


def _write_outputs(
    run_dir: Path,
    store: RawStore,
    raw: RawTable,
    outcome: CleaningOutcome,
    inputs: FolderInputs,
) -> None:
    """The logs and the report always; clean data, handoff and contract only past Gate 2."""
    for name in DELIVERABLES:
        run_file(run_dir, name).unlink(missing_ok=True)

    result = outcome.result
    run, gate = result.run, result.gate
    rows_clean = store.row_count(run.clean_table)
    run_file(run_dir, CLEANING_PLAN).write_text(
        result.plan.model_dump_json(indent=2), encoding="utf-8"
    )
    log = {
        "raw_table": raw.name,
        "clean_table": run.clean_table,
        "rows_raw": raw.n_rows,
        "rows_clean": rows_clean,
        "warnings": run.warnings,
        "records": [r.model_dump(mode="json") for r in run.records],
        "gate_2": gate.model_dump(mode="json"),
    }
    run_file(run_dir, CLEANING_LOG).write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_attempt_log(run_file(run_dir, CODER_LOG), outcome.coder_attempts, coder_prompt_version())
    run_file(run_dir, GATE_2_FILE).write_text(gate.model_dump_json(indent=2), encoding="utf-8")
    if outcome.reviews or outcome.review_failures:
        run_file(run_dir, REVIEW_LOG).write_text(
            json.dumps(_review_log(outcome), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if run.dropped_table:
        _copy(
            store,
            f"SELECT * EXCLUDE ({ROW_ID}) FROM {_qualified(run.dropped_table)} ORDER BY {ROW_ID}",
            run_file(run_dir, DROPPED_CSV),
            "HEADER, DELIMITER ','",
        )
    if not result.blocked:
        clean = f"SELECT * EXCLUDE ({ROW_ID}) FROM {_qualified(run.clean_table)} ORDER BY {ROW_ID}"
        _copy(store, clean, run_file(run_dir, CLEAN_PARQUET), "FORMAT parquet")
        _copy(store, clean, run_file(run_dir, CLEAN_CSV), "HEADER, DELIMITER ','")
        handoff = _handoff(store, raw, outcome, inputs, rows_clean=rows_clean)
        run_file(run_dir, HANDOFF_FILE).write_text(
            handoff.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
        )
        run_file(run_dir, SCHEMA_CONTRACT_FILE).write_text(
            _contract_yaml(handoff, gate.outcome), encoding="utf-8"
        )

    report = render_report(
        outcome, inputs, rows_raw=raw.n_rows, rows_clean=rows_clean, run_dir=run_dir
    )
    run_file(run_dir, CLEANING_REPORT).write_text(report, encoding="utf-8")


def _handoff(
    store: RawStore,
    raw: RawTable,
    outcome: CleaningOutcome,
    inputs: FolderInputs,
    *,
    rows_clean: int,
) -> Handoff:
    """What Stage 3 reads instead of the data: the contract, the accounting, every flag."""
    result = outcome.result
    run = result.run
    dropped_columns: dict[str, str] = {}
    for step, record in zip(result.plan.steps, run.records, strict=True):
        if record.op_name == "drop_column":
            dropped_columns[record.params["column"]] = record.params["reason"]
        elif record.op_name == "custom_code":
            for column in record.params.get("drops") or []:
                dropped_columns[column] = step.rationale
    flags = list(result.gate.flags)
    if outcome.reviews:
        flags += outcome.reviews[-1].verdict.findings
    flags += [
        Flag(code="cleaning_note", severity="warn", message=note, stage="cleaning")
        for note in outcome.notes
    ]
    flags += _decision_flags(inputs)
    return Handoff(
        run_id=inputs.manifest.run_id,
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
        profile=outcome.clean_profile,
        flags=flags,
        data_hash=store.data_hash(run.clean_table),
    )


def _decision_flags(inputs: FolderInputs) -> list[Flag]:
    """Every settled judgment call travels on, with who settled it and how sure the AI was."""
    by_id = {d.id: d for d in inputs.decisions}
    flags = []
    for call in inputs.diagnosis.judgment_calls:
        decision = by_id.get(call.id)
        if decision is None or not decision.decided:
            continue
        ai = call.ai_decision
        if decision.decided_by == "ai" and ai is not None:
            flags.append(
                Flag(
                    code="ai_decision",
                    severity="warn" if ai.confidence == "low" else "info",
                    message=f"{call.id}: '{ai.choice}' decided by the AI ({ai.confidence} "
                    f"confidence). {ai.reasoning}",
                    column=call.column,
                    stage="cleaning",
                    details={
                        "choice": ai.choice,
                        "confidence": ai.confidence,
                        "least_destructive": ai.least_destructive,
                        "alternatives": [a.option for a in ai.alternatives],
                    },
                )
            )
        else:
            flags.append(
                Flag(
                    code="user_decision",
                    severity="info",
                    message=f"{call.id}: '{decision.choice}' decided by the user.",
                    column=call.column,
                    stage="cleaning",
                )
            )
    return flags


def _review_log(outcome: CleaningOutcome) -> dict[str, Any]:
    return {
        "status": outcome.status,
        "notes": outcome.notes,
        "rounds": [
            {
                "round": r.round,
                "accepted": r.verdict.accepted,
                "weak": r.weak,
                "prompt_version": r.prompt_version,
                "verdict": r.verdict.model_dump(mode="json"),
                "outcome": r.outcome.model_dump(mode="json"),
                "dropped": r.dropped,
                "queries": [asdict(q) for q in r.queries],
                "attempts": [asdict(a) for a in r.attempts],
            }
            for r in outcome.reviews
        ],
        "failed_attempts": [asdict(a) for a in outcome.review_failures],
    }


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

GATE_VERDICT = {
    "pass": "The clean data matches the diagnosis and every check passed.",
    "flag": "The clean data is usable; the flagged checks below deserve a look.",
    "block": (
        "This result may not be used, so no clean data, handoff or schema contract was "
        "written. The failing checks are listed below."
    ),
}
REVIEW_VERDICT = {
    "accepted": "The reviewer checked the result and asks for no further fixes.",
    "flagged": "Some review points are unresolved; see 'For you to look at' below.",
    "not_reviewed": "Not run (--no-review).",
}


def render_report(
    outcome: CleaningOutcome,
    inputs: FolderInputs,
    *,
    rows_raw: int,
    rows_clean: int,
    run_dir: Path | None = None,
) -> str:
    """The cleaning report. Every number comes from the logs, never from an LLM."""
    result = outcome.result
    run, plan, gate = result.run, result.plan, result.gate
    n_custom = sum(1 for s in plan.steps if s.op == "custom_code")
    dropped = sum(r.rows_dropped for r in run.records)
    lines = [
        f"# Cleaning report: {run.clean_table}",
        "",
        f"**Gate 2: {gate.outcome.upper()}.** {GATE_VERDICT[gate.outcome]}",
    ]
    if outcome.status in REVIEW_VERDICT:
        lines += ["", f"**Review: {outcome.status.upper()}.** {REVIEW_VERDICT[outcome.status]}"]
    lines += [
        "",
        f"All {len(plan.steps)} step(s) of the final plan ran. "
        + (f"{n_custom} used custom code, shown in full below. " if n_custom else "")
        + "Nothing outside these steps changed the data, and the raw data is untouched.",
        "",
        f"- **Rows:** {rows_raw:,} raw -> {rows_clean:,} clean, {dropped:,} dropped"
        + (f" (each stored with its reason in `{DROPPED_CSV}`)" if dropped else ""),
        f"- **Columns:** {', '.join(run.columns)}",
        f"- **Plan:** {plan.summary}",
        "",
    ]
    lines += _ai_decisions(outcome, inputs, run_dir)
    lines += _for_you(outcome, inputs)
    lines += ["## Gate 2 checks", ""]
    order = {False: 0, True: 1}
    for check in sorted(gate.checks, key=lambda c: (order[c.passed], c.severity != "block")):
        mark = "pass" if check.passed else f"**FAIL ({check.severity})**"
        lines.append(f"- {mark} `{check.name}`: {check.message}")
    lines += ["", "## Steps", ""]
    lines += _steps(result)
    lines += _contract_table(result.contract)

    if plan.skipped:
        lines += ["## Not done, with reasons", ""]
        lines += [f"- **{s.id}**: {s.reason}" for s in plan.skipped]
        lines.append("")

    by_id = {d.id: d for d in inputs.decisions}
    if inputs.diagnosis.judgment_calls:
        lines += ["## Judgment calls", ""]
        for call in inputs.diagnosis.judgment_calls:
            decision = by_id.get(call.id)
            if decision is not None and decision.decided and decision.decided_by == "ai":
                lines.append(
                    f"- **{call.id}** decided by the AI: {decision.choice} "
                    f"(see 'Decided by the AI' above)"
                )
            elif decision is not None and decision.decided:
                lines.append(f"- **{call.id}** decided by you: {decision.choice}")
            else:
                lines.append(
                    f"- **{call.id}** still open, data left as it is: {call.question} "
                    f"(decide it in `{DECISIONS_FILE}` and run the cleaning again)"
                )
        lines.append("")

    if run.warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in run.warnings]
        lines.append("")

    lines += _review_section(outcome)

    coder_cost = total_cost(outcome.coder_attempts)
    review_cost = total_cost(outcome.review_attempts)
    lines.append(
        f"_Coder: {len(outcome.coder_attempts)} call(s), ${coder_cost:.4f}. "
        f"Reviewer: {len(outcome.review_attempts)} call(s), ${review_cost:.4f}. "
        f"Written by LLMs from masked profiles and logs; every number in this report "
        f"comes from the execution log or a logged query._"
    )
    return "\n".join(lines) + "\n"


def _ai_decided(inputs: FolderInputs) -> list:
    """The judgment calls the AI settled and the user has not changed, least certain first."""
    by_id = {d.id: d for d in inputs.decisions}
    calls = [
        call
        for call in inputs.diagnosis.judgment_calls
        if call.ai_decision is not None
        and by_id.get(call.id) is not None
        and by_id[call.id].decided
        and by_id[call.id].decided_by == "ai"
    ]
    order = {"low": 0, "medium": 1, "high": 2}
    return sorted(calls, key=lambda c: order[c.ai_decision.confidence])


def _ai_decisions(
    outcome: CleaningOutcome, inputs: FolderInputs, run_dir: Path | None
) -> list[str]:
    """Every choice the AI made for the user: what, why, how sure, what it did, how to undo it."""
    calls = _ai_decided(inputs)
    if not calls:
        return []
    folder = run_dir.as_posix() if run_dir is not None else "<this run folder>"
    lines = [
        "## Decided by the AI: check these",
        "",
        f"The AI settled {len(calls)} judgment call(s) from your brief "
        f"(`judgment_calls: decide`), least certain first. Each can be undone: change its "
        f"`decision:` in `{DECISIONS_FILE}` and run the cleaning again. The raw data is never "
        f"changed.",
        "",
    ]
    for call in calls:
        ai = call.ai_decision
        where = f" ({call.column})" if call.column else ""
        confidence = f"- **Confidence:** {ai.confidence}"
        if ai.least_destructive:
            confidence += (
                ". The brief does not settle this, so the least destructive option was "
                "taken. **Please check it.**"
            )
        lines += [
            f"### {call.id}{where}: {call.question}",
            "",
            f"- **Decision:** {ai.choice}",
            confidence,
            f"- **Why:** {ai.reasoning}",
            f"- **What it did (measured):** {_measured(call.id, outcome.result)}",
            f"- **Expected effect (the AI's estimate):** {ai.expected_effect}",
        ]
        if ai.alternatives:
            lines.append("- **Alternatives (the AI's estimates, not measured):**")
            lines += [f"  - {alt.option}: {alt.effect}" for alt in ai.alternatives]
        lines += [
            f"- **To change it:** set `decision:` under `{call.id}` in `{DECISIONS_FILE}` to "
            f"another option, your own words, or `undecided`, then run "
            f"`uv run python -m pipeline.clean {folder}`",
            "",
        ]
    return lines


def _measured(call_id: str, result: CoderResult) -> str:
    """What the steps carrying out a decision changed, from the execution log."""
    parts = []
    steps = zip(result.plan.steps, result.run.records, strict=True)
    for number, (step, record) in enumerate(steps, start=1):
        if call_id not in step.addresses:
            continue
        changed = ", ".join(f"{c} {n} cell(s)" for c, n in record.cells_changed.items())
        details = [changed or "no values changed"]
        if record.new_nulls:
            details.append(
                "new missing values: " + ", ".join(f"{c} {n}" for c, n in record.new_nulls.items())
            )
        if record.rows_dropped:
            details.append(f"{record.rows_dropped} row(s) dropped")
        parts.append(f"step {number} ({step.op}): " + "; ".join(details))
    if parts:
        return "; ".join(parts)
    skipped = [s.reason for s in result.plan.skipped if s.id == call_id]
    if skipped:
        return f"no change to the data. {skipped[0]}"
    return "no step carried it out"


def _for_you(outcome: CleaningOutcome, inputs: FolderInputs) -> list[str]:
    """Everything a person should look at, in one place near the top."""
    items = list(outcome.notes)
    items += [
        f"{call.id} was decided by the AI with low confidence: '{call.ai_decision.choice}', "
        f"the least destructive option. Check it under 'Decided by the AI'."
        for call in _ai_decided(inputs)
        if call.ai_decision.confidence == "low"
    ]
    # The final round saw the final data, so its questions are the current ones; earlier
    # rounds' questions stay in the Review section and review_log.json.
    if outcome.reviews:
        for ambiguity in outcome.reviews[-1].outcome.ambiguities:
            where = f" ({ambiguity.column})" if ambiguity.column else ""
            items.append(f"Question{where}: {ambiguity.question} Evidence: {ambiguity.evidence}")
    if outcome.reviews:
        final = outcome.reviews[-1]
        if not final.verdict.accepted:
            items += [
                f"Unresolved fix request ({fix.column or 'table'}): {fix.issue}. {fix.reason}"
                for fix in final.verdict.fix_requests
            ]
        items += [
            f"Review point ({finding.column or 'table'}): {finding.problem} "
            f"Evidence: {finding.evidence}"
            for finding in final.outcome.findings
            if finding.fix is None
        ]
        if final.weak:
            items.append(
                "The reviewer could not verify enough with numbers, so its verdict carries "
                "little weight."
            )
    if not items:
        return []
    return ["## For you to look at", "", *[f"- {item}" for item in items], ""]


def _steps(result: CoderResult) -> list[str]:
    lines: list[str] = []
    steps = zip(result.plan.steps, result.run.records, strict=True)
    for number, (step, record) in enumerate(steps, start=1):
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
    return lines


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


def _review_section(outcome: CleaningOutcome) -> list[str]:
    if not outcome.reviews:
        return []
    lines = ["## Review", ""]
    for reviewed in outcome.reviews:
        verdict = reviewed.verdict
        state = (
            "accepted"
            if verdict.accepted
            else f"sent back with {len(verdict.fix_requests)} fix request(s)"
        )
        lines += [f"### Round {reviewed.round}: {state}", "", reviewed.outcome.summary, ""]
        if reviewed.outcome.verified:
            lines.append("Verified:")
            lines += [f"- {v.claim} ({v.evidence})" for v in reviewed.outcome.verified]
        if reviewed.outcome.findings:
            lines.append("Findings:")
            for finding in reviewed.outcome.findings:
                where = f"step {finding.step}, " if finding.step else ""
                fix = f" Fix asked: {finding.fix.issue}." if finding.fix else ""
                lines.append(
                    f"- [{finding.severity}] {where}{finding.column or 'table'}: "
                    f"{finding.problem} ({finding.evidence}).{fix}"
                )
        if reviewed.queries:
            lines.append("Queries it ran:")
            lines += [
                f"- {q.query.get('id')} ({q.query.get('purpose')}): {q.answer.splitlines()[0]}"
                for q in reviewed.queries
            ]
        if reviewed.dropped:
            lines.append("Claims discarded for citing numbers it was never shown:")
            lines += [f"- {d}" for d in reviewed.dropped]
        lines.append("")
    return lines


# -- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.clean",
        description="Clean a diagnosed run folder: coder, Gate 2, reviewer.",
    )
    parser.add_argument("run_dir", help="A run folder the analyst has diagnosed")
    parser.add_argument("--no-review", action="store_true", help="Skip the reviewer")
    parser.add_argument(
        "--budget",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=f"Stop between rounds once this run has spent this many dollars "
        f"(default {DEFAULT_BUDGET_USD})",
    )
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)

    if not is_run_folder(run_dir):
        print(f"{run_dir} is not a run folder (no logs/manifest.json).")
        return 1
    try:
        client = connect()
        review_model = os.environ.get("LLM_REVIEW_MODEL", "").strip() or None
        review_client = connect(model=review_model) if review_model else client
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1

    reviewing = not args.no_review
    print(
        f"Cleaning with {client.config.model}"
        + (f", reviewing with {review_client.config.model}" if reviewing else ", no review")
        + "..."
    )
    try:
        outcome = clean_folder(
            run_dir,
            client,
            client.config.public(),
            review_client=review_client,
            review_provider=review_client.config.public(),
            reviewing=reviewing,
            budget_usd=args.budget,
        )
    except (RunFolderError, DecisionsError) as error:
        print(f"Cannot clean yet: {error}")
        return 1
    except CoderError as error:
        print(f"\n{error}\nEvery attempt is logged in {run_file(run_dir, CODER_LOG).resolve()}")
        return 1

    result = outcome.result
    from pipeline.reports import build_reports  # the reports read every step; import late

    report = build_reports(run_dir)
    review_line = (
        f" Review: {outcome.status.upper()} after {len(outcome.reviews)} round(s)."
        if reviewing and not result.blocked
        else ""
    )
    print(
        f"\nGate 2: {result.gate.outcome.upper()}.{review_line} "
        f"{len(result.plan.steps)} step(s), ${outcome.cost_usd:.4f} this run."
    )
    for note in outcome.notes:
        print(f"Note: {note}")
    print(f"Start here: {report.resolve()}")
    if result.blocked:
        print("No clean data was written; the report says why.")
        return 1
    print(f"Clean data: {run_file(run_dir, CLEAN_CSV).resolve()}")
    print(
        "To update the findings for the new clean data, run "
        f"`uv run python -m pipeline.analyze {run_dir.as_posix()}`."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
