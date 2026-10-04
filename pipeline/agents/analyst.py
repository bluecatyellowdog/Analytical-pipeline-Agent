"""The analyst agent: masked profile + brief in, checked diagnosis out. Changes nothing.

The LLM proposes; deterministic checks decide whether the proposal is usable.
A reply is parsed into `Diagnosis`, then checked against the actual columns and
the brief (every column covered, target and exclusions respected, target never
imputed, suggested ops exist). Problems go back to the model, at most twice.
Every attempt is logged with its model, tokens, cost and problems.

    uv run python -m pipeline.agents.analyst runs/<run folder>
"""

from __future__ import annotations

import argparse
import json
import sys
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
    load_manifest,
    profile_block,
    record_in_manifest,
    save_manifest,
    short_hash,
    total_cost,
    write_attempt_log,
)
from pipeline.agents.decisions import DECISIONS_FILE, write_template
from pipeline.agents.llm import LLMClient, LLMConfigError, Message, connect
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.diagnosis import Diagnosis
from pipeline.ops import REGISTRY, op_names

PROMPT_PATH = Path(__file__).parent / "prompts" / "analyst.md"
DIAGNOSIS_JSON = "diagnosis.json"
DIAGNOSIS_MD = "diagnosis.md"
ANALYST_LOG = "analyst_log.json"


class AnalystError(RuntimeError):
    """No usable diagnosis after the allowed retries."""

    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass
class AnalystResult:
    diagnosis: Diagnosis
    attempts: list[Attempt]
    prompt_version: str

    @property
    def cost_usd(self) -> float:
        return total_cost(self.attempts)


def system_prompt() -> str:
    ops = "\n".join(
        f"- {name}: {(REGISTRY[name].doc.strip().splitlines() or [''])[0]}" for name in op_names()
    )
    schema = json.dumps(Diagnosis.model_json_schema(), indent=1)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{ops}", ops).replace("{schema}", schema)


def prompt_version() -> str:
    """Changes whenever the prompt, the op list or the output schema changes."""
    return short_hash(system_prompt())


def build_messages(profile_llm: dict[str, Any], brief: ContextBrief) -> list[Message]:
    """The whole prompt. Built only from the compact profile and the brief."""
    user = (
        f"Context brief:\n{brief_block(brief)}\n"
        f"{profile_block(profile_llm)}\n\n"
        "Return the diagnosis as one JSON object."
    )
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def parse_reply(text: str) -> Diagnosis:
    """The JSON object in the reply, as a Diagnosis. Raises ValueError if there is none."""
    try:
        return Diagnosis.model_validate_json(extract_json_object(text))
    except ValidationError as error:
        raise ValueError(compact_validation_error(error)) from error


def check_diagnosis(
    diagnosis: Diagnosis, profile_llm: dict[str, Any], brief: ContextBrief
) -> list[str]:
    """Problems a structurally valid diagnosis can still have. Empty means usable."""
    actual = [c["name"] for c in profile_llm["columns"]]
    problems: list[str] = []

    listed = [c.name for c in diagnosis.columns]
    for name in listed:
        if name not in actual:
            problems.append(f"columns lists {name!r}, which is not in the data")
    for name in actual:
        if name not in listed:
            problems.append(f"column {name!r} has no entry in columns; every column needs one")

    for item in [*diagnosis.issues, *diagnosis.judgment_calls]:
        if item.column is not None and item.column not in actual:
            problems.append(f"{item.id} refers to column {item.column!r}, which is not in the data")

    known_ops = set(op_names())
    for issue in diagnosis.issues:
        if issue.suggested_op is not None and issue.suggested_op not in known_ops:
            problems.append(
                f"{issue.id} suggests op {issue.suggested_op!r}, which does not exist; "
                f"use one of {sorted(known_ops)} or null"
            )

    targets = [c.name for c in diagnosis.columns if c.role == "target"]
    if len(targets) > 1:
        problems.append(f"more than one column has role 'target': {targets}")
    if brief.target and brief.target in actual:
        entry = diagnosis.column(brief.target)
        if entry is not None and entry.role != "target":
            problems.append(
                f"the brief names {brief.target!r} as the target; give it role 'target'"
            )
        for issue in diagnosis.issues:
            if issue.column == brief.target and issue.suggested_op == "impute":
                problems.append(
                    f"{issue.id} would impute the target {brief.target!r}; never impute the target"
                )

    for name in brief.excluded_variables:
        entry = diagnosis.column(name)
        if name in actual and entry is not None and entry.role != "excluded":
            problems.append(f"the brief excludes {name!r}; give it role 'excluded'")

    return problems


def diagnose(
    profile_llm: dict[str, Any],
    brief: ContextBrief,
    client: LLMClient,
    *,
    max_retries: int = MAX_RETRIES,
) -> AnalystResult:
    """Ask, check, and send problems back until the diagnosis is usable or retries run out."""
    messages = build_messages(profile_llm, brief)
    attempts: list[Attempt] = []

    for number in range(1, max_retries + 2):
        response, attempt = ask(client, messages, number)
        attempts.append(attempt)

        try:
            diagnosis = parse_reply(response.text)
            attempt.problems = check_diagnosis(diagnosis, profile_llm, brief)
        except ValueError as error:
            attempt.problems = [str(error)]

        if not attempt.problems:
            return AnalystResult(diagnosis, attempts, prompt_version())

        attempt.feedback = (
            "The diagnosis cannot be used yet:\n"
            + "\n".join(f"- {p}" for p in attempt.problems)
            + "\nReturn the complete corrected JSON object."
        )
        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            {"role": "user", "content": attempt.feedback},
        ]

    raise AnalystError(
        f"no usable diagnosis after {len(attempts)} attempts; "
        f"last problems: {attempts[-1].problems}",
        attempts,
    )


def render_markdown(diagnosis: Diagnosis, result: AnalystResult | None = None) -> str:
    """A readable version for the run folder."""
    lines = ["# Diagnosis", "", diagnosis.summary, ""]

    lines += ["## Issues to fix", ""]
    if not diagnosis.issues:
        lines.append("None.")
    for issue in diagnosis.issues:
        where = f" `{issue.column}`" if issue.column else ""
        op = f" (op: `{issue.suggested_op}`)" if issue.suggested_op else ""
        lines += [
            f"- **{issue.id}**{where}, {issue.kind}: {issue.fix}{op}",
            f"  - Evidence: {issue.evidence}",
        ]

    lines += ["", "## Judgment calls (for you to decide)", ""]
    if not diagnosis.judgment_calls:
        lines.append("None.")
    for call in diagnosis.judgment_calls:
        where = f" `{call.column}`" if call.column else ""
        lines += [
            f"- **{call.id}**{where}: {call.question}",
            *(f"  - Option: {option}" for option in call.options),
            f"  - Recommendation: {call.recommendation}",
            f"  - Why it matters: {call.why_it_matters}",
        ]

    lines += [
        "",
        "## Columns",
        "",
        "| Column | Role | Clean type | Notes |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c.name} | {c.role} | {c.target_type} | {(c.notes or '').replace('|', '/')} |"
        for c in diagnosis.columns
    ]

    if result is not None:
        cost = (
            f", ${result.cost_usd:.4f}"
            if any(a.cost_usd is not None for a in result.attempts)
            else ""
        )
        lines += [
            "",
            f"_Model {result.attempts[-1].model}, prompt {result.prompt_version}, "
            f"{len(result.attempts)} attempt(s){cost}. Written by an LLM from the masked profile; "
            f"nothing in the data has been changed._",
        ]
    return "\n".join(lines) + "\n"


def run_on_folder(run_dir: Path, client: LLMClient, provider: dict[str, str]) -> AnalystResult:
    """Diagnose an ingestion run folder and write the results next to it."""
    profile_llm = json.loads((run_dir / "profile_llm.json").read_text(encoding="utf-8"))
    manifest = load_manifest(run_dir)

    try:
        result = diagnose(profile_llm, manifest.context_brief, client)
    except AnalystError as error:
        write_attempt_log(run_dir / ANALYST_LOG, error.attempts, prompt_version())
        record_in_manifest(manifest, "analyst", provider, error.attempts, prompt_version())
        manifest.stage_status["diagnosis"] = "blocked"
        save_manifest(run_dir, manifest)
        raise

    (run_dir / DIAGNOSIS_JSON).write_text(
        result.diagnosis.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / DIAGNOSIS_MD).write_text(render_markdown(result.diagnosis, result), encoding="utf-8")
    write_template(run_dir / DECISIONS_FILE, result.diagnosis)
    write_attempt_log(run_dir / ANALYST_LOG, result.attempts, result.prompt_version)

    record_in_manifest(manifest, "analyst", provider, result.attempts, result.prompt_version)
    manifest.stage_status["diagnosis"] = "done"
    save_manifest(run_dir, manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.agents.analyst",
        description="Diagnose an ingested dataset with the LLM. Changes no data.",
    )
    parser.add_argument("run_dir", help="An ingestion run folder, e.g. runs/danish_semicolon_...")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not (run_dir / "profile_llm.json").exists():
        print(f"{run_dir} is not an ingestion run folder (no profile_llm.json).")
        return 1
    try:
        client = connect()
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1

    print(f"Asking {client.config.model} for a diagnosis...")
    try:
        result = run_on_folder(run_dir, client, client.config.public())
    except AnalystError as error:
        print(f"\n{error}\nEvery attempt is logged in {(run_dir / ANALYST_LOG).resolve()}")
        return 1

    print()
    print(render_markdown(result.diagnosis, result))
    print(f"Wrote {(run_dir / DIAGNOSIS_MD).resolve()}")
    if result.diagnosis.judgment_calls:
        print(
            f"Decide the {len(result.diagnosis.judgment_calls)} judgment call(s) in "
            f"{(run_dir / DECISIONS_FILE).resolve()}, then run the coder:\n"
            f"  uv run python -m pipeline.agents.coder {run_dir.as_posix()}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
