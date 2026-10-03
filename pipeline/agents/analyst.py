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
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pipeline.agents.llm import LLMClient, LLMConfigError, Message, connect
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.diagnosis import Diagnosis
from pipeline.contracts.manifest import ModelInfo, RunManifest
from pipeline.ops import REGISTRY, op_names

PROMPT_PATH = Path(__file__).parent / "prompts" / "analyst.md"
MAX_RETRIES = 2
DIAGNOSIS_JSON = "diagnosis.json"
DIAGNOSIS_MD = "diagnosis.md"
ANALYST_LOG = "analyst_log.json"


class AnalystError(RuntimeError):
    """No usable diagnosis after the allowed retries."""

    def __init__(self, message: str, attempts: list[Attempt]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass
class Attempt:
    number: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    seconds: float
    problems: list[str] = field(default_factory=list)
    reply: str = ""
    """The model's raw reply, kept for the audit trail."""


@dataclass
class AnalystResult:
    diagnosis: Diagnosis
    attempts: list[Attempt]
    prompt_version: str

    @property
    def cost_usd(self) -> float:
        return sum(a.cost_usd or 0.0 for a in self.attempts)


def system_prompt() -> str:
    ops = "\n".join(
        f"- {name}: {(REGISTRY[name].doc.strip().splitlines() or [''])[0]}" for name in op_names()
    )
    schema = json.dumps(Diagnosis.model_json_schema(), indent=1)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{ops}", ops).replace("{schema}", schema)


def prompt_version() -> str:
    """Changes whenever the prompt, the op list or the output schema changes."""
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()[:12]


def build_messages(profile_llm: dict[str, Any], brief: ContextBrief) -> list[Message]:
    """The whole prompt. Built only from the compact profile and the brief."""
    brief_text = yaml.safe_dump(
        brief.model_dump(mode="json", exclude={"claims"}, exclude_none=True, exclude_defaults=True),
        allow_unicode=True,
        sort_keys=False,
    )
    # '<' is escaped inside the JSON so no value in the data can close the block.
    profile_text = json.dumps(profile_llm, indent=1, ensure_ascii=False).replace("<", "\\u003c")
    user = (
        f"Context brief:\n{brief_text}\n"
        f"<profile>\n{profile_text}\n</profile>\n\n"
        "Return the diagnosis as one JSON object."
    )
    return [{"role": "system", "content": system_prompt()}, {"role": "user", "content": user}]


def parse_reply(text: str) -> Diagnosis:
    """The JSON object in the reply, as a Diagnosis. Raises ValueError if there is none."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the reply contains no JSON object")
    try:
        return Diagnosis.model_validate_json(candidate[start : end + 1])
    except ValidationError as error:
        raise ValueError(_compact_validation_error(error)) from error


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
        started = time.perf_counter()
        response = client.complete(messages)
        attempt = Attempt(
            number=number,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=response.cost_usd,
            seconds=round(time.perf_counter() - started, 2),
            reply=response.text,
        )
        attempts.append(attempt)

        try:
            diagnosis = parse_reply(response.text)
            attempt.problems = check_diagnosis(diagnosis, profile_llm, brief)
        except ValueError as error:
            attempt.problems = [str(error)]

        if not attempt.problems:
            return AnalystResult(diagnosis, attempts, prompt_version())

        messages = [
            *messages,
            {"role": "assistant", "content": response.text},
            {
                "role": "user",
                "content": "The diagnosis cannot be used yet:\n"
                + "\n".join(f"- {p}" for p in attempt.problems)
                + "\nReturn the complete corrected JSON object.",
            },
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
    manifest_path = run_dir / "manifest.json"
    manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    try:
        result = diagnose(profile_llm, manifest.context_brief, client)
    except AnalystError as error:
        _write_log(run_dir, error.attempts, prompt_version())
        manifest.stage_status["diagnosis"] = "blocked"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        raise

    (run_dir / DIAGNOSIS_JSON).write_text(
        result.diagnosis.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / DIAGNOSIS_MD).write_text(render_markdown(result.diagnosis, result), encoding="utf-8")
    _write_log(run_dir, result.attempts, result.prompt_version)

    manifest.models["analyst"] = ModelInfo(
        provider=provider.get("provider", "unknown"),
        name=result.attempts[-1].model,
        temperature=0.0,
    )
    manifest.prompt_versions["analyst"] = result.prompt_version
    for attempt in result.attempts:
        manifest.cost.add(
            prompt_tokens=attempt.input_tokens,
            completion_tokens=attempt.output_tokens,
            usd=attempt.cost_usd or 0.0,
            wall_clock_s=attempt.seconds,
        )
    manifest.stage_status["diagnosis"] = "done"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return result


def _write_log(run_dir: Path, attempts: list[Attempt], version: str) -> None:
    log = {"prompt_version": version, "attempts": [asdict(a) for a in attempts]}
    (run_dir / ANALYST_LOG).write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _compact_validation_error(error: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in error.errors()[:10]
    ]
    return "the JSON does not match the schema: " + "; ".join(parts)


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
    return 0


if __name__ == "__main__":
    sys.exit(main())
