"""What every agent shares: prompt blocks, reply parsing, the attempt log and cost.

Each agent works the same way: build a prompt from masked inputs, ask, parse the
reply into a contract, check it deterministically, and send problems back at
most twice. The pieces that do not depend on what is being asked live here.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from pipeline.agents.llm import LLMClient, LLMResponse, Message
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.manifest import ModelInfo, RunManifest

MAX_RETRIES = 2


@dataclass
class Attempt:
    """One question to the model and what came of it, kept for the audit trail."""

    number: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    seconds: float
    reply: str = ""
    problems: list[str] = field(default_factory=list)
    """Blocking: why this attempt could not be used. Kept in full."""
    warnings: list[str] = field(default_factory=list)
    feedback: str = ""
    """Exactly what was sent back to the model after this attempt (masked)."""


def ask(client: LLMClient, messages: list[Message], number: int) -> tuple[LLMResponse, Attempt]:
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
    return response, attempt


def total_cost(attempts: list[Attempt]) -> float:
    return sum(a.cost_usd or 0.0 for a in attempts)


def brief_block(brief: ContextBrief) -> str:
    """The brief as YAML, without the claims (their results travel with the profile)."""
    return yaml.safe_dump(
        brief.model_dump(mode="json", exclude={"claims"}, exclude_none=True, exclude_defaults=True),
        allow_unicode=True,
        sort_keys=False,
    )


def profile_block(profile_llm: dict[str, Any]) -> str:
    """The compact profile, fenced. '<' is escaped so no value can close the block."""
    text = json.dumps(profile_llm, indent=1, ensure_ascii=False).replace("<", "\\u003c")
    return f"<profile>\n{text}\n</profile>"


def fenced(tag: str, text: str) -> str:
    """Data in a block that no value inside can close: every '<' is escaped."""
    escaped = text.replace("<", "&lt;")
    return f"<{tag}>\n{escaped}\n</{tag}>"


def extract_json_object(text: str) -> str:
    """The JSON object in a reply, with or without a code fence."""
    block = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = block.group(1) if block else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("the reply contains no JSON object")
    return candidate[start : end + 1]


def compact_validation_error(error: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in error.errors()[:10]
    ]
    return "the JSON does not match the schema: " + "; ".join(parts)


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def write_attempt_log(path: Path, attempts: list[Attempt], prompt_version: str) -> None:
    log = {"prompt_version": prompt_version, "attempts": [asdict(a) for a in attempts]}
    path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def record_in_manifest(
    manifest: RunManifest,
    agent: str,
    provider: dict[str, str],
    attempts: list[Attempt],
    prompt_version: str,
) -> None:
    """Model, prompt version and cost of every attempt, failed ones included."""
    if attempts:
        manifest.models[agent] = ModelInfo(
            provider=provider.get("provider", "unknown"),
            name=attempts[-1].model,
            temperature=0.0,
        )
    manifest.prompt_versions[agent] = prompt_version
    for attempt in attempts:
        manifest.cost.add(
            prompt_tokens=attempt.input_tokens,
            completion_tokens=attempt.output_tokens,
            usd=attempt.cost_usd or 0.0,
            wall_clock_s=attempt.seconds,
        )


def load_manifest(run_dir: Path) -> RunManifest:
    return RunManifest.model_validate_json((run_dir / "manifest.json").read_text(encoding="utf-8"))


def save_manifest(run_dir: Path, manifest: RunManifest) -> None:
    (run_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
