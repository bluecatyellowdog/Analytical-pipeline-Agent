"""Stage 1 end to end: load, profile, verify the brief, run Gate 1.

Fully deterministic, no LLM. Running this is how you find out what a file
actually contains before anything is allowed to change it.

    uv run python -m pipeline.ingest data.csv --brief brief.yaml --out runs/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline.checks.gates import gate_1_ingestion
from pipeline.contracts.audit import GateResult
from pipeline.contracts.context_brief import ClaimResult, ContextBrief, load_context_brief
from pipeline.contracts.ingestion import LoadResult
from pipeline.contracts.manifest import RunManifest
from pipeline.contracts.profile import DatasetProfile
from pipeline.ingestion.claims import verify_claims
from pipeline.ingestion.compact import compact_profile, render_profile_lines
from pipeline.ingestion.loaders import load
from pipeline.ingestion.profiler import profile_table
from pipeline.ingestion.raw import RawStore, RawTable
from pipeline.report import (
    REJECTS_FILENAME,
    REPORT_FILENAME,
    render_ingestion_report,
    write_rejected_csv,
)
from pipeline.runtime import git_is_dirty, git_sha, pipeline_version, platform_name, python_version

PLACEHOLDER_BRIEF = ContextBrief(
    description="No context brief was provided.",
    goal="Describe and profile the data.",
)
"""Stage 1 still runs without a brief; it simply has no claims to test."""


@dataclass
class IngestionResult:
    """Everything Stage 1 produced, in the order the report presents it."""

    manifest: RunManifest
    load_result: LoadResult
    table: RawTable
    profile: DatasetProfile
    claim_results: list[ClaimResult]
    gate: GateResult
    report: str
    """The readable ingestion report, rendered while the store was open."""

    @property
    def blocked(self) -> bool:
        return self.gate.blocked

    def write(self, out_dir: str | Path) -> Path:
        """Write every artifact as JSON, plus the report and the rejected lines."""
        directory = Path(out_dir)
        directory.mkdir(parents=True, exist_ok=True)

        documents = {
            "manifest.json": self.manifest,
            "load_result.json": self.load_result,
            "profile.json": self.profile,
            "gate_1.json": self.gate,
        }
        for filename, document in documents.items():
            (directory / filename).write_text(
                document.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
            )

        (directory / "claims.json").write_text(
            json.dumps(
                [r.model_dump(mode="json") for r in self.claim_results],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (directory / "profile_llm.json").write_text(
            json.dumps(
                compact_profile(self.profile, self.claim_results), indent=2, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        (directory / "profile.txt").write_text(render_profile_lines(self.profile), encoding="utf-8")
        (directory / REPORT_FILENAME).write_text(self.report, encoding="utf-8")
        write_rejected_csv(self.load_result.rejects, directory / REJECTS_FILENAME)
        return directory


def ingest(
    path: str | Path,
    brief: ContextBrief | None = None,
    *,
    store: RawStore,
    sheet: str | None = None,
    table: str | None = None,
    seed: int = 42,
) -> IngestionResult:
    """Run Stage 1 against one file in an open store."""
    brief = brief or PLACEHOLDER_BRIEF

    load_result, raw_table = load(path, store, table=table, sheet=sheet)
    profile = profile_table(store, raw_table, seed=seed)
    claim_results = verify_claims(brief, profile)
    gate = gate_1_ingestion(load_result, profile, claim_results)

    # The raw layer must be exactly what was loaded; profiling only reads.
    store.verify(raw_table)

    report = render_ingestion_report(
        load_result, profile, gate, claim_results, store=store, table=raw_table
    )

    manifest = RunManifest(
        pipeline_version=pipeline_version(),
        python_version=python_version(),
        platform=platform_name(),
        git_sha=git_sha(),
        git_dirty=git_is_dirty(),
        context_brief=brief,
        inputs=[load_result.file],
        seeds={"profiling_sample": seed},
        stage_status={"ingestion": "blocked" if gate.blocked else "done"},
    )

    return IngestionResult(
        manifest=manifest,
        load_result=load_result,
        table=raw_table,
        profile=profile,
        claim_results=claim_results,
        gate=gate,
        report=report,
    )


def _summarise(result: IngestionResult) -> str:
    load = result.load_result
    lines = [
        f"Read {load.n_loaded:,} of {load.n_source_rows:,} data rows; "
        f"{load.n_rejected:,} rejected ({load.reject_rate:.2%}).",
        "",
        render_profile_lines(result.profile),
        "",
        f"Gate 1: {result.gate.outcome.upper()}",
    ]
    for check in result.gate.checks:
        lines.append(f"  [{'ok ' if check.passed else 'FAIL'}] {check.name}: {check.message}")

    if result.claim_results:
        lines.append("")
        lines.append("Brief claims:")
        for claim in result.claim_results:
            marker = {"pass": "ok  ", "fail": "FAIL", "not_checkable": "n/a "}[claim.status]
            lines.append(f"  [{marker}] {claim.claim.text or claim.claim.kind}: {claim.message}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.ingest",
        description="Load, profile and check one tabular file. Changes nothing.",
    )
    parser.add_argument("path", help="CSV, Excel, Parquet or JSON file")
    parser.add_argument("--brief", help="Context brief YAML (see examples/context_brief.yaml)")
    parser.add_argument("--out", help="Directory to write the JSON outputs to")
    parser.add_argument("--db", help="DuckDB file for the raw layer (default: in memory)")
    parser.add_argument("--sheet", help="Excel sheet name (default: the first one)")
    parser.add_argument("--table", help="Table name in the raw schema (default: from the filename)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for profiling samples")
    args = parser.parse_args(argv)

    brief = load_context_brief(args.brief) if args.brief else None
    with RawStore.open(args.db) as store:
        result = ingest(
            args.path, brief, store=store, sheet=args.sheet, table=args.table, seed=args.seed
        )
        print(_summarise(result))
        if args.out:
            directory = result.write(args.out)
            print(f"\nWrote {len(list(directory.iterdir()))} files to {directory}")
            print(f"Read the report: {directory / REPORT_FILENAME}")
            if result.load_result.n_rejected:
                print(f"Rejected lines:  {directory / REJECTS_FILENAME}")

    return 1 if result.blocked else 0


if __name__ == "__main__":
    sys.exit(main())
