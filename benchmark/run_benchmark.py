"""Run the cleaning benchmark: break clean data on purpose, clean it, score the result.

    uv run python -m benchmark.run_benchmark                     # all scenarios
    uv run python -m benchmark.run_benchmark --scenarios mixed --no-review

For each scenario: plant its problems in the clean synthetic loans (writing the
answer key next to the broken file), run the full pipeline on it exactly as a
user would, and score the clean output against the key. Writes summary.md and
summary.json, plus each case's own run folder with its PDF report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import yaml

from benchmark.datasets.synthetic import CleanDataset, make_loans
from benchmark.injectors import Corrupted, inject
from benchmark.metrics import CaseScore, score_case
from benchmark.scenarios import SCENARIOS, Scenario
from pipeline.contracts.diagnosis import Diagnosis


@dataclass
class RunInfo:
    run_dir: Path | None
    stopped_at: str | None
    cost_usd: float
    gate_2: str | None = None
    review: str | None = None


Pipeline = Callable[[Path, Path, Path], RunInfo]
"""(broken file, brief, run folder) -> what happened."""


@dataclass
class Case:
    scenario: Scenario
    corrupted: Corrupted
    source: Path
    brief: Path


@dataclass
class CaseResult:
    scenario: str
    description: str
    score: CaseScore
    info: RunInfo

    @property
    def stopped_at(self) -> str | None:
        return self.info.stopped_at


# -- building a case --------------------------------------------------------


def build_case(scenario: Scenario, data: CleanDataset, *, seed: int, case_dir: Path) -> Case:
    """Plant the scenario's problems and write the broken file, its brief and its answer key."""
    case_dir.mkdir(parents=True, exist_ok=True)
    corrupted = Corrupted.from_clean(data)
    for number, plant in enumerate(scenario.plants):
        inject(corrupted, data, plant.kind, seed=seed * 100 + number, **plant.params)

    source = corrupted.write_csv(case_dir / f"{scenario.name}.csv", delimiter=scenario.delimiter)
    brief = case_dir / "brief.yaml"
    brief.write_text(_brief_yaml(scenario, data, corrupted), encoding="utf-8")
    key = {
        "scenario": scenario.name,
        "seed": seed,
        "rows": len(data.frame),
        "origin": corrupted.origin,
        "injections": [asdict(i) for i in corrupted.injections],
    }
    (case_dir / "answer_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return Case(scenario=scenario, corrupted=corrupted, source=source, brief=brief)


def _brief_yaml(scenario: Scenario, data: CleanDataset, corrupted: Corrupted) -> str:
    """The brief a user who knows this data would write: including the codes they know of."""
    notes = {note["name"]: dict(note) for note in data.column_notes}
    for note in corrupted.column_notes:
        notes.setdefault(note["name"], {"name": note["name"]}).update(note)
    brief = {
        "description": data.description,
        "goal": data.goal,
        "target": data.target,
        "id_column": data.id_column,
        "unit_of_observation": "one loan application",
        "column_notes": list(notes.values()),
        "judgment_calls": "decide",
        "guidance": scenario.guidance,
    }
    return yaml.safe_dump(brief, allow_unicode=True, sort_keys=False, width=100)


# -- scoring a run ----------------------------------------------------------


def load_clean_rows(path: Path) -> list[dict[str, Any]]:
    """The clean data with DuckDB's own types: ints, floats, dates, strings."""
    connection = duckdb.connect()
    cursor = connection.execute(
        f"SELECT * FROM read_parquet('{path.as_posix()}') ORDER BY _source_line"
    )
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def score_run(case: Case, data: CleanDataset, info: RunInfo) -> CaseScore:
    run_dir = info.run_dir
    clean_path = run_dir / "clean_data.parquet" if run_dir else None
    clean_rows = load_clean_rows(clean_path) if clean_path and clean_path.exists() else None
    diagnosis_path = run_dir / "diagnosis.json" if run_dir else None
    diagnosis = (
        Diagnosis.model_validate_json(diagnosis_path.read_text(encoding="utf-8"))
        if diagnosis_path and diagnosis_path.exists()
        else None
    )
    renames = {}
    handoff_path = run_dir / "handoff.json" if run_dir else None
    if handoff_path and handoff_path.exists():
        handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        renames = {
            c["source_column"]: c["name"] for c in handoff["schema"] if c.get("source_column")
        }
    return score_case(data, case.corrupted, clean_rows, diagnosis, renames=renames)


def run_benchmark(
    names: list[str],
    *,
    data: CleanDataset,
    seed: int,
    out_dir: Path,
    pipeline: Pipeline,
    progress: Callable[[str], None] | None = None,
    meta: dict[str, Any] | None = None,
) -> list[CaseResult]:
    """Build, run and score each scenario, then write the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for number, name in enumerate(names, start=1):
        scenario = SCENARIOS[name]
        if progress:
            progress(f"[{number}/{len(names)}] {name}: {scenario.description}")
        case_dir = out_dir / name
        case = build_case(scenario, data, seed=seed, case_dir=case_dir)
        info = pipeline(case.source, case.brief, case_dir / "run")
        score = score_run(case, data, info)
        results.append(CaseResult(name, scenario.description, score, info))
        (case_dir / "score.json").write_text(
            json.dumps(_case_json(results[-1]), indent=1), encoding="utf-8"
        )
    provenance = {"seed": seed, "data_seed": data.seed, "rows": len(data.frame)}
    write_summary(results, out_dir, {**provenance, **(meta or {})})
    return results


def rescore(out_dir: Path) -> list[CaseResult]:
    """Score a finished benchmark again, e.g. after the scorer changed. No LLM calls."""
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    meta = summary["meta"]
    data = make_loans(rows=int(meta["rows"]), seed=int(meta.get("data_seed", meta["seed"])))
    results = []
    for case_json in summary["cases"]:
        scenario = SCENARIOS[case_json["scenario"]]
        case_dir = out_dir / scenario.name
        case = build_case(scenario, data, seed=int(meta["seed"]), case_dir=case_dir)
        info = RunInfo(
            run_dir=case_dir / "run",
            stopped_at=case_json["stopped_at"],
            cost_usd=case_json["cost_usd"],
            gate_2=case_json["gate_2"],
            review=case_json["review"],
        )
        score = score_run(case, data, info)
        results.append(CaseResult(scenario.name, scenario.description, score, info))
        (case_dir / "score.json").write_text(
            json.dumps(_case_json(results[-1]), indent=1), encoding="utf-8"
        )
    write_summary(results, out_dir, meta)
    return results


# -- the summary ------------------------------------------------------------


def _case_json(result: CaseResult) -> dict[str, Any]:
    score = result.score
    return {
        "scenario": result.scenario,
        "stopped_at": result.info.stopped_at,
        "gate_2": result.info.gate_2,
        "review": result.info.review,
        "cost_usd": result.info.cost_usd,
        "detection_rate": score.detection_rate,
        "repair_rate": score.repair_rate,
        "damaged_cells": score.damaged_cells,
        "checked_cells": score.checked_cells,
        "wrongly_dropped_rows": score.wrongly_dropped_rows,
        "false_alarms": score.false_alarms,
        "injections": [{**asdict(s), "fix_rate": s.fix_rate} for s in score.injections],
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def write_summary(results: list[CaseResult], out_dir: Path, meta: dict[str, Any]) -> Path:
    cases = [_case_json(r) for r in results]
    injections = [s for r in results for s in r.score.injections]
    fixable = [s for s in injections if s.fixed is not None]
    totals = {
        "detection_rate": sum(s.found for s in injections) / len(injections) if injections else 1.0,
        "repair_rate": (
            sum(s.fixed for s in fixable) / sum(s.cells for s in fixable)
            if sum(s.cells for s in fixable)
            else 1.0
        ),
        "damaged_cells": sum(r.score.damaged_cells for r in results),
        "wrongly_dropped_rows": sum(r.score.wrongly_dropped_rows for r in results),
        "false_alarms": sum(len(r.score.false_alarms) for r in results),
        "cost_usd": sum(r.info.cost_usd for r in results),
    }
    (out_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "totals": totals, "cases": cases}, indent=1, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Cleaning benchmark",
        "",
        " · ".join(f"{k}: {v}" for k, v in meta.items()) or "",
        "",
        "## Overall",
        "",
        f"- **Found:** {_pct(totals['detection_rate'])} of planted problems were named in the "
        f"diagnosis",
        f"- **Fixed:** {_pct(totals['repair_rate'])} of broken cells got their right value back",
        f"- **Damage:** {totals['damaged_cells']} good cell(s) changed, "
        f"{totals['wrongly_dropped_rows']} good row(s) dropped",
        f"- **False alarms:** {totals['false_alarms']}",
        f"- **Cost:** ${totals['cost_usd']:.2f}",
        "",
        "| Scenario | Found | Fixed | Damaged cells | Rows wrongly dropped | False alarms "
        "| Gate 2 | Review | Cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for case in cases:
        lines.append(
            f"| {case['scenario']} | {_pct(case['detection_rate'])} | "
            f"{_pct(case['repair_rate'])} | {case['damaged_cells']} | "
            f"{case['wrongly_dropped_rows']} | {len(case['false_alarms'])} | "
            f"{case['gate_2'] or case['stopped_at'] or '-'} | {case['review'] or '-'} | "
            f"${case['cost_usd']:.2f} |"
        )

    lines += [
        "",
        "## By problem",
        "",
        "| Problem | Planted in | Found | Fixed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for kind in sorted({s.kind for s in injections}):
        group = [s for s in injections if s.kind == kind]
        cells = sum(s.cells for s in group if s.fixed is not None)
        fixed = sum(s.fixed for s in group if s.fixed is not None) / cells if cells else None
        lines.append(
            f"| {kind} | {len(group)} case(s) | {sum(s.found for s in group)}/{len(group)} | "
            f"{_pct(fixed) if cells else 'flag only'} |"
        )

    for result in results:
        lines += ["", f"## {result.scenario}", "", result.description, ""]
        if result.info.stopped_at:
            lines += [f"**Stopped at {result.info.stopped_at}.**", ""]
        lines += [
            "| Problem | Column | Cells | Expected | Found | Fixed |",
            "| --- | --- | ---: | --- | --- | ---: |",
        ]
        for s in result.score.injections:
            lines.append(
                f"| {s.kind} | {s.column or 'rows'} | {s.cells} | {s.expected} | "
                f"{'yes' if s.found else 'no'} | {_pct(s.fix_rate)} |"
            )
        wrong = [
            f"{s.kind} ({s.column}): {w}"
            for s in result.score.injections
            for w in s.wrong_examples[:3]
        ]
        if wrong:
            lines += ["", "Not fixed (examples):", *[f"- {w}" for w in wrong]]
        if result.score.damaged_examples:
            lines += ["", "Damage (examples):", *[f"- {d}" for d in result.score.damaged_examples]]
        if result.score.false_alarms:
            lines += ["", "False alarms:", *[f"- {a}" for a in result.score.false_alarms]]
        if result.info.run_dir:
            lines += ["", f"Run folder: `{result.info.run_dir.as_posix()}` (see run_report.pdf)"]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# -- the real pipeline ------------------------------------------------------


def real_pipeline(client, *, provider, review_client, review_provider, reviewing, budget):
    from pipeline.run import run_all

    def run(source: Path, brief: Path, out: Path) -> RunInfo:
        summary = run_all(
            source,
            brief,
            client,
            provider=provider,
            review_client=review_client,
            review_provider=review_provider,
            reviewing=reviewing,
            budget_usd=budget,
            out=out,
        )
        cleaning = summary.cleaning
        return RunInfo(
            run_dir=summary.run_dir,
            stopped_at=summary.stopped_at,
            cost_usd=summary.cost_usd,
            gate_2=cleaning.result.gate.outcome if cleaning else None,
            review=cleaning.status if cleaning else None,
        )

    return run


def main(argv: list[str] | None = None) -> int:
    from pipeline.agents.llm import LLMConfigError, connect

    parser = argparse.ArgumentParser(
        prog="python -m benchmark.run_benchmark",
        description="Break clean data on purpose, run the pipeline on it, and score the result.",
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIOS),
        help=f"Comma-separated, from: {', '.join(SCENARIOS)}",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument(
        "--budget",
        type=float,
        default=1.00,
        help="Per case; above the $0.50 product target so review rounds can finish",
    )
    parser.add_argument("--rescore", help="Score a finished benchmark folder again; no LLM calls")
    parser.add_argument("--out", help="Output folder (default: runs/benchmark_<timestamp>)")
    args = parser.parse_args(argv)

    if args.rescore:
        for result in rescore(Path(args.rescore)):
            _print_line(result)
        print(f"\nSummary: {(Path(args.rescore) / 'summary.md').resolve()}")
        return 0

    names = [n.strip() for n in args.scenarios.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"Unknown scenario(s) {unknown}; choose from {list(SCENARIOS)}")
        return 1
    try:
        client = connect()
        review_model = os.environ.get("LLM_REVIEW_MODEL", "").strip() or None
        review_client = connect(model=review_model) if review_model else client
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1

    out_dir = Path(args.out or f"runs/benchmark_{datetime.now():%Y%m%d-%H%M%S}")
    meta = {
        "seed": args.seed,
        "rows": args.rows,
        "model": client.config.model,
        "reviewer": "off" if args.no_review else review_client.config.model,
    }
    pipeline = real_pipeline(
        client,
        provider=client.config.public(),
        review_client=review_client,
        review_provider=review_client.config.public(),
        reviewing=not args.no_review,
        budget=args.budget,
    )
    results = run_benchmark(
        names,
        data=make_loans(rows=args.rows, seed=args.seed),
        seed=args.seed,
        out_dir=out_dir,
        pipeline=pipeline,
        progress=print,
        meta=meta,
    )
    print()
    for result in results:
        _print_line(result)
    print(f"\nSummary: {(out_dir / 'summary.md').resolve()}")
    return 0


def _print_line(result: CaseResult) -> None:
    score = result.score
    print(
        f"{result.scenario:14} found {score.detection_rate:.0%}, "
        f"fixed {score.repair_rate:.0%}, "
        f"damaged {score.damaged_cells}, dropped {score.wrongly_dropped_rows}, "
        f"false alarms {len(score.false_alarms)}, ${result.info.cost_usd:.2f}"
    )


if __name__ == "__main__":
    sys.exit(main())
