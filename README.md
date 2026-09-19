# Auditable agentic data science pipeline

A user provides data plus a short context brief. The system ingests and profiles the data, cleans it, validates the result, plans an analysis, gets the plan approved, then runs exploratory and predictive analysis. Every step produces a report with the exact code, parameters, before/after statistics and check results, so the output is both the analysis and a complete audit trail.

Full specification: [docs/project_spec.pdf](docs/project_spec.pdf). Working rules: [CLAUDE.md](CLAUDE.md). Design choices and their reasons: [DECISIONS.md](DECISIONS.md).

## Quickstart

```powershell
uv sync
uv run pytest
uv run ruff check .
```

Profile a file — deterministic, no LLM, changes nothing:

```powershell
uv run python -m pipeline.ingest tests/fixtures/danish_semicolon.csv `
    --brief examples/context_brief.yaml --out runs/demo
```

This prints the compact profile and the Gate 1 result, and writes `profile.json`, `profile_llm.json` (the only view an agent ever gets), `claims.json`, `load_result.json`, `gate_1.json` and `manifest.json`.

## Status

| Phase | State |
| --- | --- |
| 0 — Contracts & setup | Done, minus tracing and the Pages workflow |
| 1 — Deterministic foundation | Done |
| 2 — Cleaning agent | Not started |
| 3–7 | Not started |

Phase 1's exit criteria are met: a dataset can be cleaned by hand through the op library with a complete audit log, and replay and row accounting pass.

## What exists

- **[pipeline/contracts/](pipeline/contracts/)** — every stage boundary as a Pydantic model. The validators enforce audit invariants at construction: row accounting must balance, a rejected review must carry a fix request, an artifact without audit references cannot be built.
- **[pipeline/ingestion/](pipeline/ingestion/)** — detection, loaders, the immutable raw layer, type inference, profiling, PII detection and masking, claim verification, and the compact LLM profile.
- **[pipeline/ops/](pipeline/ops/)** — 15 logged operations and the executor that audits them. Nothing changes data outside this package.
- **[pipeline/checks/](pipeline/checks/)** — Gate 1 and the reconciliation checks: row accounting, cell diff, untouched-column comparison, distribution shift, and replay against the output hash.
- **[tests/fixtures/](tests/fixtures/)** — the ten deliberately nasty files from spec 4.5, with their expected results written out in [expected.yaml](tests/fixtures/expected.yaml). That file is the specification the loaders are tested against.

## The guarantees, and where they are enforced

| Guarantee | Enforced by |
| --- | --- |
| Raw data is immutable | `RawStore.verify()` — recomputes the table hash, raises if anything touched it |
| Nothing changes outside a logged op | `apply_op` does all accounting; ops never touch the database |
| No silent coercion | `apply_op` raises on undeclared new nulls |
| Every change is accounted for | `checks/reconciliation.py`, including replay from raw |
| The LLM never sees raw rows or personal data | `ingestion/compact.py`, with an integration test asserting no personal value appears |

## Next

Phase 2 is the cleaning agent: a LangGraph node that reads `profile_llm.json` and returns an op plan, executed through the library above, behind Gate 2. The deterministic half it depends on is finished, so the agent adds op *selection* and nothing else.
