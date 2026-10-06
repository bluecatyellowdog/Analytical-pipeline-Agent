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

This prints the compact profile and the Gate 1 result, and writes `profile.json`, `profile_llm.json` (the only view an agent ever gets), `claims.json`, `load_result.json`, `gate_1.json`, `manifest.json` and a readable report (`ingestion_report.md` and `.xlsx`). Without `--out`, each run gets its own folder under `runs/`, listed in `runs/index.md`.

Clean it with the agents. Put any LLM API key in `.env` first (see [.env.example](.env.example)):

```powershell
uv run python -m pipeline.agents.analyst runs/<run folder>   # diagnosis.md + decisions.yaml
# the AI has decided the judgment calls from your brief; change any in decisions.yaml
uv run python -m pipeline.clean runs/<run folder>            # coder -> Gate 2 -> reviewer; report, clean data, handoff.json
```

## Status

| Phase | State |
| --- | --- |
| 0 — Contracts & setup | Done, minus tracing and the Pages workflow |
| 1 — Deterministic foundation | Done |
| 2 — Cleaning agent | In progress: analyst, coder, Gate 2 and reviewer run end to end; canary rows and the 5-dataset exit run remain |
| 3–7 | Not started |

Phase 1's exit criteria are met: a dataset can be cleaned by hand through the op library with a complete audit log, and replay and row accounting pass.

## What exists

- **[pipeline/contracts/](pipeline/contracts/)** — every stage boundary as a Pydantic model. The validators enforce audit invariants at construction: row accounting must balance, a rejected review must carry a fix request, an artifact without audit references cannot be built.
- **[pipeline/ingestion/](pipeline/ingestion/)** — detection, loaders, the immutable raw layer, type inference, profiling, PII detection and masking, claim verification, and the compact LLM profile.
- **[pipeline/ops/](pipeline/ops/)** — 15 library operations, the `custom_code` op for agent-written pandas code (run in a sandboxed process), and the executor that audits both. Nothing changes data outside this package.
- **[pipeline/agents/](pipeline/agents/)** — the LLM client (any OpenAI-compatible provider, one key), the analyst (masked profile + brief → diagnosis with judgment calls), the coder (diagnosis + your decisions → a cleaning plan, run and checked by [pipeline/ops/plan.py](pipeline/ops/plan.py)) and the reviewer (evidence + its own aggregate queries → a grounded verdict, fix requests, questions for you). Every attempt, reply, query, cost and prompt version is logged.
- **[pipeline/clean.py](pipeline/clean.py)** — Stage 2 end to end: coder, Gate 2, reviewer, at most two loop-backs, a per-run budget, and every output in the run folder.
- **[pipeline/tools/queries.py](pipeline/tools/queries.py)** — the reviewer's query tool: five aggregate question kinds, no SQL from the model, personal values masked inside the query.
- **[pipeline/checks/](pipeline/checks/)** — Gates 1 and 2, the reconciliation checks (row accounting, cell diff, untouched-column comparison, distribution shift, replay against the output hash) and the schema contract that travels with the clean data ([schema.py](pipeline/checks/schema.py)).
- **[tests/fixtures/](tests/fixtures/)** — the ten deliberately nasty files from spec 4.5, with their expected results written out in [expected.yaml](tests/fixtures/expected.yaml). That file is the specification the loaders are tested against.

## The guarantees, and where they are enforced

| Guarantee | Enforced by |
| --- | --- |
| Raw data is immutable | `RawStore.verify()` — recomputes the table hash, raises if anything touched it |
| Nothing changes outside a logged op | `apply_op` does all accounting; library ops never touch the database |
| Custom code changes only what it declares | `apply_op` refuses undeclared column changes, undeclared nulls, undeclared row drops and new rows; `sandbox.py` refuses file, network and process access |
| No silent coercion | `apply_op` raises on undeclared new nulls |
| Every change is accounted for | `checks/reconciliation.py`, including replay from raw |
| The LLM never sees raw rows or personal data | `ingestion/compact.py`, with tests asserting no personal value reaches the profile or any prompt; error messages sent back to an LLM are masked by `agents/feedback.py` |
| No judgment call is settled silently | Decided by you, or by the AI from your brief (`judgment_calls: decide`, the default) with its choice, confidence, reasons and alternatives at the top of the report; low confidence must mean the least destructive option (`contracts/diagnosis.py`); every decision is reversible in `decisions.yaml`, and `agents/coder.py` refuses a plan that touches an undecided call |
| The target is never imputed or dropped | Checked on the plan before it runs, on the data after every step (`ops/plan.py`), and against raw at Gate 2, which also catches a null token turned straight into a value |
| Clean data is what the diagnosis said it would be | Gate 2's schema contract (`checks/schema.py`): types, columns, primary key; later stages re-check the data file against the saved contract |
| The reviewer never sees a row or a personal value | `tools/queries.py` answers only aggregates, masks personal and ID values in the query itself and refuses to filter on them; the evidence pack is built from logs and masked profiles |
| Review claims are grounded | `agents/reviewer.py` drops any claim whose numbers appear nowhere in what the reviewer was shown, derives `accepted` from its fix requests, and requires two verified checks |
| Ambiguities go to the user | The reviewer reports them; nothing it says can change a decision or a judgment call |
| Blocked data is never handed on | Clean data, `handoff.json` and the contract are written only when Gate 2 does not block, and old copies are removed first |

## Next

The rest of Phase 2: canary rows (synthetic rows with known correct outputs, checked on every run), then the exit run on five clean public datasets. After that, Phase 3: the corruption benchmark that measures the agents, including whether the reviewer catches anything (`--no-review` is the ablation).
