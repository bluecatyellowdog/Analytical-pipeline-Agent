# Auditable agentic data science pipeline

A user provides data plus a short context brief. The system ingests and profiles the data, cleans it, validates the result, plans an analysis, gets the plan approved, then runs exploratory and predictive analysis. Every step produces a report with the exact code, parameters, before/after statistics and check results, so the output is both the analysis and a complete audit trail.

Full specification: [docs/project_spec.pdf](docs/project_spec.pdf). Working rules: [CLAUDE.md](CLAUDE.md). Design choices: [DECISIONS.md](DECISIONS.md).

## Quickstart (development)

```powershell
uv sync
uv run pytest
uv run ruff check .
```

## Status

Phase 0 (contracts and setup) in progress.
