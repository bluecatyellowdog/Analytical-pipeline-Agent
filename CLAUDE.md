# Project: auditable agentic data science pipeline
Reference: docs/project_spec.pdf. Read the relevant section before larger changes.

## Principles (do not violate)
- The LLM decides, tools execute: no data transformation outside logged ops in pipeline/ops/.
- Every op returns (new_table, OpRecord) with params, exact code, before/after stats.
- Raw data is read-only. Never write to the raw schema or data/raw/.
- CSV/Excel are loaded as literal text; types are applied later as logged cast ops.
- No silent coercion (never errors="coerce"); new nulls must be explained.
- Numbers in reports come from logs, never from LLM text.
- The LLM never sees raw rows or personal data; only compact, masked profiles.
- Deterministic checks first, judgment second. Flag judgment calls; don't fix them.

## Working rules
- One roadmap task per session. Propose a plan before writing code.
- Tests first for ops, checks and gates; use the fixture set in tests/fixtures/.
- Run pytest before every commit. Small, focused commits.
- Contracts in pipeline/contracts/ are the source of truth; change them deliberately.
- Record design choices in DECISIONS.md (choice, alternatives, reason).

## Environment
- Python 3.12 managed with uv. Setup: `uv sync`. Run: `uv run pytest`, `uv run ruff check .`.
- Repo lives in a OneDrive folder; keep .venv out of sync if it causes problems (see DECISIONS.md).
- Shell is Windows (PowerShell / Git Bash): use forward-slash paths in code, `pathlib` everywhere.
