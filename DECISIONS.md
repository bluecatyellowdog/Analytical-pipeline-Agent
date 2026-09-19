# Design decisions

Format: choice, alternatives, reason. Newest at the bottom.

## D001 — Python 3.12 pinned via uv
- **Choice:** `requires-python = ">=3.12,<3.13"`, environment managed with uv.
- **Alternatives:** system Python 3.14 (installed on the dev machine).
- **Reason:** the ML/privacy stack (LightGBM, SHAP, Presidio/spaCy, FLAML) often lags on wheels for the newest Python. 3.12 is safe for every planned dependency.

## D002 — Dependencies added per phase
- **Choice:** `pyproject.toml` only lists what Phase 0–1 needs (pydantic, pyyaml, duckdb, pandas, pandera).
- **Alternatives:** install the whole tech stack from spec section 7 up front.
- **Reason:** keeps CI fast and avoids unused heavy dependencies (Presidio, SHAP) until the phase that uses them.

## D003 — Repo lives in OneDrive
- **Choice:** keep the repo in the OneDrive folder; `.venv/`, `data/` and `runs/` are git-ignored.
- **Alternatives:** move the repo outside OneDrive; set `UV_PROJECT_ENVIRONMENT` to a path outside OneDrive.
- **Reason:** OneDrive syncing a `.venv` (thousands of small files) and DuckDB files can cause locking and slowness. If that happens, set `UV_PROJECT_ENVIRONMENT` to a local path or exclude `.venv` from sync.

## D004 — Data never committed
- **Choice:** `data/*` is git-ignored; showcase datasets are fetched by script.
- **Reason:** raw data is immutable and may be large or licensed; publishing is aggregates-only (spec principle and publish gate).

## D005 — Contracts beyond the spec's list
- **Choice:** added `contracts/profile.py` and `contracts/ingestion.py` alongside the six the spec names.
- **Alternatives:** keep profile and file types inside `pipeline/ingestion/`.
- **Reason:** the profile crosses three stage boundaries (Stage 1 → 2 → 3) and the handoff embeds it, so it belongs with the other contracts. `FileInfo`/`LoadResult` are referenced by the run manifest for provenance.

## D006 — `renames` is a list, not a mapping
- **Choice:** `LoadResult.renames: list[ColumnRename]` with a position, not `dict[original, final]`.
- **Reason:** several blank headers share the original name `''`; only the position tells them apart. A mapping silently loses all but one.

## D007 — The raw layer is enforced by hashing, not by permissions
- **Choice:** `RawStore.verify()` recomputes the table hash and raises if it changed.
- **Alternatives:** DuckDB read-only mode (applies to the whole database, so the pipeline could not write the clean layer); filesystem permissions.
- **Reason:** immutability is the assumption every later check rests on, so it needs to be checkable at any point, not just enforced at open time.

## D008 — A file-backed store attaches under a fixed catalog alias
- **Choice:** connect in memory, then `ATTACH '<file>' AS pipeline_db; USE pipeline_db`.
- **Reason:** DuckDB names the catalog after the file, so `--db raw.duckdb` created a catalog named `raw` that collided with the `raw` schema and made every `raw.<table>` reference ambiguous. Found by a test, not in production.

## D009 — Identifier values are masked in the compact profile
- **Choice:** the LLM-facing profile masks example values for `id` columns as well as for detected PII.
- **Alternatives:** mask only detected PII.
- **Reason:** a customer number is pseudonymous personal data, and its content never informs a cleaning decision — only its shape does, which the mask keeps ('00417' → '99999'). The full profile on disk keeps the real values; it never reaches a prompt.

## D010 — A date column with mixed formats stays a date proposal
- **Choice:** below the 95% coverage threshold a date column is still proposed as `date`, marked ambiguous, with the mix reported ("4/5 match %d-%m-%Y").
- **Alternatives:** fall back to the generic `ambiguous` proposal.
- **Reason:** spec 4.4's own example does exactly this, and "date, but 1 in 5 does not parse" is far more actionable for the cleaning agent than "no confident type".

## D011 — A bare 8-digit number is not a phone number
- **Choice:** without a column-name hint, phone detection requires a `+45` prefix or grouped digits.
- **Reason:** otherwise every amount, postcode and account column in a Danish dataset is flagged as PII and masked, which costs the analysis real information for no privacy gain.
