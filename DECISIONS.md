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

## D012 — Ops declare intent; the executor does the accounting
- **Choice:** an op returns an `OpPlan` (expressions, renames, a filter) and never touches DuckDB. `apply_op` hashes before and after, counts changed cells and new nulls, captures dropped rows and builds the `OpRecord`.
- **Alternatives:** each op executes its own SQL and reports its own statistics.
- **Reason:** an op cannot then forget, understate or opt out of its own audit, and adding an op cannot weaken the guarantees. It also means `code` in the record is the statement that actually ran.

## D013 — Each op writes a new versioned table
- **Choice:** `clean.<name>__v1`, `__v2`, …, published as `clean.<name>` at the end.
- **Alternatives:** update in place.
- **Reason:** before/after diffs become a plain join, and replay reproduces every intermediate state rather than only the final one. Storage cost is acceptable for tabular data of this size.

## D014 — No silent coercion is enforced, not just documented
- **Choice:** `apply_op` raises if an op introduces nulls without `allow_new_nulls`. A cast must pass `on_error="null"` to null out values that do not parse, which also raises a flag.
- **Reason:** this is the failure mode from spec 5.3 — unparseable values become NaN and then get imputed — and a rule that is only written down does not prevent it.

## D015 — `drop_rows` takes a condition, not a predicate
- **Choice:** column + comparison + value, never a SQL string.
- **Reason:** the parameters eventually come from an LLM. A closed vocabulary means no generated SQL reaches the execution path at all, rather than relying on sanitising it.

## D016 — PSI is implemented directly rather than pulled from a library
- **Choice:** ~15 lines in `checks/reconciliation.py`, no scipy.
- **Alternatives:** scipy's KS test as the spec also suggests.
- **Reason:** the number ends up in an audit trail, so being able to read its definition next to the check matters more than the dependency. KS with a proper p-value can be added when scipy arrives for the modelling phase.

## D017 — Cell diff compares net change against claimed change
- **Choice:** a column whose cells differ from raw must have been targeted by some op, and the sum of claimed changes must be at least the net difference.
- **Reason:** per-op counts can legitimately exceed the net difference (an op changes a value, a later one changes it back), so equality would produce false failures. The direction that matters — a change nothing claimed — is still caught exactly.

## D018 -- A readable ingestion report and a saved rejected-lines file
- **Choice:** `--out` now also writes `ingestion_report.md` (what was read, what was not, column by column, flags, Gate 1, brief claims, a masked row preview) and `rejected_lines.csv` (every rejected line in full, with its source line number).
- **Alternatives:** print the rejects on request only; leave them in the DuckDB `__rejects` table, which is in-memory and disposed of at the end of the run unless `--db` is used.
- **Reason:** the JSON artifacts are complete but not something a person reads to answer "did this work". The report is built entirely from `load_result`, `profile` and `gate`, so its numbers are never a second source of truth.
- **Note:** `rejected_lines.csv` holds unmasked source text, deliberately, since the point of the file is to locate the bad line in the original. It is written only under `--out`, and the report says explicitly to keep it out of version control.

## D019 -- Every run lands in runs/, listed in runs/index.md
- **Choice:** without `--out`, ingestion writes to `runs/<file>_<timestamp>/` and adds a row (newest first) to `runs/index.md` linking the reports. An explicit `--out` is honoured as before and not indexed.
- **Alternatives:** keep `--out` mandatory for writing; overwrite a fixed `runs/latest/`.
- **Reason:** outputs were easy to lose. A timestamped folder never overwrites an earlier run, and one index file is a single place to find all of them. `runs/` stays gitignored because the rejected lines are unmasked.

## D020 -- An Excel workbook alongside the Markdown report
- **Choice:** each run also writes `ingestion_report.xlsx` with fixed sheets: Summary, Gate 1 checks, Flags, Columns, Claims, Rejected lines, Renames, Preview. Built from the same objects as the Markdown report; the masked preview is computed once (`report.preview_rows`) and shared by both.
- **Alternatives:** replace the Markdown report; export the JSON artifacts to CSV.
- **Reason:** the column profile is a table people want to sort and filter. Keeping the Markdown report means the audit record stays readable without Excel.
- **Note:** value-bearing cells in the Columns sheet go through `compact_column`, so masking matches the LLM view; numeric summaries and coded-missing values of masked columns are hidden or masked too. Every string cell is stored as text, so a rejected line beginning with `=` is never evaluated as a formula. The Rejected lines sheet is unmasked, for the same reason as D018.
