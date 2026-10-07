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

## D021 -- One OpenAI-compatible LLM client, any provider, configured by a single key
- **Choice:** `pipeline/agents/llm.py` uses the `openai` package against whichever OpenAI-compatible endpoint the key belongs to. The provider is recognised from the key prefix (OpenRouter, Anthropic, OpenAI, Gemini, Groq, xAI); `LLM_BASE_URL` and `LLM_MODEL` override. Defaults exist only for OpenRouter and Anthropic, where the model names were verified; other providers must name a model. Tests use `FakeLLM` and never touch the network.
- **Alternatives:** the Anthropic SDK (one provider only); LiteLLM (broader, but a large dependency for what is one HTTP call shape).
- **Reason:** the user wants to paste any key and have it work, and later phases compare models (stability, ablations), so switching model must be a config change. Structured output is not delegated to provider features, which vary; the reply is validated with Pydantic and errors go back through the retry loop.
- **Note:** on OpenRouter, requests are restricted to providers that do not store or train on prompts (`data_collection: deny`), and the real per-call cost is recorded. The key is kept out of `repr` and out of anything logged (`LLMConfig.public()`).

## D022 -- Agents may write their own cleaning code, through an audited `custom_code` op
- **Choice:** a 16th op, `custom_code`, runs an agent-written `transform(df)` in a separate Python process. The code declares the columns it changes, adds and drops, which columns may gain nulls (with a reason) and whether it drops rows (with a reason). `apply_op` measures what actually changed and refuses any difference. The code is the op's record, so replay re-runs it and the reconciliation checks apply unchanged. The library ops stay, as the preferred tools.
- **Alternatives:** a fixed op library only (the original principle, spec section 3); free code that is logged but not checked; the spec's escape hatch as a rarely used afterthought.
- **Reason:** real data has a long tail the 15 ops cannot cover, and forcing every problem into the nearest op makes cleaning worse. The guarantees never came from the op list itself but from the accounting around it, so free code keeps them if it goes through the same accounting. Logging code alone would prove what ran, not what it did.
- **Also changed for every op:** `apply_op` now refuses undeclared column changes (value or type) per op rather than only in the final reconciliation, so an error names the step that caused it. Nulls can be allowed per column, and an op can declare that it keeps every row.
- **Limits, stated honestly:** the sandbox is a guard rail, not a security boundary: a static check (import allowlist; no file, network, eval or dunder access; no pandas readers or writers), then a separate process with no secrets in its environment, restricted builtins, an audit hook that blocks writes, network and process creation, and a timeout. A determined escape from a Python-level sandbox is possible. The spec's container sandbox (no network, CPU and memory limits) remains the hard boundary and is a follow-up before running on sensitive data in production. The agent writing the code sees only the masked profile.
- **Open for the agent session:** sandbox and op error messages can contain data values (for example a value that failed to parse). They are kept in full in the audit log, but must be masked before being sent back to an LLM as retry feedback.

## D023 -- The analyst agent diagnoses; deterministic checks decide if the diagnosis is usable
- **Choice:** `pipeline/agents/analyst.py` sends the compact profile and the brief (nothing else) and asks for a `Diagnosis` (`pipeline/contracts/diagnosis.py`): issues with evidence and a suggested op, judgment calls with options and a recommendation, and one role and clean type per column. The reply is parsed, then checked against the data and the brief: every column covered exactly once, no unknown columns, suggested ops exist, the brief's target has role `target` and is never imputed, excluded columns are `excluded`. Problems go back to the model, at most two retries; every attempt (reply, tokens, cost, problems) goes to `analyst_log.json`, and the model, prompt version and cost go to the run manifest.
- **Alternatives:** one agent that diagnoses and writes code together; trusting the model's JSON without checks; provider-specific structured-output features.
- **Reason:** separating diagnosis from code gives an auditable "why" for every later change, and lets the diagnosis be judged on its own before anything runs. Checks catch the failures that matter (invented columns, an imputed target) whatever the model.
- **Prompt injection:** the profile goes inside `<profile>` tags with `<` escaped, so no value in the data can close the block; the prompt says its contents are data, never instructions.
- **Also fixed:** coded-missing values (sentinels) of masked columns were not masked in the compact profile; they now are.
- **Observed on the first real run (danish_semicolon + example brief, Opus 5.5, 1 attempt, $0.087):** it caught that the brief's target is absent from the file, scoped the 999 sentinel to pre-2021 rows as the brief says, and refused to impute ages because of the under-25 focus. It also surfaced a profiler oddity: `alder` carries a thousands-separator hint, to be checked.

## D024 -- The coder agent writes a whole plan; a deterministic executor runs it from raw
- **Choice:** the coder reads the diagnosis, the user's decisions, the brief and the masked profile, and returns a complete `CleaningPlan` (`pipeline/contracts/cleaning.py`): ordered steps, each a library op or `custom_code` with its parameters, the issue/decision ids it addresses and a rationale, plus items deliberately skipped with a reason. The plan is checked before it runs (ops and parameters valid, custom code passes the static check, every issue with a suggested op and every decided call handled or skipped, no undecided call touched, target never imputed or dropped). `pipeline/ops/plan.py` then runs it from raw, step by step through `apply_op`, stops at the first refused step, refuses any step that fills in or removes the target, and finalises and reconciles (including replay) only when every step succeeded. A failure goes back to the model, at most twice; each retry is a complete plan, run again from raw.
- **Alternatives:** step-by-step tool calling, where the model sees each result before choosing the next op; resuming from the failed step instead of from raw; letting the coder decide judgment calls itself.
- **Reason:** a whole plan is one auditable artifact, cheap (one call when it works: $0.06 on the first real run) and replayable as a unit; re-running from raw keeps the final log exactly the accepted plan, with no residue from failed attempts (`reset_derived_tables` clears them, never touching raw). Step-by-step calling would show the model intermediate results it does not need and cost a call per step.
- **Judgment calls stay human:** the analyst writes `decisions.yaml` with every call `undecided` and never overwrites it; the coder carries out decided calls as decided and must leave undecided ones alone. On the first real run it skipped the 999 replacement on its own, citing the undecided scope question (J1).
- **Feedback is masked:** error messages keep data values in the audit log, but the copy sent back to the model has every quoted value and long number replaced by its shape, unless it is a column name, an op name, an id or something the model wrote itself (`pipeline/agents/feedback.py`). Tested with a pandas error that quotes a real amount.
- **A step that changes nothing** is reported as a warning and earns a retry, but a plan with only warnings is kept as a fallback, so a worse retry cannot lose a working plan.
- **Shared agent code** moved to `pipeline/agents/common.py`; the analyst behaves as before.
- **Found on the way:** `impute` on a column that is still text fails with a raw DuckDB type error rather than a clear message; the executor reports it as a refused step. Worth a clearer error in the op.

## D025 -- Gate 2 and the schema contract, checked on DuckDB types
- **Choice:** Gate 2 (`gate_2_cleaning` in `pipeline/checks/gates.py`) runs on every executed plan. Contract layer: each clean column has the type the diagnosis expects, and the column set matches it. Invariant layer: primary key still unique and complete, target never imputed (compared with raw, so a null token turned straight into a value counts), row loss (under 5% passes, 5-20% flags, over 20% blocks), plus every reconciliation check. Its output includes the schema contract: one `ColumnSchema` per clean column (actual type, role and expected type from the diagnosis, missing share, distinct count, the ops that touched it, whether it is the key, and a note for any allowed deviation). `validate_against_contract` re-checks any table or Parquet file against a saved contract.
- **Grading a type mismatch:** if a step touched the column and it still has the wrong type, Gate 2 blocks (the plan tried and got it wrong, or cast an ID to a number). If the coder skipped the column's issue with a reason, or the diagnosis raised no issue for it, Gate 2 flags and writes the reason into the contract. An undecided judgment call does not by itself excuse a mismatch; the coder must say so in `skipped`.
- **In the loop:** a blocked Gate 2 goes back to the coder like a refused step, within the same two retries. A result that is still blocked is reported, not discarded: the report and `gate_2.json` say why, the stage is marked blocked, and no clean data, `handoff.json` or contract is written. Those deliverables are deleted at the start of every coder run, so a blocked re-run can never leave an older handoff behind for Stage 3.
- **Contract change (deliberate):** `ColumnSchema` gained `role`, `expected_type`, `unique` and `note`, with a validator that a key may not allow missing values; `Handoff` gained `data_file`, because the clean table lives in an in-memory database that is gone after the run, and the next stage needs the Parquet file.
- **Alternatives:** Pandera, as the spec suggests; checking only at the end of Phase 2.
- **Reason for not using Pandera:** it validates pandas DataFrames, and converting from DuckDB blurs types: a DATE and a TIMESTAMP both arrive as datetime64, so a date column could not be told from a timestamp. The data already lives in DuckDB with exact types, so the checks run there, and the contract is a plain Pydantic model saved as JSON and YAML. Pandera stays a dependency for now; nothing uses it yet.
- **First real run (Opus 5.5):** PASS on the first attempt, $0.056; key `kunde_id` unique, every type as expected.

## D026 -- The reviewer: grounded verdicts, its own aggregate queries, flags not blocks
- **Choice:** `pipeline/agents/reviewer.py` reads an evidence pack (brief, diagnosis, decisions, the plan with what each step changed, Gate 2, the schema contract, masked profiles of raw and clean data) and may ask up to two rounds of at most eight queries through `pipeline/tools/queries.py` before replying with a verdict: what it verified, findings (with a fix when the coder can repair them), and ambiguities. `pipeline/clean.py` runs coder -> Gate 2 -> reviewer, sends fix requests (R1, R2, ...) back to the coder at most twice, and ends `accepted`, `flagged` or `blocked`. `--no-review` skips it; `LLM_REVIEW_MODEL` gives it its own model.
- **Queries, decided by the user:** a closed vocabulary of five aggregate kinds (count, value_counts, summary, group_count, compare), never SQL from the model and never rows. Column names are checked and quoted, values bound as parameters. Personal-data and identifier columns are masked inside the query (grouped by shape) and can be filtered only with is_null / is_not_null, so a query cannot confirm that a person is in the data.
- **Unresolved after two rounds: flag, decided by the user.** A judgment should not overrule data that passed Gate 2; the user decides at approval. Ambiguities the reviewer spots are reported in the report and the handoff, never acted on.
- **Grounding:** a claim whose evidence quotes no number that appears in the evidence pack or a query answer is dropped and listed in the report. This catches invented numbers, not a misread real one. An accept needs two verified checks, otherwise the review is flagged as weak. `accepted` is derived from whether any finding carries a fix, never taken from the model.
- **Robustness:** if the coder cannot apply the fixes, or applying them makes Gate 2 block, the previous result is restored by re-running its plan from raw (`coder.rerun`) and the review ends flagged. A per-run budget (default $0.50, `--budget`) is checked between rounds.
- **Placement:** the spec puts review in Stage 3 and lists a separate reviewer-of-the-cleaner as out of scope unless the benchmark shows it is needed. This is the review half of Stage 3, using its existing `ReviewVerdict` / `FixRequest` contracts, built early because the user asked for it; `--no-review` makes its value measurable in Phase 3.
- **Alternatives:** a fixed evidence pack without queries; free SQL; blocking on unresolved findings; adding new judgment calls to decisions.yaml automatically.
- **Structure:** the run folder, the loop and the outputs moved from `agents/coder.py` to `pipeline/clean.py`, the Stage 2 counterpart of `pipeline/ingest.py`; agent modules hold only agent logic.
- **First real run (Opus 5.5 for both):** Gate 2 PASS, review ACCEPTED in one round, 8 queries, 7 verified checks. It established from a query that one of the three 999 values is dated after 2020, so the two J1 options differ by one row, and reported that as a question rather than deciding it. Cost: coder $0.057, reviewer $0.255; with the analyst about $0.40 for the dataset. The reviewer is the expensive part.
- **Open:** once, under heavy load, `test_imports_are_refused_at_runtime` saw a different error message than "not allowed" (the code was still refused). It did not recur in four full reruns. Most likely the sandbox process failed to start while the machine was busy (that run took twice as long; the repo and `.venv` live in OneDrive). Watch for it; a sandbox that cannot start reports "process failed" rather than a policy refusal.

## D027 -- The AI may settle judgment calls from the brief, visibly and reversibly
- **Choice:** the brief gains `judgment_calls: decide | ask` (default `decide`) and free-text `guidance`. In decide mode the analyst gives every judgment call an `ai_decision`: a choice that must be one of the call's options, a confidence, the reasoning (what in the brief or profile settled it), the expected effect, and each alternative with its effect. When the brief does not settle a call it must take the least destructive option; low confidence without `least_destructive` is rejected by the contract. `decisions.yaml` is pre-filled with `ai_choice` and `decision`; changing `decision` makes it the user's (`decided_by: user`), `undecided` leaves the data alone, and re-running the analyst refreshes the AI's choices but never an answer the user changed. The report opens with "Decided by the AI: check these", least certain first, each with what it did (measured from the log), why, the alternatives (labelled as estimates) and the exact command to re-run. Every decision also travels in the handoff as a flag (`ai_decision` / `user_decision`), and the reviewer is told to judge the AI's choices against the brief and report doubts as ambiguities.
- **Alternatives:** keep every judgment call for the user (the original principle); let the AI decide without exposing it; leave unsure calls undecided instead of taking the least destructive option.
- **Reason:** the user wants a run to finish without a manual step, with the context they gave in the brief doing the work. The guarantee the project needs is not that a human typed every answer but that no answer is silent or irreversible: every choice is shown with its basis and its measured effect, the least certain first, and undoing one is a one-line edit. Least destructive when unsure means a run always ends with a complete result that changed as little as possible where the brief was silent.
- **Principle changed (deliberately):** CLAUDE.md said "flag judgment calls; don't fix them". It now says no judgment call is settled silently. Undecided calls are still refused by the coder.
- **First real run (Titanic, Opus 5.5):** the analyst decided five calls in one attempt ($0.12): keeping high fares at high confidence (the guidance says so), the rest at low or medium confidence with the least destructive option. Cleaning: Gate 2 passed; the reviewer sent two fix requests in round 1 (Fare, Pclass, SibSp and Parch were left as text because the diagnosis gave them a clean type but raised no issue, which Gate 2 only flags) and accepted round 2. It challenged one AI decision with evidence: 5 of the 15 zero fares are first class, so "free ticket" is doubtful; it reported this as a question for the user, as designed. Cost: analyst $0.12, coder $0.14, reviewer $0.38, about $0.65 in all, above the $0.50 target; the reviewer's two rounds are most of it.
- **Follow-up:** the analyst should raise an issue for every column whose clean type differs from its raw type; here the reviewer had to catch four. And the report now lists only the final review round's questions, since an earlier round's question was repeated in different words.

## D028 -- Every type change needs an issue that converts it, checked before cleaning starts
- **Choice:** the compact profile now says how each column is stored (`stored_as`: "text" for CSV and Excel, the declared type for Parquet). The analyst must raise an issue with a converting op (cast, normalize_decimal, parse_dates or custom_code) for every column whose clean type differs from its storage; `check_diagnosis` sends a diagnosis back if one is missing. Columns marked `drop` are exempt.
- **Alternatives:** let the coder convert every column to its target type regardless of issues; leave it to Gate 2 and the reviewer.
- **Reason:** on the first Titanic run four columns kept their text type because the diagnosis gave them a clean type but raised no issue, so the coder had nothing to address. Gate 2 only flags that case (D025), and the reviewer caught it at the cost of a second review round (about $0.20). Caught at the diagnosis, it costs at most a retry of one cheap call, and every conversion stays tied to an issue in the audit trail.

## D029 -- Sandbox output keeps each column's own type when only pandas changed it
- **Choice:** when custom code returns its DataFrame, every column whose type moved only because of the trip through pandas (DATE to TIMESTAMP, integers to other integer widths or to DOUBLE, DECIMAL to DOUBLE) is cast back to its original type, provided no value changes by it. A real type change stays and must be declared.
- **Alternatives:** convert dates inside the worker; let the coder declare every column as changed.
- **Reason:** found while testing the PDF report: any custom code after a `parse_dates` step was refused, because the untouched date column came back as a timestamp and `apply_op` (correctly) refuses undeclared type changes. In a real run that would burn the coder's retries on an error it cannot fix. The fix is on the host side and checks losslessness in SQL, so it does not depend on what the code did.

## D030 -- The run report is one PDF, built from the run folder, with no personal data
- **Choice:** `pipeline/report_pdf.py` writes `run_report.pdf` at the end of `pipeline.run` and `pipeline.clean` (and on demand: `python -m pipeline.report_pdf runs/<folder>`). Page one is a summary: each step's outcome and meaning, the numbers (rows, steps, values changed, types converted, values turned missing, rows dropped, judgment calls by whom, cost) and "look at these first". Then, in workflow order: the data as received (format, rejected lines, column by column with what stands out, brief claims, Gate 1), what was wrong (issues; judgment calls with options, who decided, confidence, reasons, alternatives), what was changed (every step with its measured effect, new and removed columns, and any custom code in full), the checks on the clean data (Gate 2, schema contract), the review (verified checks, findings, questions, queries) and an audit appendix (models, prompt versions, calls, cost, files). Text written by an LLM is boxed and labelled "AI-written".
- **Alternatives:** Jinja2 HTML with a PDF export (the spec's suggestion; WeasyPrint needs GTK on Windows); Markdown only.
- **Reason:** the user wants one shareable document. reportlab is pure Python, so it works on every machine without system packages. Building from the folder's files, not from in-memory objects, means any run can be re-rendered and the PDF can only say what the logs say.
- **Privacy:** a PDF travels, so it holds only masked values: column examples come from the masked profile, and rejected lines appear as line number and reason only (their text stays in `rejected_lines.csv`). Tests assert that no personal value and no raw rejected line reaches it.

## D031 -- The cleaning benchmark: planted problems, an answer key, and four scores
- **Choice:** `benchmark/` generates a clean, seeded loan table (`datasets/synthetic.py`), plants problems with nine breakers that each record an answer key (`injectors/`), mixes them into scenarios (`scenarios.py`), runs the full pipeline on the broken file exactly as a user would (`run_benchmark.py`, via `pipeline.run`), and scores the clean output (`metrics.py`): found (named in the diagnosis), fixed (right value back, or missing where the original is unrecoverable, or duplicate removed), false alarms (problems raised where none was planted) and damage (good cells changed, good rows dropped). Values are compared by meaning: 31 = 31.0 = "31", 1 = True, and a category counts as fixed when it equals what its untouched rows became. `--rescore` re-scores a finished run without LLM calls.
- **Alternatives:** hand-built broken files (the existing fixtures); scoring only by diagnosis text; measuring against an LLM judge.
- **Reason:** the 525 tests prove the rules hold with scripted AI answers; they say nothing about how often the real AI makes the right call. Planting problems in data whose right answer is known makes that measurable, repeatably and without a human labelling anything.
- **First run (seed 1, 300 rows, Opus 5.5 for every agent, $1.32 for three scenarios):** after fixing two benchmark bugs it surfaced, found 100%, fixed 100% of 1,365 broken cells, damage 0, wrongly dropped rows 0, two false alarms (both questions about genuinely extreme incomes in the synthetic data, kept unchanged). The reviewer also caught a real mistake: a marker for impossible fees, defined as fee > loan, missed 2 of the 6 planted fees; the fix round did not run because the $0.50 budget was spent, so the benchmark now defaults to $1 per case.
- **What it found in the system:** Gate 2's distribution-shift check compared Danish-formatted raw text with clean numbers and flagged a meaningless PSI of 27 (fixed: raw columns are compared only when 90% of their values read as numbers); and the scorer missed a detection worded as "impossible fees" (fixed: a quoted planted value counts).
- **Limits, stated plainly:** one seed, three scenarios, synthetic data with tidy problems, one model. The spec warns that planted errors flatter the system; real messy datasets with hand-listed problems are the next comparison, and these numbers should not be quoted without that caveat.
