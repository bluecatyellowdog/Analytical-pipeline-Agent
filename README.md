# Auditable agentic data science pipeline

You give it a data file and a short description of the data and what you want to learn. It loads and checks the file, cleans it with AI agents, and (in later phases) plans and runs the analysis. Every step writes a report saying what was done, why, and what the numbers were before and after, so the result comes with a complete audit trail.

Three rules hold throughout:

- **The AI decides, the system executes.** The AI chooses what to do; every change runs through logged, checked operations. The original data is never modified.
- **The AI never sees rows or personal data.** It works from masked summaries and aggregate counts only.
- **Nothing is decided silently.** Every choice the AI makes is in the report, with its reasons and alternatives, and can be undone.

More detail: [specification](docs/project_spec.pdf) · [working rules](CLAUDE.md) · [design decisions and their reasons](DECISIONS.md)

## The workflow

| Step | Who | What happens | Command |
| --- | --- | --- | --- |
| 1. Describe | You | Write a brief: what the data is, what you want to learn, how unclear cases should be handled. | edit a YAML file |
| 2. Ingest | System (no AI) | Loads the file exactly as it is, profiles every column, masks personal data, and checks the claims in your brief. **Gate 1** stops here if the file was not read faithfully. | `pipeline.ingest` |
| 3. Diagnose | AI: analyst | Lists the problems to fix and the judgment calls (questions the data cannot answer on its own), and decides those calls from your brief. | `pipeline.agents.analyst` |
| 4. Check decisions | You (optional) | Change any of the AI's decisions in `decisions.yaml`. | edit a YAML file |
| 5. Clean | AI: coder and reviewer | The coder writes a cleaning plan, which runs step by step from the original data. **Gate 2** checks the result. The reviewer checks that it makes sense and can send fixes back, at most twice. | `pipeline.clean` |
| 6. Read the report | You | `run_report.pdf` opens with a one-page summary, then tells the whole story: the dirty data, what was wrong, every change (with any code the AI wrote), the checks and the review. To change something, edit `decisions.yaml` and run step 5 again. | – |

Steps 2, 3 and 5 run in one go with `pipeline.run`; the separate commands do the same work one step at a time.

Not built yet: plan the analysis → you approve the plan → run the analysis (charts, models, explanations) → final report → optional public dashboard.

### Gates

A gate is an automatic check between steps. **PASS** continues, **FLAG** continues with the issue shown at the top of the report, and **BLOCK** stops and says why.

- **Gate 1, after ingest:** every row is accounted for, fewer than 1% of lines fail to parse, column names are unique, the file is fingerprinted, and every column's type is recognised or flagged.
- **Gate 2, after cleaning:**
  - every column has the type the diagnosis asked for
  - the ID column is still unique
  - the target column is never filled in
  - rows lost: under 5% passes, 5–20% flags, over 20% blocks
  - every change is accounted for, and re-running the plan from the original data gives the identical result

## Quickstart

**1. Install and run the tests**

```powershell
uv sync
uv run pytest
```

**2. Add an LLM key.** Copy `.env.example` to `.env` and fill in `LLM_API_KEY=`. Keys from OpenRouter, Anthropic, OpenAI, Gemini, Groq and xAI are recognised automatically.

**3. Run the example**, a small Danish loan file from the test set, in one command:

```powershell
uv run python -m pipeline.run tests/fixtures/danish_semicolon.csv --brief examples/context_brief.yaml
```

It ingests, diagnoses and cleans the file, then prints a short summary and where the PDF report is. `runs/index.md` lists every run.

Or step by step, which does exactly the same:

```powershell
uv run python -m pipeline.ingest tests/fixtures/danish_semicolon.csv --brief examples/context_brief.yaml
uv run python -m pipeline.agents.analyst runs/<run folder>
uv run python -m pipeline.clean runs/<run folder>
```

Ingesting is free. Diagnosing and cleaning call the LLM: a small dataset costs about $0.40–0.65 with Claude Opus. If Gate 1 blocks the file, the run stops before any LLM call. See [Settings](#settings) to make it cheaper.

## The brief

The brief tells the agents what they cannot see in the data. A shortened version of [examples/titanic_brief.yaml](examples/titanic_brief.yaml):

```yaml
description: One row per passenger on the Titanic's maiden voyage in April 1912.
goal: Understand which passenger characteristics are associated with survival.
target: Survived
id_column: PassengerId
focus_areas: [Children (under 16), Third-class passengers]
excluded_variables: [Name, Ticket]
judgment_calls: decide   # or: ask
guidance: Missing ages must stay missing; never fill them in. High fares are real.
claims:
  - {kind: column_unique, column: PassengerId, text: PassengerId is unique}
```

- **Claims** are checked against the data during ingest. A claim that does not hold is reported, not trusted.
- **Column notes, known issues and units** can be added too; see [examples/context_brief.yaml](examples/context_brief.yaml).
- **`judgment_calls` and `guidance`** control how unclear cases are settled; see below.

## Judgment calls

A judgment call is a cleaning question the data cannot answer by itself, where the answer changes the results. For example: does 999 mean "unknown" or a real amount? Should missing ages be filled in?

- **`judgment_calls: decide`** (the default): the analyst decides each call from your brief and `guidance`. When the brief does not settle a call, it takes the option that changes the least and marks it low confidence.
- **`judgment_calls: ask`**: every call waits for you in `decisions.yaml`.

Every decision appears in two places:
- **`decisions.yaml`**, the file you edit.
- **The top of `cleaning_report.md`**, under "Decided by the AI: check these", least certain first. Each entry shows why it was decided, what it changed (measured), and the alternatives.

**To change a decision,** edit its `decision:` line in `decisions.yaml`. You can write another option, your own words, or `undecided` to leave that data untouched. Then run `pipeline.clean` again. The original data is never modified, so every decision can be undone.

## What a run folder contains

| File | Written by | What it is |
| --- | --- | --- |
| `ingestion_report.md` / `.xlsx` | ingest | What was read and what was not, column by column, and Gate 1 |
| `rejected_lines.csv` | ingest | Lines that could not be parsed, in full |
| `profile_llm.json` | ingest | The masked summary: the only view of the data the AI ever gets |
| `diagnosis.md` | analyst | The problems found, and the judgment calls with their decisions |
| `decisions.yaml` | analyst, then you | The decisions on judgment calls; edit to change |
| `run_report.pdf` | run, clean | **Start here.** A one-page summary, then the dirty data, what was wrong, every change and the AI's code, the checks and the review. No personal data. |
| `cleaning_report.md` | clean | The same story in Markdown, with the technical detail |
| `clean_data.csv` / `.parquet` | clean | The cleaned data, written only when Gate 2 does not block |
| `handoff.json`, `schema_contract.yaml` | clean | What the clean data is, for you and for later stages |
| `dropped_rows.csv` | clean | Every removed row, with the reason |
| `manifest.json` | every step | Models, prompt versions, cost and the status of each step |
| `*_log.json`, `gate_*.json`, `cleaning_plan.json` | every step | The full audit trail: every AI attempt, query, step and check |

`runs/` is not in git, because it can contain your data.

## Settings

In `.env`:

| Setting | Use |
| --- | --- |
| `LLM_API_KEY` | Required. The provider is recognised from the key. |
| `LLM_MODEL` | Optional. Defaults exist for OpenRouter and Anthropic keys. |
| `LLM_REVIEW_MODEL` | Optional. A different, e.g. cheaper, model for the reviewer. |
| `LLM_BASE_URL` | Optional. Any other OpenAI-compatible endpoint, e.g. a local Ollama server. |

Command options:

- `pipeline.ingest`:
  - `--out` writes to a chosen folder instead of `runs/`
  - `--sheet` picks an Excel sheet
  - `--db` keeps the database in a file
  - `--seed` sets the random seed for profiling
- `pipeline.clean` and `pipeline.run`:
  - `--budget` (default $0.50) stops between review rounds once cleaning has spent that much
  - `--no-review` skips the reviewer
- `pipeline.run` also takes `--brief`, `--sheet`, `--out` and `--seed`, as ingest does

## Measuring it: the benchmark

How good is the cleaning, really? The benchmark breaks clean data on purpose, runs the pipeline on it exactly as a user would, and scores the result against an answer key.

```powershell
uv run python -m benchmark.run_benchmark                       # all scenarios
uv run python -m benchmark.run_benchmark --scenarios mixed --no-review
```

1. **Clean data with known answers:** a generated loan table (IDs with leading zeros, ages, incomes, amounts, Danish regions, dates, fees, a default flag), the same every time for a given seed.
2. **Planted problems, with an answer key:** Danish decimal commas, day-first or US dates, `NA`/`-`/`?` for missing values, a documented 999 code, stray spaces and mixed case, `kr` after numbers, values a thousand times too large, duplicate rows, and a column that gives the answer away.
3. **Scenarios** mix these the way real sources do: `danish_export`, `manual_entry`, `mixed`.
4. **Scores:**
   - **found:** the diagnosis named the problem
   - **fixed:** the cell has its right value back
   - **false alarms:** problems raised where none were planted
   - **damage:** good cells changed, or good rows dropped

Results go to `runs/benchmark_<timestamp>/summary.md`, with each case's run folder and PDF next to it. A run of all three scenarios costs about $1.50–2 with Claude Opus.

## Guarantees, and where they are enforced

**Data safety**

| Guarantee | Enforced by |
| --- | --- |
| The original data is never changed | `RawStore.verify()` recomputes its fingerprint and fails if anything touched it |
| Nothing changes the data outside a logged step | `apply_op` does all the accounting for every operation |
| AI-written code changes only what it declares | `apply_op` refuses undeclared changes, missing values, dropped or new rows; `ops/sandbox.py` blocks file, network and process access |
| No value is silently lost | `apply_op` refuses undeclared new missing values |
| Every change is accounted for | `checks/reconciliation.py`, including a full replay from the original data |
| Data that failed Gate 2 is never handed on | Clean data, `handoff.json` and the contract are written only past Gate 2; old copies are removed first |

**Privacy**

| Guarantee | Enforced by |
| --- | --- |
| The AI never sees rows or personal data | `ingestion/compact.py` builds the masked summary; tests assert no personal value reaches any prompt |
| The reviewer's queries reveal no one | `tools/queries.py` returns only counts and summaries, masks personal and ID values inside the query, and refuses to filter on them |
| Error messages sent back to the AI are masked | `agents/feedback.py` |

**AI accountability**

| Guarantee | Enforced by |
| --- | --- |
| No judgment call is settled silently | Decided by you or by the AI from your brief, always shown with reasons and alternatives; low confidence must mean the option that changes the least (`contracts/diagnosis.py`); undecided calls are left alone (`agents/coder.py`) |
| The target is never filled in or dropped | Checked on the plan, after every step (`ops/plan.py`) and against the original data at Gate 2 |
| The clean data is what the diagnosis said it would be | Gate 2's schema contract (`checks/schema.py`); later stages re-check the data file against it |
| The reviewer's claims are backed by numbers | `agents/reviewer.py` drops claims citing numbers it was never shown and needs two verified checks to accept |
| Questions only you can answer reach you | The reviewer reports them in the report; nothing it says can change a decision |

## Project layout

| Folder | What it holds |
| --- | --- |
| [pipeline/contracts/](pipeline/contracts/) | The data passed between steps, as validated models. Their validators enforce the audit rules. |
| [pipeline/ingestion/](pipeline/ingestion/) | Loading, profiling, personal-data detection and masking, and claim checks |
| [pipeline/ops/](pipeline/ops/) | 15 library operations, the sandboxed `custom_code` operation, and the plan executor |
| [pipeline/agents/](pipeline/agents/) | The LLM client and the analyst, coder and reviewer agents, with their prompts |
| [pipeline/checks/](pipeline/checks/) | Gates 1 and 2, the reconciliation checks and the schema contract |
| [pipeline/tools/](pipeline/tools/) | The reviewer's query tool |
| [pipeline/run.py](pipeline/run.py) | The one command for the whole workflow |
| [pipeline/report_pdf.py](pipeline/report_pdf.py) | The PDF run report, rebuilt from the run folder (`python -m pipeline.report_pdf runs/<folder>`) |
| [pipeline/ingest.py](pipeline/ingest.py), [pipeline/clean.py](pipeline/clean.py) | The commands for steps 2 and 5 |
| [benchmark/](benchmark/) | The benchmark: synthetic clean data, the breakers, scenarios, the scorer and the runner |
| [tests/fixtures/](tests/fixtures/) | Ten deliberately difficult files, with their expected results in [expected.yaml](tests/fixtures/expected.yaml) |

## Status

| Phase | State |
| --- | --- |
| 0. Contracts and setup | Done, except tracing and the GitHub Pages workflow |
| 1. Deterministic foundation | Done |
| 2. Cleaning agents | Mostly done: analyst, coder, Gate 2, reviewer and AI decisions run end to end. Left: canary rows and a run on five public datasets. |
| 3. Corruption benchmark | In progress: synthetic data, breakers, scorer and runner built; first results in; real messy datasets to come |
| 4. Analysis planning and your approval | Not started (the review part was built early, in step 5) |
| 5. Analysis agent | Not started |
| 6. Evaluation | Not started |
| 7. App and publishing | Not started |

## Next

1. **Canary rows:** test rows with known correct answers, added and checked on every run.
2. **Analyst fix:** raise an issue for every column whose type has to change, so the reviewer does not have to catch it.
3. **Five-dataset run:** run the full workflow on five public datasets, Titanic among them.
4. **Phase 3:** deliberately break clean datasets and measure how much the agents catch and fix correctly, including whether the reviewer is worth its cost (`--no-review` gives the comparison).
