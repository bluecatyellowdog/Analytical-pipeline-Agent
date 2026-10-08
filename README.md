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
| 3. Diagnose | AI: diagnostic agent | Lists the problems to fix and the judgment calls (questions the data cannot answer on its own), and decides those calls from your brief. | `pipeline.agents.diagnostic` |
| 4. Check decisions | You (optional) | Change any of the AI's decisions in `edit/decisions.yaml`. | edit a YAML file |
| 5. Clean | AI: coder and reviewer | The coder writes a cleaning plan, which runs step by step from the original data. **Gate 2** checks the result. The reviewer checks that it makes sense and can send fixes back, at most twice. | `pipeline.clean` |
| 6. Plan the analysis | AI: planner | Writes 3–5 questions that serve your goal, each answered with the [analysis tools](#analysis-tools), and marks what this data cannot answer, with the reason. **Gate 3** checks the plan. It is approved automatically; to change it, edit `edit/analysis_plan.yaml`. | `pipeline.agents.planner` |
| 7. Analyse | AI: analysis agent | Runs the plan with the tools in up to three rounds, following up on whatever looks interesting, then writes up what it found: an answer per question and findings that cite the tool results behind them. **Gate 4** checks that every number in the write-up is one a tool reported. | `pipeline.analyze` |
| 8. Read the report | You | `REPORT.pdf`: the answer, how good the data is, the decisions made for you, and a self-critical wrap-up, in three or four pages. Each section links to a detail document. To change a cleaning decision, edit `edit/decisions.yaml` and run step 5 again; to change the analysis, edit `edit/analysis_plan.yaml` and run step 7 again. | – |

Steps 2, 3 and 5–7 run in one go with `pipeline.run`; the separate commands do the same work one step at a time.

Not built yet: an optional public dashboard.

### Gates

A gate is an automatic check between steps. **PASS** continues, **FLAG** continues with the issue shown at the top of the report, and **BLOCK** stops and says why.

- **Gate 1, after ingest:** every row is accounted for, fewer than 1% of lines fail to parse, column names are unique, the file is fingerprinted, and every column's type is recognised or flagged.
- **Gate 2, after cleaning:**
  - every column has the type the diagnosis asked for
  - the ID column is still unique
  - the target column is never filled in
  - rows lost: under 5% passes, 5–20% flags, over 20% blocks
  - every change is accounted for, and re-running the plan from the original data gives the identical result
- **Gate 3, after planning:**
  - every column and tool in the plan exists, and the target is the one in your brief
  - no ID, personal data, free text or excluded column is analysed
  - a column that predicts the outcome almost perfectly on its own is left out: it probably gives the answer away
  - every question has enough rows (10 rows with the rarer outcome per regression feature, two groups of 30 to compare), or is marked "not possible" with the reason
- **Gate 4, after the analysis:**
  - every plan question is answered, or marked "not possible" with the reason
  - every finding rests on tool calls that ran, and every number in the write-up is one those calls reported (rounding is fine; a number the AI worked out itself is not)
  - a finding that rests only on a model doing no better than no model at all cannot be called strong
  - model warnings (a suspiciously good fit, too few rows) are shown at the top of the report

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

It ingests, diagnoses and cleans the file, then plans and runs the analysis, and prints a short summary and where `REPORT.pdf` is. `runs/index.md` lists every run.

Or step by step, which does exactly the same:

```powershell
uv run python -m pipeline.ingest tests/fixtures/danish_semicolon.csv --brief examples/context_brief.yaml
uv run python -m pipeline.agents.diagnostic runs/<run folder>
uv run python -m pipeline.clean runs/<run folder>
uv run python -m pipeline.analyze runs/<run folder>
```

Ingesting is free. The other steps call the LLM: a small dataset costs about $0.80–1.10 with Claude Opus, of which planning about $0.10 and the analysis about $0.30. If Gate 1 blocks the file, the run stops before any LLM call. See [Settings](#settings) to make it cheaper.

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

- **`judgment_calls: decide`** (the default): the diagnostic agent decides each call from your brief and `guidance`. When the brief does not settle a call, it takes the option that changes the least and marks it low confidence.
- **`judgment_calls: ask`**: every call waits for you in `edit/decisions.yaml`.

Every decision appears in three places:
- **`edit/decisions.yaml`**, the file you edit.
- **`REPORT.pdf`, section 3**: a short table, least certain first.
- **`details/3_decisions.pdf`**: each decision with its options, reasons, alternatives and what it changed in the data (measured).

**To change a decision,** edit its `decision:` line in `edit/decisions.yaml`. You can write another option, your own words, or `undecided` to leave that data untouched. Then run `pipeline.clean` and `pipeline.analyze` again. The original data is never modified, so every decision can be undone.

## What a run folder contains

```
runs/<file>_<date-time>/
├── REPORT.pdf     ← start here
├── details/       one document per section of the report
├── edit/          the two files you change to change the result
├── data/          the clean data and the rows that were left out
└── logs/          the machine-readable audit trail
```

**`REPORT.pdf`** is three or four pages:
- **In brief:** your question, the answer in a few lines, and the status of every check.
- **1. What we found:** each question's answer and findings, with a key chart.
- **2. How good the data is:** rows, problems fixed, what is still missing, and what that means for the findings.
- **3. Decisions made for you:** every judgment call, who decided it and how sure they were.
- **4. Wrap-up:** what might be wrong and what we recommend. This comes from the checks, the decisions and the review, plus the analysis's own doubts (marked AI).

Each section ends with a link to its detail document.

| Folder | File | What it is |
| --- | --- | --- |
| `details/` | `1_analysis.pdf` | Every finding with all its charts and tool results, and the plan behind the analysis |
| | `2_data_quality.pdf` | The data as received, what was wrong, every change (with any code the AI wrote), the checks, the review |
| | `2_data_columns.xlsx` | The data as received, column by column, in Excel |
| | `3_decisions.pdf` | Every decision with its options, reasons, alternatives and measured effect |
| | `4_audit.pdf` | Models, costs, prompt versions and every file |
| | `charts/` | Every chart as an image |
| `edit/` | `decisions.yaml` | The cleaning decisions; change one, then run `pipeline.clean` and `pipeline.analyze` |
| | `analysis_plan.yaml` | The analysis questions; change them, then run `pipeline.analyze` |
| `data/` | `clean_data.csv` / `.parquet` | The cleaned data, written only when Gate 2 does not block |
| | `dropped_rows.csv` | Every row removed while cleaning, with the reason |
| | `rejected_lines.csv` | Every line of the file that could not be read, in full |
| `logs/` | `manifest.json` | Models, prompt versions, cost and the status of each step |
| | everything else | The audit trail: every AI attempt, query, step, check and result, as JSON, plus Markdown versions of each step's report |

Old run folders, from before this layout, are moved into it the first time a command opens them (or all at once with `uv run python -m pipeline.layout runs`). The reports can be rebuilt at any time with `uv run python -m pipeline.reports runs/<folder>`.

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

## Analysis tools

The analysis agent does not write code: it picks from a fixed set of tested tools, and every result is saved with a pointer to the call that made it. Linear regression and `target_by_group` can also take another column as the outcome, for side questions such as "what goes with the fare?".

| Tool | Answers |
| --- | --- |
| `summary_table` | The basic numbers for every column: averages, spread and ranges, yes/no shares, categories, date ranges. |
| `distribution` | How are a column's values spread? |
| `target_by_group` | What is the outcome in each group or range? The rate for a yes/no outcome, the average for a number, with 95% intervals. |
| `target_by_two_groups` | Does a difference hold in every group? The outcome for every combination of two columns, e.g. sex within each class. |
| `target_by_ratio` | Does the outcome follow one number relative to another, e.g. loan amount ÷ income? |
| `trend_over_time` | How does the outcome change per year, quarter or month of a date? Rows per period when there is no outcome. |
| `compare_by_target` | How does a number differ between the two outcomes of a yes/no target? |
| `correlation` | Which numbers move together? |
| `missingness` | Where are values missing, and does that relate to the outcome? |
| `linear_regression` | For a number outcome: how much does it change with each feature, the others held fixed? |
| `logistic_regression` | For a yes/no outcome: how do the odds change with each feature, the others held fixed? |

Rules every tool follows:

- **Totals only, never rows.** Groups under 10 rows are merged or hidden.
- **Classic, simple statistics.** Each regression is fitted once on all usable rows and reports 95% intervals and p-values. There is no machine learning and no cross-validation. Finding no effect is a valid result.
- **Readable effects.** Numbers are compared in round steps ("alder per 20", "beloeb per 100,000"), and categories against the most common one.
- **Warnings.** An almost perfect fit (R² or AUC above 0.95) is flagged as possible leakage: a column that gives the answer away. Too few rows for the number of effects is flagged as unstable.
- **Subgroups and your own ranges.** Every tool can be limited to one subgroup (`subset`, e.g. third class only, or ages below 16; at least 30 rows), and numbers can be grouped at your own cut points (`cuts`, e.g. under and over 16).
- **Refused as inputs:** the outcome itself, IDs, dates, free text, personal data and excluded columns.

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
| Analysis results reveal no one | `tools/eda.py` and `tools/modeling.py` return only totals, rates and regression effects; groups under 10 rows are merged or hidden |
| The planner sees no rows | `tools/context.py` describes the clean data in totals only: nothing from ID, personal or free-text columns, no category with fewer than 10 rows |

**AI accountability**

| Guarantee | Enforced by |
| --- | --- |
| No judgment call is settled silently | Decided by you or by the AI from your brief, always shown with reasons and alternatives; low confidence must mean the option that changes the least (`contracts/diagnosis.py`); undecided calls are left alone (`agents/coder.py`) |
| The target is never filled in or dropped | Checked on the plan, after every step (`ops/plan.py`) and against the original data at Gate 2 |
| The clean data is what the diagnosis said it would be | Gate 2's schema contract (`checks/schema.py`); later stages re-check the data file against it |
| The reviewer's claims are backed by numbers | `agents/reviewer.py` drops claims citing numbers it was never shown and needs two verified checks to accept |
| Questions only you can answer reach you | The reviewer reports them in the report; nothing it says can change a decision |
| Every analysis result can be traced | Each one is saved as an artifact pointing to the tool call and the plan question behind it (`tools/registry.py`); each finding points to the calls it rests on (`agents/analysis.py`) |
| Every number in the findings was computed by a tool | Gate 4 (`checks/analysis_gate.py`, `checks/traceability.py`) matches every number in the AI's write-up against the results of the calls it cites |
| No finding claims more than its evidence | Gate 4 refuses "strong" for a finding that rests only on models doing no better than no model at all |
| The plan only uses what exists and is allowed | Gate 3 (`checks/plan_gate.py`) checks every column, tool, target and exclusion, and sends problems back to the planner |
| A column that gives the answer away is left out | Gate 3 tests each column on its own against the target; the planner is told before it plans, and must exclude it |
| Your edits to the plan are never overwritten | `agents/planner.py` will not plan again over an edited `edit/analysis_plan.yaml` |

## Project layout

| Folder | What it holds |
| --- | --- |
| [pipeline/contracts/](pipeline/contracts/) | The data passed between steps, as validated models. Their validators enforce the audit rules. |
| [pipeline/ingestion/](pipeline/ingestion/) | Loading, profiling, personal-data detection and masking, and claim checks |
| [pipeline/ops/](pipeline/ops/) | 15 library operations, the sandboxed `custom_code` operation, and the plan executor |
| [pipeline/agents/](pipeline/agents/) | The LLM client and the diagnostic, coder, reviewer, planner and analysis agents, with their prompts |
| [pipeline/checks/](pipeline/checks/) | Gates 1 to 4, the reconciliation checks, the schema contract and the number check |
| [pipeline/tools/](pipeline/tools/) | The reviewer's query tool, the analysis tools (statistics, charts, regressions), and the loader that prepares the clean data for analysis |
| [pipeline/run.py](pipeline/run.py) | The one command for the whole workflow |
| [pipeline/reports/](pipeline/reports/) | `REPORT.pdf` and the detail documents, rebuilt from the run folder (`python -m pipeline.reports runs/<folder>`) |
| [pipeline/layout.py](pipeline/layout.py) | Where every file of a run folder lives |
| [pipeline/ingest.py](pipeline/ingest.py), [pipeline/clean.py](pipeline/clean.py), [pipeline/analyze.py](pipeline/analyze.py) | The commands for steps 2, 5 and 6–7 |
| [benchmark/](benchmark/) | The benchmark: synthetic clean data, the breakers, scenarios, the scorer and the runner |
| [tests/fixtures/](tests/fixtures/) | Ten deliberately difficult files, with their expected results in [expected.yaml](tests/fixtures/expected.yaml) |

## Status

| Phase | State |
| --- | --- |
| 0. Contracts and setup | Done, except tracing and the GitHub Pages workflow |
| 1. Deterministic foundation | Done |
| 2. Cleaning agents | Mostly done: diagnostic agent, coder, Gate 2, reviewer and AI decisions run end to end. Left: canary rows and a run on five public datasets. |
| 3. Corruption benchmark | In progress: synthetic data, breakers, scorer and runner built; first results in; real messy datasets to come |
| 4. Analysis planning and your approval | Done: the planner, Gate 3 and automatic approval; edit the plan and run `pipeline.analyze` to change it |
| 5. Analysis agent | Done: the tools, the agent, Gate 4 and the findings in the PDF. Left: looking inside one subgroup, or crossing two groupings |
| 6. Evaluation | Not started |
| 7. App and publishing | Not started |

## Next

1. **Five-dataset run:** the whole workflow on five public datasets, to see how the findings hold up on data other than Titanic.

Later: canary rows (test rows with known answers, checked on every run) and real messy datasets for the benchmark.
