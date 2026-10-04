"""The schema contract: what the clean data is, checked against what the diagnosis expects.

Gate 2 builds the contract from the clean table (actual types, missing values,
distinct counts, which ops touched each column) and the diagnosis (role and
expected type per column), then checks the two against each other. The result
is saved with the handoff, and `validate_against_contract` lets any later stage
re-check data against it before using it.

Types are checked on DuckDB's own column types. Converting to pandas first
(as a Pandera schema would) blurs them: a DATE and a TIMESTAMP both arrive as
datetime64, so a date column could not be told apart from a timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb

from pipeline.checks.reconciliation import final_name_of
from pipeline.contracts.audit import CheckResult, OpRecord
from pipeline.contracts.cleaning import CleaningPlan
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.diagnosis import Diagnosis, TargetType
from pipeline.contracts.handoff import ColumnSchema
from pipeline.ingestion.raw import RESERVED_COLUMNS, ROW_ID, RawStore, RawTable, quote_ident
from pipeline.ops.library import DEFAULT_NULL_TOKENS

ROW_LOSS_WARN = 0.05
ROW_LOSS_BLOCK = 0.20
"""Spec Gate 2: under 5% passes, 5-20% is flagged, over 20% blocks."""

_INTEGER = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT")
_INTEGER += ("UINTEGER", "UBIGINT", "UHUGEINT")
TYPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "integer": _INTEGER,
    "float": ("FLOAT", "DOUBLE", "REAL", "DECIMAL", *_INTEGER),
    "boolean": ("BOOLEAN",),
    "date": ("DATE", "TIMESTAMP"),
    "datetime": ("TIMESTAMP", "DATE"),
    "categorical": ("VARCHAR", "ENUM"),
    "text": ("VARCHAR",),
    "id": ("VARCHAR",),
}
"""Which DuckDB types satisfy each clean type. An ID must stay text: '00417' is not 417."""


def type_fits(expected: TargetType, actual: str) -> bool:
    name = actual.upper()
    return any(
        name == family or name.startswith((f"{family}(", f"{family}_", f"{family} "))
        for family in TYPE_FAMILIES[expected]
    )


def key_column(brief: ContextBrief, diagnosis: Diagnosis, columns: Sequence[str]) -> str | None:
    """The brief's ID column, or else the diagnosis's only identifier."""
    if brief.id_column and brief.id_column in columns:
        return brief.id_column
    identifiers = [
        c.name for c in diagnosis.columns if c.role == "identifier" and c.name in columns
    ]
    return identifiers[0] if len(identifiers) == 1 else None


# -- building the contract --------------------------------------------------


def build_contract(
    store: RawStore,
    clean_table: str,
    *,
    raw_columns: Sequence[str],
    records: Sequence[OpRecord],
    diagnosis: Diagnosis,
    key: str | None,
) -> list[ColumnSchema]:
    """One entry per clean column, describing what is actually there."""
    origin = {}
    for column in raw_columns:
        final = final_name_of(column, records)
        if final is not None:
            origin[final] = column
    final_key = final_name_of(key, records) if key else None
    table = _qualified(clean_table)

    contract: list[ColumnSchema] = []
    for name, dtype in store.column_types_of(clean_table).items():
        identifier = quote_ident(name)
        rows, missing, distinct = store.connection.execute(
            f"SELECT count(*), count(*) FILTER (WHERE {identifier} IS NULL), "
            f"count(DISTINCT {identifier}) FROM {table}"
        ).fetchone()
        source = origin.get(name)
        entry = diagnosis.column(source or name)
        touched_by = [
            r.op_id
            for r in records
            if {source, name} & set(r.target_columns) or name in (r.params.get("adds") or [])
        ]
        contract.append(
            ColumnSchema(
                name=name,
                dtype=dtype,
                nullable=missing > 0,
                null_rate=missing / rows if rows else 0.0,
                n_distinct=distinct,
                source_column=source if source and source != name else None,
                transformations=touched_by,
                role=entry.role if entry else None,
                expected_type=entry.target_type if entry else None,
                unique=name == final_key and missing == 0 and distinct == rows,
            )
        )
    return contract


# -- the Gate 2 checks ------------------------------------------------------


def check_types(
    contract: list[ColumnSchema], plan: CleaningPlan, diagnosis: Diagnosis
) -> tuple[list[ColumnSchema], CheckResult]:
    """Every column has the type the diagnosis expects, or a stated reason why not.

    A column a step touched and left in the wrong type blocks: the plan tried
    and got it wrong (or cast an ID to a number). A column whose issue was
    explicitly skipped, or for which the diagnosis raised no issue at all, is
    only flagged, with the reason written into the contract.
    """
    skipped: dict[str, list[str]] = {}
    issues = {issue.id: issue for issue in diagnosis.issues}
    for item in plan.skipped:
        issue = issues.get(item.id)
        if issue is not None and issue.column:
            skipped.setdefault(issue.column, []).append(f"{item.id} was skipped: {item.reason}")

    updated: list[ColumnSchema] = []
    blocking: list[str] = []
    explained: list[str] = []
    for column in contract:
        expected = column.expected_type
        if expected is None or type_fits(expected, column.dtype):
            updated.append(column)
            continue
        source = column.source_column or column.name
        mismatch = f"{column.name} is {column.dtype}, the diagnosis expects {expected}"
        if source in skipped:
            note = f"{mismatch}: " + "; ".join(skipped[source])
        elif column.transformations:
            blocking.append(mismatch)
            updated.append(column)
            continue
        else:
            note = f"{mismatch}: the diagnosis raised no issue for it, so nothing converted it"
        explained.append(note)
        updated.append(column.model_copy(update={"note": note}))

    if blocking:
        check = _check("schema_types_match_diagnosis", "contract", False, "block", blocking)
    elif explained:
        check = _check("schema_types_match_diagnosis", "contract", False, "warn", explained)
    else:
        check = CheckResult(
            name="schema_types_match_diagnosis",
            layer="contract",
            passed=True,
            message=f"All {len(contract)} column(s) have the type the diagnosis expects.",
        )
    return updated, check


def check_columns(
    contract: list[ColumnSchema], diagnosis: Diagnosis, records: Sequence[OpRecord]
) -> CheckResult:
    """The column set matches the diagnosis; every difference is listed with its step."""
    present = {c.source_column or c.name for c in contract}
    diagnosed = {c.name for c in diagnosis.columns}
    removed_by = _removals(records)
    notes: list[str] = []

    for entry in diagnosis.columns:
        if entry.role == "drop" and entry.name in present:
            notes.append(f"{entry.name} is marked for dropping in the diagnosis but is still there")
        elif entry.role != "drop" and entry.name not in present:
            notes.append(f"{entry.name} was removed ({removed_by.get(entry.name, 'by a step')})")
    for column in contract:
        if (column.source_column or column.name) not in diagnosed:
            notes.append(f"{column.name} was added by cleaning")

    if notes:
        return _check("columns_match_diagnosis", "contract", False, "warn", notes)
    return CheckResult(
        name="columns_match_diagnosis",
        layer="contract",
        passed=True,
        message="The clean columns are exactly the ones the diagnosis expects.",
    )


def check_primary_key(
    store: RawStore,
    raw: RawTable,
    clean_table: str,
    key: str | None,
    records: Sequence[OpRecord],
) -> CheckResult:
    """The ID stays unique and complete, if it was in raw."""
    name = "primary_key_intact"
    if key is None:
        return CheckResult(
            name=name,
            layer="invariant",
            passed=True,
            message="No ID column is named in the brief or the diagnosis; nothing to check.",
        )
    final = final_name_of(key, records)
    if final is None:
        return _check(name, "invariant", False, "block", [f"the ID column {key!r} was removed"])

    raw_dupes, raw_missing = _key_faults(store, _qualified(raw.name), key, blank_is_missing=True)
    dupes, missing = _key_faults(store, _qualified(clean_table), final, blank_is_missing=False)
    if not dupes and not missing:
        return CheckResult(
            name=name,
            layer="invariant",
            passed=True,
            message=f"{final} is unique and never missing in every clean row.",
        )
    detail = f"{final} has {dupes} duplicate value(s) and {missing} missing value(s) after cleaning"
    if not raw_dupes and not raw_missing:
        return _check(name, "invariant", False, "block", [f"{detail}; it was unique in raw"])
    return _check(
        name,
        "invariant",
        False,
        "warn",
        [
            f"{detail}. It was not unique or complete in raw either ({raw_dupes} duplicate(s), "
            f"{raw_missing} missing), so this is a data issue, not one cleaning caused"
        ],
    )


def check_target(
    store: RawStore,
    raw: RawTable,
    clean_table: str,
    target: str | None,
    records: Sequence[OpRecord],
    null_tokens: Sequence[str],
) -> CheckResult:
    """No row whose target was missing in raw has a value now, whatever route it took.

    The per-step check in the executor catches a missing value being filled in.
    This one also catches a null token ('NA') turned straight into a value, which
    never passes through missing and so looks like an ordinary change step by step.
    """
    name = "target_not_imputed"
    if not target or target not in raw.columns:
        return CheckResult(
            name=name, layer="invariant", passed=True, message="No target column to protect."
        )
    final = final_name_of(target, records)
    if final is None:
        return _check(name, "invariant", False, "block", [f"the target {target!r} was removed"])

    tokens = sorted({t.strip().lower() for t in [*null_tokens, *DEFAULT_NULL_TOKENS]})
    token_list = ", ".join("'" + t.replace("'", "''") + "'" for t in tokens)
    raw_value = f"r.{quote_ident(target)}"
    missing_in_raw = (
        f"({raw_value} IS NULL OR lower(trim(CAST({raw_value} AS VARCHAR))) IN ({token_list}))"
    )
    filled, was_missing = store.connection.execute(
        f"SELECT count(*) FILTER (WHERE {missing_in_raw} AND c.{quote_ident(final)} IS NOT NULL), "
        f"count(*) FILTER (WHERE {missing_in_raw}) "
        f"FROM {_qualified(raw.name)} r JOIN {_qualified(clean_table)} c USING ({ROW_ID})"
    ).fetchone()
    if filled:
        return _check(
            name,
            "invariant",
            False,
            "block",
            [
                f"{filled} row(s) where the target {target!r} was missing in raw now have a "
                f"value; the target is never imputed"
            ],
        )
    return CheckResult(
        name=name,
        layer="invariant",
        passed=True,
        message=f"None of the {was_missing} missing target value(s) in kept rows was filled in.",
    )


def check_row_loss(rows_raw: int, records: Sequence[OpRecord]) -> CheckResult:
    dropped = sum(r.rows_dropped for r in records)
    share = dropped / rows_raw if rows_raw else 0.0
    message = (
        f"{dropped:,} of {rows_raw:,} rows dropped ({share:.1%}); under 5% passes, "
        f"5-20% is flagged, over 20% blocks."
    )
    severity = "block" if share > ROW_LOSS_BLOCK else "warn"
    return CheckResult(
        name="row_loss_within_threshold",
        layer="invariant",
        passed=share < ROW_LOSS_WARN,
        severity=severity,
        message=message,
        observed={"rows_dropped": dropped, "rows_raw": rows_raw, "share": round(share, 4)},
    )


# -- re-checking data against a saved contract ------------------------------


def validate_against_contract(
    connection: duckdb.DuckDBPyConnection, source: str, contract: Sequence[ColumnSchema]
) -> list[CheckResult]:
    """Does the data in `source` (a table, or e.g. read_parquet('...')) still match?

    For any stage that receives clean data: same columns, same types, no missing
    values where none were promised, and the key still unique.
    """
    described = connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    actual = {row[0]: row[1] for row in described if row[0] not in RESERVED_COLUMNS}
    promised = {c.name: c for c in contract}

    missing = [name for name in promised if name not in actual]
    extra = [name for name in actual if name not in promised]
    wrong_type = [
        f"{c.name} is {actual[c.name]}, the contract says {c.dtype}"
        for c in contract
        if c.name in actual and actual[c.name] != c.dtype
    ]
    unexpected_missing: list[str] = []
    not_unique: list[str] = []
    for column in contract:
        if column.name not in actual:
            continue
        identifier = quote_ident(column.name)
        rows, nulls, distinct = connection.execute(
            f"SELECT count(*), count(*) FILTER (WHERE {identifier} IS NULL), "
            f"count(DISTINCT {identifier}) FROM {source}"
        ).fetchone()
        if not column.nullable and nulls:
            unexpected_missing.append(f"{column.name} has {nulls} missing value(s)")
        if column.unique and distinct != rows:
            not_unique.append(f"{column.name} has {rows - distinct} duplicate or missing value(s)")

    column_problems = [f"missing column {n}" for n in missing] + [
        f"extra column {n}" for n in extra
    ]
    return [
        _verdict("contract_columns", column_problems, "The columns match the contract."),
        _verdict("contract_types", wrong_type, "Every column has its contracted type."),
        _verdict(
            "contract_missing_values",
            unexpected_missing,
            "No missing values where the contract promises none.",
        ),
        _verdict("contract_key_unique", not_unique, "The key is still unique."),
    ]


# -- helpers ----------------------------------------------------------------


def _key_faults(
    store: RawStore, table: str, column: str, *, blank_is_missing: bool
) -> tuple[int, int]:
    identifier = quote_ident(column)
    missing = f"{identifier} IS NULL"
    if blank_is_missing:
        missing += f" OR trim(CAST({identifier} AS VARCHAR)) = ''"
    rows, n_missing, distinct = store.connection.execute(
        f"SELECT count(*), count(*) FILTER (WHERE {missing}), "
        f"count(DISTINCT {identifier}) FILTER (WHERE NOT ({missing})) FROM {table}"
    ).fetchone()
    return rows - n_missing - distinct, n_missing


def _removals(records: Sequence[OpRecord]) -> dict[str, str]:
    removed: dict[str, str] = {}
    for number, record in enumerate(records, start=1):
        if record.op_name == "drop_column":
            removed[record.params["column"]] = (
                f"step {number}, drop_column: {record.params['reason']}"
            )
        elif record.op_name == "custom_code":
            for column in record.params.get("drops") or []:
                removed[column] = f"step {number}, custom code: {record.rationale}"
    return removed


def _check(name: str, layer: str, passed: bool, severity: str, parts: list[str]) -> CheckResult:
    return CheckResult(
        name=name,
        layer=layer,  # type: ignore[arg-type]
        passed=passed,
        severity=severity,  # type: ignore[arg-type]
        message="; ".join(parts) + ".",
    )


def _verdict(name: str, problems: list[str], ok: str) -> CheckResult:
    return CheckResult(
        name=name,
        layer="contract",
        passed=not problems,
        severity="block",
        message="; ".join(problems) + "." if problems else ok,
    )


def _qualified(table: str) -> str:
    schema, _, name = table.rpartition(".")
    return f"{schema}.{quote_ident(name)}"
