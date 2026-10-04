"""The fixed op library: the only vocabulary for changing data.

An agent may ask for these operations and nothing else. Each one is narrow on
purpose — `drop_rows` takes a column and a condition rather than a SQL
predicate, so a model cannot write arbitrary SQL — and each one that makes a
judgment call raises a flag rather than quietly making it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline.contracts.common import Flag
from pipeline.ingestion.raw import quote_ident
from pipeline.ops.base import OpContext, OpError, OpPlan, register, require_columns

DEFAULT_NULL_TOKENS = ("", "NA", "N/A", "null", "none", "-", "?")
"""What tokens_to_null treats as missing unless told otherwise; Gate 2 uses the same list."""

BOOLEAN_TRUE = ("true", "yes", "ja", "y", "t", "1")
BOOLEAN_FALSE = ("false", "no", "nej", "n", "f", "0")


class Params(BaseModel):
    """Op parameters are a contract too: unknown keys are a mistake, not a hint."""

    model_config = ConfigDict(extra="forbid")


def sql_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(value)
    return sql_str(str(value))


def judgment_flag(code: str, message: str, column: str | None, **details: Any) -> Flag:
    """Spec: flag rather than fix. A judgment call is recorded where a human sees it."""
    return Flag(
        code=code,
        severity="warn",
        message=message,
        column=column,
        stage="cleaning",
        details=details,
    )


# -- types --------------------------------------------------------------


class CastParams(Params):
    column: str
    to: Literal["integer", "float", "boolean", "date", "datetime", "text"]
    format: str | None = Field(default=None, description="strptime format for date/datetime.")
    on_error: Literal["error", "null"] = Field(
        default="error",
        description="'error' refuses to lose values; 'null' records the loss deliberately.",
    )

    @model_validator(mode="after")
    def _date_needs_format(self) -> CastParams:
        if self.to in {"date", "datetime"} and not self.format:
            raise ValueError("casting to a date requires an explicit format; guessing is banned")
        return self


@register("cast", CastParams)
def cast(ctx: OpContext, params: CastParams) -> OpPlan:
    """Convert a column to a type, with no format guessed and no value lost silently."""
    require_columns(ctx, params.column)
    identifier = quote_ident(params.column)

    if params.to == "integer":
        expression = f"TRY_CAST({identifier} AS BIGINT)"
    elif params.to == "float":
        expression = f"TRY_CAST({identifier} AS DOUBLE)"
    elif params.to == "text":
        expression = f"CAST({identifier} AS VARCHAR)"
    elif params.to == "boolean":
        true_list = ", ".join(sql_str(v) for v in BOOLEAN_TRUE)
        false_list = ", ".join(sql_str(v) for v in BOOLEAN_FALSE)
        expression = (
            f"CASE WHEN lower(trim(CAST({identifier} AS VARCHAR))) IN ({true_list}) THEN TRUE "
            f"WHEN lower(trim(CAST({identifier} AS VARCHAR))) IN ({false_list}) THEN FALSE "
            f"ELSE NULL END"
        )
    else:
        cast_to = "DATE" if params.to == "date" else "TIMESTAMP"
        expression = (
            f"CAST(try_strptime(CAST({identifier} AS VARCHAR), {sql_str(params.format)}) "
            f"AS {cast_to})"
        )

    return OpPlan(
        expressions={params.column: expression},
        target_columns=[params.column],
        allow_new_nulls=params.on_error == "null",
        note=f"Cast {params.column} to {params.to}.",
        flags=(
            [
                judgment_flag(
                    "cast_drops_unparseable",
                    f"Cast of {params.column!r} to {params.to} was allowed to null out "
                    f"values that do not parse.",
                    params.column,
                    on_error="null",
                )
            ]
            if params.on_error == "null"
            else []
        ),
    )


class ParseDatesParams(Params):
    column: str
    format: str
    on_error: Literal["error", "null"] = "error"


@register("parse_dates", ParseDatesParams)
def parse_dates(ctx: OpContext, params: ParseDatesParams) -> OpPlan:
    """Parse a text column into dates using one explicit format."""
    return cast(
        ctx,
        CastParams(column=params.column, to="date", format=params.format, on_error=params.on_error),
    )


class NormalizeDecimalParams(Params):
    column: str
    decimal_separator: Literal[",", "."] = ","
    thousands_separator: Literal[",", ".", " ", ""] = "."
    on_error: Literal["error", "null"] = "error"

    @model_validator(mode="after")
    def _separators_differ(self) -> NormalizeDecimalParams:
        if self.decimal_separator == self.thousands_separator:
            raise ValueError("decimal and thousands separators must differ")
        return self


@register("normalize_decimal", NormalizeDecimalParams)
def normalize_decimal(ctx: OpContext, params: NormalizeDecimalParams) -> OpPlan:
    """Read locale-formatted numbers: '1.234,50' becomes 1234.50, not 1.23450."""
    require_columns(ctx, params.column)
    identifier = f"CAST({quote_ident(params.column)} AS VARCHAR)"

    stripped = identifier
    if params.thousands_separator:
        stripped = f"replace({stripped}, {sql_str(params.thousands_separator)}, '')"
    if params.decimal_separator != ".":
        stripped = f"replace({stripped}, {sql_str(params.decimal_separator)}, '.')"

    return OpPlan(
        expressions={params.column: f"TRY_CAST(trim({stripped}) AS DOUBLE)"},
        target_columns=[params.column],
        allow_new_nulls=params.on_error == "null",
        note=(
            f"Parsed {params.column} with decimal {params.decimal_separator!r} and "
            f"thousands {params.thousands_separator!r}."
        ),
    )


class StripCurrencyParams(Params):
    column: str
    symbols: list[str] = Field(default_factory=lambda: ["kr.", "kr", "DKK", "€", "$", "£"])


@register("strip_currency", StripCurrencyParams)
def strip_currency(ctx: OpContext, params: StripCurrencyParams) -> OpPlan:
    """Remove currency markers so a column can later be read as a number."""
    require_columns(ctx, params.column)
    expression = f"CAST({quote_ident(params.column)} AS VARCHAR)"
    for symbol in params.symbols:
        expression = f"replace({expression}, {sql_str(symbol)}, '')"
    return OpPlan(
        expressions={params.column: f"trim({expression})"},
        target_columns=[params.column],
        note=f"Stripped {params.symbols} from {params.column}.",
    )


# -- text ---------------------------------------------------------------


class ColumnsParams(Params):
    columns: list[str] = Field(min_length=1)


@register("trim_whitespace", ColumnsParams)
def trim_whitespace(ctx: OpContext, params: ColumnsParams) -> OpPlan:
    """Remove leading and trailing whitespace, a common export artefact."""
    require_columns(ctx, *params.columns)
    return OpPlan(
        expressions={c: f"trim(CAST({quote_ident(c)} AS VARCHAR))" for c in params.columns},
        target_columns=list(params.columns),
        note="Trimmed whitespace.",
    )


class NormalizeCaseParams(Params):
    columns: list[str] = Field(min_length=1)
    case: Literal["lower", "upper"] = "lower"


@register("normalize_case", NormalizeCaseParams)
def normalize_case(ctx: OpContext, params: NormalizeCaseParams) -> OpPlan:
    """Fold case so 'Nord' and 'nord' stop being two categories."""
    require_columns(ctx, *params.columns)
    function = "lower" if params.case == "lower" else "upper"
    return OpPlan(
        expressions={c: f"{function}(CAST({quote_ident(c)} AS VARCHAR))" for c in params.columns},
        target_columns=list(params.columns),
        note=f"Normalised case to {params.case}.",
    )


class StandardizeCategoriesParams(Params):
    column: str
    mapping: dict[str, str] = Field(min_length=1)
    case_insensitive: bool = True


@register("standardize_categories", StandardizeCategoriesParams)
def standardize_categories(ctx: OpContext, params: StandardizeCategoriesParams) -> OpPlan:
    """Map category spellings onto one label each. Unmapped values are left alone."""
    require_columns(ctx, params.column)
    identifier = f"CAST({quote_ident(params.column)} AS VARCHAR)"
    subject = f"lower(trim({identifier}))" if params.case_insensitive else f"trim({identifier})"

    branches = "".join(
        f" WHEN {sql_str(key.lower() if params.case_insensitive else key)} THEN {sql_str(value)}"
        for key, value in params.mapping.items()
    )
    return OpPlan(
        expressions={params.column: f"CASE {subject}{branches} ELSE {identifier} END"},
        target_columns=[params.column],
        note=f"Standardised {len(params.mapping)} category spelling(s) in {params.column}.",
    )


# -- missing values -----------------------------------------------------


class TokensToNullParams(Params):
    columns: list[str] = Field(min_length=1)
    tokens: list[str] = Field(default_factory=lambda: list(DEFAULT_NULL_TOKENS))
    case_insensitive: bool = True


@register("tokens_to_null", TokensToNullParams)
def tokens_to_null(ctx: OpContext, params: TokensToNullParams) -> OpPlan:
    """Turn null tokens into real nulls, so 'NA' stops counting as a category."""
    require_columns(ctx, *params.columns)
    tokens = [t.lower() if params.case_insensitive else t for t in params.tokens]
    token_list = ", ".join(sql_str(t) for t in tokens)

    expressions = {}
    for column in params.columns:
        identifier = f"CAST({quote_ident(column)} AS VARCHAR)"
        subject = f"lower(trim({identifier}))" if params.case_insensitive else f"trim({identifier})"
        expressions[column] = (
            f"CASE WHEN {subject} IN ({token_list}) THEN NULL ELSE {identifier} END"
        )

    return OpPlan(
        expressions=expressions,
        target_columns=list(params.columns),
        allow_new_nulls=True,
        note=f"Converted {params.tokens} to NULL.",
    )


class ReplaceSentinelParams(Params):
    column: str
    values: list[str] = Field(min_length=1)


@register("replace_sentinel", ReplaceSentinelParams)
def replace_sentinel(ctx: OpContext, params: ReplaceSentinelParams) -> OpPlan:
    """Reinterpret coded missing values (999, 1900-01-01) as missing.

    A judgment call: only the brief or an explicit decision justifies it, so it
    is always flagged.
    """
    require_columns(ctx, params.column)
    identifier = quote_ident(params.column)
    values = ", ".join(sql_str(v) for v in params.values)

    return OpPlan(
        expressions={
            params.column: (
                f"CASE WHEN CAST({identifier} AS VARCHAR) IN ({values}) THEN NULL "
                f"ELSE {identifier} END"
            )
        },
        target_columns=[params.column],
        allow_new_nulls=True,
        note=f"Treated {params.values} in {params.column} as missing.",
        flags=[
            judgment_flag(
                "sentinel_reinterpreted",
                f"Values {params.values} in {params.column!r} were treated as missing. "
                f"This changes every statistic on the column.",
                params.column,
                values=params.values,
            )
        ],
    )


class ImputeParams(Params):
    column: str
    strategy: Literal["constant", "mean", "median", "mode"]
    value: Any = Field(default=None, description="Required for the 'constant' strategy.")

    @model_validator(mode="after")
    def _constant_needs_value(self) -> ImputeParams:
        if self.strategy == "constant" and self.value is None:
            raise ValueError("the 'constant' strategy needs a value")
        return self


@register("impute", ImputeParams)
def impute(ctx: OpContext, params: ImputeParams) -> OpPlan:
    """Fill missing values. Always a judgment call, so always flagged.

    The fill value is computed here and written into the SQL as a literal, so
    the record shows the number that was actually used rather than a recipe.
    """
    require_columns(ctx, params.column)
    identifier = quote_ident(params.column)
    source = ctx.table.rpartition(".")
    qualified = f"{source[0] or 'clean'}.{quote_ident(source[2])}"

    n_missing = ctx.store.connection.execute(
        f"SELECT count(*) FILTER (WHERE {identifier} IS NULL) FROM {qualified}"
    ).fetchone()[0]

    if params.strategy == "constant":
        fill = params.value
    elif params.strategy == "mode":
        fill = ctx.store.connection.execute(
            f"SELECT mode({identifier}) FROM {qualified}"
        ).fetchone()[0]
    else:
        # Cast inside the aggregate so asking for the mean of a text column
        # fails as "no value could be computed" rather than as a SQL error.
        aggregate = "avg" if params.strategy == "mean" else "median"
        fill = ctx.store.connection.execute(
            f"SELECT {aggregate}(TRY_CAST({identifier} AS DOUBLE)) FROM {qualified}"
        ).fetchone()[0]

    if fill is None:
        raise OpError(
            f"cannot impute {params.column!r} with strategy {params.strategy!r}: "
            f"no value could be computed (the column may be entirely missing)"
        )

    share = n_missing / max(ctx.store.row_count(ctx.table), 1)
    return OpPlan(
        expressions={params.column: f"coalesce({identifier}, {sql_value(fill)})"},
        target_columns=[params.column],
        note=f"Imputed {n_missing} missing value(s) in {params.column} with {fill!r}.",
        flags=[
            judgment_flag(
                "imputation",
                f"{n_missing} missing value(s) ({share:.1%}) in {params.column!r} were filled "
                f"with {fill!r} ({params.strategy}). Distributions and any model change too.",
                params.column,
                strategy=params.strategy,
                fill_value=str(fill),
                n_filled=n_missing,
                share=share,
            )
        ],
    )


# -- rows ---------------------------------------------------------------


class DropDuplicatesParams(Params):
    subset: list[str] | None = Field(
        default=None, description="Columns that define a duplicate; all of them by default."
    )


@register("drop_duplicates", DropDuplicatesParams)
def drop_duplicates(ctx: OpContext, params: DropDuplicatesParams) -> OpPlan:
    """Keep the first row per duplicate key, in file order."""
    subset = params.subset or list(ctx.columns)
    require_columns(ctx, *subset)
    return OpPlan(
        distinct_on=subset,
        drop_reason=f"duplicate of an earlier row on {subset}",
        note=f"Dropped duplicate rows on {subset}, keeping the first occurrence.",
    )


class DropRowsParams(Params):
    column: str
    condition: Literal[
        "is_null", "is_not_null", "equals", "not_equals", "greater_than", "less_than"
    ]
    value: Any = None
    reason: str = Field(min_length=1, description="Why these rows do not belong in the analysis.")

    @model_validator(mode="after")
    def _comparison_needs_value(self) -> DropRowsParams:
        if self.condition not in {"is_null", "is_not_null"} and self.value is None:
            raise ValueError(f"condition {self.condition!r} needs a value")
        return self


@register("drop_rows", DropRowsParams)
def drop_rows(ctx: OpContext, params: DropRowsParams) -> OpPlan:
    """Remove rows matching one simple condition.

    Deliberately not a SQL predicate: an agent picks a column, a comparison and
    a value, which keeps generated SQL out of the execution path entirely.
    """
    require_columns(ctx, params.column)
    identifier = quote_ident(params.column)
    literal = sql_value(params.value)

    # The predicate describes the rows to KEEP, so each condition is negated.
    keep = {
        "is_null": f"{identifier} IS NOT NULL",
        "is_not_null": f"{identifier} IS NULL",
        "equals": f"{identifier} IS DISTINCT FROM {literal}",
        "not_equals": f"{identifier} IS NOT DISTINCT FROM {literal}",
        "greater_than": f"NOT ({identifier} > {literal}) OR {identifier} IS NULL",
        "less_than": f"NOT ({identifier} < {literal}) OR {identifier} IS NULL",
    }[params.condition]

    return OpPlan(
        where=keep,
        drop_reason=params.reason,
        note=f"Dropped rows where {params.column} {params.condition} {params.value!r}.",
        flags=[
            judgment_flag(
                "rows_dropped",
                f"Rows were removed because {params.column} {params.condition} "
                f"{params.value!r}: {params.reason}",
                params.column,
                condition=params.condition,
            )
        ],
    )


class ClipOutliersParams(Params):
    column: str
    lower: float | None = None
    upper: float | None = None
    method: Literal["bounds", "iqr"] = "bounds"
    factor: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def _bounds_need_a_limit(self) -> ClipOutliersParams:
        if self.method == "bounds" and self.lower is None and self.upper is None:
            raise ValueError("method 'bounds' needs a lower or upper limit")
        return self


@register("clip_outliers", ClipOutliersParams)
def clip_outliers(ctx: OpContext, params: ClipOutliersParams) -> OpPlan:
    """Pull extreme values back to a limit instead of dropping their rows.

    A judgment call, and one that hides information: the flag says how many
    values moved so a reader can weigh it.
    """
    require_columns(ctx, params.column)
    identifier = quote_ident(params.column)
    source = ctx.table.rpartition(".")
    qualified = f"{source[0] or 'clean'}.{quote_ident(source[2])}"

    lower, upper = params.lower, params.upper
    if params.method == "iqr":
        q1, q3 = ctx.store.connection.execute(
            f"SELECT quantile_cont({identifier}, 0.25), quantile_cont({identifier}, 0.75) "
            f"FROM {qualified}"
        ).fetchone()
        if q1 is None or q3 is None:
            raise OpError(f"cannot compute IQR bounds for {params.column!r}: no numeric values")
        spread = float(q3) - float(q1)
        lower, upper = float(q1) - params.factor * spread, float(q3) + params.factor * spread

    expression = identifier
    if lower is not None:
        expression = f"greatest({expression}, {sql_value(lower)})"
    if upper is not None:
        expression = f"least({expression}, {sql_value(upper)})"

    n_clipped = ctx.store.connection.execute(
        f"SELECT count(*) FROM {qualified} WHERE "
        f"({sql_value(lower)} IS NOT NULL AND {identifier} < {sql_value(lower)}) OR "
        f"({sql_value(upper)} IS NOT NULL AND {identifier} > {sql_value(upper)})"
    ).fetchone()[0]

    return OpPlan(
        expressions={params.column: expression},
        target_columns=[params.column],
        note=f"Clipped {params.column} to [{lower}, {upper}]; {n_clipped} value(s) moved.",
        flags=[
            judgment_flag(
                "outliers_clipped",
                f"{n_clipped} value(s) in {params.column!r} were pulled to "
                f"[{lower}, {upper}]. The original values are only in the raw layer.",
                params.column,
                lower=lower,
                upper=upper,
                n_clipped=n_clipped,
            )
        ],
    )


# -- columns ------------------------------------------------------------


class DropColumnParams(Params):
    column: str
    reason: str = Field(min_length=1)


@register("drop_column", DropColumnParams)
def drop_column(ctx: OpContext, params: DropColumnParams) -> OpPlan:
    """Remove a column. Always a judgment call, so always flagged."""
    require_columns(ctx, params.column)
    if len(ctx.columns) == 1:
        raise OpError("refusing to drop the last remaining column")
    return OpPlan(
        drop_columns=[params.column],
        target_columns=[params.column],
        note=f"Dropped column {params.column}: {params.reason}",
        flags=[
            judgment_flag(
                "column_dropped",
                f"Column {params.column!r} was removed: {params.reason}",
                params.column,
                reason=params.reason,
            )
        ],
    )


class RenameColumnParams(Params):
    column: str
    to: str = Field(min_length=1)


@register("rename_column", RenameColumnParams)
def rename_column(ctx: OpContext, params: RenameColumnParams) -> OpPlan:
    """Rename a column, refusing to collide with an existing one."""
    require_columns(ctx, params.column)
    if params.to in ctx.columns and params.to != params.column:
        raise OpError(f"cannot rename {params.column!r} to {params.to!r}: that column exists")
    return OpPlan(
        renames={params.column: params.to},
        target_columns=[params.column],
        note=f"Renamed {params.column} to {params.to}.",
    )
