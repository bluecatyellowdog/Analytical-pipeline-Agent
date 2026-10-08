"""Descriptive analysis tools: summary statistics, distributions, the target by group,
comparisons, correlation and missingness.

Every result is an aggregate. Groups below the context's minimum size are
merged into 'other' or hidden, and missing values get their own group rather
than disappearing, because who is missing is often part of the answer. A
yes/no target is described by its rate, a numeric target by its mean.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from scipy import stats

from pipeline.tools import charts
from pipeline.tools.registry import (
    FEATURE_KINDS,
    AnalysisContext,
    Params,
    ToolError,
    ToolResult,
    as_yes_no,
    register,
)

MISSING = "(missing)"
Z_95 = 1.959964


def wilson(events: int, n: int) -> tuple[float, float]:
    """95% Wilson interval for a proportion: honest at small n and near 0 or 1."""
    if n == 0:
        return 0.0, 1.0
    p = events / n
    centre = (p + Z_95**2 / (2 * n)) / (1 + Z_95**2 / n)
    spread = Z_95 * math.sqrt(p * (1 - p) / n + Z_95**2 / (4 * n * n)) / (1 + Z_95**2 / n)
    return max(0.0, centre - spread), min(1.0, centre + spread)


def mean_interval(mean: float, squares: float, n: int) -> tuple[float, float]:
    """95% t-interval for a mean, from the group's sum of squared values."""
    if n < 2:
        return mean, mean
    variance = max((squares - n * mean * mean) / (n - 1), 0.0)
    half = float(stats.t.ppf(0.975, n - 1)) * math.sqrt(variance / n)
    return mean - half, mean + half


def fmt(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.3g}"


def _target(ctx: AnalysisContext) -> tuple[pd.Series, str]:
    """The target as numbers, and 'rate' for a yes/no target or 'mean' for a numeric one."""
    if ctx.target_type() == "binary":
        return ctx.binary_target(), "rate"
    return ctx.numeric_target(), "mean"


def _show(value: float, measure: str) -> str:
    return f"{value:.1%}" if measure == "rate" else fmt(value)


def _groups(
    ctx: AnalysisContext, column: str, kind: str, bins: int, cuts: list[float] | None = None
) -> tuple[pd.Series, list[str] | None]:
    """Each row's group label (a value, a numeric range, or (missing)) and, for numbers,
    the labels in ascending order. `cuts` are the user's own cut points for a number."""
    values = ctx.frame[column]
    if cuts and kind != "numeric":
        raise ToolError(f"cuts are for number columns; {column} is {kind}")
    if kind != "numeric":
        return values.map(lambda v: str(v) if pd.notna(v) else MISSING).astype(object), None
    numbers = pd.to_numeric(values, errors="raise").astype(float)
    present = numbers.dropna()
    labels = pd.Series(MISSING, index=values.index, dtype=object)
    if cuts:
        edges = sorted({float(c) for c in cuts})
        names = [f"below {fmt(edges[0])}"]
        names += [f"{fmt(a)} to below {fmt(b)}" for a, b in zip(edges, edges[1:], strict=False)]
        names.append(f"{fmt(edges[-1])} and over")
        binned = pd.cut(present, [-math.inf, *edges, math.inf], right=False, labels=names)
        labels.loc[present.index] = binned.astype(object)
        return labels, [*names, MISSING]
    if present.nunique() <= bins:
        ordered = [fmt(v) for v in sorted(present.unique())]
        labels.loc[present.index] = present.map(fmt)
    else:
        binned = pd.qcut(present, q=bins, duplicates="drop")
        whole = bool((present == present.round()).all())
        names = {
            iv: _range_label(iv, first=i == 0, lowest=float(present.min()), whole=whole)
            for i, iv in enumerate(binned.cat.categories)
        }
        ordered = list(names.values())
        labels.loc[present.index] = binned.map(names).astype(object)
    return labels, [*ordered, MISSING]


def _range_label(interval: pd.Interval, *, first: bool, lowest: float, whole: bool) -> str:
    """A range as people read it: from the real lowest value, and '2-6' rather than '(1, 6]'
    for whole numbers."""
    low, high = (lowest if first else interval.left), interval.right
    if whole:
        low = lowest if first else math.floor(interval.left) + 1
        high = math.floor(high)
    return fmt(low) if low == high else f"{fmt(low)}-{fmt(high)}"


def _collapse(table: pd.DataFrame, min_n: int, *, keep_order: bool) -> tuple[pd.DataFrame, int]:
    """Merge groups below min_n into 'other'; hide 'other' too if it is still too small."""
    small = table[table["n"] < min_n]
    if small.empty:
        return table, 0
    kept = table[table["n"] >= min_n]
    merged = small.drop(columns=["group"]).sum(numeric_only=True)
    hidden = 0
    if merged["n"] >= min_n:
        other = pd.DataFrame([{"group": f"other ({len(small)} groups)", **merged.to_dict()}])
        kept = pd.concat([kept, other], ignore_index=True)
    else:
        hidden = int(merged["n"])
    if not keep_order:
        kept = kept.sort_values("n", ascending=False, kind="stable")
    return kept.reset_index(drop=True), hidden


# -- summary table ------------------------------------------------------------


class SummaryParams(Params):
    pass


def _iso(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def _numeric_summary(x: pd.Series) -> dict:
    if x.empty:
        return dict.fromkeys(("mean", "std", "min", "p25", "median", "p75", "max"))
    q = x.quantile([0.25, 0.5, 0.75])
    return {
        "mean": float(x.mean()),
        "std": float(x.std()) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "p25": float(q[0.25]),
        "median": float(q[0.5]),
        "p75": float(q[0.75]),
        "max": float(x.max()),
    }


@register("summary_table", SummaryParams, ("distribution", "data_quality"))
def summary_table(ctx: AnalysisContext, params: SummaryParams) -> list[ToolResult]:
    """Descriptive statistics for every column: numbers, yes/no columns, categories and dates."""
    numbers, yes_no, categories, dates = [], [], [], []
    smallest: list[int] = []
    for column, kind in ctx.kinds.items():
        if column in ctx.excluded or kind not in (*FEATURE_KINDS, "date"):
            continue
        values = ctx.frame[column]
        present = values.dropna()
        n = int(len(present))
        base = {"column": column, "n": n, "missing": int(values.isna().sum())}
        flags = as_yes_no(values) if kind == "boolean" else None
        if kind == "numeric":
            numbers.append({**base, **_numeric_summary(pd.to_numeric(present).astype(float))})
        elif kind == "date":
            dates.append(
                {
                    **base,
                    "earliest": _iso(present.min()) if n else None,
                    "latest": _iso(present.max()) if n else None,
                }
            )
        elif flags is not None:
            yes = int(flags.sum())
            if n:
                smallest += [yes, n - yes]
            yes_no.append({**base, "yes": yes, "share_yes": yes / n if n else None})
        else:  # a category, or a yes/no column whose values do not read as yes/no
            counts = present.astype(str).value_counts()
            row = {**base, "values": int(len(counts)), "most_common": None, "share": None}
            if len(counts) and counts.iloc[0] >= ctx.min_group_n:
                row["most_common"] = str(counts.index[0])
                row["share"] = float(counts.iloc[0] / base["n"])
                smallest.append(int(counts.iloc[0]))
            categories.append(row)
    if not (numbers or yes_no or categories or dates):
        raise ToolError("no columns to describe")

    parts = []
    if numbers:
        parts.append(
            "Numbers: "
            + "; ".join(
                f"{r['column']} mean {fmt(r['mean'])} (sd {fmt(r['std'])}, median "
                f"{fmt(r['median'])}, range {fmt(r['min'])}-{fmt(r['max'])}, "
                f"missing {r['missing']})"
                for r in numbers
            )
        )
    if yes_no:
        parts.append(
            "Yes/no: "
            + "; ".join(
                f"{r['column']} {r['share_yes']:.1%} yes (n={r['n']})"
                if r["n"]
                else f"{r['column']} no values"
                for r in yes_no
            )
        )
    if categories:
        parts.append(
            "Categories: "
            + "; ".join(
                f"{r['column']} {r['values']} values"
                + (
                    f", most common {r['most_common']} ({r['share']:.1%})"
                    if r["most_common"] is not None
                    else ""
                )
                for r in categories
            )
        )
    if dates:
        parts.append(
            "Dates: "
            + "; ".join(
                f"{r['column']} {r['earliest']} to {r['latest']}"
                if r["n"]
                else f"{r['column']} no values"
                for r in dates
            )
        )
    return [
        ToolResult(
            title="Summary statistics",
            data={
                "rows": int(len(ctx.frame)),
                "numbers": numbers,
                "yes_no": yes_no,
                "categories": categories,
                "dates": dates,
            },
            summary=f"{len(ctx.frame)} rows. " + ". ".join(parts),
            min_group_n=min(smallest, default=None),
        )
    ]


# -- distribution -----------------------------------------------------------


class DistributionParams(Params):
    column: str
    bins: int = Field(default=10, ge=2, le=30, description="Histogram bins for a number.")


@register("distribution", DistributionParams, ("distribution", "data_quality"))
def distribution(ctx: AnalysisContext, params: DistributionParams) -> list[ToolResult]:
    """How the values of one column are spread: a histogram for numbers, counts for categories."""
    column = params.column
    kind = ctx.require(column)
    if kind in ("id", "text", "personal") or column in ctx.excluded:
        raise ToolError(f"{column} is a {kind} or excluded column; its values are not shown")
    values = ctx.frame[column]
    missing = int(values.isna().sum())

    if kind == "numeric":
        numbers = pd.to_numeric(values.dropna()).astype(float)
        counts, edges = np.histogram(numbers, bins=params.bins)
        quantiles = numbers.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        summary = {
            "n": int(len(numbers)),
            "missing": missing,
            "mean": float(numbers.mean()),
            "std": float(numbers.std()),
            "min": float(numbers.min()),
            "p5": float(quantiles[0.05]),
            "p25": float(quantiles[0.25]),
            "median": float(quantiles[0.5]),
            "p75": float(quantiles[0.75]),
            "p95": float(quantiles[0.95]),
            "max": float(numbers.max()),
        }
        bins = [
            {"from": float(a), "to": float(b), "count": int(c)}
            for a, b, c in zip(edges[:-1], edges[1:], counts, strict=True)
        ]
        small = f"<{ctx.min_group_n}"  # a tail of a few rows is shown, not counted
        shown = [
            f"{fmt(b['from'])}-{fmt(b['to'])}: "
            + (str(b["count"]) if b["count"] == 0 or b["count"] >= ctx.min_group_n else small)
            for b in bins
        ]
        text = (
            f"{column}: n={summary['n']}, missing={missing}, median {fmt(summary['median'])}, "
            f"middle half {fmt(summary['p25'])}-{fmt(summary['p75'])}, "
            f"range {fmt(summary['min'])}-{fmt(summary['max'])}; rows per range: "
            + "; ".join(shown)
        )
        figure = charts.histogram(edges, counts, title=f"Distribution of {column}", xlabel=column)
        return [
            ToolResult(
                title=f"Distribution of {column}",
                data={"column": column, "summary": summary, "bins": bins},
                summary=text,
                figure=figure,
                min_group_n=int(min(counts)) if len(counts) else None,
            )
        ]

    labels = values.map(lambda v: str(v) if pd.notna(v) else MISSING)
    table = labels.value_counts().rename_axis("group").reset_index(name="n")
    table, hidden = _collapse(table, ctx.min_group_n, keep_order=False)
    total = int(len(values))
    groups = [
        {"group": row.group, "n": int(row.n), "share": float(row.n / total)}
        for row in table.itertuples()
    ]
    text = f"{column}: " + "; ".join(f"{g['group']} {g['share']:.1%} (n={g['n']})" for g in groups)
    if hidden:
        text += f"; {hidden} row(s) in groups too small to show"
    figure = charts.bars(
        [g["group"] for g in groups],
        [g["share"] for g in groups],
        title=f"Distribution of {column}",
        ylabel="share of rows",
        percent=True,
    )
    return [
        ToolResult(
            title=f"Distribution of {column}",
            data={"column": column, "groups": groups, "hidden_rows": hidden},
            summary=text,
            figure=figure,
            min_group_n=min((g["n"] for g in groups), default=None),
        )
    ]


# -- the target by group ------------------------------------------------------


class GroupParams(Params):
    column: str
    bins: int = Field(
        default=5,
        ge=2,
        le=10,
        description="Ranges a number is split into. A number with this many distinct values "
        "or fewer (a count such as 0-6) is grouped value by value instead.",
    )
    cuts: list[float] | None = Field(
        default=None,
        description="Your own cut points for a number, e.g. [16, 30] gives 'below 16', "
        "'16 to below 30' and '30 and over'. Used instead of bins.",
    )
    outcome: str | None = Field(
        default=None,
        description="A numeric or yes/no column to compare across the groups instead of the "
        "target, for a side question.",
    )


@register("target_by_group", GroupParams, ("relationship", "segment"))
def target_by_group(ctx: AnalysisContext, params: GroupParams) -> list[ToolResult]:
    """The target's rate (yes/no) or mean (number) per group of one column, with 95% intervals."""
    ctx = ctx.with_outcome(params.outcome)
    column = params.column
    kind = ctx.feature(column)
    target, measure = _target(ctx)
    present = target.notna()
    labels, order = _groups(ctx, column, kind, params.bins, params.cuts)
    values = target[present].astype(float)

    table = (
        pd.DataFrame({"group": labels[present], "value": values, "square": values**2})
        .groupby("group", sort=False)
        .agg(n=("value", "size"), total=("value", "sum"), squares=("square", "sum"))
        .reset_index()
    )
    if order is not None:
        position = {label: index for index, label in enumerate(order)}
        table = table.sort_values("group", key=lambda g: g.map(position), kind="stable")
    table, hidden = _collapse(
        table.reset_index(drop=True), ctx.min_group_n, keep_order=order is not None
    )

    overall = float(values.mean())
    rows = []
    for row in table.itertuples():
        n = int(row.n)
        value = float(row.total / n)
        if measure == "rate":
            events = int(round(row.total))
            low, high = wilson(events, n)
            rows.append({"group": row.group, "n": n, "events": events, "rate": value})
        else:
            low, high = mean_interval(value, float(row.squares), n)
            rows.append({"group": row.group, "n": n, "mean": value})
        rows[-1].update({"ci_low": low, "ci_high": high})
    text = (
        f"{measure.title()} of {ctx.target} by {column} (overall {_show(overall, measure)}, "
        f"n={int(present.sum())}): "
        + "; ".join(
            f"{r['group']}: {_show(r[measure], measure)} (95% interval "
            f"{_show(r['ci_low'], measure)} to {_show(r['ci_high'], measure)}, n={r['n']})"
            for r in rows
        )
    )
    if hidden:
        text += f"; {hidden} row(s) in groups too small to show"
    figure = charts.bars(
        [r["group"] for r in rows],
        [r[measure] for r in rows],
        lower=[r["ci_low"] for r in rows],
        upper=[r["ci_high"] for r in rows],
        reference=overall,
        title=f"{ctx.target} {measure} by {column} (95% interval)",
        ylabel=f"{ctx.target} {measure}",
        percent=measure == "rate",
    )
    return [
        ToolResult(
            title=f"{ctx.target} {measure} by {column}",
            data={
                "column": column,
                "target": ctx.target,
                "measure": measure,
                "overall": overall,
                "groups": rows,
                "hidden_rows": hidden,
            },
            summary=text,
            figure=figure,
            min_group_n=min((r["n"] for r in rows), default=None),
        )
    ]


# -- the target by two groups at once ------------------------------------------


class TwoGroupParams(Params):
    column_a: str
    column_b: str
    bins: int = Field(default=3, ge=2, le=6, description="Ranges a number is split into.")
    cuts_a: list[float] | None = Field(default=None, description="Own cut points for column_a.")
    cuts_b: list[float] | None = Field(default=None, description="Own cut points for column_b.")
    outcome: str | None = Field(
        default=None, description="Another numeric or yes/no column to compare instead."
    )


@register("target_by_two_groups", TwoGroupParams, ("segment", "relationship"))
def target_by_two_groups(ctx: AnalysisContext, params: TwoGroupParams) -> list[ToolResult]:
    """The target's rate or mean for every combination of two columns, e.g. sex within class."""
    ctx = ctx.with_outcome(params.outcome)
    a, b = params.column_a, params.column_b
    if a == b:
        raise ToolError("column_a and column_b must be different columns")
    kind_a, kind_b = ctx.feature(a), ctx.feature(b)
    target, measure = _target(ctx)
    present = target.notna()
    labels_a, order_a = _groups(ctx, a, kind_a, params.bins, params.cuts_a)
    labels_b, order_b = _groups(ctx, b, kind_b, params.bins, params.cuts_b)
    values = target[present].astype(float)
    table = (
        pd.DataFrame(
            {
                "a": labels_a[present],
                "b": labels_b[present],
                "value": values,
                "square": values**2,
            }
        )
        .groupby(["a", "b"], sort=False)
        .agg(n=("value", "size"), total=("value", "sum"), squares=("square", "sum"))
        .reset_index()
    )
    small = table["n"] < ctx.min_group_n
    hidden = int(table.loc[small, "n"].sum())
    table = table[~small]
    order_a = order_a or list(labels_a[present].value_counts().index)
    order_b = order_b or list(labels_b[present].value_counts().index)
    rank_a = {label: i for i, label in enumerate(order_a)}
    rank_b = {label: i for i, label in enumerate(order_b)}
    table = table.assign(_a=table["a"].map(rank_a), _b=table["b"].map(rank_b)).sort_values(
        ["_a", "_b"], kind="stable"
    )

    overall = float(values.mean())
    cells = []
    for row in table.itertuples():
        n = int(row.n)
        value = float(row.total / n)
        cell = {"group_a": row.a, "group_b": row.b, "n": n}
        if measure == "rate":
            events = int(round(row.total))
            low, high = wilson(events, n)
            cell |= {"events": events, "rate": value}
        else:
            low, high = mean_interval(value, float(row.squares), n)
            cell |= {"mean": value}
        cells.append(cell | {"ci_low": low, "ci_high": high})

    text = (
        f"{measure.title()} of {ctx.target} by {a} and {b} (overall {_show(overall, measure)}): "
        + "; ".join(
            f"{c['group_a']} / {c['group_b']}: {_show(c[measure], measure)} (95% interval "
            f"{_show(c['ci_low'], measure)} to {_show(c['ci_high'], measure)}, n={c['n']})"
            for c in cells
        )
    )
    if hidden:
        text += f"; {hidden} row(s) in combinations too small to show"
    shown_a = [g for g in order_a if any(c["group_a"] == g for c in cells)]
    shown_b = [g for g in order_b if any(c["group_b"] == g for c in cells)]
    figure = charts.grouped_bars(
        shown_a,
        shown_b,
        {(c["group_a"], c["group_b"]): c[measure] for c in cells},
        title=f"{ctx.target} {measure} by {a} and {b}",
        ylabel=f"{ctx.target} {measure}",
        legend=b,
        reference=overall,
        percent=measure == "rate",
    )
    return [
        ToolResult(
            title=f"{ctx.target} {measure} by {a} and {b}",
            data={
                "column_a": a,
                "column_b": b,
                "target": ctx.target,
                "measure": measure,
                "overall": overall,
                "groups": cells,
                "hidden_rows": hidden,
            },
            summary=text,
            figure=figure,
            min_group_n=min((c["n"] for c in cells), default=None),
        )
    ]


# -- the target by a ratio of two columns ----------------------------------------


class RatioParams(Params):
    numerator: str
    denominator: str
    bins: int = Field(default=5, ge=2, le=10, description="Ranges the ratio is split into.")
    cuts: list[float] | None = Field(
        default=None, description="Your own cut points for the ratio, e.g. [0.5, 1]."
    )
    outcome: str | None = Field(
        default=None, description="Another numeric or yes/no column to compare instead."
    )


@register("target_by_ratio", RatioParams, ("relationship", "segment", "drivers"))
def target_by_ratio(ctx: AnalysisContext, params: RatioParams) -> list[ToolResult]:
    """The target per group of one number divided by another, e.g. loan amount / income."""
    ctx = ctx.with_outcome(params.outcome)
    top_name, bottom_name = params.numerator, params.denominator
    if top_name == bottom_name:
        raise ToolError("a ratio needs two different columns")
    for column in (top_name, bottom_name):
        if ctx.feature(column) != "numeric":
            raise ToolError(f"{column} is not a number; a ratio needs two number columns")
    top = pd.to_numeric(ctx.frame[top_name]).astype(float)
    bottom = pd.to_numeric(ctx.frame[bottom_name]).astype(float)
    usable = top.notna() & bottom.notna() & (bottom != 0)
    left_out = int((~usable).sum())
    name = f"{top_name} / {bottom_name}"
    frame = ctx.frame.loc[usable].copy()
    frame[name] = top[usable] / bottom[usable]
    ratio_ctx = replace(ctx, frame=frame, kinds={**ctx.kinds, name: "numeric"})
    results = target_by_group(
        ratio_ctx, GroupParams(column=name, bins=params.bins, cuts=params.cuts)
    )
    for result in results:
        result.data = {
            **result.data,
            "ratio": name,
            "numerator": top_name,
            "denominator": bottom_name,
            "left_out": left_out,
        }
        if left_out:
            result.summary += (
                f"; {left_out} row(s) left out because {bottom_name} is missing or 0, or "
                f"{top_name} is missing"
            )
    return results


# -- the target over time ----------------------------------------------------------

PERIOD_CODES = {"year": "Y", "quarter": "Q", "month": "M"}


class TrendParams(Params):
    column: str
    period: Literal["year", "quarter", "month"] = "year"
    outcome: str | None = Field(
        default=None, description="Another numeric or yes/no column to follow instead."
    )


@register("trend_over_time", TrendParams, ("distribution", "relationship", "segment"))
def trend_over_time(ctx: AnalysisContext, params: TrendParams) -> list[ToolResult]:
    """The target's rate or mean per year, quarter or month; rows per period without a target."""
    ctx = ctx.with_outcome(params.outcome)
    column = params.column
    if ctx.require(column) != "date":
        raise ToolError(f"{column} is not a date column; trend_over_time needs one")
    if column in ctx.excluded:
        raise ToolError(f"{column} is excluded from the analysis and cannot be used")
    try:
        dates = pd.to_datetime(ctx.frame[column])
    except (TypeError, ValueError) as error:
        raise ToolError(f"{column} could not be read as dates") from error
    try:
        target, measure = _target(ctx)
    except ToolError:
        target, measure = None, "rows"

    keep = dates.notna() if target is None else dates.notna() & target.notna()
    periods = dates[keep].dt.to_period(PERIOD_CODES[params.period]).astype(str)
    if target is None:
        table = periods.value_counts().rename_axis("period").reset_index(name="n")
        table["total"] = table["squares"] = 0.0
    else:
        values = target[keep].astype(float)
        table = (
            pd.DataFrame({"period": periods, "value": values, "square": values**2})
            .groupby("period")
            .agg(n=("value", "size"), total=("value", "sum"), squares=("square", "sum"))
            .reset_index()
        )
    table = table.sort_values("period")
    small = table["n"] < ctx.min_group_n
    hidden = int(table.loc[small, "n"].sum())
    table = table[~small]

    rows = []
    for row in table.itertuples():
        n = int(row.n)
        entry = {"period": str(row.period), "n": n}
        if measure == "rate":
            events = int(round(row.total))
            low, high = wilson(events, n)
            entry |= {"events": events, "rate": row.total / n, "ci_low": low, "ci_high": high}
        elif measure == "mean":
            mean = float(row.total / n)
            low, high = mean_interval(mean, float(row.squares), n)
            entry |= {"mean": mean, "ci_low": low, "ci_high": high}
        rows.append(entry)

    if measure == "rows":
        head = f"Rows per {params.period} of {column}: "
        body = "; ".join(f"{r['period']}: {r['n']}" for r in rows)
        overall = None
    else:
        overall = float(target[keep].astype(float).mean())
        head = (
            f"{measure.title()} of {ctx.target} per {params.period} of {column} (overall "
            f"{_show(overall, measure)}): "
        )
        body = "; ".join(
            f"{r['period']}: {_show(r[measure], measure)} (95% interval "
            f"{_show(r['ci_low'], measure)} to {_show(r['ci_high'], measure)}, n={r['n']})"
            for r in rows
        )
    text = head + (body or "no period has enough rows to show")
    if hidden:
        text += f"; {hidden} row(s) in periods too small to show"
    key = "n" if measure == "rows" else measure
    figure = charts.trend(
        [r["period"] for r in rows],
        [r[key] for r in rows],
        lower=None if measure == "rows" else [r["ci_low"] for r in rows],
        upper=None if measure == "rows" else [r["ci_high"] for r in rows],
        title=(
            f"Rows per {params.period}"
            if measure == "rows"
            else f"{ctx.target} {measure} per {params.period} (95% interval)"
        ),
        ylabel="rows" if measure == "rows" else f"{ctx.target} {measure}",
        reference=overall,
        percent=measure == "rate",
    )
    return [
        ToolResult(
            title=f"{'Rows' if measure == 'rows' else f'{ctx.target} {measure}'} per "
            f"{params.period} of {column}",
            data={
                "column": column,
                "period": params.period,
                "target": ctx.target,
                "measure": measure,
                "overall": overall,
                "periods": rows,
                "hidden_rows": hidden,
                "missing_dates": int(ctx.frame[column].isna().sum()),
            },
            summary=text,
            figure=figure,
            min_group_n=min((r["n"] for r in rows), default=None),
        )
    ]


# -- compare a number across the target --------------------------------------


class CompareParams(Params):
    column: str


@register("compare_by_target", CompareParams, ("relationship", "drivers"))
def compare_by_target(ctx: AnalysisContext, params: CompareParams) -> list[ToolResult]:
    """How a numeric column differs between the two groups of a yes/no target."""
    column = params.column
    if ctx.feature(column) != "numeric":
        raise ToolError(f"{column} is not numeric; use target_by_group for categories")
    if ctx.target_type() != "binary":
        raise ToolError(
            f"compare_by_target needs a yes/no target; {ctx.target} is a number: "
            "use correlation or linear_regression"
        )
    target = ctx.binary_target()
    numbers = pd.to_numeric(ctx.frame[column]).astype(float)
    keep = target.notna() & numbers.notna()
    classes = []
    box_stats = []
    for value in (0, 1):
        group = numbers[keep & (target == value)]
        if len(group) < ctx.min_group_n:
            raise ToolError(f"fewer than {ctx.min_group_n} rows with {ctx.target} = {value}")
        q = group.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        classes.append(
            {
                "target": value,
                "n": int(len(group)),
                "mean": float(group.mean()),
                "std": float(group.std()),
                "p5": float(q[0.05]),
                "p25": float(q[0.25]),
                "median": float(q[0.5]),
                "p75": float(q[0.75]),
                "p95": float(q[0.95]),
            }
        )
        box_stats.append(
            {
                "label": f"{ctx.target}={value}",
                "whislo": q[0.05],
                "q1": q[0.25],
                "med": q[0.5],
                "q3": q[0.75],
                "whishi": q[0.95],
            }
        )
    pooled = math.sqrt((classes[0]["std"] ** 2 + classes[1]["std"] ** 2) / 2) or 1.0
    effect = (classes[1]["mean"] - classes[0]["mean"]) / pooled
    text = (
        f"{column} by {ctx.target}: mean {fmt(classes[0]['mean'])} when 0 (n={classes[0]['n']}) "
        f"vs {fmt(classes[1]['mean'])} when 1 (n={classes[1]['n']}); medians "
        f"{fmt(classes[0]['median'])} vs {fmt(classes[1]['median'])}; standardised "
        f"difference {effect:+.2f}"
    )
    figure = charts.boxes(
        box_stats, title=f"{column} by {ctx.target} (5th-95th percentile)", ylabel=column
    )
    return [
        ToolResult(
            title=f"{column} by {ctx.target}",
            data={"column": column, "classes": classes, "standardised_difference": effect},
            summary=text,
            figure=figure,
            min_group_n=min(c["n"] for c in classes),
        )
    ]


# -- correlation --------------------------------------------------------------


class CorrelationParams(Params):
    columns: list[str] | None = Field(default=None, description="Numeric columns; default all.")
    method: Literal["spearman", "pearson"] = "spearman"


@register("correlation", CorrelationParams, ("relationship", "drivers"))
def correlation(ctx: AnalysisContext, params: CorrelationParams) -> list[ToolResult]:
    """Pairwise correlation of numeric columns, including the target when yes/no or numeric."""
    columns = [c for c in ctx.features(params.columns) if ctx.kinds[c] in ("numeric", "boolean")]
    frame = ctx.frame[columns].apply(pd.to_numeric).astype(float)
    if ctx.target:
        try:
            target, _ = _target(ctx)
        except ToolError:
            pass  # a target that is neither yes/no nor a number has no correlation
        else:
            frame[ctx.target] = target.astype(float)
    if frame.shape[1] < 2:
        raise ToolError("correlation needs at least two numeric columns")
    matrix = frame.corr(method=params.method)
    labels = list(matrix.columns)
    pairs = sorted(
        (
            {"a": a, "b": b, "r": float(matrix.loc[a, b])}
            for i, a in enumerate(labels)
            for b in labels[i + 1 :]
            if pd.notna(matrix.loc[a, b])
        ),
        key=lambda pair: -abs(pair["r"]),
    )
    text = f"{params.method} correlations, strongest first: " + "; ".join(
        f"{p['a']}~{p['b']} {p['r']:+.2f}" for p in pairs[:6]
    )
    figure = charts.heatmap(
        labels, matrix.fillna(0).values.tolist(), title=f"{params.method.title()} correlation"
    )
    return [
        ToolResult(
            title=f"{params.method.title()} correlation",
            data={"method": params.method, "pairs": pairs, "n": int(len(frame))},
            summary=text,
            figure=figure,
        )
    ]


# -- missingness ----------------------------------------------------------------


class MissingnessParams(Params):
    pass


@register("missingness", MissingnessParams, ("data_quality", "relationship"))
def missingness(ctx: AnalysisContext, params: MissingnessParams) -> list[ToolResult]:
    """Missing values per column, and whether the target differs where a value is missing."""
    try:
        target, measure = _target(ctx)
    except ToolError:
        target, measure = None, None
    rows = []
    for column in ctx.kinds:
        missing = ctx.frame[column].isna()
        count = int(missing.sum())
        row = {"column": column, "missing": count, "share": float(missing.mean())}
        if target is not None and column != ctx.target and count:
            known = target.notna()
            when_missing = target[known & missing]
            when_present = target[known & ~missing]
            if len(when_missing) >= ctx.min_group_n and len(when_present) >= ctx.min_group_n:
                row["target_when_missing"] = float(when_missing.mean())
                row["target_when_present"] = float(when_present.mean())
        rows.append(row)
    rows.sort(key=lambda r: -r["missing"])
    with_gaps = [r for r in rows if r["missing"]]
    text = "Missing values: " + (
        "; ".join(
            f"{r['column']} {r['missing']} ({r['share']:.1%})"
            + (
                f", {ctx.target} {measure} {_show(r['target_when_missing'], measure)} when "
                f"missing vs {_show(r['target_when_present'], measure)} when present"
                if "target_when_missing" in r
                else ""
            )
            for r in with_gaps
        )
        or "none"
    )
    return [
        ToolResult(
            title="Missing values",
            data={
                "columns": rows,
                "rows": int(len(ctx.frame)),
                "target": ctx.target,
                "target_measure": measure,
            },
            summary=text,
        )
    ]
