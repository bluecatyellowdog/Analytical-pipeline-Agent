"""Regression tools: linear regression for a numeric target, logistic regression for a yes/no one.

Classic regressions, fitted once on all usable rows (statsmodels), with 95%
intervals and p-values and no cross-validation: the aim is to describe how the
target moves with each feature while the others are held fixed, not to build a
prediction model (DECISIONS D033).

To keep effects readable and comparable, a number is measured in a round step
close to its standard deviation ('per 10', 'per 100,000'), a yes/no column as
'yes vs no', and a category against its most common value. Rows with a missing
target or number are left out and counted; a missing category is its own value;
values with fewer than MIN_GROUP_N rows are pooled.

Flags the gate reads:
- suspiciously_good_model: R² or AUC above 0.95, or a perfect prediction, which
  usually means a column gives the answer away
- few_rows_per_effect: fewer than 10 rows (logistic: rows with the rarer outcome)
  per estimated effect, so the estimates are unstable
- did_not_converge: the logistic fit did not settle; very wide intervals are unreliable

Models never change the data.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import Field
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationWarning,
)

from pipeline.contracts.common import Flag
from pipeline.tools import charts
from pipeline.tools.eda import MISSING, fmt
from pipeline.tools.registry import (
    AnalysisContext,
    Params,
    ToolError,
    ToolResult,
    as_yes_no,
    register,
)

SUSPICIOUS_FIT = 0.95
"""Spec Gate 4: a fit this good (R² or AUC) triggers a leakage investigation."""
ROWS_PER_EFFECT = 10
"""Rule of thumb: with fewer rows per estimated effect the estimates are unstable."""
OTHER = "(other)"


@dataclass
class _Design:
    """What a regression is fitted on: complete rows, numbers in steps, categories as 0/1."""

    x: pd.DataFrame
    """One column per effect, plus the constant."""
    y: pd.Series
    terms: dict[str, dict]
    """Effect name -> the column it comes from and its unit ('per 10', 'vs Hovedstaden')."""
    left_out: dict[str, int]
    notes: list[str] = field(default_factory=list)
    smallest_group: int | None = None


def _step(sd: float) -> float:
    """A round number near one standard deviation, so effects 'per step' are comparable."""
    if not sd or not math.isfinite(sd):
        return 1.0
    exponent = math.floor(math.log10(sd))
    candidates = [m * 10.0**e for e in (exponent, exponent + 1) for m in (1, 2, 5)]
    return min(candidates, key=lambda c: abs(math.log10(c / sd)))


def _design(ctx: AnalysisContext, features: list[str], target: pd.Series) -> _Design:
    keep = target.notna()
    left_out: dict[str, int] = {}
    if int((~keep).sum()):
        left_out[f"missing {ctx.target}"] = int((~keep).sum())

    numbers: dict[str, pd.Series] = {}
    for feature in features:
        if ctx.kinds[feature] == "boolean":
            values = as_yes_no(ctx.frame[feature])
            if values is None:
                raise ToolError(f"{feature} does not read as yes/no")
            numbers[feature] = values.astype(float)
        elif ctx.kinds[feature] == "numeric":
            numbers[feature] = pd.to_numeric(ctx.frame[feature]).astype(float)
    for feature, values in numbers.items():
        missing = int((keep & values.isna()).sum())
        if missing:
            left_out[f"missing {feature}"] = missing
    for values in numbers.values():
        keep &= values.notna()

    columns: dict[str, pd.Series] = {}
    terms: dict[str, dict] = {}
    notes: list[str] = []
    groups: list[int] = []
    for feature, values in numbers.items():
        values = values[keep]
        if values.nunique() < 2:
            raise ToolError(f"{feature} has the same value in every row used; it explains nothing")
        if ctx.kinds[feature] == "boolean":
            columns[feature] = values
            terms[feature] = {"column": feature, "per": "yes vs no"}
            yes = int(values.sum())
            groups += [yes, len(values) - yes]
        else:
            step = _step(float(values.std()))
            columns[feature] = values / step
            terms[feature] = {"column": feature, "per": f"per {fmt(step)}", "step": step}

    for feature in (f for f in features if ctx.kinds[f] == "categorical"):
        labels = ctx.frame[feature][keep].map(lambda v: str(v) if pd.notna(v) else MISSING)
        counts = labels.value_counts()
        rare = list(counts[counts < ctx.min_group_n].index)
        if len(rare) == len(counts):
            raise ToolError(f"no value of {feature} has {ctx.min_group_n} or more rows")
        if rare:
            pooled = int(counts[rare].sum())
            into = OTHER if pooled >= ctx.min_group_n else str(counts.index[0])
            labels = labels.where(~labels.isin(rare), into)
            notes.append(f"{feature}: {len(rare)} rare value(s), {pooled} rows, counted as {into}")
            counts = labels.value_counts()
        if len(counts) < 2:
            raise ToolError(f"{feature} has the same value in every row used; it explains nothing")
        reference = str(counts.index[0])
        groups += [int(c) for c in counts]
        for level in counts.index[1:]:
            name = f"{feature} = {level}"
            columns[name] = (labels == level).astype(float)
            terms[name] = {"column": feature, "per": f"vs {reference}"}

    x = sm.add_constant(pd.DataFrame(columns), has_constant="add")
    used = int(keep.sum())
    if used < max(ctx.min_group_n, x.shape[1] + 1):
        raise ToolError(f"only {used} usable rows for {len(terms)} effects")
    if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
        raise ToolError(
            "some features carry the same information (for example one column is a multiple "
            "of another, or two categories always go together); leave one of them out"
        )
    return _Design(
        x=x,
        y=target[keep].astype(float),
        terms=terms,
        left_out=left_out,
        notes=notes,
        smallest_group=min(groups, default=None),
    )


def _signed(value: float) -> str:
    return ("+" if value >= 0 else "-") + fmt(abs(value))


def _p(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "p n/a"
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def _finite(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _flag(code: str, message: str) -> Flag:
    return Flag(code=code, severity="warn", message=message, stage="analysis")


def _left_out_text(design: _Design) -> str:
    if not design.left_out:
        return "none left out"
    reasons = ", ".join(f"{reason}: {count}" for reason, count in design.left_out.items())
    return f"rows left out ({reasons})"


def auc(y: pd.Series, score) -> float:
    """The chance that a random row with outcome 1 scores above one with outcome 0."""
    ranks = pd.Series(np.asarray(score, dtype=float)).rank().to_numpy()
    positive = np.asarray(y) == 1
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _few_rows(rows: int, effects: int, what: str) -> Flag | None:
    if rows >= ROWS_PER_EFFECT * effects:
        return None
    return _flag(
        "few_rows_per_effect",
        f"Only {rows} {what} for {effects} effects; with fewer than {ROWS_PER_EFFECT} per "
        f"effect the estimates are unstable. Use fewer features.",
    )


# -- linear regression ------------------------------------------------------


class LinearParams(Params):
    features: list[str] | None = Field(default=None, description="Default: every usable column.")


@register("linear_regression", LinearParams, ("drivers", "relationship"))
def linear_regression(ctx: AnalysisContext, params: LinearParams) -> list[ToolResult]:
    """How a numeric target moves with each feature, the others held fixed (95% intervals)."""
    if ctx.target_type() != "numeric":
        raise ToolError(
            f"linear_regression needs a numeric target; {ctx.target} is yes/no: "
            "use logistic_regression"
        )
    features = ctx.features(params.features)
    design = _design(ctx, features, ctx.numeric_target())
    fit = sm.OLS(design.y, design.x).fit()
    intervals = fit.conf_int(0.05)
    rows = sorted(
        (
            {
                "feature": name,
                "column": term["column"],
                "per": term["per"],
                "effect": float(fit.params[name]),
                "ci_low": float(intervals.loc[name, 0]),
                "ci_high": float(intervals.loc[name, 1]),
                "p_value": _finite(fit.pvalues[name]),
            }
            for name, term in design.terms.items()
        ),
        key=lambda r: r["p_value"] if r["p_value"] is not None else 1.0,
    )
    metrics = {
        "model": "linear regression",
        "target": ctx.target,
        "features": features,
        "rows_used": len(design.y),
        "rows_left_out": design.left_out,
        "effects": len(rows),
        "r_squared": float(fit.rsquared),
        "adjusted_r_squared": float(fit.rsquared_adj),
        "f_test_p": _finite(fit.f_pvalue),
        "residual_std": float(np.sqrt(fit.scale)),
        "intercept": float(fit.params["const"]),
        "notes": design.notes,
    }
    flags = []
    if metrics["r_squared"] > SUSPICIOUS_FIT:
        flags.append(
            _flag(
                "suspiciously_good_model",
                f"R² {metrics['r_squared']:.3f} is suspiciously high: check for a column that is "
                f"recorded after {ctx.target} or is {ctx.target} in another form.",
            )
        )
    if few := _few_rows(len(design.y), len(rows), "rows"):
        flags.append(few)

    fit_text = (
        f"linear regression of {ctx.target} on {', '.join(features)}: {len(design.y)} rows, "
        f"{_left_out_text(design)}; R² {metrics['r_squared']:.2f} (adjusted "
        f"{metrics['adjusted_r_squared']:.2f}); model vs no model (F-test) "
        f"{_p(metrics['f_test_p'])}; typical error {fmt(metrics['residual_std'])}"
    )
    effects_text = f"Change in {ctx.target} (95% interval): " + "; ".join(
        f"{r['feature']} {_signed(r['effect'])} {r['per']} "
        f"({_signed(r['ci_low'])} to {_signed(r['ci_high'])}, {_p(r['p_value'])})"
        for r in rows[:8]
    )
    figure = charts.forest(
        [f"{r['feature']} ({r['per']})" for r in rows[:15]],
        [r["effect"] for r in rows[:15]],
        [r["ci_low"] for r in rows[:15]],
        [r["ci_high"] for r in rows[:15]],
        title=f"Linear regression: change in {ctx.target} (95% interval)",
        xlabel=f"change in {ctx.target}",
        reference=0.0,
    )
    return [
        ToolResult(
            title="Linear regression: fit",
            kind="metric",
            data={"metrics": metrics},
            summary=fit_text,
            flags=flags,
        ),
        ToolResult(
            title="Linear regression: effects",
            data={"target": ctx.target, "effects": rows},
            summary=effects_text,
            figure=figure,
            min_group_n=design.smallest_group,
        ),
    ]


# -- logistic regression ----------------------------------------------------


class LogisticParams(Params):
    features: list[str] | None = Field(default=None, description="Default: every usable column.")


def _separating(design: _Design, y: pd.Series) -> list[str]:
    """Columns that on their own split the outcome perfectly."""
    by_column: dict[str, list[str]] = {}
    for name, term in design.terms.items():
        by_column.setdefault(term["column"], []).append(name)
    culprits = []
    for column, names in by_column.items():
        if design.terms[names[0]]["per"].startswith("vs "):
            key = design.x[names].astype(int).astype(str).agg("".join, axis=1)
            if y.groupby(key).transform("mean").isin([0.0, 1.0]).all():
                culprits.append(column)
        else:
            separation = auc(y, design.x[names[0]])
            if max(separation, 1 - separation) > 0.999:
                culprits.append(column)
    return culprits


@register("logistic_regression", LogisticParams, ("drivers", "relationship"))
def logistic_regression(ctx: AnalysisContext, params: LogisticParams) -> list[ToolResult]:
    """How the odds of a yes/no target move with each feature, others held fixed (95% intervals)."""
    if ctx.target_type() != "binary":
        raise ToolError(
            f"logistic_regression needs a yes/no target; {ctx.target} is a number: "
            "use linear_regression"
        )
    features = ctx.features(params.features)
    design = _design(ctx, features, ctx.binary_target())
    y = design.y.astype(int)
    events = int(y.sum())
    rarer = min(events, len(y) - events)
    if rarer < ctx.min_group_n:
        raise ToolError(
            f"only {rarer} rows with the rarer outcome of {ctx.target}; "
            f"at least {ctx.min_group_n} are needed"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            fit = sm.Logit(y, design.x).fit(disp=0, maxiter=200)
        except np.linalg.LinAlgError as error:
            raise ToolError(
                "the model could not be estimated: some features carry nearly the same "
                "information, or a category has only one outcome; leave one of them out"
            ) from error
    separated = any(issubclass(w.category, PerfectSeparationWarning) for w in caught)
    unsettled = any(
        issubclass(w.category, (ConvergenceWarning, HessianInversionWarning)) for w in caught
    )
    converged = bool(fit.mle_retvals.get("converged", False)) and not unsettled

    metrics = {
        "model": "logistic regression",
        "target": ctx.target,
        "features": features,
        "rows_used": len(y),
        "events": events,
        "rows_left_out": design.left_out,
        "effects": len(design.terms),
        "pseudo_r_squared": _finite(fit.prsquared),
        "lr_test_p": _finite(fit.llr_pvalue),
        "auc": auc(y, fit.predict(design.x)),
        "converged": converged,
        "notes": design.notes,
    }
    fit_text = (
        f"logistic regression of {ctx.target} on {', '.join(features)}: {len(y)} rows "
        f"({events} with {ctx.target} = 1), {_left_out_text(design)}"
    )

    if separated:
        culprits = _separating(design, y)
        message = (
            f"{ctx.target} is predicted perfectly"
            + (f" by {', '.join(culprits)}" if culprits else "")
            + ", so no effects can be estimated. This usually means a column is recorded after "
            f"the outcome or is {ctx.target} in another form; check it and leave it out."
        )
        metrics["separating_columns"] = culprits
        return [
            ToolResult(
                title="Logistic regression: fit",
                kind="metric",
                data={"metrics": metrics},
                summary=f"{fit_text}. {message}",
                flags=[_flag("suspiciously_good_model", message)],
            )
        ]

    intervals = fit.conf_int(0.05)
    with np.errstate(over="ignore"):
        rows = sorted(
            (
                {
                    "feature": name,
                    "column": term["column"],
                    "per": term["per"],
                    "odds_ratio": float(np.exp(fit.params[name])),
                    "ci_low": float(np.exp(intervals.loc[name, 0])),
                    "ci_high": float(np.exp(intervals.loc[name, 1])),
                    "p_value": _finite(fit.pvalues[name]),
                    "coefficient": float(fit.params[name]),
                }
                for name, term in design.terms.items()
            ),
            key=lambda r: r["p_value"] if r["p_value"] is not None else 1.0,
        )
    flags = []
    if metrics["auc"] > SUSPICIOUS_FIT:
        flags.append(
            _flag(
                "suspiciously_good_model",
                f"AUC {metrics['auc']:.3f} is suspiciously high: check for a column that is "
                f"recorded after {ctx.target} or is {ctx.target} in another form.",
            )
        )
    if few := _few_rows(rarer, len(rows), "rows with the rarer outcome"):
        flags.append(few)
    if not converged:
        flags.append(
            _flag(
                "did_not_converge",
                "The model did not settle on stable estimates, often because a category has "
                "only one outcome. Effects with very wide intervals are unreliable.",
            )
        )

    fit_text += (
        f"; model vs no model (likelihood ratio) {_p(metrics['lr_test_p'])}; pseudo R² "
        f"{metrics['pseudo_r_squared'] or 0:.2f}; AUC {metrics['auc']:.2f} on the same rows"
    )
    effects_text = "Odds ratios (1 = no effect; 95% interval): " + "; ".join(
        f"{r['feature']} x{r['odds_ratio']:.2f} {r['per']} "
        f"({r['ci_low']:.2f}-{r['ci_high']:.2f}, {_p(r['p_value'])})"
        for r in rows[:8]
    )
    figure = charts.forest(
        [f"{r['feature']} ({r['per']})" for r in rows[:15]],
        [r["odds_ratio"] for r in rows[:15]],
        [r["ci_low"] for r in rows[:15]],
        [r["ci_high"] for r in rows[:15]],
        title=f"Logistic regression: odds of {ctx.target} (95% interval)",
        xlabel="odds ratio (log scale)",
        reference=1.0,
        log=True,
    )
    return [
        ToolResult(
            title="Logistic regression: fit",
            kind="metric",
            data={"metrics": metrics},
            summary=fit_text,
            flags=flags,
        ),
        ToolResult(
            title="Logistic regression: odds ratios",
            data={"target": ctx.target, "effects": rows},
            summary=effects_text,
            figure=figure,
            min_group_n=design.smallest_group,
        ),
    ]
