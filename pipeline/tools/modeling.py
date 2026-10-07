"""Model tools: a baseline, an interpretable model and a strong one, all scored the same way.

Every model is scored with the same stratified 5-fold cross-validation, so its
numbers can be compared with the baseline's: a model that does not beat
'always predict the most common outcome' has found nothing. Each result reports
the spread across folds and the gap between training and test scores, and
raises flags the gate reads: a suspiciously good model (often a column that
gives the answer away) and a large train-test gap (overfitting).

Models never change the data. Missing values are handled inside each training
fold (median plus a 'was missing' marker for numbers, a 'missing' category for
categories; gradient boosting handles them natively), and that handling is
recorded with the result. Gradient boosting uses small trees and early stopping
by default, because with the library defaults it memorises a few hundred rows
(training AUC 1.0) and scores worse on the test folds (DECISIONS D032).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pipeline.contracts.common import Flag
from pipeline.tools import charts
from pipeline.tools.registry import AnalysisContext, Params, ToolError, ToolResult, register

FOLDS = 5
SUSPICIOUS_AUC = 0.95
"""Spec Gate 4: performance this high triggers a leakage investigation."""
MAX_TRAIN_TEST_GAP = 0.10
"""Spec Gate 4: a larger gap between training and test AUC is flagged as overfitting."""


def _data(
    ctx: AnalysisContext, features: list[str], missing: str
) -> tuple[pd.DataFrame, pd.Series, str]:
    """Features and 0/1 target for a model, plus a sentence on how missing values were handled."""
    target = ctx.binary_target()
    frame = ctx.frame[features].copy()
    keep = target.notna()
    note = "rows with a missing target are left out"
    if missing == "drop_rows":
        keep &= frame.notna().all(axis=1)
        note = "rows with any missing feature or target are left out"
    else:
        note += "; missing numbers get the training fold's median plus a 'was missing' marker"
    frame, y = frame[keep], target[keep].astype(int)
    if y.nunique() < 2:
        raise ToolError("the target has only one value in the usable rows")
    if int(y.value_counts().min()) < 2 * FOLDS:
        raise ToolError(
            f"only {int(y.value_counts().min())} rows in the smaller target group; "
            f"{2 * FOLDS} are needed for {FOLDS}-fold cross-validation"
        )
    return frame, y, note


def _numeric(ctx: AnalysisContext, features: list[str]) -> list[str]:
    return [f for f in features if ctx.kinds[f] in ("numeric", "boolean")]


def _categorical(ctx: AnalysisContext, features: list[str]) -> list[str]:
    return [f for f in features if ctx.kinds[f] == "categorical"]


def _cross_validate(make_model, frame: pd.DataFrame, y: pd.Series, seed: int):
    """Train and test on every fold; return per-fold test and train scores and fitted models."""
    folds = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    results = []
    for train, test in folds.split(frame, y):
        model = make_model()
        model.fit(frame.iloc[train], y.iloc[train])
        test_p = model.predict_proba(frame.iloc[test])[:, 1]
        train_p = model.predict_proba(frame.iloc[train])[:, 1]
        results.append(
            {
                "model": model,
                "test": test,
                "auc": float(roc_auc_score(y.iloc[test], test_p)),
                "train_auc": float(roc_auc_score(y.iloc[train], train_p)),
                "accuracy": float(accuracy_score(y.iloc[test], test_p >= 0.5)),
            }
        )
    return results


def _metrics(folds: list[dict], y: pd.Series, note: str, model: str) -> dict:
    auc = [f["auc"] for f in folds]
    train = [f["train_auc"] for f in folds]
    return {
        "model": model,
        "auc": float(np.mean(auc)),
        "auc_std": float(np.std(auc)),
        "fold_auc": auc,
        "accuracy": float(np.mean([f["accuracy"] for f in folds])),
        "train_auc": float(np.mean(train)),
        "train_test_gap": float(np.mean(train) - np.mean(auc)),
        "rows": int(len(y)),
        "events": int(y.sum()),
        "folds": FOLDS,
        "missing_values": note,
    }


def _flags(metrics: dict) -> list[Flag]:
    flags = []
    if metrics["auc"] > SUSPICIOUS_AUC:
        flags.append(
            Flag(
                code="suspiciously_good_model",
                severity="warn",
                message=f"AUC {metrics['auc']:.3f} is suspiciously high: check for a column that "
                f"is recorded after the outcome or gives the answer away.",
                stage="analysis",
            )
        )
    if metrics["train_test_gap"] > MAX_TRAIN_TEST_GAP:
        flags.append(
            Flag(
                code="overfitting",
                severity="warn",
                message=f"Training AUC is {metrics['train_test_gap']:.2f} above test AUC: the "
                f"model fits noise in the training data.",
                stage="analysis",
            )
        )
    return flags


def _metrics_text(metrics: dict) -> str:
    return (
        f"{metrics['model']}: AUC {metrics['auc']:.3f} (+/- {metrics['auc_std']:.3f} across "
        f"{metrics['folds']} folds), accuracy {metrics['accuracy']:.1%}, train-test gap "
        f"{metrics['train_test_gap']:+.3f}, {metrics['rows']} rows, {metrics['events']} events"
    )


# -- baseline ---------------------------------------------------------------


class BaselineParams(Params):
    pass


@register("baseline_model", BaselineParams, ("prediction",))
def baseline_model(ctx: AnalysisContext, params: BaselineParams) -> list[ToolResult]:
    """Always predict the most common outcome: the score any real model must beat."""
    features = ctx.features(None)
    frame, y, note = _data(ctx, features, "indicator")
    folds = _cross_validate(lambda: DummyClassifier(strategy="most_frequent"), frame, y, ctx.seed)
    metrics = _metrics(folds, y, note, "baseline (most common outcome)")
    return [
        ToolResult(
            title="Baseline model",
            kind="metric",
            data={"metrics": metrics},
            summary=_metrics_text(metrics),
        )
    ]


# -- logistic regression ----------------------------------------------------


class LogisticParams(Params):
    features: list[str] | None = Field(default=None, description="Default: every usable column.")
    missing: Literal["indicator", "drop_rows"] = Field(
        default="indicator", description="How missing feature values are handled in the model."
    )
    regularisation: float = Field(default=1.0, gt=0, description="Inverse strength (C).")


def _logistic(ctx: AnalysisContext, features: list[str], c: float) -> Pipeline:
    numeric, categorical = _numeric(ctx, features), _categorical(ctx, features)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="constant", fill_value="(missing)"),
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist", min_frequency=ctx.min_group_n
                    ),
                ),
                categorical,
            )
        )
    return Pipeline(
        [
            ("prepare", ColumnTransformer(transformers)),
            ("model", LogisticRegression(C=c, max_iter=2000)),
        ]
    )


def _as_object(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = (
            frame[column].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        )
    return frame


@register("logistic_regression", LogisticParams, ("drivers", "prediction"))
def logistic_regression(ctx: AnalysisContext, params: LogisticParams) -> list[ToolResult]:
    """An interpretable model: how much each feature moves the odds of the target."""
    features = ctx.features(params.features)
    frame, y, note = _data(ctx, features, params.missing)
    frame = _as_object(frame, _categorical(ctx, features))
    for column in _numeric(ctx, features):
        frame[column] = pd.to_numeric(frame[column]).astype(float)

    folds = _cross_validate(
        lambda: _logistic(ctx, features, params.regularisation), frame, y, ctx.seed
    )
    metrics = _metrics(folds, y, note, "logistic regression")

    full = _logistic(ctx, features, params.regularisation).fit(frame, y)
    names = full.named_steps["prepare"].get_feature_names_out()
    coefficients = full.named_steps["model"].coef_[0]
    rows = []
    for name, value in zip(names, coefficients, strict=True):
        part, _, rest = name.partition("__")
        if part == "num":
            readable = rest.replace("missingindicator_", "") + (
                " (was missing)" if rest.startswith("missingindicator_") else ""
            )
            unit = (
                "per standard deviation"
                if not rest.startswith("missingindicator_")
                else "if missing"
            )
        else:
            column, _, level = rest.partition("_")
            readable = f"{column} = {level}"
            unit = "vs other values"
        rows.append(
            {
                "feature": readable,
                "coefficient": float(value),
                "odds_ratio": float(np.exp(value)),
                "unit": unit,
            }
        )
    rows.sort(key=lambda r: -abs(r["coefficient"]))
    text = "Odds ratios (1 = no effect): " + "; ".join(
        f"{r['feature']} x{r['odds_ratio']:.2f} {r['unit']}" for r in rows[:8]
    )
    figure = charts.horizontal_bars(
        [r["feature"] for r in rows[:15]],
        [r["coefficient"] for r in rows[:15]],
        title="Logistic regression: effect on the log-odds",
        xlabel="coefficient (log-odds)",
    )
    return [
        ToolResult(
            title="Logistic regression: performance",
            kind="metric",
            data={"metrics": metrics},
            summary=_metrics_text(metrics),
            flags=_flags(metrics),
        ),
        ToolResult(
            title="Logistic regression: effects",
            data={"coefficients": rows},
            summary=text,
            figure=figure,
        ),
    ]


# -- gradient boosting ------------------------------------------------------


class BoostingParams(Params):
    features: list[str] | None = Field(default=None, description="Default: every usable column.")
    max_iter: int = Field(
        default=300,
        ge=10,
        le=1000,
        description="Most boosting rounds; training stops sooner when a held-out tenth of "
        "the training fold stops improving.",
    )
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    max_leaf_nodes: int = Field(
        default=8, ge=2, le=64, description="Size of each tree; small trees overfit less."
    )


@register("gradient_boosting", BoostingParams, ("drivers", "prediction"))
def gradient_boosting(ctx: AnalysisContext, params: BoostingParams) -> list[ToolResult]:
    """A strong model, and which features it relies on (permutation importance on test folds)."""
    features = ctx.features(params.features)
    frame, y, _ = _data(ctx, features, "indicator")
    note = "rows with a missing target are left out; the model handles missing features itself"
    frame = frame.copy()
    for column in _numeric(ctx, features):
        frame[column] = pd.to_numeric(frame[column]).astype(float)
    for column in _categorical(ctx, features):
        frame[column] = (
            frame[column].map(lambda v: str(v) if pd.notna(v) else np.nan).astype("category")
        )

    def make():
        # Small, regularised trees that stop early on a tenth of the training fold: the
        # test fold is never used to decide when to stop.
        return HistGradientBoostingClassifier(
            max_iter=params.max_iter,
            learning_rate=params.learning_rate,
            max_leaf_nodes=params.max_leaf_nodes,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            categorical_features="from_dtype",
            random_state=ctx.seed,
        )

    folds = _cross_validate(make, frame, y, ctx.seed)
    metrics = _metrics(folds, y, note, "gradient boosting")
    metrics["rounds_used"] = [int(fold["model"].n_iter_) for fold in folds]

    drops = {feature: [] for feature in features}
    for fold in folds:
        test = fold["test"]
        result = permutation_importance(
            fold["model"],
            frame.iloc[test],
            y.iloc[test],
            scoring="roc_auc",
            n_repeats=5,
            random_state=ctx.seed,
        )
        for feature, value in zip(features, result.importances_mean, strict=True):
            drops[feature].append(float(value))
    importance = sorted(
        (
            {"feature": f, "importance": float(np.mean(v)), "std": float(np.std(v))}
            for f, v in drops.items()
        ),
        key=lambda r: -r["importance"],
    )
    text = "Permutation importance (drop in AUC when a feature is shuffled): " + "; ".join(
        f"{r['feature']} {r['importance']:.3f}" for r in importance[:8]
    )
    figure = charts.horizontal_bars(
        [r["feature"] for r in importance],
        [r["importance"] for r in importance],
        title="Gradient boosting: what the model relies on",
        xlabel="drop in AUC when shuffled",
    )
    rounds = metrics["rounds_used"]
    return [
        ToolResult(
            title="Gradient boosting: performance",
            kind="metric",
            data={"metrics": metrics},
            summary=_metrics_text(metrics)
            + f", stopped after {min(rounds)}-{max(rounds)} of {params.max_iter} rounds",
            flags=_flags(metrics),
        ),
        ToolResult(
            title="Gradient boosting: feature importance",
            data={"importance": importance},
            summary=text,
            figure=figure,
        ),
    ]
