"""Gate 3: is the analysis plan executable, safe and feasible on this data?

Spec Gate 3, checked on the plan and the clean data (the analysis context), never
on the planner's word:
- invariant: every column exists; every method is a registry tool; the target is
  the brief's and the task type fits it; no identifier, personal data, free text
  or excluded column is analysed; every excluded column is listed with its reason;
  every method can run on this target
- judgment: a column that on its own predicts the target almost perfectly is
  excluded (leakage); each question has enough rows or is marked infeasible (10
  rows with the rarer outcome per regression feature, two segments of 30 rows);
  at least one question can be answered; and, as a note that never flags, whether
  each method suits its question's kind

For the AI's plan every failure blocks and goes back to the planner to fix. A plan
the user edited keeps their judgment: leakage and feasibility only warn.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from pipeline.checks.gates import build_gate, checked
from pipeline.contracts.audit import GateResult
from pipeline.contracts.common import Severity
from pipeline.contracts.context_brief import ContextBrief
from pipeline.contracts.plan import AnalysisPlan, PlanQuestion
from pipeline.tools import REGISTRY
from pipeline.tools.context import task_type_of
from pipeline.tools.modeling import ROWS_PER_EFFECT, auc
from pipeline.tools.registry import FEATURE_KINDS, AnalysisContext, ToolError, as_yes_no

LEAKAGE_STRENGTH = 0.95
"""A column this strongly tied to the target on its own probably gives the answer away."""
MIN_SEGMENT_ROWS = 30
"""Spec Gate 3: a segment comparison needs groups of at least 30 rows."""
MIN_ROWS = 30
"""Fewer rows than this say nothing reliable about how two columns move together."""

NEVER_ANALYSED = {"id": "an identifier", "personal": "personal data", "text": "free text"}
TARGET_METHODS = {
    "logistic_regression": "binary",
    "compare_by_target": "binary",
    "linear_regression": "numeric",
    "target_by_group": "any",
    "target_by_two_groups": "any",
}
FEATURE_METHODS = (*TARGET_METHODS, "correlation")
REGRESSIONS = ("logistic_regression", "linear_regression")


def gate_3_plan(
    plan: AnalysisPlan,
    ctx: AnalysisContext,
    brief: ContextBrief,
    *,
    edited_by_user: bool = False,
) -> GateResult:
    """Every check, in one gate. `ctx` is the clean data as delivered, before the plan."""
    judged: Severity = "warn" if edited_by_user else "block"
    planned = replace(ctx, target=plan.target)
    target_kind = _target_kind(planned)
    feasible = [q for q in plan.questions if q.feasible]

    checks = [
        checked(
            "columns_exist",
            "invariant",
            "block",
            _unknown_columns(plan, ctx),
            "Every column exists.",
        ),
        checked(
            "methods_exist", "invariant", "block", _unknown_methods(plan), "Every method is a tool."
        ),
        checked(
            "target_as_briefed",
            "invariant",
            "block",
            _target_problems(plan, ctx, brief),
            "The target is the brief's and can be analysed.",
        ),
        checked(
            "task_type_fits_target",
            "invariant",
            "block",
            _task_type_problems(plan, planned),
            "The task type fits the target.",
        ),
        checked(
            "usable_columns",
            "invariant",
            "block",
            _unusable_columns(feasible, plan, ctx, brief),
            "No identifier, personal, free-text or excluded column is analysed.",
        ),
        checked(
            "exclusions_listed",
            "invariant",
            "block",
            _unlisted_exclusions(plan, ctx, brief),
            "Every excluded column is listed with its reason.",
        ),
        checked(
            "methods_fit_target",
            "invariant",
            "block",
            _method_target_problems(feasible, plan, planned, target_kind),
            "Every method can run on this target.",
        ),
        checked(
            "leaky_columns_excluded",
            "judgment",
            judged,
            _leaks_not_excluded(plan, planned),
            "No column that gives the answer away is used.",
        ),
        checked(
            "feasible",
            "judgment",
            judged,
            [p for q in feasible for p in _infeasible(q, planned, target_kind)],
            "Every question has enough rows, or is marked infeasible.",
        ),
        checked(
            "methods_fit_question",
            "judgment",
            "info",
            _method_kind_mismatches(feasible),
            "Every method suits its question.",
        ),
        checked(
            "something_to_do",
            "judgment",
            "warn",
            [] if feasible else ["no question can be answered with this data"],
            "At least one question can be answered.",
        ),
    ]
    return build_gate("gate_3_plan", checks)


def leak_suspects(ctx: AnalysisContext) -> dict[str, float]:
    """Usable columns that on their own predict the target almost perfectly: column -> strength.

    Strength is the AUC (or 1 - AUC) of the column against a yes/no target, the
    absolute rank correlation with a numeric one; categories are scored by their
    own target rate or mean, with values under the minimum group size pooled.
    """
    kind = _target_kind(ctx)
    if kind is None:
        return {}
    target = (ctx.binary_target() if kind == "binary" else ctx.numeric_target()).astype(float)
    suspects = {}
    for column, column_kind in ctx.kinds.items():
        if column == ctx.target or column in ctx.excluded or column_kind not in FEATURE_KINDS:
            continue
        strength = _strength(ctx, column, column_kind, target, binary=kind == "binary")
        if strength is not None and strength >= LEAKAGE_STRENGTH:
            suspects[column] = round(strength, 3)
    return suspects


# -- the checks ---------------------------------------------------------------


def _target_kind(ctx: AnalysisContext) -> str | None:
    """'binary' or 'numeric' as the tools see the target; None when they cannot use it."""
    if not ctx.target or ctx.target not in ctx.kinds:
        return None
    try:
        return ctx.target_type()
    except ToolError:
        return None


def _unknown_columns(plan: AnalysisPlan, ctx: AnalysisContext) -> list[str]:
    problems = [
        f"{q.id} uses column {column!r}, which is not in the clean data"
        for q in plan.questions
        for column in q.columns
        if column not in ctx.kinds
    ]
    if plan.target and plan.target not in ctx.kinds:
        problems.append(f"the target {plan.target!r} is not in the clean data")
    return problems


def _unknown_methods(plan: AnalysisPlan) -> list[str]:
    problems = [
        f"{q.id} uses method {method!r}, which is not a tool; the tools are {sorted(REGISTRY)}"
        for q in plan.questions
        for method in q.methods
        if method not in REGISTRY
    ]
    problems += [
        f"{q.id} names no method; give it at least one tool, or mark it infeasible"
        for q in plan.questions
        if q.feasible and not q.methods
    ]
    return problems


def _target_problems(plan: AnalysisPlan, ctx: AnalysisContext, brief: ContextBrief) -> list[str]:
    problems = []
    if brief.target and brief.target in ctx.kinds and plan.target != brief.target:
        problems.append(f"the brief names {brief.target!r} as the target, the plan {plan.target!r}")
    kind = ctx.kinds.get(plan.target) if plan.target else None
    if kind in NEVER_ANALYSED or kind == "date":
        problems.append(f"the target {plan.target!r} is a {kind} column and cannot be an outcome")
    if plan.target and plan.target in ctx.excluded | set(brief.excluded_variables):
        problems.append(f"the target {plan.target!r} is excluded from the analysis")
    return problems


def _task_type_problems(plan: AnalysisPlan, planned: AnalysisContext) -> list[str]:
    expected = task_type_of(planned)
    if plan.task_type == expected:
        return []
    return [
        f"task_type is {plan.task_type!r}, but the target {plan.target!r} makes it {expected!r}"
    ]


def _unusable_columns(
    questions: list[PlanQuestion], plan: AnalysisPlan, ctx: AnalysisContext, brief: ContextBrief
) -> list[str]:
    excluded = ctx.excluded | set(brief.excluded_variables) | plan.excluded_columns
    problems = []
    for q in questions:
        for column in q.columns:
            if column == plan.target or column not in ctx.kinds:
                continue
            kind = ctx.kinds[column]
            if kind in NEVER_ANALYSED:
                problems.append(
                    f"{q.id} uses {column}, which is {NEVER_ANALYSED[kind]} and is never analysed"
                )
            elif column in excluded:
                problems.append(f"{q.id} uses {column}, which is excluded from the analysis")
            elif kind == "date" and any(m in FEATURE_METHODS for m in q.methods):
                problems.append(
                    f"{q.id} uses the date {column} with a method that needs numbers or "
                    f"categories; describe dates with distribution or summary_table"
                )
    return problems


def _unlisted_exclusions(
    plan: AnalysisPlan, ctx: AnalysisContext, brief: ContextBrief
) -> list[str]:
    required = (ctx.excluded | set(brief.excluded_variables)) & set(ctx.kinds)
    return [
        f"{column} is excluded by the brief or the diagnosis; list it in excluded_features "
        f"with the reason"
        for column in sorted(required - plan.excluded_columns)
    ]


def _method_target_problems(
    questions: list[PlanQuestion],
    plan: AnalysisPlan,
    planned: AnalysisContext,
    target_kind: str | None,
) -> list[str]:
    problems = []
    for q in questions:
        for method in q.methods:
            needs = TARGET_METHODS.get(method)
            if needs is None or (method == "linear_regression" and _side_outcomes(q, planned)):
                continue
            if target_kind is None:
                problems.append(f"{q.id}: {method} needs a yes/no or numeric target")
            elif needs == "binary" and target_kind != "binary":
                problems.append(
                    f"{q.id}: {method} needs a yes/no target; {plan.target} is a number "
                    f"(use linear_regression or target_by_group)"
                )
            elif needs == "numeric" and target_kind != "numeric":
                problems.append(
                    f"{q.id}: {method} needs a numeric outcome: the target, or for a side "
                    f"question another number column in the question; {plan.target} is yes/no "
                    f"(use logistic_regression)"
                )
    return problems


def _side_outcomes(q: PlanQuestion, ctx: AnalysisContext) -> list[str]:
    """Number columns a side question could explain, when the target is not a number."""
    if _target_kind(ctx) == "numeric":
        return []
    return [c for c in q.columns if c != ctx.target and ctx.kinds.get(c) == "numeric"]


def _leaks_not_excluded(plan: AnalysisPlan, planned: AnalysisContext) -> list[str]:
    return [
        f"{column} predicts {plan.target} almost perfectly on its own (strength {strength:.2f}), "
        f"so it probably gives the answer away; list it in excluded_features with source "
        f"'leakage'"
        for column, strength in leak_suspects(planned).items()
        if column not in plan.excluded_columns
    ]


def _method_kind_mismatches(questions: list[PlanQuestion]) -> list[str]:
    return [
        f"{q.id} is a {q.kind} question, but {method} is meant for "
        f"{' or '.join(REGISTRY[method].question_kinds)} questions"
        for q in questions
        for method in q.methods
        if method in REGISTRY and q.kind not in REGISTRY[method].question_kinds
    ]


# -- feasibility --------------------------------------------------------------


def _infeasible(q: PlanQuestion, ctx: AnalysisContext, target_kind: str | None) -> list[str]:
    """Why this question cannot be answered well on this data; empty when it can."""
    features = [c for c in q.columns if c != ctx.target and c in ctx.kinds]
    advice = "use fewer columns or mark the question infeasible"
    problems = []
    for method in q.methods:
        if method in REGRESSIONS and target_kind == TARGET_METHODS[method]:
            target = _target(ctx, target_kind)
            k = len(features) or len(_usable_features(ctx))
            if target_kind == "binary":
                events = int(target.sum())
                rows = min(events, len(target) - events)
                what = f"rows with the rarer outcome of {ctx.target}"
            else:
                rows, what = len(target), f"rows with {ctx.target}"
            if rows < ROWS_PER_EFFECT * k:
                problems.append(
                    f"{q.id}: {method} with {k} feature(s) needs {ROWS_PER_EFFECT * k} {what}; "
                    f"there are {rows}: {advice}"
                )
        elif method == "linear_regression" and _side_outcomes(q, ctx):
            k = max(1, len(features) - 1)
            if len(ctx.frame) < ROWS_PER_EFFECT * k:
                problems.append(
                    f"{q.id}: linear_regression with {k} feature(s) needs "
                    f"{ROWS_PER_EFFECT * k} rows; there are {len(ctx.frame)}: {advice}"
                )
        elif method == "target_by_group" and target_kind is not None:
            present = ctx.frame[ctx.target].notna()
            problems += [
                f"{q.id}: {column} has no two groups with {MIN_SEGMENT_ROWS} or more rows; "
                f"group by another column or mark the question infeasible"
                for column in features
                if ctx.kinds[column] in FEATURE_KINDS and not _two_segments(ctx, column, present)
            ]
        elif method == "compare_by_target" and target_kind == "binary":
            target = _target(ctx, target_kind)
            smaller = int(min(target.sum(), len(target) - target.sum()))
            if smaller < MIN_SEGMENT_ROWS:
                problems.append(
                    f"{q.id}: compare_by_target needs {MIN_SEGMENT_ROWS} rows with each outcome; "
                    f"the smaller group has {smaller}: mark the question infeasible"
                )
        elif method == "distribution":
            problems += [
                f"{q.id}: {column} has fewer than {ctx.min_group_n} values to describe"
                for column in q.columns
                if column in ctx.kinds and ctx.frame[column].notna().sum() < ctx.min_group_n
            ]
        elif method == "correlation" and len(ctx.frame) < MIN_ROWS:
            problems.append(
                f"{q.id}: correlation needs {MIN_ROWS} rows; there are {len(ctx.frame)}: {advice}"
            )
    return problems


def _target(ctx: AnalysisContext, kind: str) -> pd.Series:
    values = ctx.binary_target() if kind == "binary" else ctx.numeric_target()
    return values.dropna().astype(float)


def _usable_features(ctx: AnalysisContext) -> list[str]:
    try:
        return ctx.features(None)
    except ToolError:
        return []


def _two_segments(ctx: AnalysisContext, column: str, present: pd.Series) -> bool:
    values = ctx.frame.loc[present, column]
    if ctx.kinds[column] == "numeric":
        return int(values.notna().sum()) >= 2 * MIN_SEGMENT_ROWS
    counts = values.map(lambda v: str(v) if pd.notna(v) else "(missing)").value_counts()
    return int((counts >= MIN_SEGMENT_ROWS).sum()) >= 2


# -- leakage ------------------------------------------------------------------


def _strength(
    ctx: AnalysisContext, column: str, kind: str, target: pd.Series, *, binary: bool
) -> float | None:
    """How well one column alone tells the target: 0.5-1 (yes/no) or 0-1 (numeric)."""
    if kind == "categorical":
        labels = ctx.frame[column].map(lambda v: str(v) if pd.notna(v) else "(missing)")
        keep = target.notna()
        labels, y = labels[keep], target[keep]
        counts = labels.map(labels.value_counts())
        labels = labels.where(counts >= ctx.min_group_n, "(other)")
        if labels.nunique() < 2:
            return None
        score = y.groupby(labels).transform("mean")
    else:
        numbers = (
            as_yes_no(ctx.frame[column])
            if kind == "boolean"
            else pd.to_numeric(ctx.frame[column]).astype("Float64")
        )
        if numbers is None:
            return None
        keep = target.notna() & numbers.notna()
        score, y = numbers[keep].astype(float), target[keep]
        if score.nunique() < 2:
            return None

    if binary:
        events = int(y.sum())
        if min(events, len(y) - events) < ctx.min_group_n:
            return None
        area = auc(y, score)
        return max(area, 1 - area)
    if len(y) < MIN_ROWS:
        return None
    method = "pearson" if kind == "categorical" else "spearman"
    r = score.corr(y, method=method)
    return None if pd.isna(r) else abs(float(r))
