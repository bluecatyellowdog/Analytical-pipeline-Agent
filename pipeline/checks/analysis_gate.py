"""Gate 4: is the write-up complete, grounded and honest about what the tools found?

Spec Gate 4, checked on the write-up and the logged tool calls, never on the
agent's word:
- invariant: every plan question has an answer, and a question the plan marked
  infeasible stays infeasible; every finding rests on tool calls that ran;
  every number in the text is one the cited calls reported (a summary or a
  limitation may use any logged number)
- judgment: a finding that rests only on models doing no better than no model
  at all cannot be called strong or moderate ('no signal' is a valid result);
  model warnings (a suspiciously good fit, too few rows, a fit that did not
  settle) are flagged for the report

Failures block and go back to the agent; warnings continue, shown at the top
of the report.
"""

from __future__ import annotations

from collections.abc import Iterable

from pipeline.checks.gates import build_gate, checked
from pipeline.checks.traceability import untraced, values_of
from pipeline.contracts.analysis import CallRecord, WriteUp
from pipeline.contracts.artifact import Artifact, ArtifactCatalog
from pipeline.contracts.audit import GateResult
from pipeline.contracts.plan import AnalysisPlan

REGRESSIONS = ("linear_regression", "logistic_regression")
NO_SIGNAL_P = 0.05


def gate_4_analysis(
    write_up: WriteUp,
    plan: AnalysisPlan,
    calls: list[CallRecord],
    catalog: ArtifactCatalog,
    *,
    context: str = "",
) -> GateResult:
    """Every check, in one gate. `context` is text whose numbers may be used anywhere,
    such as the brief and the plan."""
    artifacts = {c.call_id: _artifacts(c, catalog) for c in calls}
    flags = [flag for found in artifacts.values() for a in found for flag in a.flags]
    checks = [
        checked(
            "every_question_handled",
            "invariant",
            "block",
            _unanswered(write_up, plan),
            "Every plan question is answered or marked infeasible.",
        ),
        checked(
            "evidence_exists",
            "invariant",
            "block",
            _bad_evidence(write_up, plan, calls),
            "Every finding rests on tool calls that ran.",
        ),
        checked(
            "numbers_traceable",
            "invariant",
            "block",
            _untraced_numbers(write_up, calls, artifacts, context),
            "Every number in the write-up is one the tools reported.",
        ),
        checked(
            "no_overclaiming",
            "judgment",
            "block",
            _overclaims(write_up, calls, artifacts),
            "No finding claims more than its evidence shows.",
        ),
        checked(
            "model_warnings",
            "judgment",
            "warn",
            [
                f"{c.call_id} {c.tool}: {flag.message}"
                for c in calls
                for a in artifacts[c.call_id]
                for flag in a.flags
            ],
            "No model raised a warning.",
        ),
    ]
    return build_gate("gate_4_analysis", checks, flags=flags)


def _artifacts(call: CallRecord, catalog: ArtifactCatalog) -> list[Artifact]:
    return [a for a in (catalog.get(i) for i in call.artifact_ids) if a is not None]


def _unanswered(write_up: WriteUp, plan: AnalysisPlan) -> list[str]:
    answers = {a.question_id: a for a in write_up.answers}
    problems = []
    for q in plan.questions:
        answer = answers.get(q.id)
        if answer is None:
            problems.append(f"{q.id} has no answer; answer it, or say why it could not be")
        elif not q.feasible and answer.status != "infeasible":
            problems.append(
                f"{q.id} was marked infeasible in the plan ({q.infeasible_reason}); give it "
                f"status 'infeasible'"
            )
    known = {q.id for q in plan.questions}
    problems += [f"there is no plan question {qid}" for qid in answers if qid not in known]
    return problems


def _bad_evidence(write_up: WriteUp, plan: AnalysisPlan, calls: list[CallRecord]) -> list[str]:
    status = {c.call_id: c.ok for c in calls}
    questions = {q.id for q in plan.questions}
    problems = []
    for finding in write_up.findings:
        for call_id in finding.evidence:
            if call_id not in status:
                problems.append(f"{finding.id} cites {call_id}, which was never run")
            elif not status[call_id]:
                problems.append(f"{finding.id} cites {call_id}, which the tool refused")
        if finding.question_id and finding.question_id not in questions:
            problems.append(f"{finding.id} answers {finding.question_id}, not a plan question")
    return problems


def _untraced_numbers(
    write_up: WriteUp,
    calls: list[CallRecord],
    artifacts: dict[str, list[Artifact]],
    context: str,
) -> list[str]:
    by_id = {c.call_id: c for c in calls if c.ok}

    def logged(call_ids: Iterable[str]) -> list[float]:
        chosen = [by_id[i] for i in call_ids if i in by_id]
        return values_of(
            [c.summary for c in chosen], [a.data for c in chosen for a in artifacts[c.call_id]]
        )

    allowed = values_of(context)
    everything = logged(by_id)
    problems: list[str] = []

    def check(where: str, texts: Iterable[str | None], values: list[float], cited: str) -> None:
        for text in texts:
            for raw in untraced(text or "", values, allowed=allowed):
                problems.append(f"{where}: {raw} is not in the results {cited}")

    hint = "; quote a number those calls reported, or cite the call that did"
    for finding in write_up.findings:
        check(
            finding.id,
            [finding.headline, finding.detail, *finding.caveats],
            logged(finding.evidence),
            f"of {', '.join(finding.evidence)}{hint}",
        )
    for answer in write_up.answers:
        evidence = [
            call_id
            for finding_id in answer.finding_ids
            if (finding := write_up.finding(finding_id)) is not None
            for call_id in finding.evidence
        ]
        values = logged(evidence) if evidence else everything
        cited = f"its findings cite ({', '.join(evidence)})" if evidence else "of any call"
        check(answer.question_id, [answer.answer, answer.reason], values, cited)
    check("summary", [write_up.summary], everything, "of any call")
    check("limitations", write_up.limitations, everything, "of any call")
    check("next_questions", write_up.next_questions, everything, "of any call")
    return list(dict.fromkeys(problems))


def _model_test_p(artifacts: list[Artifact]) -> float | None:
    for artifact in artifacts:
        metrics = artifact.data.get("metrics") if isinstance(artifact.data, dict) else None
        if metrics:
            return metrics.get("f_test_p", metrics.get("lr_test_p"))
    return None


def _overclaims(
    write_up: WriteUp, calls: list[CallRecord], artifacts: dict[str, list[Artifact]]
) -> list[str]:
    by_id = {c.call_id: c for c in calls if c.ok}
    problems = []
    for finding in write_up.findings:
        if finding.strength not in ("strong", "moderate"):
            continue
        cited = [by_id[i] for i in finding.evidence if i in by_id]
        if not cited or any(c.tool not in REGRESSIONS for c in cited):
            continue
        tests = {c.call_id: _model_test_p(artifacts[c.call_id]) for c in cited}
        if all(p is not None and p > NO_SIGNAL_P for p in tests.values()):
            shown = ", ".join(f"{i} p={p:.2f}" for i, p in tests.items())
            problems.append(
                f"{finding.id} is called {finding.strength}, but its model does no better than "
                f"no model at all ({shown}): call it weak or none"
            )
    return problems
