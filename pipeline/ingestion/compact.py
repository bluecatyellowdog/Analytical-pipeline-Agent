"""The compact profile: the only view of the data an agent ever gets.

Spec 4.1 step 8: one or two lines per column, roughly 100-150 tokens, with
personal values reduced to their shape. This module is the boundary where the
'LLM never sees raw rows' principle is actually enforced, so it builds its
output from the profile rather than from the data, and masks on the way out.
"""

from __future__ import annotations

from typing import Any

from pipeline.contracts.context_brief import ClaimResult
from pipeline.contracts.profile import ColumnProfile, DatasetProfile
from pipeline.ingestion.pii import mask_shape

MAX_EXAMPLES = 3


def compact_profile(
    profile: DatasetProfile, claim_results: list[ClaimResult] | None = None
) -> dict[str, Any]:
    """A small, JSON-serialisable description of the data for a prompt."""
    return {
        "table": profile.table,
        "n_rows": profile.n_rows,
        "n_columns": profile.n_columns,
        "duplicate_rows": profile.n_duplicate_rows,
        "candidate_keys": profile.candidate_keys,
        "sampled": profile.sampled,
        "columns": [compact_column(column) for column in profile.columns],
        "flags": [
            {"code": f.code, "severity": f.severity, "column": f.column, "message": f.message}
            for f in _all_flags(profile)
        ],
        "brief_claims": [
            {"claim": r.claim.text or r.claim.kind, "status": r.status, "message": r.message}
            for r in (claim_results or [])
        ],
    }


def compact_column(column: ColumnProfile) -> dict[str, Any]:
    """One column, small enough to fit many of them in a prompt."""
    proposal = column.type_proposal
    # Identifier values are masked alongside detected PII. A customer number is
    # pseudonymous personal data, and its content never helps a cleaning
    # decision — only its shape does, which the mask preserves.
    masked = column.pii_level != "none" or proposal.proposed == "id"

    entry: dict[str, Any] = {
        "name": column.name,
        "type": proposal.proposed,
        "coverage": round(proposal.coverage, 3),
        "missing": f"{column.missing_share:.1%}",
        "distinct": column.n_distinct,
    }

    if proposal.ambiguous:
        entry["ambiguous"] = True
    if column.numeric:
        entry["range"] = [
            _round(column.numeric.min),
            _round(column.numeric.median),
            _round(column.numeric.max),
        ]
        if column.numeric.n_outliers:
            entry["far_outliers"] = column.numeric.n_outliers
    if column.length:
        entry["length"] = [column.length.min, column.length.max]

    examples = [v.value for v in column.top_values[:MAX_EXAMPLES]]
    if examples:
        # Values in the profile are already masked for PII columns; masking again
        # is harmless and means a future change upstream cannot leak through here.
        entry["examples"] = [mask_shape(v) if masked else v for v in examples]

    if column.null_tokens:
        entry["null_tokens"] = column.null_tokens
    if column.sentinel_candidates:
        entry["sentinel_candidates"] = [s.value for s in column.sentinel_candidates]
    if column.pii_level != "none":
        entry["pii"] = {
            "level": column.pii_level,
            "kinds": sorted({f.kind for f in column.pii}),
            "masked": True,
        }
    elif masked:
        entry["masked"] = "identifier: only the shape of the values is shown"
    if proposal.failing_examples:
        entry["unparsed_examples"] = [
            mask_shape(v) if masked else v for v in proposal.failing_examples[:MAX_EXAMPLES]
        ]
    if proposal.notes:
        entry["notes"] = proposal.notes
    if proposal.details:
        entry["format"] = {k: v for k, v in proposal.details.items() if k != "unique"}
    return entry


def render_profile_lines(profile: DatasetProfile) -> str:
    """The same information as plain text, one or two lines per column.

    Used in prompts, where text costs fewer tokens than JSON, and in the data
    report, where a human reads it.
    """
    lines = [
        f"Table {profile.table}: {profile.n_rows:,} rows x {profile.n_columns} columns"
        + (f", {profile.n_duplicate_rows} duplicate rows" if profile.n_duplicate_rows else "")
        + (
            f", candidate keys: {', '.join(profile.candidate_keys)}"
            if profile.candidate_keys
            else ""
        )
    ]

    for column in profile.columns:
        entry = compact_column(column)
        head = (
            f"- {entry['name']}: {entry['type']}"
            f" (coverage {entry['coverage']:.0%}, missing {entry['missing']},"
            f" {entry['distinct']} distinct)"
        )
        if "range" in entry:
            head += f" range {entry['range'][0]}..{entry['range'][2]}"
        lines.append(head)

        detail: list[str] = []
        if entry.get("examples"):
            detail.append("e.g. " + ", ".join(str(v) for v in entry["examples"]))
        if entry.get("pii"):
            detail.append(
                f"PII {entry['pii']['level']} ({', '.join(entry['pii']['kinds'])}), masked"
            )
        elif entry.get("masked"):
            detail.append(entry["masked"])
        if entry.get("sentinel_candidates"):
            detail.append("possible sentinels: " + ", ".join(entry["sentinel_candidates"]))
        if entry.get("ambiguous"):
            detail.append("AMBIGUOUS")
        if entry.get("notes"):
            detail.extend(entry["notes"])
        if detail:
            lines.append("    " + "; ".join(detail))

    flags = _all_flags(profile)
    if flags:
        lines.append(f"Flags ({len(flags)}):")
        lines.extend(f"- [{f.severity}] {f.code}: {f.message}" for f in flags)
    return "\n".join(lines)


def _all_flags(profile: DatasetProfile):
    return [*profile.flags, *(f for column in profile.columns for f in column.flags)]


def _round(value: float) -> float:
    return round(value, 4)
