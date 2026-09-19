"""Contracts are the source of truth for every stage boundary.

Change them deliberately: a change here is a change to what one stage promises
the next, and every gate, report and test reads from these types.
"""

from pipeline.contracts.artifact import (
    Artifact,
    ArtifactCatalog,
    ArtifactKind,
    AuditRef,
    PublishConfig,
    PublishSelection,
)
from pipeline.contracts.audit import (
    AuditLog,
    CheckResult,
    ColumnStats,
    GateResult,
    OpRecord,
)
from pipeline.contracts.common import (
    CheckLayer,
    Flag,
    GateOutcome,
    Severity,
    Strict,
    new_id,
    utcnow,
)
from pipeline.contracts.context_brief import (
    Claim,
    ClaimKind,
    ClaimResult,
    ClaimStatus,
    ColumnNote,
    ContextBrief,
    load_context_brief,
)
from pipeline.contracts.handoff import ColumnSchema, Handoff
from pipeline.contracts.ingestion import (
    ColumnRename,
    FileFormat,
    FileInfo,
    LoadResult,
    RejectedLine,
)
from pipeline.contracts.manifest import (
    CostRecord,
    ModelInfo,
    RunManifest,
    SamplingRecord,
)
from pipeline.contracts.plan import (
    AnalysisPlan,
    ExcludedFeature,
    FixRequest,
    PlanQuestion,
    QuestionKind,
    ReviewVerdict,
    TaskType,
)
from pipeline.contracts.profile import (
    ColumnProfile,
    DatasetProfile,
    LengthSummary,
    NumericSummary,
    PIIFinding,
    PIIKind,
    PIILevel,
    ProposedType,
    SentinelCandidate,
    TypeProposal,
    ValueCount,
)

__all__ = [
    "AnalysisPlan",
    "Artifact",
    "ArtifactCatalog",
    "ArtifactKind",
    "AuditLog",
    "AuditRef",
    "CheckLayer",
    "CheckResult",
    "Claim",
    "ClaimKind",
    "ClaimResult",
    "ClaimStatus",
    "ColumnNote",
    "ColumnProfile",
    "ColumnRename",
    "ColumnSchema",
    "ColumnStats",
    "ContextBrief",
    "CostRecord",
    "DatasetProfile",
    "ExcludedFeature",
    "FileFormat",
    "FileInfo",
    "FixRequest",
    "Flag",
    "GateOutcome",
    "GateResult",
    "Handoff",
    "LengthSummary",
    "LoadResult",
    "ModelInfo",
    "NumericSummary",
    "OpRecord",
    "PIIFinding",
    "PIIKind",
    "PIILevel",
    "PlanQuestion",
    "ProposedType",
    "PublishConfig",
    "PublishSelection",
    "QuestionKind",
    "RejectedLine",
    "ReviewVerdict",
    "RunManifest",
    "SamplingRecord",
    "SentinelCandidate",
    "Severity",
    "Strict",
    "TaskType",
    "TypeProposal",
    "ValueCount",
    "load_context_brief",
    "new_id",
    "utcnow",
]
