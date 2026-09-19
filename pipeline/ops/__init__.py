"""The op library. Importing this package registers every op.

No data transformation exists outside this package: that is the whole point of
the design, and `pipeline/checks/reconciliation.py` is what proves it.
"""

from pipeline.ops import library  # noqa: F401  (import registers the ops)
from pipeline.ops.base import (
    REGISTRY,
    OpContext,
    OpError,
    OpPlan,
    apply_op,
    finalize,
    op_names,
    register,
    validate_params,
)

__all__ = [
    "REGISTRY",
    "OpContext",
    "OpError",
    "OpPlan",
    "apply_op",
    "finalize",
    "op_names",
    "register",
    "validate_params",
]
