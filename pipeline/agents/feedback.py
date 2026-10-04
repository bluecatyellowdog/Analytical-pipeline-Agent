"""Masking what goes back to an LLM after a failure.

Error messages from ops, DuckDB and pandas often quote the value that broke
something: "could not convert string to float: '12.345,50'". The full message
stays in the audit log; the copy sent to the model has data values replaced by
their shape ('99.999,99'), which keeps the format information the model needs
and drops the content. Column names and anything the model itself wrote stay
readable, otherwise it could not correct its own typo.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pipeline.ingestion.pii import mask_shape

MAX_FEEDBACK_CHARS = 800
MIN_ECHO_LENGTH = 3
"""Shorter strings occur in almost any reply by chance, so they never count as 'its own'."""

_QUOTED = re.compile(r"'([^'\n]{1,200})'|\"([^\"\n]{1,200})\"")
_LONG_DIGITS = re.compile(r"\d{6,}")


def mask_feedback(text: str, *, known: str = "", keep: Iterable[str] = ()) -> str:
    """`text` with every quoted value and long number masked unless it is safe.

    Safe means: in `keep` (column names, op names, ids) or written by the model
    itself (a substring of `known`, its reply).
    """
    keep_set = {k for k in keep if k}

    def safe(value: str) -> bool:
        return value in keep_set or (len(value) >= MIN_ECHO_LENGTH and value in known)

    def mask_quoted(match: re.Match[str]) -> str:
        quote = match.group(0)[0]
        value = match.group(1) if match.group(1) is not None else match.group(2)
        return match.group(0) if safe(value) else f"{quote}{mask_shape(value)}{quote}"

    masked = _QUOTED.sub(mask_quoted, text)
    masked = _LONG_DIGITS.sub(
        lambda m: m.group(0) if safe(m.group(0)) else "9" * len(m.group(0)), masked
    )
    if len(masked) > MAX_FEEDBACK_CHARS:
        masked = masked[: MAX_FEEDBACK_CHARS - 1] + "…"
    return masked
