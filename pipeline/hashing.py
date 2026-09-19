"""Canonical hashing for the audit trail.

Every reconciliation check compares hashes, so the encoding has to be pinned
down once and used everywhere: the same values must always produce the same
digest, and two different values must never collide by being stringified the
same way.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Any

_FIELD_SEP = b"\x1f"
_RECORD_SEP = b"\x1e"
_NULL = b"\x00NULL\x00"
"""Distinct from the empty string: a missing value and a blank value are not the same."""


def canonical(value: Any) -> bytes:
    """One stable byte representation per value.

    Types are part of the representation, so the integer 1 and the string "1"
    hash differently — otherwise a cast could change a column without changing
    its hash.
    """
    if value is None:
        return _NULL
    if isinstance(value, str):
        return b"s:" + value.encode("utf-8")
    if isinstance(value, bool):
        return b"b:1" if value else b"b:0"
    if isinstance(value, int):
        return b"i:" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"f:" + repr(value).encode("ascii")
    if isinstance(value, Decimal):
        return b"d:" + str(value).encode("ascii")
    if isinstance(value, dt.datetime):
        return b"dt:" + value.isoformat().encode("ascii")
    if isinstance(value, dt.date):
        return b"da:" + value.isoformat().encode("ascii")
    if isinstance(value, dt.time):
        return b"ti:" + value.isoformat().encode("ascii")
    if isinstance(value, bytes | bytearray):
        return b"x:" + bytes(value).hex().encode("ascii")
    return b"r:" + repr(value).encode("utf-8")


def hash_values(values: Iterable[Any]) -> str:
    """Order-sensitive hash of a sequence of values, e.g. one column."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(canonical(value))
        digest.update(_FIELD_SEP)
    return digest.hexdigest()


def hash_rows(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Hash of a whole table: column names first, then every row in order.

    Column names are included because renaming a column changes the data's
    meaning even when every value stays the same.
    """
    digest = hashlib.sha256()
    digest.update(_FIELD_SEP.join(name.encode("utf-8") for name in columns))
    digest.update(_RECORD_SEP)
    for row in rows:
        digest.update(_FIELD_SEP.join(canonical(value) for value in row))
        digest.update(_RECORD_SEP)
    return digest.hexdigest()
