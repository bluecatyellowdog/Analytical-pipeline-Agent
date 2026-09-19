"""Find personal data, flag it, and mask it before anything leaves the machine.

Spec principle: the LLM never sees raw rows or personal data. Detection is
deliberately conservative about *dropping* — nothing is ever removed here,
because that is a judgment call for a later, logged stage — and deliberately
strict about *masking*, because a masked value cannot be unmasked by accident.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Sequence
from typing import Any

from pipeline.contracts.profile import PIIFinding, PIIKind, PIILevel
from pipeline.ingestion.types import is_missing

LIKELY_SHARE = 0.5
"""At least half the values match: treat the column as personal data."""

POSSIBLE_SHARE = 0.1
"""A tenth is enough to warrant masking and a flag."""

NAME_HINTS: dict[PIIKind, tuple[str, ...]] = {
    "email": ("email", "mail", "epost"),
    "cpr": ("cpr", "personnummer", "ssn", "personnr"),
    "phone": ("telefon", "phone", "tlf", "mobil", "mobile"),
    "iban": ("iban", "konto", "account", "reg_nr", "regnr"),
    "name": ("navn", "name", "fornavn", "efternavn", "kunde_navn"),
    "address": ("adresse", "address", "vej", "gade", "postnr", "postnummer", "zip"),
    "ip": ("ip", "ip_adresse", "ip_address"),
}

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
CPR_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})-?(\d{4})$")
PHONE_RE = re.compile(r"^(\+45\s?)?(\d{2}[\s.-]?){3}\d{2}$")
PHONE_FORMATTED_RE = re.compile(
    r"^(\+45[\s.-]?\d{8}|\+45[\s.-]?(\d{2}[\s.-]){3}\d{2}|(\d{2}[\s.-]){3}\d{2})$"
)
"""A phone number that announces itself: a country code or grouped digits.

Without the column name saying 'telefon', a bare '12345678' is far more likely
to be an amount or a postcode, and masking those would cost the analysis real
information for no privacy gain.
"""
IBAN_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$")
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def is_valid_cpr(value: str) -> bool:
    """A CPR number's first six digits must be a real date (spec 4.1).

    Without that check every 10-digit code looks like a CPR number, and the
    fixture set contains look-alikes precisely to keep this honest. The modulus
    11 check is deliberately not used: it has not been mandatory since 2007, so
    valid numbers fail it.
    """
    match = CPR_RE.match(value.strip())
    if not match:
        return False
    day, month, year = (int(match.group(i)) for i in (1, 2, 3))
    for century in (1900, 2000):
        try:
            dt.date(century + year, month, day)
        except ValueError:
            continue
        return True
    return False


_VALUE_TESTS: dict[PIIKind, Any] = {
    "email": lambda v: bool(EMAIL_RE.match(v)),
    "cpr": is_valid_cpr,
    "iban": lambda v: bool(IBAN_RE.match(v.replace(" ", ""))),
    "ip": lambda v: bool(IP_RE.match(v)),
    "phone": lambda v: bool(PHONE_RE.match(v)),
}


def name_suggests(column: str, kind: PIIKind) -> bool:
    lowered = column.lower()
    return any(hint in lowered for hint in NAME_HINTS.get(kind, ()))


def _value_test(kind: PIIKind, *, name_hint: bool):
    """The matcher for one kind, tightened when the column name gives no support."""
    if kind == "phone" and not name_hint:
        return lambda value: bool(PHONE_FORMATTED_RE.match(value.strip()))
    return _VALUE_TESTS[kind]


def detect_pii(column: str, values: Sequence[Any]) -> list[PIIFinding]:
    """Findings for one column, strongest first. Never drops, never rewrites."""
    present = [str(v) for v in values if not is_missing(v)]
    findings: list[PIIFinding] = []

    for kind in _VALUE_TESTS:
        name_hint = name_suggests(column, kind)
        test = _value_test(kind, name_hint=name_hint)
        matches = sum(1 for value in present if test(value)) if present else 0
        share = matches / len(present) if present else 0.0
        level = _level(share, name_hint=name_hint)
        if level == "none":
            continue
        findings.append(
            PIIFinding(
                kind=kind,
                level=level,
                n_matches=matches,
                share=share,
                detector=f"regex:{kind}",
                matched_column_name=name_hint,
            )
        )

    # Free-text names and addresses have no reliable pattern, so only the column
    # name can raise them here. Presidio can take over later (spec 4.1).
    for kind in ("name", "address"):
        if name_suggests(column, kind):  # type: ignore[arg-type]
            findings.append(
                PIIFinding(
                    kind=kind,  # type: ignore[arg-type]
                    level="possible",
                    n_matches=0,
                    share=0.0,
                    detector="column_name",
                    matched_column_name=True,
                )
            )

    order: dict[PIILevel, int] = {"likely": 0, "possible": 1, "none": 2}
    return sorted(findings, key=lambda f: (order[f.level], -f.share))


def _level(share: float, *, name_hint: bool) -> PIILevel:
    if share >= LIKELY_SHARE:
        return "likely"
    if share >= POSSIBLE_SHARE:
        return "possible"
    if name_hint and share > 0:
        return "possible"
    return "none"


def mask_shape(value: Any) -> str:
    """Replace a value with its shape: letters become 'a', digits become '9'.

    The shape is what the agent actually needs — it shows that a column holds
    '999999-9999' rather than free text — while the content is gone. This is
    applied before any value reaches a prompt.
    """
    if value is None:
        return ""
    out = []
    for char in str(value):
        if char.isdigit():
            out.append("9")
        elif char.isalpha():
            out.append("a" if char.islower() else "A")
        else:
            out.append(char)
    return "".join(out)


def mask_values(values: Sequence[Any]) -> list[str]:
    return [mask_shape(v) for v in values]


def strongest_level(findings: Sequence[PIIFinding]) -> PIILevel:
    levels = [f.level for f in findings]
    if "likely" in levels:
        return "likely"
    if "possible" in levels:
        return "possible"
    return "none"
