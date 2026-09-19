"""Facts about the machine and the checkout, recorded in every run manifest.

Spec 6: a run is reproducible only if you can tell which code produced it, so
the git SHA and whether the tree was dirty are part of the audit trail.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from functools import cache
from importlib import metadata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@cache
def pipeline_version() -> str:
    try:
        return metadata.version("agentic-ds-pipeline")
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


def python_version() -> str:
    return sys.version.split()[0]


def platform_name() -> str:
    return f"{platform.system()}-{platform.release()}"


def _git(*args: str) -> str | None:
    """Run a git command in the repo, or return None outside a checkout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_sha() -> str | None:
    return _git("rev-parse", "HEAD")


def git_is_dirty() -> bool:
    """True when uncommitted changes mean the SHA does not identify the code."""
    status = _git("status", "--porcelain")
    return bool(status)
