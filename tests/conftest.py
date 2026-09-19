"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, so tests can reach examples/ and tests/fixtures/."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixtures_dir(repo_root: Path) -> Path:
    """The deliberately nasty files from spec 4.5."""
    return repo_root / "tests" / "fixtures"
