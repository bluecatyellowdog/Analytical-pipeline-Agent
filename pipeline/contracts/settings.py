"""Project settings: which AI model each agent uses, read from settings.yaml.

Model names are not secret, so the file can be shared; the API key stays in .env.
Without the file, every agent uses LLM_MODEL from .env or the provider's default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import yaml
from pydantic import Field, ValidationError, field_validator

from pipeline.contracts.common import Strict

AgentName = Literal["diagnostic", "coder", "reviewer", "planner", "analysis"]
AGENTS: tuple[AgentName, ...] = get_args(AgentName)
"""In the order they run."""

SETTINGS_FILE = Path("settings.yaml")
"""In the project folder, like the paths in a brief."""


class SettingsError(ValueError):
    """settings.yaml cannot be read, with what to fix."""


class ModelSettings(Strict):
    """A model name per agent; an agent without one uses `default`."""

    default: str | None = Field(default=None, description="For every agent without its own line.")
    diagnostic: str | None = None
    coder: str | None = None
    reviewer: str | None = None
    planner: str | None = None
    analysis: str | None = None

    @field_validator("*")
    @classmethod
    def _blank_is_unset(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip() or None


class Settings(Strict):
    models: ModelSettings = Field(default_factory=ModelSettings)


def load_settings(path: str | Path = SETTINGS_FILE) -> Settings:
    """The settings in `path`; no file, or an empty one, means no choices."""
    path = Path(path)
    if not path.exists():
        return Settings()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise SettingsError(f"{path.as_posix()} is not valid YAML: {error}") from error
    if raw is None:
        return Settings()
    if not isinstance(raw, dict):
        raise SettingsError(f"{path.as_posix()} must hold settings such as 'models:'")
    try:
        return Settings.model_validate(raw)
    except ValidationError as error:
        raise SettingsError(f"{path.as_posix()}: {_explain(error)}") from error


def _explain(error: ValidationError) -> str:
    problems = []
    for e in error.errors():
        loc = [str(p) for p in e["loc"]]
        if e["type"] == "extra_forbidden" and loc[:1] == ["models"]:
            problems.append(
                f"'{loc[-1]}' is not an agent. The agents are: {', '.join(AGENTS)} (and default)"
            )
        elif e["type"] == "extra_forbidden":
            problems.append(f"'{loc[-1]}' is not a setting. The only setting is 'models'")
        else:
            problems.append(f"{'.'.join(loc)}: {e['msg']}")
    return "; ".join(problems)
