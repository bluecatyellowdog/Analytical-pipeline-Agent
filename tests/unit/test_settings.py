"""settings.yaml: each agent can have its own model, with a default for the rest."""

from __future__ import annotations

import pytest

from pipeline.agents.llm import (
    LLMConfigError,
    connect_agents,
    describe_models,
    model_for,
)
from pipeline.contracts.settings import AGENTS, Settings, SettingsError, load_settings

KEY = {"LLM_API_KEY": "sk-or-v1-abc"}


def settings(**models: str) -> Settings:
    return Settings.model_validate({"models": models})


class TestLoadSettings:
    def test_no_file_means_no_choices(self, tmp_path) -> None:
        loaded = load_settings(tmp_path / "settings.yaml")
        assert loaded == Settings()

    def test_an_empty_file_means_no_choices(self, tmp_path) -> None:
        path = tmp_path / "settings.yaml"
        path.write_text("# only comments\n", encoding="utf-8")
        assert load_settings(path) == Settings()

    def test_reads_a_model_per_agent(self, tmp_path) -> None:
        path = tmp_path / "settings.yaml"
        path.write_text("models:\n  default: m-cheap\n  analysis: m-strong\n", encoding="utf-8")
        loaded = load_settings(path)
        assert loaded.models.default == "m-cheap"
        assert loaded.models.analysis == "m-strong"
        assert loaded.models.coder is None

    def test_a_misspelt_agent_is_named_with_the_right_names(self, tmp_path) -> None:
        path = tmp_path / "settings.yaml"
        path.write_text("models:\n  analyst: m-strong\n", encoding="utf-8")
        with pytest.raises(SettingsError) as error:
            load_settings(path)
        message = str(error.value)
        assert "'analyst' is not an agent" in message
        for agent in AGENTS:
            assert agent in message

    def test_broken_yaml_says_so(self, tmp_path) -> None:
        path = tmp_path / "settings.yaml"
        path.write_text("models:\n  analysis: [unclosed\n", encoding="utf-8")
        with pytest.raises(SettingsError, match="not valid YAML"):
            load_settings(path)

    def test_a_blank_model_counts_as_not_set(self, tmp_path) -> None:
        path = tmp_path / "settings.yaml"
        path.write_text("models:\n  coder: '  '\n", encoding="utf-8")
        assert load_settings(path).models.coder is None


class TestModelFor:
    def test_the_agents_own_line_wins(self) -> None:
        chosen = settings(default="m-cheap", analysis="m-strong")
        assert model_for("analysis", chosen, KEY) == "m-strong"

    def test_an_agent_without_a_line_uses_the_default(self) -> None:
        assert model_for("coder", settings(default="m-cheap"), KEY) == "m-cheap"

    def test_without_settings_the_env_decides(self) -> None:
        """None: connect() then falls back to LLM_MODEL, then the provider's default."""
        assert model_for("planner", Settings(), KEY) is None

    def test_the_old_reviewer_variable_still_works(self) -> None:
        env = {**KEY, "LLM_REVIEW_MODEL": "m-other"}
        assert model_for("reviewer", settings(default="m-cheap"), env) == "m-other"
        assert model_for("reviewer", settings(reviewer="m-own"), env) == "m-own"


class TestConnectAgents:
    def test_every_agent_gets_its_model(self) -> None:
        chosen = settings(default="m-cheap", analysis="m-strong", diagnostic="m-strong")
        clients = connect_agents(KEY, chosen)
        assert set(clients) == set(AGENTS)
        assert clients["analysis"].config.model == "m-strong"
        assert clients["diagnostic"].config.model == "m-strong"
        assert clients["coder"].config.model == "m-cheap"
        assert clients["reviewer"].config.model == "m-cheap"

    def test_without_settings_every_agent_uses_llm_model(self) -> None:
        clients = connect_agents({**KEY, "LLM_MODEL": "m-env"}, Settings())
        assert {c.config.model for c in clients.values()} == {"m-env"}

    def test_a_bad_settings_file_is_a_configuration_error(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "settings.yaml").write_text("models:\n  analyst: x\n", encoding="utf-8")
        with pytest.raises(LLMConfigError, match="'analyst' is not an agent"):
            connect_agents(KEY)

    def test_the_summary_groups_agents_by_model(self) -> None:
        chosen = settings(default="m-cheap", analysis="m-strong", diagnostic="m-strong")
        text = describe_models(connect_agents(KEY, chosen))
        assert text == "diagnostic, analysis: m-strong | coder, reviewer, planner: m-cheap"
