"""One API key, any provider: detection, configuration and the fake client."""

from __future__ import annotations

import pytest

from pipeline.agents.llm import (
    FakeLLM,
    LLMConfigError,
    LLMResponse,
    connect,
    detect_provider,
    load_config,
)


class TestDetectProvider:
    @pytest.mark.parametrize(
        ("key", "provider"),
        [
            ("sk-or-v1-abc", "openrouter"),
            ("sk-ant-api03-abc", "anthropic"),
            ("sk-proj-abc", "openai"),
            ("sk-abc", "openai"),
            ("gsk_abc", "groq"),
            ("AIzaSyabc", "gemini"),
            ("xai-abc", "xai"),
        ],
    )
    def test_known_prefixes(self, key, provider) -> None:
        assert detect_provider(key).name == provider

    def test_unknown_key_without_base_url_is_an_error(self) -> None:
        with pytest.raises(LLMConfigError, match="LLM_BASE_URL"):
            load_config({"LLM_API_KEY": "mystery-key", "LLM_MODEL": "m"})


class TestLoadConfig:
    def test_openrouter_key_alone_is_enough(self) -> None:
        config = load_config({"LLM_API_KEY": "sk-or-v1-abc"})
        assert config.provider == "openrouter"
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.model == "anthropic/claude-sonnet-5.5"

    def test_anthropic_key_alone_is_enough(self) -> None:
        config = load_config({"LLM_API_KEY": "sk-ant-api03-abc"})
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-5-5"

    def test_model_and_base_url_can_be_overridden(self) -> None:
        config = load_config(
            {
                "LLM_API_KEY": "anything",
                "LLM_BASE_URL": "http://localhost:11434/v1",
                "LLM_MODEL": "llama3",
            }
        )
        assert config.provider == "custom"
        assert config.base_url == "http://localhost:11434/v1"
        assert config.model == "llama3"

    def test_provider_without_a_default_model_asks_for_one(self) -> None:
        with pytest.raises(LLMConfigError, match="LLM_MODEL"):
            load_config({"LLM_API_KEY": "sk-proj-abc"})

    def test_missing_key_says_where_to_put_it(self) -> None:
        with pytest.raises(LLMConfigError, match=r"\.env"):
            load_config({})

    def test_the_key_never_appears_in_the_repr(self) -> None:
        config = load_config({"LLM_API_KEY": "sk-or-v1-secret123"})
        assert "secret123" not in repr(config)
        assert "secret123" not in str(config.public())


class TestFakeLLM:
    def test_returns_scripted_replies_in_order_and_records_prompts(self) -> None:
        fake = FakeLLM(["first", '{"a": 1}'])
        assert fake.complete([{"role": "user", "content": "hi"}]).text == "first"
        reply = fake.complete([{"role": "user", "content": "again"}])
        assert isinstance(reply, LLMResponse)
        assert reply.text == '{"a": 1}'
        assert [m[-1]["content"] for m in fake.calls] == ["hi", "again"]

    def test_running_out_of_replies_is_loud(self) -> None:
        fake = FakeLLM([])
        with pytest.raises(AssertionError):
            fake.complete([{"role": "user", "content": "hi"}])


def test_connect_can_use_another_model_for_one_agent() -> None:
    """The reviewer may run on a different model than the coder (LLM_REVIEW_MODEL)."""
    client = connect({"LLM_API_KEY": "sk-or-v1-abc"}, model="openai/gpt-6-sol")
    assert client.config.model == "openai/gpt-6-sol"
    assert client.config.provider == "openrouter"
