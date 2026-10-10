"""One LLM client for every provider: put an API key in `.env` and it works.

Almost every provider now speaks the OpenAI chat-completions format, so a
single client pointed at the right base URL covers OpenRouter, Anthropic,
OpenAI, Gemini, Groq, xAI and local servers such as Ollama. The provider is
recognised from the key's prefix; `LLM_BASE_URL` and `LLM_MODEL` override it.

    LLM_API_KEY=sk-or-v1-...          # required
    LLM_MODEL=anthropic/claude-opus-5.5   # optional where a default exists
    LLM_BASE_URL=http://localhost:11434/v1  # optional, for anything else

Which model each agent uses is set in settings.yaml (`connect_agents`); LLM_MODEL
is the fallback for an agent it names no model for.

This module only moves text. What the text may contain (masked profiles,
never raw rows) is enforced where prompts are built, not here.

    uv run python -m pipeline.agents.llm     # check the key and every model work
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from pipeline.contracts.settings import (
    AGENTS,
    AgentName,
    Settings,
    SettingsError,
    load_settings,
)

Message = dict[str, str]
"""{'role': 'system' | 'user' | 'assistant', 'content': '...'}"""


class LLMConfigError(RuntimeError):
    """The key, model or endpoint is missing or not recognised."""


@dataclass(frozen=True)
class Provider:
    name: str
    prefix: str
    base_url: str
    default_model: str | None = None
    """Only where the name is known to exist; otherwise LLM_MODEL is required."""


# Longest prefix first: "sk-or-" and "sk-ant-" must win over plain "sk-".
PROVIDERS: tuple[Provider, ...] = (
    Provider("openrouter", "sk-or-", "https://openrouter.ai/api/v1", "anthropic/claude-sonnet-5.5"),
    Provider("anthropic", "sk-ant-", "https://api.anthropic.com/v1/", "claude-sonnet-5-5"),
    Provider("groq", "gsk_", "https://api.groq.com/openai/v1"),
    Provider("gemini", "AIza", "https://generativelanguage.googleapis.com/v1beta/openai/"),
    Provider("xai", "xai-", "https://api.x.ai/v1"),
    Provider("openai", "sk-", "https://api.openai.com/v1"),
)


def detect_provider(api_key: str) -> Provider | None:
    """Which provider issued this key, judged by its prefix."""
    return next((p for p in PROVIDERS if api_key.startswith(p.prefix)), None)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)

    def public(self) -> dict[str, str]:
        """What may be logged in the run manifest: everything but the key."""
        return {"provider": self.provider, "base_url": self.base_url, "model": self.model}


def load_config(env: Mapping[str, str] | None = None) -> LLMConfig:
    """Read LLM_API_KEY, LLM_MODEL and LLM_BASE_URL from `env`, or from .env + the environment."""
    if env is None:
        from dotenv import load_dotenv

        load_dotenv()
        env = os.environ

    api_key = (env.get("LLM_API_KEY") or "").strip()
    if not api_key:
        raise LLMConfigError(
            "No API key found. Add a line `LLM_API_KEY=<your key>` to the .env file in the "
            "repository root (see .env.example)."
        )

    base_url = (env.get("LLM_BASE_URL") or "").strip()
    model = (env.get("LLM_MODEL") or "").strip()
    provider = detect_provider(api_key)

    if base_url:
        name = provider.name if provider and provider.base_url == base_url else "custom"
    elif provider:
        name, base_url = provider.name, provider.base_url
    else:
        raise LLMConfigError(
            "The key's provider was not recognised from its prefix. Set LLM_BASE_URL to the "
            "provider's OpenAI-compatible endpoint, and LLM_MODEL to a model name."
        )

    if not model:
        default = provider.default_model if provider and name == provider.name else None
        if not default:
            raise LLMConfigError(
                f"No default model for provider '{name}'. Set LLM_MODEL in .env to one of the "
                f"provider's model names."
            )
        model = default

    return LLMConfig(provider=name, base_url=base_url, model=model, api_key=api_key)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    """Reported by OpenRouter; None where the provider does not say."""


class LLMClient(Protocol):
    def complete(self, messages: Sequence[Message]) -> LLMResponse: ...


class OpenAICompatibleClient:
    """Any provider with an OpenAI-style /chat/completions endpoint."""

    def __init__(self, config: LLMConfig, *, timeout: float = 120.0) -> None:
        from openai import OpenAI

        self.config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)

    def complete(self, messages: Sequence[Message]) -> LLMResponse:
        extra: dict[str, Any] = {}
        if self.config.provider == "openrouter":
            # Route only to providers that neither store nor train on prompts,
            # and have OpenRouter report the real cost of the call.
            extra = {"provider": {"data_collection": "deny"}, "usage": {"include": True}}

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            temperature=0,
            extra_body=extra or None,
        )
        usage = response.usage
        cost = (usage.model_extra or {}).get("cost") if usage else None
        return LLMResponse(
            text=response.choices[0].message.content or "",
            model=response.model or self.config.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=float(cost) if cost is not None else None,
        )


class FakeLLM:
    """Scripted replies for tests: no network, no key, no cost."""

    def __init__(self, replies: Sequence[str], model: str = "fake") -> None:
        self._replies = list(replies)
        self.model = model
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> LLMResponse:
        self.calls.append(list(messages))
        assert self._replies, "FakeLLM ran out of scripted replies"
        return LLMResponse(text=self._replies.pop(0), model=self.model)


def connect(
    env: Mapping[str, str] | None = None, *, model: str | None = None
) -> OpenAICompatibleClient:
    """A client for whatever key is configured; `model` overrides the configured model."""
    config = load_config(env)
    if model:
        config = replace(config, model=model)
    return OpenAICompatibleClient(config)


def model_for(agent: AgentName, settings: Settings, env: Mapping[str, str]) -> str | None:
    """The agent's own line in settings.yaml, else (reviewer only) LLM_REVIEW_MODEL, else the
    settings' default. None leaves it to connect(): LLM_MODEL, then the provider's default."""
    own = getattr(settings.models, agent)
    if own:
        return own
    if agent == "reviewer":
        review_model = (env.get("LLM_REVIEW_MODEL") or "").strip()
        if review_model:
            return review_model
    return settings.models.default


def connect_agents(
    env: Mapping[str, str] | None = None, settings: Settings | None = None
) -> dict[AgentName, OpenAICompatibleClient]:
    """One client per agent, each on the model settings.yaml gives it."""
    if env is None:
        from dotenv import load_dotenv

        load_dotenv()
        env = os.environ
    if settings is None:
        try:
            settings = load_settings()
        except SettingsError as error:
            raise LLMConfigError(str(error)) from error
    by_model: dict[str | None, OpenAICompatibleClient] = {}
    clients: dict[AgentName, OpenAICompatibleClient] = {}
    for agent in AGENTS:
        model = model_for(agent, settings, env)
        if model not in by_model:
            by_model[model] = connect(env, model=model)
        clients[agent] = by_model[model]
    return clients


def connect_agent(agent: AgentName) -> OpenAICompatibleClient:
    """The client for one agent, for the commands that run a single step."""
    return connect_agents()[agent]


def describe_models(clients: Mapping[str, OpenAICompatibleClient]) -> str:
    """'diagnostic, analysis: <model> | coder, reviewer, planner: <model>', in running order."""
    groups: dict[str, list[str]] = {}
    for agent, client in clients.items():
        groups.setdefault(client.config.model, []).append(agent)
    return " | ".join(f"{', '.join(agents)}: {model}" for model, agents in groups.items())


def public_info(client: LLMClient, fallback: dict[str, str] | None = None) -> dict[str, str]:
    """What the manifest records about a client (never the key); `fallback` for test fakes."""
    config = getattr(client, "config", None)
    return config.public() if isinstance(config, LLMConfig) else dict(fallback or {})


def main() -> int:
    try:
        clients = connect_agents()
    except LLMConfigError as error:
        print(f"Not configured: {error}")
        return 1

    cfg = next(iter(clients.values())).config
    print(f"Provider: {cfg.provider}\nEndpoint: {cfg.base_url}\n")
    for agent, client in clients.items():
        print(f"  {agent:<11}{client.config.model}")
    print()

    failed = False
    for client in {id(c): c for c in clients.values()}.values():
        model = client.config.model
        try:
            reply = client.complete([{"role": "user", "content": "Reply with the single word: ok"}])
        except Exception as error:  # noqa: BLE001 - report any failure from any provider plainly
            print(f"{model}: FAILED, {type(error).__name__}: {error}")
            failed = True
            continue
        cost = f", ${reply.cost_usd:.6f}" if reply.cost_usd is not None else ""
        print(
            f"{model}: answered {reply.text.strip()!r} ({reply.input_tokens} in, "
            f"{reply.output_tokens} out{cost})"
        )

    if failed:
        print(
            "\nA model failed. Check its name in settings.yaml "
            "(for OpenRouter: openrouter.ai/models)."
        )
        return 1
    print("\nThe key works, and every model answers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
