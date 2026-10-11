"""Optional Langfuse tracing: off without keys, one generation per AI call, never breaks a run.

No test contacts Langfuse: a fake client records the calls, and one test uses the
real SDK with an in-memory exporter to pin down what Langfuse would receive.
"""

from __future__ import annotations

import json

import pytest

from pipeline.agents import tracing
from pipeline.agents.llm import FakeLLM, LLMResponse, TracedClient, connect_agents
from pipeline.agents.tracing import LangfuseTracing, Tracing, tracing_from_env
from pipeline.contracts.settings import AGENTS, Settings

KEYS = {"LANGFUSE_PUBLIC_KEY": "pk-lf-1", "LANGFUSE_SECRET_KEY": "sk-lf-1"}
LLM_KEY = {"LLM_API_KEY": "sk-or-v1-abc"}
MESSAGES = [{"role": "user", "content": "hi"}]


class FakeGeneration:
    def __init__(self, **started) -> None:
        self.started = started
        self.updates: list[dict] = []
        self.ended = False

    def update(self, **fields):
        self.updates.append(fields)
        return self

    def end(self):
        self.ended = True
        return self


class FakeLangfuse:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.generations: list[FakeGeneration] = []
        self.flushed = 0

    def create_trace_id(self) -> str:
        return "a" * 32

    def start_observation(self, **started):
        if self.fail:
            raise ConnectionError("Langfuse is down")
        self.generations.append(FakeGeneration(**started))
        return self.generations[-1]

    def flush(self) -> None:
        self.flushed += 1

    def get_trace_url(self, *, trace_id: str) -> str:
        return f"https://cloud.langfuse.com/project/p/traces/{trace_id}"


@pytest.fixture(autouse=True)
def no_tracing_left_on():
    yield
    tracing.start_tracing({}, Tracing())


class TestSwitchingItOn:
    def test_off_without_keys(self) -> None:
        assert tracing_from_env({}).status() == "Tracing: off"

    def test_off_when_paused(self) -> None:
        off = tracing_from_env({**KEYS, "LANGFUSE": "off"})
        assert not off.enabled and "LANGFUSE=off" in off.status()

    def test_one_key_is_not_enough_and_says_so(self) -> None:
        off = tracing_from_env({"LANGFUSE_PUBLIC_KEY": "pk-lf-1"})
        assert not off.enabled and "only one" in off.status()

    def test_on_with_both_keys(self) -> None:
        on = tracing_from_env({**KEYS, "LANGFUSE_HOST": "http://localhost:9"})
        assert on.enabled and on.status() == "Tracing: Langfuse on (http://localhost:9)"


class TestRecording:
    def traced(self, client: FakeLangfuse, model: str = "m-1"):
        active = LangfuseTracing(client)
        llm = FakeLLM(["first", "second"], model=model)
        llm.config = type("Config", (), {"model": model})()  # what OpenAICompatibleClient has
        return active, TracedClient(llm, "diagnostic", active)

    def test_each_call_is_a_generation_named_after_its_agent(self) -> None:
        client = FakeLangfuse()
        active, llm = self.traced(client)
        active.label_run("telco_20261011-101010", "run")
        assert llm.complete(MESSAGES).text == "first"
        llm.complete(MESSAGES)

        assert len(client.generations) == 2
        started = client.generations[0].started
        assert started["name"] == "diagnostic"
        assert started["as_type"] == "generation"
        assert started["model"] == "m-1"
        assert started["input"] == MESSAGES
        assert started["trace_context"] == {"trace_id": "a" * 32}
        assert {g.started["trace_context"]["trace_id"] for g in client.generations} == {"a" * 32}
        assert all(g.ended for g in client.generations)
        assert client.generations[0].updates[0]["output"] == "first"

    def test_tokens_and_the_reported_cost_are_recorded(self) -> None:
        client = FakeLangfuse()
        active = LangfuseTracing(client)
        response = LLMResponse("ok", "m-1", input_tokens=120, output_tokens=30, cost_usd=0.0042)
        active.call("analysis", "m-1", MESSAGES, lambda: response)
        update = client.generations[0].updates[0]
        assert update["usage_details"] == {"input": 120, "output": 30}
        assert update["cost_details"] == {"total": 0.0042}

    def test_a_tracing_failure_never_breaks_the_run(self, capsys) -> None:
        active, llm = self.traced(FakeLangfuse(fail=True))
        assert llm.complete(MESSAGES).text == "first"
        assert llm.complete(MESSAGES).text == "second"
        out = capsys.readouterr().out
        assert out.count("Tracing problem") == 1 and "Langfuse is down" in out

    def test_a_failed_ai_call_is_marked_and_still_raised(self) -> None:
        client = FakeLangfuse()
        active = LangfuseTracing(client)

        def broken():
            raise TimeoutError("the provider timed out")

        with pytest.raises(TimeoutError):
            active.call("coder", "m-1", MESSAGES, broken)
        update = client.generations[0].updates[0]
        assert update["level"] == "ERROR" and "timed out" in update["status_message"]

    def test_finish_sends_everything_and_says_where_to_look(self) -> None:
        client = FakeLangfuse()
        active = LangfuseTracing(client)
        assert active.finish() is None  # no AI call, nothing to look at
        active.call("planner", "m-1", MESSAGES, lambda: LLMResponse("ok", "m-1"))
        assert active.finish() == f"Trace: https://cloud.langfuse.com/project/p/traces/{'a' * 32}"
        assert client.flushed >= 2


class TestWiring:
    def test_with_tracing_on_every_agent_is_traced_under_its_own_name(self) -> None:
        active = LangfuseTracing(FakeLangfuse())
        clients = connect_agents(LLM_KEY, Settings(), tracing=active)
        assert set(clients) == set(AGENTS)
        assert all(isinstance(c, TracedClient) for c in clients.values())
        assert {agent: c.agent for agent, c in clients.items()} == {a: a for a in AGENTS}
        assert clients["coder"].config.model == "anthropic/claude-sonnet-5.5"

    def test_with_tracing_off_the_clients_are_untouched(self) -> None:
        clients = connect_agents(LLM_KEY, Settings(), tracing=Tracing())
        assert not any(isinstance(c, TracedClient) for c in clients.values())


def test_the_real_sdk_receives_trace_name_session_usage_and_cost() -> None:
    """The Langfuse SDK itself, exporting to memory: what the dashboard would show."""
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        host="http://localhost:9",
        span_exporter=exporter,
    )
    active = LangfuseTracing(client, "http://localhost:9")
    active.label_run("telco_20261011-101010", "run")
    response = LLMResponse("ok", "m-1", input_tokens=10, output_tokens=2, cost_usd=0.001)
    active.call("diagnostic", "m-1", MESSAGES, lambda: response)
    client.flush()

    [span] = [s for s in exporter.get_finished_spans() if s.name == "diagnostic"]
    attributes = dict(span.attributes)
    assert format(span.context.trace_id, "032x") == active.trace_id
    assert attributes["langfuse.trace.name"] == "telco_20261011-101010"
    assert attributes["session.id"] == "telco_20261011-101010"
    assert attributes["langfuse.observation.type"] == "generation"
    usage = json.loads(attributes["langfuse.observation.usage_details"])
    assert usage == {"input": 10, "output": 2}
    assert json.loads(attributes["langfuse.observation.cost_details"]) == {"total": 0.001}
