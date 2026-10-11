"""Optional tracing of every AI call to Langfuse; on only when its keys are in .env.

    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com
    LANGFUSE=off                 # optional: pause tracing without removing the keys

Each command is one trace, named after its run folder; the run folder is also the
session, so a later `pipeline.analyze` on the same folder sits next to the first run.
Each AI call is one generation named after its agent, with the model, the messages,
the reply, the tokens and the cost the provider reported.

Tracing can never break a run: an error in it is reported once, then ignored. What
it records is what the AI is sent (masked profiles and aggregates, never rows), so
switching it on means that also goes to Langfuse.
"""

from __future__ import annotations

import atexit
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline.agents.llm import LLMResponse, Message

DEFAULT_HOST = "https://cloud.langfuse.com"
OFF_WORDS = ("off", "false", "0", "no")
MAX_STATUS_MESSAGE = 500


class Tracing:
    """No tracing: every call simply goes through."""

    enabled = False

    def __init__(self, reason: str = "") -> None:
        self.reason = reason

    def status(self) -> str:
        return f"Tracing: off ({self.reason})" if self.reason else "Tracing: off"

    def label_run(self, run_folder: str, command: str) -> None:
        """Name this command's trace after its run folder."""

    def call(
        self,
        agent: str,
        model: str,
        messages: Sequence[Message],
        send: Callable[[], LLMResponse],
    ) -> LLMResponse:
        return send()

    def finish(self) -> str | None:
        """Send what is left; a line saying where to see the trace, or None."""
        return None


class LangfuseTracing(Tracing):
    """Records each AI call as a Langfuse generation in one trace per command."""

    enabled = True

    def __init__(self, client: Any, host: str = DEFAULT_HOST) -> None:
        super().__init__()
        self.client = client
        self.host = host
        self.trace_id = client.create_trace_id()
        self.run_folder: str | None = None
        self.command = "run"
        self.calls = 0
        self._warned = False
        atexit.register(self._flush_quietly)

    def status(self) -> str:
        return f"Tracing: Langfuse on ({self.host.removeprefix('https://')})"

    def label_run(self, run_folder: str, command: str) -> None:
        self.run_folder, self.command = run_folder, command

    def call(self, agent, model, messages, send):
        generation = self._start(agent, model, messages)
        try:
            response = send()
        except Exception as error:
            if generation is not None:
                message = str(error)[:MAX_STATUS_MESSAGE]
                self._safely(lambda: generation.update(level="ERROR", status_message=message).end())
            raise
        if generation is not None:
            self._safely(lambda: self._record(generation, response))
        return response

    def finish(self) -> str | None:
        self._flush_quietly()
        if not self.calls:
            return None
        try:
            url = self.client.get_trace_url(trace_id=self.trace_id)
        except Exception:  # noqa: BLE001 - the link is a convenience; the trace is sent anyway
            url = None
        return f"Trace: {url}" if url else f"Trace: in Langfuse ({self.host}), {self.label}"

    @property
    def label(self) -> str:
        return self.run_folder or "pipeline run"

    # -- inside -------------------------------------------------------------------

    def _start(self, agent: str, model: str, messages: Sequence[Message]) -> Any:
        from langfuse import propagate_attributes

        def start():
            with propagate_attributes(
                trace_name=self.label,
                session_id=self.label,
                tags=[self.command],
                metadata={"run_folder": self.label, "command": self.command},
            ):
                return self.client.start_observation(
                    trace_context={"trace_id": self.trace_id},
                    name=agent,
                    as_type="generation",
                    model=model,
                    input=list(messages),
                    metadata={"agent": agent},
                )

        generation = self._safely(start)
        if generation is not None:
            self.calls += 1
        return generation

    @staticmethod
    def _record(generation: Any, response: LLMResponse) -> None:
        generation.update(
            output=response.text,
            model=response.model,
            usage_details={"input": response.input_tokens, "output": response.output_tokens},
            cost_details={"total": response.cost_usd} if response.cost_usd is not None else None,
        ).end()

    def _safely(self, action: Callable[[], Any]) -> Any:
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - tracing must never break a run
            if not self._warned:
                self._warned = True
                print(f"Tracing problem, carrying on without it: {type(error).__name__}: {error}")
            return None

    def _flush_quietly(self) -> None:
        try:
            self.client.flush()
        except Exception:  # noqa: BLE001 - nothing to do about it at the end of a run
            pass


_current: Tracing = Tracing()


def tracing_from_env(env: Mapping[str, str]) -> Tracing:
    """Langfuse tracing when both keys are set and LANGFUSE is not 'off'; otherwise none."""
    public = (env.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret = (env.get("LANGFUSE_SECRET_KEY") or "").strip()
    if (env.get("LANGFUSE") or "").strip().lower() in OFF_WORDS:
        return Tracing("LANGFUSE=off in .env")
    if not public and not secret:
        return Tracing()
    if not (public and secret):
        return Tracing("only one of the two Langfuse keys is set in .env")
    host = (env.get("LANGFUSE_HOST") or "").strip() or DEFAULT_HOST
    try:
        from langfuse import Langfuse

        client = Langfuse(public_key=public, secret_key=secret, host=host)
    except Exception as error:  # noqa: BLE001 - a broken tracing setup must not stop a run
        return Tracing(f"Langfuse could not start: {type(error).__name__}: {error}")
    _quiet_exporter_logs()
    return LangfuseTracing(client, host)


def start_tracing(env: Mapping[str, str], tracing: Tracing | None = None) -> Tracing:
    """Make `tracing` (or the one .env asks for) the active one; one per process."""
    global _current
    if tracing is not None:
        _current = tracing
    elif not _current.enabled:
        _current = tracing_from_env(env)
    return _current


def current() -> Tracing:
    return _current


def label_run(run_folder: str, command: str) -> None:
    current().label_run(run_folder, command)


def announce(run_folder: str, command: str) -> str:
    """Name the trace after the run folder; the line a command prints about tracing."""
    label_run(run_folder, command)
    return current().status()


def finish() -> str | None:
    return current().finish()


def check(tracing: Tracing) -> str:
    """Whether Langfuse accepts the keys: for the check command, not for every run."""
    if not isinstance(tracing, LangfuseTracing):
        return tracing.status()
    try:
        accepted = tracing.client.auth_check()
    except Exception as error:  # noqa: BLE001 - report any failure plainly
        return f"Langfuse: could not connect to {tracing.host}: {type(error).__name__}: {error}"
    if not accepted:
        return f"Langfuse: {tracing.host} did not accept the keys. Check them in .env."
    return f"Langfuse: keys accepted by {tracing.host}. Every AI call will be traced."


def _quiet_exporter_logs() -> None:
    """A failed upload is logged with a long traceback; keep the run's output readable."""
    for name in ("langfuse", "opentelemetry.sdk", "opentelemetry.exporter"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
