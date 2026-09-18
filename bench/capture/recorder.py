"""Observe-only recorder around the app's OpenAI client.

Swaps `app.openai_client.client.chat` for a proxy that forwards every
chat.completions.create call unchanged and records what was sent and what
came back. The app's behaviour is not altered (spec §1 hard rule). The patch
point is the same one the app's own tests use (`client.chat`), so it composes
with test fakes.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.agents import _INTENT_CLASSIFIER_SYSTEM_PROMPT
from app.conversation import CONDENSE_SYSTEM_PROMPT
from app.guardrails import INPUT_CLASSIFIER_SYSTEM_PROMPT, OUTPUT_CLASSIFIER_SYSTEM_PROMPT
from app.openai_client import client

CALL_ROLES = ("guardrail_input", "guardrail_output", "classifier", "condense", "agent")

_SYSTEM_PROMPT_ROLES = {
    INPUT_CLASSIFIER_SYSTEM_PROMPT: "guardrail_input",
    OUTPUT_CLASSIFIER_SYSTEM_PROMPT: "guardrail_output",
    _INTENT_CLASSIFIER_SYSTEM_PROMPT: "classifier",
    CONDENSE_SYSTEM_PROMPT: "condense",
}


@dataclass(frozen=True)
class QueryContext:
    query_id: str
    conversation_id: str | None
    turn_index: int


QUERY_CONTEXT: ContextVar[QueryContext | None] = ContextVar("bench_query_context", default=None)


@dataclass
class CallRecord:
    record_id: str
    query_id: str
    conversation_id: str | None
    turn_index: int
    call_index: int
    call_role: str
    model: str
    messages: list[dict[str, Any]]
    response_format: dict | None
    temperature: float | None
    max_tokens: int | None
    response_text: str
    finish_reason: str | None
    prompt_tokens_openai: int | None
    completion_tokens_openai: int | None
    latency_ms: float
    captured_at: str
    provenance: str = "generated"
    # Filled in by driver.py after the call, from the generator + RouteCapture.
    intent: str | None = None
    phrasing: str | None = None
    query_text: str | None = None
    routed_agent: str | None = None
    routed_via: str | None = None
    route_status: str | None = None
    app_git_sha: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def classify_call_role(messages: list[dict]) -> str:
    """Identify which of the app's call sites produced `messages`, by the
    system prompt it uses. Anything not matching a known auxiliary prompt is
    an agent generation call."""
    if messages and messages[0].get("role") == "system":
        return _SYSTEM_PROMPT_ROLES.get(messages[0].get("content"), "agent")
    return "agent"


def _extract(response) -> tuple[str, str | None, int | None, int | None]:
    choice = response.choices[0]
    text = getattr(choice.message, "content", None) or ""
    finish = getattr(choice, "finish_reason", None)
    usage = getattr(response, "usage", None)
    p = getattr(usage, "prompt_tokens", None) if usage is not None else None
    c = getattr(usage, "completion_tokens", None) if usage is not None else None
    return text, finish if isinstance(finish, str) else None, p if isinstance(p, int) else None, c if isinstance(c, int) else None


class _RecordingCompletions:
    def __init__(self, inner, recorder: "Recorder"):
        self._inner = inner
        self._recorder = recorder

    def create(self, *args, **kwargs):
        t0 = time.perf_counter()
        response = self._inner.create(*args, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        self._recorder._record(kwargs, response, latency_ms)
        return response


class _RecordingChat:
    def __init__(self, inner_chat, recorder: "Recorder"):
        self._inner = inner_chat
        self.completions = _RecordingCompletions(inner_chat.completions, recorder)

    def __getattr__(self, name):  # anything else (e.g. future sub-APIs) passes through
        return getattr(self._inner, name)


class Recorder:
    """Context manager. While active, every chat.completions.create call on the
    app's shared client is recorded into `.records`. Thread-safe; the query
    context is a ContextVar so worker threads each see their own."""

    def __init__(self):
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()
        self._call_counts: dict[str, int] = {}
        self._original_chat = None

    def __enter__(self) -> "Recorder":
        self._original_chat = client.chat
        client.chat = _RecordingChat(self._original_chat, self)
        return self

    def __exit__(self, *exc):
        client.chat = self._original_chat
        return False

    def _record(self, kwargs: dict, response, latency_ms: float) -> None:
        ctx = QUERY_CONTEXT.get()
        query_id = ctx.query_id if ctx else "unknown"
        text, finish, p_tok, c_tok = _extract(response)
        with self._lock:
            call_index = self._call_counts.get(query_id, 0)
            self._call_counts[query_id] = call_index + 1
            self.records.append(CallRecord(
                record_id=f"{query_id}-c{call_index}",
                query_id=query_id,
                conversation_id=ctx.conversation_id if ctx else None,
                turn_index=ctx.turn_index if ctx else 0,
                call_index=call_index,
                call_role=classify_call_role(kwargs.get("messages", [])),
                model=kwargs.get("model", ""),
                messages=[dict(m) for m in kwargs.get("messages", [])],
                response_format=kwargs.get("response_format"),
                temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"),
                response_text=text,
                finish_reason=finish,
                prompt_tokens_openai=p_tok,
                completion_tokens_openai=c_tok,
                latency_ms=latency_ms,
                captured_at=datetime.now(timezone.utc).isoformat(),
            ))


_ROUTE_RE = re.compile(r"coordinator_route agent=(\S+) via=(\S+) status=(\S+)")


class RouteCapture(logging.Handler):
    """Captures CoordinatorAgent's `coordinator_route ...` log line per query so
    export can tag records with the agent that actually handled the query.
    Reads the app's existing log output; installs nothing in app code."""

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.routes: dict[str, dict] = {}
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        m = _ROUTE_RE.search(record.getMessage())
        if not m:
            return
        ctx = QUERY_CONTEXT.get()
        query_id = ctx.query_id if ctx else "unknown"
        with self._lock:
            self.routes[query_id] = {"agent": m.group(1), "via": m.group(2), "status": m.group(3)}

    @classmethod
    def installed(cls):
        """Context manager: install a RouteCapture on the app.agents logger for
        the duration of the block and yield it."""
        import contextlib

        @contextlib.contextmanager
        def _cm():
            handler = cls()
            logger = logging.getLogger("app.agents")
            previous_level = logger.level
            logger.addHandler(handler)
            if logger.level > logging.INFO or logger.level == logging.NOTSET:
                logger.setLevel(logging.INFO)
            try:
                yield handler
            finally:
                logger.removeHandler(handler)
                logger.setLevel(previous_level)
        return _cm()
