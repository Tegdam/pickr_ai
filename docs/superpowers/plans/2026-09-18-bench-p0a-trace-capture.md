# Bench P0a — Trace Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce versioned, validated trace files for workloads A/B/C and three multi-turn depth profiles by driving Pickr's real `CoordinatorAgent` and condensation path in-process with a seeded query generator, recording every OpenAI call it makes, and exporting in `vllm bench serve`'s custom-dataset format.

**Architecture:** A `bench/capture/` package with five stages behind one CLI: **generate** (seeded queries and conversations from the catalog CSVs) → **capture** (drive the app with an observe-only recorder around `app.openai_client.client.chat`, write raw JSONL) → **export** (raw → per-workload trace JSONL rendered through the Qwen2.5 chat template, plus meta with quantiles and hashes) → **langsmith** (pull real emitted prompts into the same raw format) → **validate** (generated-vs-real quantile report). `app/` is never modified.

**Tech Stack:** Python 3.12 (WSL2 Ubuntu), pytest, `transformers` (Qwen2.5 tokenizer only), `langsmith` SDK, pandas/numpy (already installed). No vLLM/SGLang in this phase.

**Spec:** `docs/superpowers/specs/2026-09-17-pickr-inference-benchmark-design-v2.md` — §3.2 (trace capture), §3.1 prediction 1 (re-evaluated at P0a exit), §4 (client format), §9 P0a exit criteria.

## Global Constraints

- **`app/` is read-only for this project.** The recorder observes; it never changes app behaviour. (spec §1 hard rule)
- **`bench/` dependencies never enter `requirements.txt` or `Dockerfile`.** They live in `bench/requirements.txt`. (spec §1)
- **Trace files are immutable once versioned.** A change is a new `_v<N+1>` file; export refuses to overwrite. (spec §3.2)
- **Sample size: 1,500–2,000 queries, captured once.** (spec §3.2)
- **Every trace record carries `provenance: generated|real`.** Real traffic is described as "developer testing traffic through the deployed app," never "production." (spec §3.2)
- **Prompt tokens counted with the Qwen2.5 tokenizer; OpenAI's `usage` counts kept alongside.** (spec §3.2)
- **Trace format = `vllm bench serve` custom dataset:** JSONL with `prompt` (string) and `output_tokens` (int); extra columns allowed and ignored by the tool. Prompts are pre-rendered through the Qwen2.5 chat template so the run uses the completions endpoint with `--skip-chat-template`. (verified against vLLM `main` `vllm/benchmarks/datasets/datasets.py::CustomDataset` on 2026-09-18; P0b re-verifies at the pin)
- **All commands run from the repo root inside WSL2** (`app/db.py` reads `data/*.csv` by relative path).
- **Git hygiene:** stage files by name, never `git add -A`; no Co-Authored-By trailer; commit on `benchmarking` only.

## File Structure

```
bench/
  __init__.py
  requirements.txt                 # transformers, langsmith, numpy, pandas, pytest (bench-only)
  README.md                        # how to run the five stages
  capture/
    __init__.py
    __main__.py                    # python -m bench.capture <stage>
    cli.py                         # argparse; one function per stage
    recorder.py                    # Recorder (observe-only wrapper), CallRecord, classify_call_role, RouteCapture
    generator.py                   # Catalog, GeneratedQuery, Conversation, generate_queries, generate_conversations
    driver.py                      # run_single_turn, run_conversation, capture_all (raw JSONL writer, resumable)
    tokens.py                      # QwenTokenizer: render(messages), count(text), template_sha256
    export.py                      # assign_workload, build_trace_rows, write_trace, write_meta, export_all
    langsmith_export.py            # fetch_llm_runs → raw CallRecords with provenance="real"
    validate.py                    # compare quantiles generated vs real → report dict + markdown
  traces/
    schemas/product_card.schema.json
    raw/                           # capture_<seed>.jsonl (committed; source of truth incl. responses)
    validation/                    # langsmith_<date>.jsonl (committed)
    chat_v1.jsonl, chat_v1.meta.json
    summarization_v1.jsonl, summarization_v1.meta.json
    structured_v1.jsonl, structured_v1.meta.json
    multiturn_shallow_v1.jsonl (+meta), multiturn_medium_v1.jsonl (+meta), multiturn_deep_v1.jsonl (+meta)
  docs/
    p0a-writeup.md
tests/bench/
  __init__.py
  conftest.py                      # fake OpenAI client + small catalog fixtures shared by bench tests
  test_recorder.py
  test_generator.py
  test_driver.py
  test_tokens.py
  test_export.py
  test_langsmith_export.py
  test_validate.py
  test_cli.py
```

Responsibilities are one-per-file. `driver.py` is the only module that imports `app.*` for execution; `recorder.py` imports `app.*` only for the system-prompt constants it classifies by. `export.py`, `validate.py`, `tokens.py` know nothing about the app.

## Raw capture record (the contract between capture and export)

One JSON object per LLM call, written by `driver.py`, read by `export.py` and produced identically by `langsmith_export.py`:

```json
{
  "record_id": "q000123-c2",
  "query_id": "q000123",
  "conversation_id": null,
  "turn_index": 0,
  "call_index": 2,
  "call_role": "agent",
  "intent": "recommendation_category_price",
  "phrasing": "keyword",
  "query_text": "Recommend a laptop under $900",
  "routed_agent": "ProductRecommendationAgent",
  "routed_via": "keyword",
  "route_status": "ok",
  "model": "gpt-3.5-turbo",
  "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
  "response_format": null,
  "temperature": null,
  "max_tokens": null,
  "response_text": "...",
  "finish_reason": "stop",
  "prompt_tokens_openai": 412,
  "completion_tokens_openai": 96,
  "latency_ms": 1432.7,
  "captured_at": "2026-09-19T10:11:12+00:00",
  "provenance": "generated",
  "app_git_sha": "a333eca4..."
}
```

`call_role` ∈ `{guardrail_input, guardrail_output, classifier, condense, agent}`. Moderation calls (`client.moderations.create`) are not chat completions and are not recorded; the writeup notes that each turn also makes two moderation calls.

---

### Task 1: Package scaffold, bench requirements, test fixtures

**Files:**
- Create: `bench/__init__.py`, `bench/capture/__init__.py`, `bench/requirements.txt`, `bench/README.md`
- Create: `tests/bench/__init__.py`, `tests/bench/conftest.py`
- Modify: `.gitignore` (append)

**Interfaces:**
- Produces: `tests/bench/conftest.py` fixtures `fake_chat` (records kwargs, returns canned responses keyed by call role) and `small_catalog` (patches `app.agents.load_*`) used by Tasks 2, 4, 6.

- [ ] **Step 1: Create the package files and requirements**

`bench/__init__.py` and `bench/capture/__init__.py`: empty.

`bench/requirements.txt`:
```
# Bench-only dependencies. Never merged into the app's requirements.txt (spec §1).
transformers>=4.45   # Qwen2.5 tokenizer + chat template rendering only
langsmith            # already an app dep; listed so bench installs standalone
numpy
pandas
pytest
```

`bench/README.md`:
```markdown
# bench/ — Pickr inference-serving benchmark

Separate deliverable from the app (see docs/superpowers/specs/2026-09-17-pickr-inference-benchmark-design-v2.md).

## P0a: trace capture

All commands from the repo root, inside WSL2, with `OPENAI_API_KEY` set:

    python -m bench.capture generate  --seed 20260919 --out bench/traces/raw/queries_20260919.json
    python -m bench.capture capture   --queries bench/traces/raw/queries_20260919.json --out bench/traces/raw/capture_20260919.jsonl --workers 4
    python -m bench.capture export    --raw bench/traces/raw/capture_20260919.jsonl --version 1
    python -m bench.capture langsmith --project <LANGSMITH_PROJECT> --out bench/traces/validation/langsmith_20260919.jsonl
    python -m bench.capture validate  --generated bench/traces/raw/capture_20260919.jsonl --real bench/traces/validation/langsmith_20260919.jsonl --out bench/docs/p0a-validation.md

Tests: `pytest tests/bench -q`
```

Append to `.gitignore`:
```
# Benchmark run artifacts (large, regenerable from configs); traces are committed on purpose
bench/results/
bench/figures/
```

- [ ] **Step 2: Write the shared test fixtures**

`tests/bench/conftest.py`:
```python
"""Fixtures shared by bench tests. Nothing here touches the network."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app import agents
from app.guardrails import INPUT_CLASSIFIER_SYSTEM_PROMPT, OUTPUT_CLASSIFIER_SYSTEM_PROMPT
from app.conversation import CONDENSE_SYSTEM_PROMPT
from app.agents import _INTENT_CLASSIFIER_SYSTEM_PROMPT
from app.models import Product, Review, StorePolicy


PRODUCTS = [
    Product(id="P1", name="Alpha Laptop", brand="Acme", category="laptop",
            price=500.0, description="Budget laptop.", stock=10, rating=4.5),
    Product(id="P2", name="Beta Laptop", brand="Acme", category="laptop",
            price=1200.0, description="Premium laptop.", stock=5, rating=4.8),
    Product(id="P3", name="Gamma Phone", brand="Zenith", category="smartphone",
            price=300.0, description="Entry phone.", stock=4, rating=4.0),
    Product(id="P4", name="Delta TV", brand="Zenith", category="smart_tv",
            price=900.0, description="4K TV.", stock=3, rating=3.9),
    Product(id="P5", name="Echo Speaker", brand="Acme", category="speaker",
            price=80.0, description="Bluetooth speaker.", stock=7, rating=4.1),
]
REVIEWS = [
    Review(product_id="P1", rating=5.0, text="Great value.", date="01-01-2025"),
    Review(product_id="P1", rating=4.0, text="Battery could be better.", date="02-01-2025"),
    Review(product_id="P3", rating=3.0, text="Fine for the price.", date="03-01-2025"),
]
POLICIES = [
    StorePolicy(policy_type="returns", description="Laptop Return Policy",
                conditions="Unopened.", timeframe="14"),
    StorePolicy(policy_type="warranty", description="Standard Warranty",
                conditions="Defects only.", timeframe="365"),
]


def _response(text, prompt_tokens=100, completion_tokens=20, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeCompletions:
    """Stands in for client.chat.completions. Answers by system prompt so the
    app's control flow (guardrails pass, classifier picks a category) works."""

    def __init__(self):
        self.calls = []
        self.classifier_category = "recommendation"

    def create(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs["messages"][0]["content"]
        if system == INPUT_CLASSIFIER_SYSTEM_PROMPT:
            return _response('{"is_injection": false, "is_off_topic": false}', 60, 12)
        if system == OUTPUT_CLASSIFIER_SYSTEM_PROMPT:
            return _response('{"is_hallucination": false}', 300, 8)
        if system == _INTENT_CLASSIFIER_SYSTEM_PROMPT:
            return _response('{"category": "%s"}' % self.classifier_category, 250, 6)
        if system == CONDENSE_SYSTEM_PROMPT:
            follow_up = kwargs["messages"][1]["content"].rsplit("Follow-up message: ", 1)[1]
            return _response("standalone: " + follow_up, 180, 15)
        return _response("agent answer about products", 400, 90)


@pytest.fixture
def fake_chat(monkeypatch):
    """Swap client.chat for a fake whose .completions.create records calls.
    Same patch point the app's own tests use (tests/test_agents.py::mock_openai)."""
    completions = FakeCompletions()
    fake = SimpleNamespace(completions=completions)
    monkeypatch.setattr(agents.client, "chat", fake)
    # moderation is a separate endpoint; never flag anything in tests
    fake_mod = MagicMock()
    fake_mod.create.return_value = SimpleNamespace(results=[SimpleNamespace(flagged=False)])
    monkeypatch.setattr(agents.client, "moderations", fake_mod)
    return completions


@pytest.fixture
def small_catalog(monkeypatch):
    monkeypatch.setattr(agents, "load_products", lambda: list(PRODUCTS))
    monkeypatch.setattr(agents, "load_reviews", lambda: list(REVIEWS))
    monkeypatch.setattr(agents, "load_store_policies", lambda: list(POLICIES))
    return SimpleNamespace(products=PRODUCTS, reviews=REVIEWS, policies=POLICIES)
```

`tests/bench/__init__.py`: empty.

- [ ] **Step 3: Verify the fixtures import and the app's own suite is untouched**

Run: `pytest tests/bench -q` (collects zero tests, no import errors) and `pytest tests -q -x`
Expected: bench collects 0 tests without error; app suite passes as before.

- [ ] **Step 4: Commit**

```bash
git add bench/__init__.py bench/capture/__init__.py bench/requirements.txt bench/README.md tests/bench/__init__.py tests/bench/conftest.py .gitignore
git commit -m "bench: scaffold the capture package and test fixtures"
```

---

### Task 2: Recorder — observe-only wrapper around the OpenAI client

**Files:**
- Create: `bench/capture/recorder.py`
- Test: `tests/bench/test_recorder.py`

**Interfaces:**
- Produces:
  - `@dataclass CallRecord` with the raw-record fields listed above (all except `intent`, `phrasing`, `query_text`, `routed_*`, `app_git_sha`, which `driver.py` fills in afterwards).
  - `QueryContext(query_id: str, conversation_id: str | None, turn_index: int)`; `QUERY_CONTEXT: ContextVar[QueryContext | None]`.
  - `classify_call_role(messages: list[dict]) -> str`.
  - `class Recorder` — context manager; `.records: list[CallRecord]`; thread-safe.
  - `class RouteCapture(logging.Handler)` — `.routes: dict[str, dict]` keyed by `query_id` with `agent`, `via`, `status`; installed on logger `app.agents`.

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_recorder.py`:
```python
import logging
import threading

from app import agents
from app.guardrails import INPUT_CLASSIFIER_SYSTEM_PROMPT, OUTPUT_CLASSIFIER_SYSTEM_PROMPT
from app.conversation import CONDENSE_SYSTEM_PROMPT
from app.agents import _INTENT_CLASSIFIER_SYSTEM_PROMPT

from bench.capture.recorder import (
    QUERY_CONTEXT, QueryContext, Recorder, RouteCapture, classify_call_role,
)


def _msgs(system, user="hi"):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def test_classify_call_role_by_system_prompt():
    assert classify_call_role(_msgs(INPUT_CLASSIFIER_SYSTEM_PROMPT)) == "guardrail_input"
    assert classify_call_role(_msgs(OUTPUT_CLASSIFIER_SYSTEM_PROMPT)) == "guardrail_output"
    assert classify_call_role(_msgs(_INTENT_CLASSIFIER_SYSTEM_PROMPT)) == "classifier"
    assert classify_call_role(_msgs(CONDENSE_SYSTEM_PROMPT)) == "condense"
    assert classify_call_role(_msgs("You are a helpful shopping assistant.")) == "agent"
    assert classify_call_role([{"role": "user", "content": "no system"}]) == "agent"


def test_recorder_records_and_passes_through(fake_chat):
    with Recorder() as rec:
        QUERY_CONTEXT.set(QueryContext("q1", None, 0))
        resp = agents.client.chat.completions.create(
            model="gpt-3.5-turbo", temperature=0,
            response_format={"type": "json_object"},
            messages=_msgs(INPUT_CLASSIFIER_SYSTEM_PROMPT, "is this ok?"),
        )
    assert resp.choices[0].message.content.startswith('{"is_injection"')
    assert len(rec.records) == 1
    r = rec.records[0]
    assert r.record_id == "q1-c0"
    assert r.query_id == "q1" and r.turn_index == 0 and r.call_index == 0
    assert r.call_role == "guardrail_input"
    assert r.model == "gpt-3.5-turbo"
    assert r.temperature == 0 and r.response_format == {"type": "json_object"}
    assert r.messages[1]["content"] == "is this ok?"
    assert r.response_text.startswith('{"is_injection"')
    assert r.finish_reason == "stop"
    assert r.prompt_tokens_openai == 60 and r.completion_tokens_openai == 12
    assert r.latency_ms >= 0
    assert r.provenance == "generated"


def test_recorder_restores_client_on_exit(fake_chat):
    before = agents.client.chat
    with Recorder():
        assert agents.client.chat is not before
    assert agents.client.chat is before


def test_call_index_increments_per_query_and_resets_across_queries(fake_chat):
    with Recorder() as rec:
        QUERY_CONTEXT.set(QueryContext("q1", None, 0))
        agents.client.chat.completions.create(model="m", messages=_msgs("a"))
        agents.client.chat.completions.create(model="m", messages=_msgs("b"))
        QUERY_CONTEXT.set(QueryContext("q2", "conv1", 3))
        agents.client.chat.completions.create(model="m", messages=_msgs("c"))
    ids = [r.record_id for r in rec.records]
    assert ids == ["q1-c0", "q1-c1", "q2-c0"]
    assert rec.records[2].conversation_id == "conv1" and rec.records[2].turn_index == 3


def test_recorder_is_thread_safe_and_context_is_per_thread(fake_chat):
    with Recorder() as rec:
        def work(qid):
            QUERY_CONTEXT.set(QueryContext(qid, None, 0))
            for _ in range(20):
                agents.client.chat.completions.create(model="m", messages=_msgs("x"))
        threads = [threading.Thread(target=work, args=(f"q{i}",)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
    assert len(rec.records) == 80
    per_query = {}
    for r in rec.records:
        per_query.setdefault(r.query_id, []).append(r.call_index)
    assert all(sorted(v) == list(range(20)) for v in per_query.values())


def test_recorder_without_context_uses_unknown_query_id(fake_chat):
    QUERY_CONTEXT.set(None)
    with Recorder() as rec:
        agents.client.chat.completions.create(model="m", messages=_msgs("x"))
    assert rec.records[0].query_id == "unknown"


def test_route_capture_parses_coordinator_log_line():
    cap = RouteCapture()
    logger = logging.getLogger("app.agents")
    logger.addHandler(cap)
    logger.setLevel(logging.INFO)
    try:
        QUERY_CONTEXT.set(QueryContext("q9", None, 0))
        logger.info(
            "coordinator_route agent=%s via=%s status=%s reason=%s elapsed_ms=%.1f query=%r",
            "ProductRecommendationAgent", "llm_fallback", "ok", None, 12.5, "hello",
        )
    finally:
        logger.removeHandler(cap)
    assert cap.routes["q9"] == {"agent": "ProductRecommendationAgent", "via": "llm_fallback", "status": "ok"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/bench/test_recorder.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.capture.recorder'`

- [ ] **Step 3: Implement the recorder**

`bench/capture/recorder.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/bench/test_recorder.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add bench/capture/recorder.py tests/bench/test_recorder.py
git commit -m "bench: add observe-only recorder around the app's OpenAI client"
```

---

### Task 3: Query and conversation generator

**Files:**
- Create: `bench/capture/generator.py`
- Test: `tests/bench/test_generator.py`

**Interfaces:**
- Produces:
  - `@dataclass GeneratedQuery(query_id: str, text: str, intent: str, phrasing: str)`; `phrasing ∈ {"keyword", "natural"}`.
  - `@dataclass Conversation(conversation_id: str, profile: str, turns: list[GeneratedQuery])`; `profile ∈ {"shallow", "medium", "deep"}`.
  - `class Catalog` built from `app.db.load_products/load_reviews/load_store_policies` (so tests patch `app.agents.load_*`? No — `Catalog.from_app()` imports the loaders from `app.db` directly; tests construct `Catalog(products, reviews, policies)` explicitly).
  - `generate_queries(catalog, n, seed, mix=DEFAULT_MIX) -> list[GeneratedQuery]`
  - `generate_conversations(catalog, n_per_profile, seed) -> list[Conversation]`
  - `DEFAULT_MIX: dict[str, float]` and `PROFILE_DEPTHS = {"shallow": (2, 3), "medium": (4, 5), "deep": (6, 8)}`
  - `to_json(queries, conversations) -> dict` / `from_json(d) -> tuple[list[GeneratedQuery], list[Conversation]]`

Intent names and the routing each is designed to hit (keyword rules from `app/agents.py::CoordinatorAgent.handle_query`):

| intent | keyword phrasing (routes by rule) | natural phrasing (no rule matches → LLM classifier) |
|---|---|---|
| `review` | "What do the reviews say about {product}?" | "What do customers think of {product}?" |
| `price_comparison` | "Which is cheaper, {p1} or {p2}?" | "Is {p1} much more expensive than {p2}?" |
| `comparison` | "Compare {p1} and {p2}." | "{p1} vs {p2} — which should I get?" |
| `recommendation_category_price` | "Recommend a {category} under ${price}" | "I need a {category}, budget is about ${price}" |
| `recommendation_brand` | "Recommend a {brand} {category}" | "Got anything from {brand}?" |
| `recommendation_browse` | "What do you recommend?" | "Just browsing, what's popular?" |
| `store_policy` | "What is your {policy_type} policy?" / "Can I return a {product}?" | "Do you price match?" |
| `stock` | "Is {product} in stock?" | "Do you have {product} right now?" |
| `capabilities` | "What can you do?" | — (keyword only; phrase list is tight) |

Follow-up templates (used by conversations; `{prev}` is resolved by the app's condense call, not here):

```
after recommendation_*: "What about something cheaper?", "Any of those from {brand}?",
                        "What do the reviews say about the first one?", "Is the first one in stock?"
after review:           "Is it in stock?", "Can I return it if I don't like it?", "Compare it with {p2}."
after comparison / price_comparison: "Which one is cheaper?", "What's the warranty on the second one?",
                        "What do reviews say about the first one?"
after store_policy:     "Does that apply to a {category}?", "What about exchanges?"
after stock:            "What do the reviews say about it?", "Recommend something similar under ${price}."
```

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_generator.py`:
```python
from collections import Counter

import pytest

from bench.capture.generator import (
    DEFAULT_MIX, PROFILE_DEPTHS, Catalog, Conversation, GeneratedQuery,
    from_json, generate_conversations, generate_queries, to_json,
)
from tests.bench.conftest import POLICIES, PRODUCTS, REVIEWS


@pytest.fixture
def catalog():
    return Catalog(products=PRODUCTS, reviews=REVIEWS, policies=POLICIES)


def test_generate_queries_is_deterministic_for_a_seed(catalog):
    a = generate_queries(catalog, n=50, seed=7)
    b = generate_queries(catalog, n=50, seed=7)
    c = generate_queries(catalog, n=50, seed=8)
    assert [q.text for q in a] == [q.text for q in b]
    assert [q.text for q in a] != [q.text for q in c]


def test_generate_queries_count_ids_and_mix(catalog):
    qs = generate_queries(catalog, n=200, seed=1)
    assert len(qs) == 200
    assert [q.query_id for q in qs] == [f"q{i:06d}" for i in range(200)]
    counts = Counter(q.intent for q in qs)
    for intent, share in DEFAULT_MIX.items():
        assert abs(counts[intent] / 200 - share) < 0.06, (intent, counts[intent])
    assert {q.phrasing for q in qs} <= {"keyword", "natural"}
    natural_share = sum(q.phrasing == "natural" for q in qs) / 200
    assert 0.10 <= natural_share <= 0.30


def test_slots_are_filled_from_the_catalog(catalog):
    qs = generate_queries(catalog, n=300, seed=3)
    names = {p.name for p in PRODUCTS}
    brands = {p.brand for p in PRODUCTS}
    for q in qs:
        assert "{" not in q.text and "}" not in q.text, q.text
    review_qs = [q for q in qs if q.intent == "review"]
    assert review_qs and all(any(n in q.text for n in names) for q in review_qs)
    # review queries only name products that actually have reviews
    reviewed = {p.name for p in PRODUCTS if any(r.product_id == p.id for r in REVIEWS)}
    assert all(any(n in q.text for n in reviewed) for q in review_qs)
    brand_qs = [q for q in qs if q.intent == "recommendation_brand"]
    assert brand_qs and all(any(b in q.text for b in brands) for q in brand_qs)


def test_keyword_phrasings_hit_the_intended_keyword_rule(catalog):
    qs = generate_queries(catalog, n=300, seed=5)
    for q in qs:
        if q.phrasing != "keyword":
            continue
        t = q.text.lower()
        if q.intent == "review":
            assert "review" in t
        elif q.intent == "price_comparison":
            assert "cheaper" in t or ("price" in t and any(w in t for w in ("compare", "difference", "cost")))
        elif q.intent == "comparison":
            assert "compare" in t
        elif q.intent == "stock":
            assert "stock" in t or "availab" in t
        elif q.intent == "capabilities":
            assert "what can you do" in t


def test_generate_conversations_profiles_and_depths(catalog):
    convs = generate_conversations(catalog, n_per_profile=5, seed=11)
    assert len(convs) == 15
    assert Counter(c.profile for c in convs) == {"shallow": 5, "medium": 5, "deep": 5}
    for c in convs:
        lo, hi = PROFILE_DEPTHS[c.profile]
        assert lo <= len(c.turns) <= hi
        assert c.turns[0].phrasing in ("keyword", "natural")
        assert all(t.query_id.startswith(c.conversation_id) for t in c.turns)
        assert [t.query_id for t in c.turns] == [f"{c.conversation_id}-t{i}" for i in range(len(c.turns))]
    assert convs == generate_conversations(catalog, n_per_profile=5, seed=11)


def test_json_round_trip(catalog):
    qs = generate_queries(catalog, n=10, seed=2)
    convs = generate_conversations(catalog, n_per_profile=1, seed=2)
    qs2, convs2 = from_json(to_json(qs, convs))
    assert qs2 == qs and convs2 == convs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/bench/test_generator.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the generator**

`bench/capture/generator.py`:
```python
"""Seeded query and conversation generator over Pickr's real catalog.

Templates are written to land on each of CoordinatorAgent's keyword rules
("keyword" phrasing) and, for most intents, to miss them so the LLM intent
classifier fallback fires ("natural" phrasing). Slots are filled only from
products/brands/categories/policies that exist in the catalog, so the app's
retrieval steps find real context.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from app.models import Product, Review, StorePolicy


@dataclass(frozen=True)
class GeneratedQuery:
    query_id: str
    text: str
    intent: str
    phrasing: str  # "keyword" | "natural"


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    profile: str  # "shallow" | "medium" | "deep"
    turns: tuple[GeneratedQuery, ...]


@dataclass
class Catalog:
    products: list[Product]
    reviews: list[Review]
    policies: list[StorePolicy]

    @classmethod
    def from_app(cls) -> "Catalog":
        from app.db import load_products, load_reviews, load_store_policies
        return cls(load_products(), load_reviews(), load_store_policies())

    @property
    def in_stock(self) -> list[Product]:
        return [p for p in self.products if p.name and p.stock and p.stock > 0]

    @property
    def reviewed(self) -> list[Product]:
        ids = {r.product_id for r in self.reviews}
        return [p for p in self.products if p.name and p.id in ids]

    @property
    def categories(self) -> list[str]:
        return sorted({p.category for p in self.products if p.category})

    @property
    def brands(self) -> list[str]:
        return sorted({p.brand for p in self.products if p.brand})

    @property
    def policy_types(self) -> list[str]:
        return sorted({p.policy_type for p in self.policies if p.policy_type})


DEFAULT_MIX: dict[str, float] = {
    "recommendation_category_price": 0.20,
    "recommendation_brand": 0.07,
    "recommendation_browse": 0.03,
    "review": 0.22,
    "comparison": 0.14,
    "price_comparison": 0.10,
    "store_policy": 0.14,
    "stock": 0.07,
    "capabilities": 0.03,
}
NATURAL_SHARE = 0.20  # fraction of queries using the no-keyword phrasing, where one exists

PROFILE_DEPTHS = {"shallow": (2, 3), "medium": (4, 5), "deep": (6, 8)}

# (keyword templates, natural templates). Slots: {product} {p1} {p2} {category}
# {category_text} {brand} {price} {policy_type_text}
TEMPLATES: dict[str, tuple[list[str], list[str]]] = {
    "review": (
        ["What do the reviews say about {product}?", "Summarize the reviews for {product}.",
         "Show me reviews of {product}."],
        ["What do customers think of {product}?", "Is {product} any good according to buyers?"],
    ),
    "price_comparison": (
        ["Which is cheaper, {p1} or {p2}?", "What's the price difference between {p1} and {p2}?",
         "Compare the price of {p1} and {p2}."],
        ["Is {p1} much more expensive than {p2}?"],
    ),
    "comparison": (
        ["Compare {p1} and {p2}.", "How does {p1} compare to {p2}?"],
        ["{p1} vs {p2} — which should I get?"],
    ),
    "recommendation_category_price": (
        ["Recommend a {category_text} under ${price}", "Can you recommend a {category_text} below ${price}?",
         "Suggest a good {category_text} for less than ${price}"],
        ["I need a {category_text}, budget is about ${price}"],
    ),
    "recommendation_brand": (
        ["Recommend a {brand} {category_text}", "Which {brand} {category_text} would you suggest?"],
        ["Got anything from {brand}?"],
    ),
    "recommendation_browse": (
        ["What do you recommend?", "What products do you carry?"],
        ["Just browsing, what's popular?"],
    ),
    "store_policy": (
        ["What is your {policy_type_text} policy?", "Can I return a {product}?",
         "What's the warranty on {product}?"],
        ["Do you price match?", "How long does delivery take?"],
    ),
    "stock": (
        ["Is {product} in stock?", "How many {product} are available?"],
        ["Do you have {product} right now?"],
    ),
    "capabilities": (
        ["What can you do?", "What kind of questions can you answer?"],
        [],
    ),
}

FOLLOW_UPS: dict[str, list[str]] = {
    "recommendation": ["What about something cheaper?", "Any of those from {brand}?",
                       "What do the reviews say about the first one?", "Is the first one in stock?"],
    "review": ["Is it in stock?", "Can I return it if I don't like it?", "Compare it with {p2}."],
    "comparison": ["Which one is cheaper?", "What's the warranty on the second one?",
                   "What do reviews say about the first one?"],
    "price_comparison": ["Which one is cheaper?", "What's the warranty on the second one?"],
    "store_policy": ["Does that apply to a {category_text}?", "What about exchanges?"],
    "stock": ["What do the reviews say about it?", "Recommend something similar under ${price}."],
    "capabilities": ["Recommend a {category_text} under ${price}"],
}


def _fill(template: str, catalog: Catalog, rng: random.Random) -> str:
    in_stock = catalog.in_stock
    reviewed = catalog.reviewed or in_stock
    p1, p2 = rng.sample(in_stock, 2)
    prices = sorted(p.price for p in in_stock if p.price is not None)
    price = int(rng.choice(prices[len(prices) // 4:]) // 10 * 10) if prices else 500
    category = rng.choice(catalog.categories) if catalog.categories else "laptop"
    policy_type = rng.choice(catalog.policy_types) if catalog.policy_types else "returns"
    return template.format(
        product=rng.choice(reviewed).name if "{product}" in template and "review" in template.lower()
        else rng.choice(in_stock).name,
        p1=p1.name, p2=p2.name,
        category=category, category_text=category.replace("_", " "),
        brand=rng.choice(catalog.brands) if catalog.brands else "Acme",
        price=price,
        policy_type_text=policy_type.replace("_", " "),
    )


def _fill_review(template: str, catalog: Catalog, rng: random.Random) -> str:
    """Review queries must name a product that has reviews, or the agent
    short-circuits without an LLM call."""
    pool = catalog.reviewed or catalog.in_stock
    return template.format(product=rng.choice(pool).name)


def _make_query(query_id: str, intent: str, catalog: Catalog, rng: random.Random) -> GeneratedQuery:
    keyword, natural = TEMPLATES[intent]
    use_natural = bool(natural) and rng.random() < NATURAL_SHARE
    template = rng.choice(natural if use_natural else keyword)
    text = _fill_review(template, catalog, rng) if intent == "review" else _fill(template, catalog, rng)
    return GeneratedQuery(query_id, text, intent, "natural" if use_natural else "keyword")


def generate_queries(catalog: Catalog, n: int, seed: int, mix: dict[str, float] = DEFAULT_MIX) -> list[GeneratedQuery]:
    rng = random.Random(seed)
    intents = list(mix)
    weights = [mix[i] for i in intents]
    # Allocate counts proportionally then shuffle, so the mix is exact rather than sampled.
    counts = {i: int(n * w) for i, w in zip(intents, weights)}
    for i in intents[: n - sum(counts.values())]:
        counts[i] += 1
    schedule = [i for i in intents for _ in range(counts[i])]
    rng.shuffle(schedule)
    return [_make_query(f"q{k:06d}", intent, catalog, rng) for k, intent in enumerate(schedule)]


def _family(intent: str) -> str:
    return "recommendation" if intent.startswith("recommendation") else intent


def generate_conversations(catalog: Catalog, n_per_profile: int, seed: int) -> list[Conversation]:
    rng = random.Random(seed)
    openers = [i for i in DEFAULT_MIX if i != "capabilities"]
    convs: list[Conversation] = []
    for profile, (lo, hi) in PROFILE_DEPTHS.items():
        for k in range(n_per_profile):
            cid = f"conv-{profile}-{k:04d}"
            depth = rng.randint(lo, hi)
            opener_intent = rng.choice(openers)
            turns = [_make_query(f"{cid}-t0", opener_intent, catalog, rng)]
            family = _family(opener_intent)
            for t in range(1, depth):
                template = rng.choice(FOLLOW_UPS[family])
                text = _fill(template, catalog, rng)
                turns.append(GeneratedQuery(f"{cid}-t{t}", text, f"followup_{family}", "natural"))
                # Follow-ups drift the conversation's topic the way real ones do.
                family = rng.choice(list(FOLLOW_UPS))
            convs.append(Conversation(cid, profile, tuple(turns)))
    return convs


def to_json(queries: list[GeneratedQuery], conversations: list[Conversation]) -> dict:
    return {
        "queries": [asdict(q) for q in queries],
        "conversations": [
            {"conversation_id": c.conversation_id, "profile": c.profile, "turns": [asdict(t) for t in c.turns]}
            for c in conversations
        ],
    }


def from_json(d: dict) -> tuple[list[GeneratedQuery], list[Conversation]]:
    qs = [GeneratedQuery(**q) for q in d["queries"]]
    convs = [Conversation(c["conversation_id"], c["profile"], tuple(GeneratedQuery(**t) for t in c["turns"]))
             for c in d["conversations"]]
    return qs, convs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/bench/test_generator.py -q`
Expected: 6 passed. If `test_generate_queries_count_ids_and_mix` fails on the natural share, the mix has few intents with natural variants at that seed — the bound is 0.10–0.30 with `NATURAL_SHARE = 0.20`, so check that `capabilities` (no natural variant) is the only exception before adjusting the bound.

- [ ] **Step 5: Commit**

```bash
git add bench/capture/generator.py tests/bench/test_generator.py
git commit -m "bench: add seeded query and conversation generator over the catalog"
```

---

### Task 4: Driver — run queries through the app and write raw JSONL

**Files:**
- Create: `bench/capture/driver.py`
- Test: `tests/bench/test_driver.py`

**Interfaces:**
- Consumes: `Recorder`, `RouteCapture`, `QUERY_CONTEXT`, `QueryContext`, `CallRecord` (Task 2); `GeneratedQuery`, `Conversation` (Task 3).
- Produces:
  - `run_single_turn(coordinator, query: GeneratedQuery, recorder, routes) -> list[CallRecord]`
  - `run_conversation(coordinator, conv: Conversation, recorder, routes) -> list[CallRecord]` — replicates `app.conversation.handle_conversational_query` **without the database**: in-memory `(role, content)` history trimmed to the last `HISTORY_WINDOW` rows, `condense_query`, `coordinator.handle_query(UserQuery(query=resolved, raw_query=raw))`.
  - `capture_all(queries, conversations, out_path, workers=4, app_git_sha=None) -> int` — resumable: skips `query_id`s / `conversation_id`s already present in `out_path`; appends one JSON line per record; returns records written.
  - `read_raw(path) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_driver.py`:
```python
import json

from app import agents
from app.conversation import HISTORY_WINDOW

from bench.capture.generator import Conversation, GeneratedQuery
from bench.capture.driver import capture_all, read_raw, run_conversation, run_single_turn
from bench.capture.recorder import Recorder, RouteCapture


def _q(qid, text, intent="review", phrasing="keyword"):
    return GeneratedQuery(qid, text, intent, phrasing)


def test_single_turn_records_guardrail_agent_guardrail(fake_chat, small_catalog):
    coordinator = agents.CoordinatorAgent()
    with Recorder() as rec, RouteCapture.installed() as routes:
        records = run_single_turn(coordinator, _q("q1", "What do the reviews say about Alpha Laptop?"), rec, routes)
    roles = [r.call_role for r in records]
    assert roles == ["guardrail_input", "agent", "guardrail_output"]
    assert all(r.query_id == "q1" and r.routed_agent == "ReviewSummarizationAgent" for r in records)
    assert records[0].routed_via == "keyword" and records[0].route_status == "ok"
    assert records[0].intent == "review" and records[0].phrasing == "keyword"
    assert records[0].query_text == "What do the reviews say about Alpha Laptop?"


def test_natural_phrasing_goes_through_classifier(fake_chat, small_catalog):
    fake_chat.classifier_category = "review"
    coordinator = agents.CoordinatorAgent()
    with Recorder() as rec, RouteCapture.installed() as routes:
        records = run_single_turn(coordinator, _q("q2", "What do customers think of Alpha Laptop?", phrasing="natural"), rec, routes)
    assert [r.call_role for r in records] == ["guardrail_input", "classifier", "agent", "guardrail_output"]
    assert records[0].routed_via == "llm_fallback"


def test_non_llm_agent_records_only_guardrail_input(fake_chat, small_catalog):
    coordinator = agents.CoordinatorAgent()
    with Recorder() as rec, RouteCapture.installed() as routes:
        records = run_single_turn(coordinator, _q("q3", "Is Alpha Laptop in stock?", intent="stock"), rec, routes)
    assert [r.call_role for r in records] == ["guardrail_input"]
    assert records[0].routed_agent == "StockAvailabilityAgent"


def test_conversation_condenses_from_turn_one_and_windows_history(fake_chat, small_catalog):
    turns = tuple(_q(f"c1-t{i}", t, intent="followup_review", phrasing="natural") for i, t in enumerate([
        "What do the reviews say about Alpha Laptop?",
        "Is it in stock?", "Can I return it?", "What about the warranty?", "Is Beta Laptop in stock?",
    ]))
    turns = (_q("c1-t0", turns[0].text),) + turns[1:]
    conv = Conversation("c1", "medium", turns)
    coordinator = agents.CoordinatorAgent()
    with Recorder() as rec, RouteCapture.installed() as routes:
        records = run_conversation(coordinator, conv, rec, routes)
    by_turn = {}
    for r in records:
        by_turn.setdefault(r.turn_index, []).append(r)
    assert set(by_turn) == {0, 1, 2, 3, 4}
    assert by_turn[0][0].call_role == "guardrail_input"          # no condense on turn 0
    assert all(by_turn[t][0].call_role == "condense" for t in (1, 2, 3, 4))
    assert all(r.conversation_id == "c1" for r in records)
    # the condense prompt at turn 4 carries at most HISTORY_WINDOW rows of transcript
    transcript = by_turn[4][0].messages[1]["content"]
    assert transcript.count("\nCustomer: ") + transcript.count("\nAssistant: ") + transcript.startswith("Conversation so far:\nCustomer") <= HISTORY_WINDOW + 1
    # the resolved (condensed) query is what the coordinator saw
    assert by_turn[1][1].call_role == "guardrail_input"
    assert by_turn[1][1].messages[1]["content"] == "Is it in stock?"  # guardrail input sees the RAW query
    assert by_turn[1][0].query_text == "Is it in stock?"


def test_capture_all_writes_jsonl_and_resumes(fake_chat, small_catalog, tmp_path):
    out = tmp_path / "raw.jsonl"
    qs = [_q("q1", "What do the reviews say about Alpha Laptop?"), _q("q2", "Is Alpha Laptop in stock?", intent="stock")]
    convs = [Conversation("c1", "shallow", (_q("c1-t0", "What do the reviews say about Alpha Laptop?"),
                                              _q("c1-t1", "Is it in stock?", "followup_review", "natural")))]
    n1 = capture_all(qs, convs, out, workers=2, app_git_sha="abc")
    rows = read_raw(out)
    assert n1 == len(rows) == 3 + 1 + (1 + 1)   # q1: 3 calls, q2: 1, c1: t0 3 + t1 (condense+guardrail_input)... see below
    assert all(r["app_git_sha"] == "abc" and r["provenance"] == "generated" for r in rows)
    n2 = capture_all(qs, convs, out, workers=2, app_git_sha="abc")
    assert n2 == 0 and len(read_raw(out)) == len(rows)
    # every line is valid JSON with the contract's keys
    required = {"record_id", "query_id", "conversation_id", "turn_index", "call_index", "call_role", "intent",
                "phrasing", "query_text", "routed_agent", "routed_via", "route_status", "model", "messages",
                "response_format", "temperature", "max_tokens", "response_text", "finish_reason",
                "prompt_tokens_openai", "completion_tokens_openai", "latency_ms", "captured_at", "provenance", "app_git_sha"}
    for line in out.read_text().splitlines():
        assert required <= set(json.loads(line))
```

Fix the arithmetic in `test_capture_all_writes_jsonl_and_resumes` before running: `q1` → 3 records (guardrail_input, agent, guardrail_output); `q2` (stock, no LLM agent) → 1; `c1-t0` review → 3; `c1-t1` "Is it in stock?" → condense + guardrail_input = 2 (the fake condense returns "standalone: Is it in stock?", which routes to stock, no LLM). Total 9. Replace the expression with `assert n1 == len(rows) == 9`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/bench/test_driver.py -q`
Expected: FAIL with `ModuleNotFoundError` (and `RouteCapture.installed` missing — added in step 3).

- [ ] **Step 3: Implement the driver, and add `RouteCapture.installed()` to the recorder**

Add to `bench/capture/recorder.py` inside `class RouteCapture`:
```python
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
```

`bench/capture/driver.py`:
```python
"""Drives Pickr's real request path in-process and writes raw call records.

Single-turn queries go straight through CoordinatorAgent.handle_query.
Conversations replicate app.conversation.handle_conversational_query exactly
— condense against the last HISTORY_WINDOW rows, route the resolved query,
append the exchange — but with an in-memory history instead of RDS, so no
database is needed and nothing is persisted to the app's tables.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.agents import CoordinatorAgent
from app.conversation import HISTORY_WINDOW, condense_query
from app.models import UserQuery

from .generator import Conversation, GeneratedQuery
from .recorder import QUERY_CONTEXT, CallRecord, QueryContext, Recorder, RouteCapture


def _stamp(records: list[CallRecord], q: GeneratedQuery, route: dict | None, app_git_sha: str | None) -> list[CallRecord]:
    for r in records:
        r.intent = q.intent
        r.phrasing = q.phrasing
        r.query_text = q.text
        r.routed_agent = route.get("agent") if route else None
        r.routed_via = route.get("via") if route else None
        r.route_status = route.get("status") if route else None
        r.app_git_sha = app_git_sha
    return records


def _records_for(recorder: Recorder, query_id: str) -> list[CallRecord]:
    with recorder._lock:
        return [r for r in recorder.records if r.query_id == query_id]


def run_single_turn(coordinator: CoordinatorAgent, query: GeneratedQuery, recorder: Recorder,
                    routes: RouteCapture, app_git_sha: str | None = None) -> list[CallRecord]:
    QUERY_CONTEXT.set(QueryContext(query.query_id, None, 0))
    coordinator.handle_query(UserQuery(query=query.text))
    return _stamp(_records_for(recorder, query.query_id), query, routes.routes.get(query.query_id), app_git_sha)


def run_conversation(coordinator: CoordinatorAgent, conv: Conversation, recorder: Recorder,
                     routes: RouteCapture, app_git_sha: str | None = None) -> list[CallRecord]:
    history: list[tuple[str, str]] = []
    out: list[CallRecord] = []
    for turn_index, turn in enumerate(conv.turns):
        QUERY_CONTEXT.set(QueryContext(turn.query_id, conv.conversation_id, turn_index))
        window = history[-HISTORY_WINDOW:]                     # load_history returns the last HISTORY_WINDOW rows
        resolved = condense_query(window, turn.text)           # no-op on turn 0 (empty history)
        result = coordinator.handle_query(UserQuery(query=resolved, raw_query=turn.text))
        history.append(("user", turn.text))                    # save_exchange persists the RAW query
        history.append(("assistant", result["response"]))
        out.extend(_stamp(_records_for(recorder, turn.query_id), turn, routes.routes.get(turn.query_id), app_git_sha))
    return out


def read_raw(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _done_ids(path: Path) -> set[str]:
    done = set()
    for r in read_raw(path):
        done.add(r["conversation_id"] or r["query_id"])
    return done


def capture_all(queries: list[GeneratedQuery], conversations: list[Conversation], out_path: Path,
                workers: int = 4, app_git_sha: str | None = None) -> int:
    """Run everything not already in out_path; append records as each unit
    finishes so an interrupted capture resumes where it stopped."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids(out_path)
    pending_q = [q for q in queries if q.query_id not in done]
    pending_c = [c for c in conversations if c.conversation_id not in done]
    coordinator = CoordinatorAgent()
    written = 0
    with Recorder() as recorder, RouteCapture.installed() as routes, out_path.open("a", encoding="utf-8") as fh:
        def unit_q(q):
            return run_single_turn(coordinator, q, recorder, routes, app_git_sha)

        def unit_c(c):
            return run_conversation(coordinator, c, recorder, routes, app_git_sha)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(unit_q, q) for q in pending_q] + [pool.submit(unit_c, c) for c in pending_c]
            for fut in futures:
                for rec in fut.result():
                    fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                    written += 1
                fh.flush()
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/bench/test_driver.py tests/bench/test_recorder.py -q`
Expected: all pass. If `test_conversation_condenses...` fails on the window assertion, print `transcript` and count rows directly: at turn 4 the history holds 8 rows and the window must show only the last 6.

- [ ] **Step 5: Commit**

```bash
git add bench/capture/driver.py bench/capture/recorder.py tests/bench/test_driver.py
git commit -m "bench: add capture driver replicating the app's turn flow without the database"
```

---

### Task 5: Qwen tokenizer wrapper

**Files:**
- Create: `bench/capture/tokens.py`
- Test: `tests/bench/test_tokens.py`

**Interfaces:**
- Produces: `class QwenTokenizer` with `render(messages: list[dict]) -> str` (chat template applied, `add_generation_prompt=True`), `count(text: str) -> int`, properties `model_id`, `revision`, `template_sha256`; `QwenTokenizer.load(model_id="Qwen/Qwen2.5-3B-Instruct", revision=None)`; constructor accepts any object with `apply_chat_template` and `__call__` so tests inject a stub.

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_tokens.py`:
```python
import hashlib
import os

import pytest

from bench.capture.tokens import QwenTokenizer


class StubHF:
    chat_template = "{% for m in messages %}<|{{m.role}}|>{{m.content}}{% endfor %}<|assistant|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "".join(f"<|{m['role']}|>{m['content']}" for m in messages) + "<|assistant|>"

    def __call__(self, text):
        class R:
            input_ids = text.split()
        return R()


def test_render_and_count_with_stub():
    tok = QwenTokenizer(StubHF(), model_id="stub", revision="r1")
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello there"}]
    assert tok.render(msgs) == "<|system|>sys<|user|>hello there<|assistant|>"
    assert tok.count("a b c") == 3
    assert tok.model_id == "stub" and tok.revision == "r1"
    assert tok.template_sha256 == hashlib.sha256(StubHF.chat_template.encode()).hexdigest()


@pytest.mark.skipif(not os.environ.get("BENCH_HF_TESTS"), reason="needs the Qwen tokenizer in the HF cache; set BENCH_HF_TESTS=1")
def test_real_qwen_template_has_chatml_markers():
    tok = QwenTokenizer.load()
    text = tok.render([{"role": "system", "content": "S"}, {"role": "user", "content": "U"}])
    assert text.startswith("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n")
    assert tok.count(text) > 0 and tok.revision
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/bench/test_tokens.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`bench/capture/tokens.py`:
```python
"""Qwen2.5 tokenizer wrapper: renders messages through the model's own chat
template and counts tokens the way the benchmark's engines will.

Traces are exported pre-rendered so `vllm bench serve` can send raw text to
the completions endpoint with --skip-chat-template. That removes the
cross-engine chat-template parity risk (spec §4) by construction: both
engines receive byte-identical prompts.
"""
from __future__ import annotations

import hashlib


class QwenTokenizer:
    def __init__(self, hf_tokenizer, model_id: str, revision: str | None):
        self._tok = hf_tokenizer
        self.model_id = model_id
        self.revision = revision

    @classmethod
    def load(cls, model_id: str = "Qwen/Qwen2.5-3B-Instruct", revision: str | None = None) -> "QwenTokenizer":
        from huggingface_hub import model_info
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        resolved = revision or model_info(model_id).sha  # pin the exact commit for meta.json
        return cls(tok, model_id, resolved)

    def render(self, messages: list[dict]) -> str:
        return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count(self, text: str) -> int:
        return len(self._tok(text).input_ids)

    @property
    def template_sha256(self) -> str:
        template = getattr(self._tok, "chat_template", "") or ""
        return hashlib.sha256(template.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/bench/test_tokens.py -q` (1 passed, 1 skipped); then once, in WSL with network: `BENCH_HF_TESTS=1 pytest tests/bench/test_tokens.py -q` → 2 passed (this also populates the HF cache the export step needs).

- [ ] **Step 5: Commit**

```bash
git add bench/capture/tokens.py tests/bench/test_tokens.py
git commit -m "bench: add Qwen chat-template renderer and token counter"
```

---

### Task 6: Export — raw records → workload trace files + meta

**Files:**
- Create: `bench/capture/export.py`, `bench/traces/schemas/product_card.schema.json`
- Test: `tests/bench/test_export.py`

**Interfaces:**
- Consumes: raw record dicts (`read_raw`), `QwenTokenizer` (Task 5).
- Produces:
  - `assign_workload(record: dict) -> str | None` — `"B"` if `routed_agent == "ReviewSummarizationAgent"` and `call_role in {"agent", "guardrail_output"}`; `None` for `condense` (multi-turn files only); else `"A"`.
  - `STRUCTURED_AGENTS = {"ProductRecommendationAgent", "ProductComparisonAgent"}`; `structured_messages(messages, schema: dict) -> list[dict]` appends the JSON instruction + schema to the last user message.
  - `build_rows(records, tokenizer, workload, schema=None) -> list[dict]` — one trace row per record: `prompt`, `output_tokens`, plus `record_id, query_id, conversation_id, turn_index, call_role, workload, routed_agent, intent, phrasing, app_temperature, app_response_format, prompt_tokens_qwen, prompt_tokens_openai, output_tokens_openai, provenance, schema_file`.
  - `write_trace(rows, path) -> str` (sha256 of the file; refuses if `path` exists), `write_meta(rows, path, *, tokenizer, seed, app_git_sha, raw_path, trace_sha256, extra) -> dict`.
  - `quantiles(values) -> dict` with `p10 p50 p90 p99 mean n`.
  - `export_all(raw_path, out_dir, version, tokenizer, seed, app_git_sha) -> list[Path]` — writes `chat`, `summarization`, `structured`, `multiturn_{shallow,medium,deep}` files.

- [ ] **Step 1: Create the product-card schema**

`bench/traces/schemas/product_card.schema.json`:
```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "ProductCards",
  "type": "object",
  "properties": {
    "cards": {
      "type": "array",
      "minItems": 1,
      "maxItems": 5,
      "items": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "brand": {"type": "string"},
          "category": {"type": "string"},
          "price_usd": {"type": "number"},
          "rating": {"type": "number", "minimum": 0, "maximum": 5},
          "why": {"type": "string", "maxLength": 240}
        },
        "required": ["name", "brand", "price_usd", "why"],
        "additionalProperties": false
      }
    }
  },
  "required": ["cards"],
  "additionalProperties": false
}
```

- [ ] **Step 2: Write the failing tests**

`tests/bench/test_export.py`:
```python
import hashlib
import json

import pytest

from bench.capture.export import (
    STRUCTURED_AGENTS, assign_workload, build_rows, export_all, quantiles,
    structured_messages, write_meta, write_trace,
)
from bench.capture.tokens import QwenTokenizer
from tests.bench.test_tokens import StubHF


def _rec(**over):
    base = dict(record_id="q1-c0", query_id="q1", conversation_id=None, turn_index=0, call_index=0,
                call_role="agent", intent="review", phrasing="keyword", query_text="q",
                routed_agent="ReviewSummarizationAgent", routed_via="keyword", route_status="ok",
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "user text"}],
                response_format=None, temperature=None, max_tokens=None,
                response_text="one two three four", finish_reason="stop",
                prompt_tokens_openai=10, completion_tokens_openai=4, latency_ms=1.0,
                captured_at="t", provenance="generated", app_git_sha="abc")
    base.update(over)
    return base


@pytest.fixture
def tok():
    return QwenTokenizer(StubHF(), "stub", "r1")


def test_assign_workload():
    assert assign_workload(_rec()) == "B"
    assert assign_workload(_rec(call_role="guardrail_output")) == "B"
    assert assign_workload(_rec(call_role="guardrail_input")) == "A"
    assert assign_workload(_rec(routed_agent="ProductRecommendationAgent")) == "A"
    assert assign_workload(_rec(call_role="classifier", routed_agent="ProductRecommendationAgent")) == "A"
    assert assign_workload(_rec(call_role="condense", conversation_id="c1")) is None


def test_build_rows_renders_prompt_and_counts_output_with_qwen(tok):
    rows = build_rows([_rec()], tok, workload="B")
    assert len(rows) == 1
    r = rows[0]
    assert r["prompt"] == "<|system|>sys<|user|>user text<|assistant|>"
    assert r["output_tokens"] == 4 and r["output_tokens_openai"] == 4
    assert r["prompt_tokens_qwen"] == tok.count(r["prompt"]) and r["prompt_tokens_openai"] == 10
    assert r["workload"] == "B" and r["call_role"] == "agent" and r["record_id"] == "q1-c0"
    assert r["app_temperature"] is None and r["app_response_format"] is None
    assert r["provenance"] == "generated" and r["schema_file"] is None


def test_build_rows_drops_empty_responses(tok):
    rows = build_rows([_rec(response_text="")], tok, workload="B")
    assert rows == []


def test_structured_messages_appends_schema_to_last_user_turn():
    schema = {"type": "object", "properties": {"cards": {}}}
    msgs = structured_messages(_rec()["messages"], schema)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1]["content"].startswith("user text\n\nRespond with a JSON object")
    assert json.dumps(schema) in msgs[1]["content"]


def test_write_trace_is_immutable_and_returns_sha(tmp_path, tok):
    rows = build_rows([_rec()], tok, workload="B")
    path = tmp_path / "summarization_v1.jsonl"
    sha = write_trace(rows, path)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text().splitlines()[0])["prompt"].startswith("<|system|>")
    with pytest.raises(FileExistsError):
        write_trace(rows, path)


def test_quantiles():
    q = quantiles([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert q["n"] == 10 and q["p50"] == 5.5 and q["mean"] == 5.5 and q["p10"] < q["p90"]
    assert quantiles([]) == {"n": 0, "p10": None, "p50": None, "p90": None, "p99": None, "mean": None}


def test_write_meta_reports_by_call_role(tmp_path, tok):
    rows = build_rows([_rec(), _rec(record_id="q1-c1", call_index=1, call_role="guardrail_output")], tok, workload="B")
    meta = write_meta(rows, tmp_path / "m.json", tokenizer=tok, seed=5, app_git_sha="abc",
                      raw_path="raw.jsonl", trace_sha256="deadbeef", extra={"workload": "B"})
    assert meta["workload"] == "B" and meta["seed"] == 5 and meta["trace_sha256"] == "deadbeef"
    assert meta["tokenizer"] == {"model_id": "stub", "revision": "r1", "template_sha256": tok.template_sha256}
    assert meta["counts"]["total"] == 2 and meta["counts"]["by_provenance"] == {"generated": 2}
    assert set(meta["by_call_role"]) == {"agent", "guardrail_output"}
    assert meta["by_call_role"]["agent"]["prompt_tokens_qwen"]["n"] == 1
    assert meta["prompt_tokens_qwen"]["n"] == 2 and meta["output_tokens"]["n"] == 2
    assert json.loads((tmp_path / "m.json").read_text()) == meta


def test_export_all_writes_every_workload_file(tmp_path, tok):
    raw = tmp_path / "raw.jsonl"
    records = [
        _rec(),                                                                         # B agent
        _rec(record_id="q2-c0", query_id="q2", routed_agent="ProductRecommendationAgent",
             intent="recommendation_brand", call_role="agent"),                          # A agent -> also C
        _rec(record_id="q2-c1", query_id="q2", routed_agent="ProductRecommendationAgent",
             call_role="guardrail_input", call_index=1),                                 # A
        _rec(record_id="c-shallow-0-t1-c0", query_id="c-shallow-0-t1", conversation_id="conv-shallow-0000",
             turn_index=1, call_role="condense", intent="followup_review", phrasing="natural"),  # multiturn shallow
        _rec(record_id="c-deep-0-t3-c0", query_id="c-deep-0-t3", conversation_id="conv-deep-0000",
             turn_index=3, call_role="condense"),                                        # multiturn deep
    ]
    raw.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out = tmp_path / "traces"
    (out / "schemas").mkdir(parents=True)
    (out / "schemas" / "product_card.schema.json").write_text(json.dumps({"type": "object"}))
    paths = export_all(raw, out, version=1, tokenizer=tok, seed=5, app_git_sha="abc")
    names = sorted(p.name for p in paths)
    assert names == sorted([
        "chat_v1.jsonl", "chat_v1.meta.json", "summarization_v1.jsonl", "summarization_v1.meta.json",
        "structured_v1.jsonl", "structured_v1.meta.json",
        "multiturn_shallow_v1.jsonl", "multiturn_shallow_v1.meta.json",
        "multiturn_medium_v1.jsonl", "multiturn_medium_v1.meta.json",
        "multiturn_deep_v1.jsonl", "multiturn_deep_v1.meta.json",
    ])
    chat = [json.loads(l) for l in (out / "chat_v1.jsonl").read_text().splitlines()]
    assert [r["record_id"] for r in chat] == ["q2-c0", "q2-c1"]
    structured = [json.loads(l) for l in (out / "structured_v1.jsonl").read_text().splitlines()]
    assert [r["record_id"] for r in structured] == ["q2-c0"]
    assert structured[0]["schema_file"] == "schemas/product_card.schema.json"
    assert "Respond with a JSON object" in structured[0]["prompt"]
    shallow = [json.loads(l) for l in (out / "multiturn_shallow_v1.jsonl").read_text().splitlines()]
    assert shallow[0]["record_id"] == "c-shallow-0-t1-c0" and shallow[0]["turn_index"] == 1
    medium = (out / "multiturn_medium_v1.jsonl").read_text()
    assert medium == ""   # no medium conversations in this fixture; file still written (empty) with meta n=0
    meta = json.loads((out / "structured_v1.meta.json").read_text())
    assert meta["workload"] == "C" and meta["source_workload"] == "A" and meta["schema_sha256"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/bench/test_export.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement export**

`bench/capture/export.py`:
```python
"""Raw capture records -> versioned trace files in vllm bench serve's custom
dataset format (JSONL: `prompt`, `output_tokens`, extra columns ignored by
the tool) plus a meta sidecar with distributions and provenance.

Workload mapping (spec §3.2):
  A  interactive: every call on the interactive path except B's
  B  summarisation: ReviewSummarizationAgent's agent + guardrail_output calls
  C  structured: A's recommendation/comparison agent prompts re-issued with a
     product-card JSON schema appended to the user message
  multiturn_<profile>: the condense calls of each conversation, in turn order
"""
from __future__ import annotations

import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .driver import read_raw
from .tokens import QwenTokenizer

STRUCTURED_AGENTS = {"ProductRecommendationAgent", "ProductComparisonAgent"}
PROFILES = ("shallow", "medium", "deep")
SCHEMA_REL = "schemas/product_card.schema.json"

STRUCTURED_INSTRUCTION = (
    "\n\nRespond with a JSON object matching this JSON Schema exactly, and nothing else:\n{schema}"
)


def assign_workload(record: dict) -> str | None:
    role = record["call_role"]
    if role == "condense":
        return None
    if record.get("routed_agent") == "ReviewSummarizationAgent" and role in ("agent", "guardrail_output"):
        return "B"
    return "A"


def structured_messages(messages: list[dict], schema: dict) -> list[dict]:
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m["role"] == "user":
            m["content"] = m["content"] + STRUCTURED_INSTRUCTION.format(schema=json.dumps(schema))
            break
    return out


def build_rows(records: list[dict], tokenizer: QwenTokenizer, workload: str, schema: dict | None = None) -> list[dict]:
    rows = []
    for r in records:
        if not r.get("response_text"):
            continue  # nothing to size the output by; the app got an empty completion
        messages = structured_messages(r["messages"], schema) if schema is not None else r["messages"]
        prompt = tokenizer.render(messages)
        rows.append({
            "prompt": prompt,
            "output_tokens": tokenizer.count(r["response_text"]),
            "record_id": r["record_id"],
            "query_id": r["query_id"],
            "conversation_id": r.get("conversation_id"),
            "turn_index": r.get("turn_index", 0),
            "call_role": r["call_role"],
            "workload": workload,
            "routed_agent": r.get("routed_agent"),
            "intent": r.get("intent"),
            "phrasing": r.get("phrasing"),
            "app_temperature": r.get("temperature"),
            "app_response_format": r.get("response_format"),
            "prompt_tokens_qwen": tokenizer.count(prompt),
            "prompt_tokens_openai": r.get("prompt_tokens_openai"),
            "output_tokens_openai": r.get("completion_tokens_openai"),
            "provenance": r.get("provenance", "generated"),
            "schema_file": SCHEMA_REL if schema is not None else None,
        })
    return rows


def write_trace(rows: list[dict], path: Path) -> str:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; trace files are immutable — bump the version instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p10": None, "p50": None, "p90": None, "p99": None, "mean": None}
    vals = sorted(values)
    q = statistics.quantiles(vals, n=100, method="inclusive") if len(vals) > 1 else [vals[0]] * 99
    return {"n": len(vals), "p10": q[9], "p50": statistics.median(vals), "p90": q[89], "p99": q[98],
            "mean": statistics.fmean(vals)}


def _dist(rows: list[dict]) -> dict:
    return {
        "prompt_tokens_qwen": quantiles([r["prompt_tokens_qwen"] for r in rows]),
        "output_tokens": quantiles([r["output_tokens"] for r in rows]),
    }


def write_meta(rows: list[dict], path: Path, *, tokenizer: QwenTokenizer, seed: int, app_git_sha: str | None,
               raw_path: str, trace_sha256: str, extra: dict | None = None) -> dict:
    by_role: dict[str, list[dict]] = {}
    for r in rows:
        by_role.setdefault(r["call_role"], []).append(r)
    by_prov: dict[str, int] = {}
    for r in rows:
        by_prov[r["provenance"]] = by_prov.get(r["provenance"], 0) + 1
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "app_git_sha": app_git_sha,
        "raw_path": str(raw_path),
        "trace_sha256": trace_sha256,
        "tokenizer": {"model_id": tokenizer.model_id, "revision": tokenizer.revision,
                      "template_sha256": tokenizer.template_sha256},
        "counts": {"total": len(rows), "by_provenance": by_prov,
                   "by_call_role": {k: len(v) for k, v in by_role.items()}},
        **_dist(rows),
        "by_call_role": {k: _dist(v) for k, v in by_role.items()},
        "format": "vllm bench serve custom dataset (prompt, output_tokens); prompts pre-rendered with the "
                  "Qwen2.5 chat template for the completions endpoint with --skip-chat-template",
    }
    meta.update(extra or {})
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _emit(rows, out_dir: Path, name: str, version: int, **meta_kw) -> list[Path]:
    trace = out_dir / f"{name}_v{version}.jsonl"
    meta = out_dir / f"{name}_v{version}.meta.json"
    sha = write_trace(rows, trace)
    write_meta(rows, meta, trace_sha256=sha, **meta_kw)
    return [trace, meta]


def export_all(raw_path: Path, out_dir: Path, version: int, tokenizer: QwenTokenizer, seed: int,
               app_git_sha: str | None) -> list[Path]:
    out_dir = Path(out_dir)
    records = read_raw(raw_path)
    schema_path = out_dir / SCHEMA_REL
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_sha = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    common = dict(tokenizer=tokenizer, seed=seed, app_git_sha=app_git_sha, raw_path=str(raw_path))

    a = [r for r in records if assign_workload(r) == "A"]
    b = [r for r in records if assign_workload(r) == "B"]
    c_src = [r for r in a if r["call_role"] == "agent" and r.get("routed_agent") in STRUCTURED_AGENTS]
    written: list[Path] = []
    written += _emit(build_rows(a, tokenizer, "A"), out_dir, "chat", version, extra={"workload": "A"}, **common)
    written += _emit(build_rows(b, tokenizer, "B"), out_dir, "summarization", version, extra={"workload": "B"}, **common)
    written += _emit(build_rows(c_src, tokenizer, "C", schema=schema), out_dir, "structured", version,
                     extra={"workload": "C", "source_workload": "A", "schema_file": SCHEMA_REL,
                            "schema_sha256": schema_sha}, **common)
    for profile in PROFILES:
        conv = [r for r in records if r["call_role"] == "condense"
                and (r.get("conversation_id") or "").startswith(f"conv-{profile}-")]
        conv.sort(key=lambda r: (r["conversation_id"], r["turn_index"]))
        written += _emit(build_rows(conv, tokenizer, f"multiturn_{profile}"), out_dir, f"multiturn_{profile}",
                         version, extra={"workload": f"multiturn_{profile}", "profile": profile,
                                         "note": "condense-call prompts in turn order; the app never resends "
                                                 "history to the agent (HISTORY_WINDOW bounds the transcript)"},
                         **common)
    return written
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/bench/test_export.py -q`
Expected: 8 passed

- [ ] **Step 6: Commit**

```bash
git add bench/capture/export.py bench/traces/schemas/product_card.schema.json tests/bench/test_export.py
git commit -m "bench: export raw captures to versioned per-workload trace files"
```

---

### Task 7: LangSmith export (real traffic → same raw format)

**Files:**
- Create: `bench/capture/langsmith_export.py`
- Test: `tests/bench/test_langsmith_export.py`

**Interfaces:**
- Produces:
  - `run_to_record(run) -> dict | None` — maps one LangSmith LLM run (as produced by `wrap_openai`) to the raw-record contract with `provenance="real"`, `query_id = "ls-" + str(run.trace_id)`, `call_index` from ordering within the trace, `call_role` via `classify_call_role`, `routed_agent=None` (unknown for real traffic; the writeup says so). Returns `None` for non-chat runs (embeddings, moderation).
  - `fetch_llm_runs(project_name, since: datetime | None, client=None) -> list` — thin wrapper over `langsmith.Client().list_runs(project_name=..., run_type="llm", start_time=since)`.
  - `export_langsmith(project_name, out_path, since=None, client=None) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_langsmith_export.py`:
```python
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from app.guardrails import INPUT_CLASSIFIER_SYSTEM_PROMPT

from bench.capture.langsmith_export import export_langsmith, run_to_record


def _run(messages, text="ok", trace="t1", start="2026-09-10T10:00:00+00:00", usage=(50, 5), name="ChatOpenAI"):
    return SimpleNamespace(
        id="r1", trace_id=trace, name=name, run_type="llm",
        start_time=datetime.fromisoformat(start),
        inputs={"messages": messages, "model": "gpt-3.5-turbo", "temperature": 0,
                "response_format": {"type": "json_object"}},
        outputs={"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}},
    )


def test_run_to_record_maps_fields():
    msgs = [{"role": "system", "content": INPUT_CLASSIFIER_SYSTEM_PROMPT}, {"role": "user", "content": "hi"}]
    rec = run_to_record(_run(msgs), call_index=2)
    assert rec["provenance"] == "real" and rec["query_id"] == "ls-t1" and rec["record_id"] == "ls-t1-c2"
    assert rec["call_role"] == "guardrail_input" and rec["messages"] == msgs
    assert rec["response_text"] == "ok" and rec["finish_reason"] == "stop"
    assert rec["prompt_tokens_openai"] == 50 and rec["completion_tokens_openai"] == 5
    assert rec["temperature"] == 0 and rec["response_format"] == {"type": "json_object"}
    assert rec["routed_agent"] is None and rec["intent"] is None
    assert rec["captured_at"] == "2026-09-10T10:00:00+00:00"


def test_run_to_record_skips_non_chat_runs():
    emb = SimpleNamespace(id="e", trace_id="t", name="Embeddings", run_type="llm", start_time=datetime.now(timezone.utc),
                          inputs={"input": ["x"], "model": "text-embedding-3-small"}, outputs={"data": []})
    assert run_to_record(emb, call_index=0) is None


def test_export_langsmith_orders_calls_within_a_trace_and_writes_jsonl(tmp_path):
    a = _run([{"role": "system", "content": "s"}, {"role": "user", "content": "1"}], trace="t1", start="2026-09-10T10:00:01+00:00")
    b = _run([{"role": "system", "content": "s"}, {"role": "user", "content": "2"}], trace="t1", start="2026-09-10T10:00:00+00:00")
    c = _run([{"role": "system", "content": "s"}, {"role": "user", "content": "3"}], trace="t2")
    fake_client = SimpleNamespace(list_runs=lambda **kw: iter([a, b, c]))
    out = tmp_path / "ls.jsonl"
    n = export_langsmith("proj", out, client=fake_client)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert n == 3
    assert [(r["query_id"], r["call_index"], r["messages"][1]["content"]) for r in rows] == [
        ("ls-t1", 0, "2"), ("ls-t1", 1, "1"), ("ls-t2", 0, "3")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/bench/test_langsmith_export.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`bench/capture/langsmith_export.py`:
```python
"""Pull the app's real emitted prompts from LangSmith into the raw-record
contract, tagged provenance="real". wrap_openai (app/openai_client.py) logs
each chat.completions call as an LLM run whose inputs/outputs mirror the
OpenAI request/response, so the mapping is mechanical.

Real traffic to date is developer testing; it is used as a validation set
for the generator's distributions, not as the benchmark corpus (spec §3.2).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .recorder import classify_call_role


def _first_choice(outputs: dict) -> tuple[str, str | None]:
    choices = (outputs or {}).get("choices") or []
    if not choices:
        return "", None
    ch = choices[0]
    msg = ch.get("message") or {}
    return msg.get("content") or "", ch.get("finish_reason")


def run_to_record(run, call_index: int) -> dict | None:
    inputs = run.inputs or {}
    messages = inputs.get("messages")
    if not isinstance(messages, list):
        return None  # embeddings / moderation / non-chat runs
    text, finish = _first_choice(run.outputs or {})
    usage = (run.outputs or {}).get("usage") or {}
    query_id = f"ls-{run.trace_id}"
    return {
        "record_id": f"{query_id}-c{call_index}",
        "query_id": query_id,
        "conversation_id": None,
        "turn_index": 0,
        "call_index": call_index,
        "call_role": classify_call_role(messages),
        "intent": None,
        "phrasing": None,
        "query_text": None,
        "routed_agent": None,
        "routed_via": None,
        "route_status": None,
        "model": inputs.get("model", ""),
        "messages": messages,
        "response_format": inputs.get("response_format"),
        "temperature": inputs.get("temperature"),
        "max_tokens": inputs.get("max_tokens"),
        "response_text": text,
        "finish_reason": finish,
        "prompt_tokens_openai": usage.get("prompt_tokens"),
        "completion_tokens_openai": usage.get("completion_tokens"),
        "latency_ms": 0.0,
        "captured_at": run.start_time.isoformat() if isinstance(run.start_time, datetime) else str(run.start_time),
        "provenance": "real",
        "app_git_sha": None,
    }


def fetch_llm_runs(project_name: str, since: datetime | None = None, client=None) -> list:
    if client is None:
        from langsmith import Client
        client = Client()
    kwargs = {"project_name": project_name, "run_type": "llm"}
    if since is not None:
        kwargs["start_time"] = since
    return list(client.list_runs(**kwargs))


def export_langsmith(project_name: str, out_path: Path, since: datetime | None = None, client=None) -> int:
    runs = fetch_llm_runs(project_name, since, client)
    runs.sort(key=lambda r: (str(r.trace_id), r.start_time))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    call_index: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for run in runs:
            idx = call_index.get(str(run.trace_id), 0)
            rec = run_to_record(run, idx)
            if rec is None:
                continue
            call_index[str(run.trace_id)] = idx + 1
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/bench/test_langsmith_export.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add bench/capture/langsmith_export.py tests/bench/test_langsmith_export.py
git commit -m "bench: export real LangSmith-traced calls into the raw record format"
```

---

### Task 8: Validation — generated vs real quantiles

**Files:**
- Create: `bench/capture/validate.py`
- Test: `tests/bench/test_validate.py`

**Interfaces:**
- Consumes: raw record dicts from both sources; `QwenTokenizer`; `assign_workload`, `quantiles` (Task 6).
- Produces:
  - `compare(generated: list[dict], real: list[dict], tokenizer, tolerance=0.25, min_n=10) -> dict` — per `(workload, call_role)` cell: `n_generated`, `n_real`, `p50_generated`, `p50_real`, `p90_*`, `rel_diff_p50`, `rel_diff_p90`, `status ∈ {"pass","fail","insufficient_real"}`; top-level `overall_status` (fail if any cell fails), `tolerance`, `min_n`.
  - `to_markdown(report) -> str`.
  - Prompt length for both sources is `tokenizer.count(tokenizer.render(messages))` — measured identically, so the only difference is who wrote the query.

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_validate.py`:
```python
from bench.capture.tokens import QwenTokenizer
from bench.capture.validate import compare, to_markdown
from tests.bench.test_tokens import StubHF


def _rec(n_words, role="agent", agent="ProductRecommendationAgent", prov="generated"):
    return {"call_role": role, "routed_agent": agent, "provenance": prov, "response_text": "x",
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": " ".join(["w"] * n_words)}]}


def test_compare_passes_when_quantiles_agree():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    gen = [_rec(n) for n in range(90, 111)] * 2
    real = [_rec(n, prov="real") for n in range(95, 106)]
    rep = compare(gen, real, tok, tolerance=0.25, min_n=10)
    cell = rep["cells"]["A/agent"]
    assert cell["n_generated"] == 42 and cell["n_real"] == 11
    assert cell["status"] == "pass" and rep["overall_status"] == "pass"
    assert abs(cell["rel_diff_p50"]) < 0.05


def test_compare_fails_when_generated_is_far_off():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    gen = [_rec(300) for _ in range(20)]
    real = [_rec(100, prov="real") for _ in range(20)]
    rep = compare(gen, real, tok)
    assert rep["cells"]["A/agent"]["status"] == "fail" and rep["overall_status"] == "fail"


def test_compare_marks_thin_real_cells_and_does_not_fail_on_them():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    gen = [_rec(100) for _ in range(20)] + [_rec(50, role="guardrail_input") for _ in range(20)]
    real = [_rec(100, prov="real") for _ in range(12)] + [_rec(500, role="guardrail_input", prov="real") for _ in range(3)]
    rep = compare(gen, real, tok, min_n=10)
    assert rep["cells"]["A/guardrail_input"]["status"] == "insufficient_real"
    assert rep["overall_status"] == "pass"


def test_real_records_without_routing_are_assigned_by_call_role_only():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    gen = [_rec(100, agent="ReviewSummarizationAgent") for _ in range(20)]      # B in generated
    real = [_rec(100, agent=None, prov="real") for _ in range(20)]               # unknown agent -> A
    rep = compare(gen, real, tok)
    assert "B/agent" in rep["cells"] and rep["cells"]["B/agent"]["n_real"] == 0
    assert rep["cells"]["A/agent"]["n_real"] == 20
    assert "routed_agent is unknown for real traffic" in rep["notes"][0]


def test_markdown_has_a_row_per_cell():
    tok = QwenTokenizer(StubHF(), "stub", "r")
    rep = compare([_rec(10)], [_rec(10, prov="real")], tok)
    md = to_markdown(rep)
    assert "| A/agent |" in md and "overall" in md.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/bench/test_validate.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`bench/capture/validate.py`:
```python
"""Compare the generator's prompt-length distributions against real emitted
prompts, per (workload, call_role). Both sides are rendered and counted with
the same tokenizer, so the only difference is who wrote the query.

Real traffic carries no routing information (LangSmith runs don't include the
coordinator's log line), so real records are assigned to workloads by
call_role alone: they all land in A. B's real count is therefore 0 and its
cell reads "insufficient_real" — the report says so rather than hiding it.
"""
from __future__ import annotations

from .export import assign_workload, quantiles
from .tokens import QwenTokenizer

NOTE_ROUTING = ("routed_agent is unknown for real traffic, so real records are assigned by call_role only "
                "(all to A); B and C cells have no real counterpart and are reported as insufficient_real.")


def _lengths(records: list[dict], tokenizer: QwenTokenizer) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in records:
        w = assign_workload(r)
        if w is None:
            continue
        key = f"{w}/{r['call_role']}"
        out.setdefault(key, []).append(tokenizer.count(tokenizer.render(r["messages"])))
    return out


def _rel(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b


def compare(generated: list[dict], real: list[dict], tokenizer: QwenTokenizer,
            tolerance: float = 0.25, min_n: int = 10) -> dict:
    g = _lengths(generated, tokenizer)
    r = _lengths(real, tokenizer)
    cells = {}
    for key in sorted(set(g) | set(r)):
        qg, qr = quantiles(g.get(key, [])), quantiles(r.get(key, []))
        d50, d90 = _rel(qg["p50"], qr["p50"]), _rel(qg["p90"], qr["p90"])
        if qr["n"] < min_n:
            status = "insufficient_real"
        elif abs(d50) <= tolerance and abs(d90) <= tolerance:
            status = "pass"
        else:
            status = "fail"
        cells[key] = {"n_generated": qg["n"], "n_real": qr["n"],
                      "p50_generated": qg["p50"], "p50_real": qr["p50"],
                      "p90_generated": qg["p90"], "p90_real": qr["p90"],
                      "rel_diff_p50": d50, "rel_diff_p90": d90, "status": status}
    overall = "fail" if any(c["status"] == "fail" for c in cells.values()) else "pass"
    return {"tolerance": tolerance, "min_n": min_n, "cells": cells, "overall_status": overall,
            "notes": [NOTE_ROUTING]}


def _fmt(x):
    return "—" if x is None else (f"{x:+.0%}" if isinstance(x, float) and abs(x) < 10 else f"{x:.0f}")


def to_markdown(report: dict) -> str:
    lines = ["# Generated vs real prompt-length validation", "",
             f"Tolerance ±{report['tolerance']:.0%} on p50 and p90; cells with < {report['min_n']} real records "
             f"are marked insufficient_real and do not affect the overall result.", "",
             "| cell | n gen | n real | p50 gen | p50 real | Δp50 | p90 gen | p90 real | Δp90 | status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in report["cells"].items():
        lines.append(f"| {key} | {c['n_generated']} | {c['n_real']} | {_fmt(c['p50_generated'])} | {_fmt(c['p50_real'])} | "
                     f"{_fmt(c['rel_diff_p50'])} | {_fmt(c['p90_generated'])} | {_fmt(c['p90_real'])} | "
                     f"{_fmt(c['rel_diff_p90'])} | {c['status']} |")
    lines += ["", f"**Overall: {report['overall_status']}**", ""]
    lines += [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/bench/test_validate.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add bench/capture/validate.py tests/bench/test_validate.py
git commit -m "bench: add generated-vs-real prompt-length validation report"
```

---

### Task 9: CLI

**Files:**
- Create: `bench/capture/cli.py`, `bench/capture/__main__.py`
- Test: `tests/bench/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `main(argv) -> int` with subcommands `generate`, `capture`, `export`, `langsmith`, `validate`; `git_sha() -> str | None`.

- [ ] **Step 1: Write the failing tests**

`tests/bench/test_cli.py`:
```python
import json

from bench.capture import cli
from bench.capture.generator import from_json


def test_generate_writes_queries_json(small_catalog, tmp_path, monkeypatch):
    from bench.capture.generator import Catalog
    monkeypatch.setattr(cli.Catalog, "from_app", classmethod(lambda c: Catalog(small_catalog.products, small_catalog.reviews, small_catalog.policies)))
    out = tmp_path / "queries.json"
    rc = cli.main(["generate", "--seed", "3", "--n-single", "20", "--n-per-profile", "2", "--out", str(out)])
    assert rc == 0
    qs, convs = from_json(json.loads(out.read_text()))
    assert len(qs) == 20 and len(convs) == 6


def test_capture_then_export_end_to_end(fake_chat, small_catalog, tmp_path, monkeypatch):
    from bench.capture.generator import Catalog
    from bench.capture.tokens import QwenTokenizer
    from tests.bench.test_tokens import StubHF
    monkeypatch.setattr(cli.Catalog, "from_app", classmethod(lambda c: Catalog(small_catalog.products, small_catalog.reviews, small_catalog.policies)))
    monkeypatch.setattr(cli.QwenTokenizer, "load", classmethod(lambda c, **kw: QwenTokenizer(StubHF(), "stub", "r")))
    q = tmp_path / "queries.json"
    raw = tmp_path / "raw.jsonl"
    traces = tmp_path / "traces"
    (traces / "schemas").mkdir(parents=True)
    (traces / "schemas" / "product_card.schema.json").write_text("{}")
    assert cli.main(["generate", "--seed", "1", "--n-single", "10", "--n-per-profile", "1", "--out", str(q)]) == 0
    assert cli.main(["capture", "--queries", str(q), "--out", str(raw), "--workers", "2"]) == 0
    assert raw.read_text().count("\n") > 10
    assert cli.main(["export", "--raw", str(raw), "--out-dir", str(traces), "--version", "1", "--seed", "1"]) == 0
    assert (traces / "chat_v1.jsonl").exists() and (traces / "multiturn_deep_v1.meta.json").exists()
    # immutability: a second export at the same version fails loudly
    assert cli.main(["export", "--raw", str(raw), "--out-dir", str(traces), "--version", "1", "--seed", "1"]) == 1


def test_validate_writes_markdown(tmp_path, monkeypatch):
    from bench.capture.tokens import QwenTokenizer
    from tests.bench.test_tokens import StubHF
    monkeypatch.setattr(cli.QwenTokenizer, "load", classmethod(lambda c, **kw: QwenTokenizer(StubHF(), "stub", "r")))
    rec = {"call_role": "agent", "routed_agent": "X", "provenance": "generated", "response_text": "x",
           "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "a b c"}]}
    gen, real = tmp_path / "g.jsonl", tmp_path / "r.jsonl"
    gen.write_text(json.dumps(rec) + "\n")
    real.write_text(json.dumps({**rec, "provenance": "real", "routed_agent": None}) + "\n")
    out = tmp_path / "v.md"
    assert cli.main(["validate", "--generated", str(gen), "--real", str(real), "--out", str(out)]) == 0
    assert "| A/agent |" in out.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/bench/test_cli.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`bench/capture/__main__.py`:
```python
import sys

from .cli import main

sys.exit(main(sys.argv[1:]))
```

`bench/capture/cli.py`:
```python
"""python -m bench.capture <generate|capture|export|langsmith|validate>"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .driver import capture_all, read_raw
from .export import export_all
from .generator import Catalog, from_json, generate_conversations, generate_queries, to_json
from .langsmith_export import export_langsmith
from .tokens import QwenTokenizer
from .validate import compare, to_markdown


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _generate(a) -> int:
    catalog = Catalog.from_app()
    qs = generate_queries(catalog, n=a.n_single, seed=a.seed)
    convs = generate_conversations(catalog, n_per_profile=a.n_per_profile, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"seed": a.seed, **to_json(qs, convs)}, indent=1), encoding="utf-8")
    print(f"wrote {len(qs)} queries and {len(convs)} conversations ({sum(len(c.turns) for c in convs)} turns) to {a.out}")
    return 0


def _capture(a) -> int:
    qs, convs = from_json(json.loads(Path(a.queries).read_text(encoding="utf-8")))
    n = capture_all(qs, convs, Path(a.out), workers=a.workers, app_git_sha=git_sha())
    print(f"wrote {n} new records to {a.out} ({len(read_raw(Path(a.out)))} total)")
    return 0


def _export(a) -> int:
    tok = QwenTokenizer.load(model_id=a.tokenizer, revision=a.tokenizer_revision)
    try:
        paths = export_all(Path(a.raw), Path(a.out_dir), version=a.version, tokenizer=tok, seed=a.seed,
                           app_git_sha=git_sha())
    except FileExistsError as e:
        print(f"refusing to overwrite: {e}", file=sys.stderr)
        return 1
    for p in paths:
        print(p)
    return 0


def _langsmith(a) -> int:
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc) if a.since else None
    n = export_langsmith(a.project, Path(a.out), since=since)
    print(f"wrote {n} real records to {a.out}")
    return 0


def _validate(a) -> int:
    tok = QwenTokenizer.load(model_id=a.tokenizer, revision=a.tokenizer_revision)
    report = compare(read_raw(Path(a.generated)), read_raw(Path(a.real)), tok, tolerance=a.tolerance, min_n=a.min_n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_markdown(report), encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"overall: {report['overall_status']} -> {a.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench.capture")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--n-single", type=int, default=1400)
    g.add_argument("--n-per-profile", type=int, default=40)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=_generate)

    c = sub.add_parser("capture")
    c.add_argument("--queries", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=4)
    c.set_defaults(fn=_capture)

    e = sub.add_parser("export")
    e.add_argument("--raw", required=True)
    e.add_argument("--out-dir", default="bench/traces")
    e.add_argument("--version", type=int, required=True)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    e.add_argument("--tokenizer-revision", default=None)
    e.set_defaults(fn=_export)

    l = sub.add_parser("langsmith")
    l.add_argument("--project", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--since", default=None, help="ISO date, UTC")
    l.set_defaults(fn=_langsmith)

    v = sub.add_parser("validate")
    v.add_argument("--generated", required=True)
    v.add_argument("--real", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--tolerance", type=float, default=0.25)
    v.add_argument("--min-n", type=int, default=10)
    v.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    v.add_argument("--tokenizer-revision", default=None)
    v.set_defaults(fn=_validate)
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
```

- [ ] **Step 4: Run the whole bench suite**

Run: `pytest tests/bench -q`
Expected: all pass. Then `pytest tests -q` to confirm the app suite is still green.

- [ ] **Step 5: Commit**

```bash
git add bench/capture/cli.py bench/capture/__main__.py tests/bench/test_cli.py
git commit -m "bench: add the capture CLI"
```

---

### Task 10: The capture run (manual runbook — real OpenAI calls)

This task executes P0a. It costs money (~$5–10 at gpt-3.5-turbo prices for ~2,000 turns × 3–5 calls) and ~30–45 minutes with 4 workers. Run it once.

**Files:**
- Create (generated): `bench/traces/raw/queries_<seed>.json`, `bench/traces/raw/capture_<seed>.jsonl`, `bench/traces/*_v1.jsonl`, `bench/traces/*_v1.meta.json`, `bench/traces/validation/langsmith_<date>.jsonl`, `bench/docs/p0a-validation.md`

- [ ] **Step 1: Environment**

In WSL2, from the repo root, with the app's venv active and `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` exported (the `.env` the app uses is fine — `python-dotenv` is not auto-loaded by `bench`, so `set -a; source .env; set +a` first):
```bash
pip install -r bench/requirements.txt
BENCH_HF_TESTS=1 pytest tests/bench -q      # also warms the Qwen tokenizer cache
git rev-parse --abbrev-ref HEAD              # must print: benchmarking
```
Set `LANGSMITH_TRACING=false` for the capture run so the generated traffic is **not** logged into the same LangSmith project the validation export reads from (otherwise the validation set would contain the generator's own queries).

- [ ] **Step 2: Generate and inspect**

```bash
SEED=20260919
python -m bench.capture generate --seed $SEED --n-single 1400 --n-per-profile 40 --out bench/traces/raw/queries_$SEED.json
python - <<'EOF'
import json, collections
d = json.load(open("bench/traces/raw/queries_20260919.json"))
print(collections.Counter(q["intent"] for q in d["queries"]))
print(collections.Counter(q["phrasing"] for q in d["queries"]))
print(sum(len(c["turns"]) for c in d["conversations"]), "conversation turns")
for q in d["queries"][:8]: print(q["intent"], "|", q["text"])
EOF
```
Expected: ~1,400 single queries + 120 conversations (~560 turns) ≈ 1,960 app turns; every printed query reads like a real customer message. If any template renders oddly (a policy type with an underscore, a product name that doesn't parse), fix the template in `generator.py` and regenerate with the **same** seed before capturing.

- [ ] **Step 3: Capture (resumable)**

```bash
LANGSMITH_TRACING=false python -m bench.capture capture --queries bench/traces/raw/queries_$SEED.json --out bench/traces/raw/capture_$SEED.jsonl --workers 4
```
If interrupted, re-run the same command; finished queries/conversations are skipped. When done:
```bash
python - <<'EOF'
import json, collections
rows = [json.loads(l) for l in open("bench/traces/raw/capture_20260919.jsonl")]
print(len(rows), "records")
print(collections.Counter(r["call_role"] for r in rows))
print(collections.Counter(r["routed_agent"] for r in rows))
print(collections.Counter(r["route_status"] for r in rows))
print("blocked by guardrails:", sum(r["route_status"] == "blocked" for r in rows))
EOF
```
Expected: `route_status == "ok"` for the vast majority; a handful of `blocked` is fine (real guardrail behaviour) and is reported. If `routed_agent` is `None` for many records, the `app.agents` logger level was raised by something else — check `RouteCapture.installed()` set it to INFO.

- [ ] **Step 4: Export traces v1**

```bash
python -m bench.capture export --raw bench/traces/raw/capture_$SEED.jsonl --out-dir bench/traces --version 1 --seed $SEED
cat bench/traces/chat_v1.meta.json | python -c "import json,sys; m=json.load(sys.stdin); print(m['counts']); print('prompt p50/p90:', m['prompt_tokens_qwen']['p50'], m['prompt_tokens_qwen']['p90']); print('output p50/p90:', m['output_tokens']['p50'], m['output_tokens']['p90'])"
```
Repeat the `cat` for `summarization_v1`, `structured_v1`, and the three `multiturn_*` metas. Record the numbers for the writeup.

- [ ] **Step 5: LangSmith validation set**

```bash
python -m bench.capture langsmith --project "$LANGSMITH_PROJECT" --out bench/traces/validation/langsmith_$(date +%Y%m%d).jsonl
python -m bench.capture validate --generated bench/traces/raw/capture_$SEED.jsonl --real bench/traces/validation/langsmith_$(date +%Y%m%d).jsonl --out bench/docs/p0a-validation.md
cat bench/docs/p0a-validation.md
```
A `fail` cell means the generator's prompts differ from what the deployed app emits by more than 25% at p50 or p90 — investigate the template mix for that call role before accepting v1. `insufficient_real` cells are expected for B, C, and any role with < 10 real records; the writeup lists them.

- [ ] **Step 6: Commit the traces**

```bash
du -sh bench/traces
git add bench/traces/raw/queries_$SEED.json bench/traces/raw/capture_$SEED.jsonl bench/traces/*_v1.jsonl bench/traces/*_v1.meta.json bench/traces/validation/ bench/docs/p0a-validation.md
git commit -m "bench: capture trace set v1 (seed $SEED)"
```
If `du` reports more than ~80 MB, stop and raise it before committing — git-lfs for `bench/traces/raw/` is the fallback, and it is a decision, not a default.

---

### Task 11: P0a writeup and spec re-evaluation

**Files:**
- Create: `bench/docs/p0a-writeup.md`
- Modify: `docs/superpowers/specs/2026-09-17-pickr-inference-benchmark-design-v2.md` (§3.1 prediction 1 status; §3.2 cost line)

- [ ] **Step 1: Write the P0a writeup**

`bench/docs/p0a-writeup.md` sections, each filled from the meta files and validation report — no placeholders:

1. **What was captured.** Seed, app git SHA, counts by intent/phrasing/call role/routed agent, guardrail-blocked count, moderation calls per turn (2, not recorded), total OpenAI cost from the usage sums.
2. **Per-turn call structure.** Table: for each routed agent, the sequence of call roles observed (e.g. `ProductRecommendationAgent: guardrail_input → agent → guardrail_output`; `StockAvailabilityAgent: guardrail_input` only; natural phrasing adds `classifier`; turns ≥ 1 add `condense`). This is the "one user turn ≠ one LLM call" finding.
3. **Length distributions.** One table per trace file: n, prompt tokens (Qwen) p10/p50/p90/p99, output tokens p10/p50/p90/p99, by call role. State plainly that **Workload B prompts are short in this catalog** (median 2 reviews per product, ~51 characters each) and that the longest call on a summarisation turn is the `guardrail_output` faithfulness check, not the summary itself.
4. **Multi-turn shape.** Condense-prompt length vs `turn_index` for each profile, showing growth to turn 3 and the plateau once `HISTORY_WINDOW` (6 rows) slides. Note that the app never resends history to the agent, so the brief's "TTFT grows linearly with turn" hypothesis applies only to the condense call and only up to the window.
5. **Validation against real traffic.** Paste `p0a-validation.md`. Describe the real set as developer testing traffic through the deployed app; list cells that were insufficient.
6. **Spec §3.1 prediction 1 re-evaluated.** Chat median prompt tokens vs the 2k threshold; 32 × p50 and 32 × p90 vs the ~70k-token KV budget; conclusion: compute-bound or KV-bound expectation for P1, stated before P1 runs.
7. **Exit criteria.** Traces v1 exist, hashes recorded in meta, validation report committed.

- [ ] **Step 2: Update the spec**

In the spec's §3.1, after prediction 1, append one line: `**P0a result (YYYY-MM-DD):** measured chat median = N tokens → prediction [stands | invalidated]; see bench/docs/p0a-writeup.md §6.` In §3.2 replace the `(~$2–3)` cost estimate with the measured cost.

- [ ] **Step 3: Commit**

```bash
git add bench/docs/p0a-writeup.md docs/superpowers/specs/2026-09-17-pickr-inference-benchmark-design-v2.md
git commit -m "bench: P0a writeup and spec re-evaluation against measured traces"
```

---

## Open decisions carried to P0b (not blocking P0a)

These were discovered while reading the client source and the app; they are P0b/runner decisions, recorded here so they are not lost:

1. **One sampling configuration per run.** `vllm bench serve` sets `--temperature/--top-p/--top-k` globally, not per request, and with no flags the *server's* defaults apply — vLLM applies the model's `generation_config.json` (Qwen2.5: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05) while SGLang may not, which would be a phantom engine difference. P0b must pin explicit sampling parameters on every run. Trace rows carry `app_temperature` per record so the writeup can state the deviation: the app runs guardrail/classifier/condense calls at temperature 0 and agent calls at the SDK default; the benchmark runs each file at one pre-registered setting.
2. **Structured output for P3** is passed via `--extra-body` (global per run), with engine-specific keys (vLLM `structured_outputs`/`guided_json` vs SGLang `json_schema`) verified at the pin. The trace's `schema_file` names the schema; the prompt already asks for JSON matching it.
3. **`--skip-chat-template` + completions endpoint** is the intended run mode for all trace files; P0b's parity check reduces to confirming both engines report the same `prompt_tokens` for one trace row.

## Self-review

- **Spec coverage:** §3.2 pipeline (Tasks 2–4), one-turn-many-calls tagging (Task 2 `call_role`, Task 4 `RouteCapture`), workload tagging (Task 6 `assign_workload`), C overlay + schema (Task 6), output-length policy — `output_tokens` per record from the Qwen count of the reference response (Task 6), Qwen + OpenAI token counts (Task 6 rows), export format + meta + immutability (Task 6), three multi-turn profiles (Tasks 3, 6), validation set + tolerance check + "developer testing traffic" wording (Tasks 7, 8, 11), sample size 1,500–2,000 captured once (Task 10), §3.1 prediction 1 re-evaluation (Task 11), §9 P0a exit criteria (Tasks 10–11). No `app/` modification anywhere.
- **Placeholder scan:** none; Task 10/11 are runbooks with exact commands and the writeup's section list names the specific numbers to fill from named files.
- **Type consistency:** `CallRecord.to_dict()` keys == the raw-record contract == `required` set in `test_driver.py` == keys read by `build_rows`/`run_to_record`. `assign_workload` and `quantiles` are imported from `export` by `validate`. `RouteCapture.installed()` is added in Task 4 and used by Task 4 and Task 9 (via `capture_all`). `QwenTokenizer(hf, model_id, revision)` positional order matches every test's construction.
