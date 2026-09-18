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
