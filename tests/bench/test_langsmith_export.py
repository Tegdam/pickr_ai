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
