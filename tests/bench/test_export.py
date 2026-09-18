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
