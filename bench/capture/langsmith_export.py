"""Pull the app's real emitted prompts from LangSmith into the raw-record
contract, tagged provenance="real". wrap_openai (app/openai_client.py) logs
each chat.completions call as an LLM run whose inputs/outputs mirror the
OpenAI request/response, so the mapping is mechanical.

Real traffic to date is developer testing; it is used as a validation set
for the generator's distributions, not as the benchmark corpus (spec §3.2).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .recorder import classify_call_role


def _first_choice(outputs: dict) -> tuple[str, str | None]:
    choices = (outputs or {}).get("choices") or []
    if not choices:
        return "", None
    ch = choices[0]
    msg = ch.get("message") or {}
    return msg.get("content") or "", ch.get("finish_reason")


def _usage(outputs: dict) -> tuple[int | None, int | None]:
    # langsmith's wrap_openai pops "usage" from the response body and writes
    # "usage_metadata": {input_tokens, output_tokens, total_tokens} instead
    # (langsmith 0.11.0, _openai.py:284-289). Support both shapes.
    usage = (outputs or {}).get("usage")
    if isinstance(usage, dict):
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    usage_metadata = (outputs or {}).get("usage_metadata")
    if isinstance(usage_metadata, dict):
        return usage_metadata.get("input_tokens"), usage_metadata.get("output_tokens")
    return None, None


def run_to_record(run, call_index: int) -> dict | None:
    if getattr(run, "error", None):
        return None  # errored runs never produced a real response
    inputs = run.inputs or {}
    messages = inputs.get("messages")
    if not isinstance(messages, list):
        return None  # embeddings / moderation / non-chat runs
    text, finish = _first_choice(run.outputs or {})
    prompt_tokens, completion_tokens = _usage(run.outputs or {})
    query_id = f"ls-{run.trace_id}"

    start_time = run.start_time
    end_time = getattr(run, "end_time", None)
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        latency_ms = (end_time - start_time).total_seconds() * 1000
    else:
        latency_ms = None

    if isinstance(start_time, datetime):
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        captured_at = start_time.isoformat()
    else:
        captured_at = str(run.start_time)

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
        "prompt_tokens_openai": prompt_tokens,
        "completion_tokens_openai": completion_tokens,
        "latency_ms": latency_ms,
        "captured_at": captured_at,
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
