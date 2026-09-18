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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except FileExistsError:
        raise FileExistsError(f"{path} exists; trace files are immutable — bump the version instead")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_empty_responses(records: list[dict]) -> int:
    return sum(1 for r in records if not r.get("response_text"))


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
    extra = dict(extra or {})
    for k in ("raw_records", "dropped_empty_response"):
        if k in extra:
            meta["counts"][k] = extra.pop(k)
    meta.update(extra)
    path = Path(path)
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, indent=2))
    except FileExistsError:
        raise FileExistsError(f"{path} exists; meta files are immutable — bump the version instead")
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
    if not records:
        raise ValueError(f"no records in {raw_path}; refusing to burn version {version}")

    names = ["chat", "summarization", "structured"] + [f"multiturn_{p}" for p in PROFILES]
    targets = [out_dir / f"{name}_v{version}{ext}" for name in names for ext in (".jsonl", ".meta.json")]
    existing = [p for p in targets if p.exists()]
    if existing:
        raise FileExistsError(f"refusing to export v{version}: {existing} already exist")

    a = [r for r in records if assign_workload(r) == "A"]
    b = [r for r in records if assign_workload(r) == "B"]
    c_src = [r for r in a if r["call_role"] == "agent" and r.get("routed_agent") in STRUCTURED_AGENTS]
    condense = [r for r in records if r["call_role"] == "condense"]
    conv_by_profile = {}
    for profile in PROFILES:
        conv = [r for r in condense if (r.get("conversation_id") or "").startswith(f"conv-{profile}-")]
        conv.sort(key=lambda r: (r["conversation_id"], r["turn_index"]))
        conv_by_profile[profile] = conv

    unassigned = [r["record_id"] for r in condense
                 if not any((r.get("conversation_id") or "").startswith(f"conv-{p}-") for p in PROFILES)]
    if unassigned:
        raise ValueError(f"{len(unassigned)} condense records match no profile prefix: {unassigned[:5]}...")

    schema_path = out_dir / SCHEMA_REL
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_sha = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    common = dict(tokenizer=tokenizer, seed=seed, app_git_sha=app_git_sha, raw_path=str(raw_path))

    written: list[Path] = []
    written += _emit(build_rows(a, tokenizer, "A"), out_dir, "chat", version,
                     extra={"workload": "A", "raw_records": len(records),
                            "dropped_empty_response": count_empty_responses(a)}, **common)
    written += _emit(build_rows(b, tokenizer, "B"), out_dir, "summarization", version,
                     extra={"workload": "B", "raw_records": len(records),
                            "dropped_empty_response": count_empty_responses(b)}, **common)
    written += _emit(build_rows(c_src, tokenizer, "C", schema=schema), out_dir, "structured", version,
                     extra={"workload": "C", "source_workload": "A", "schema_file": SCHEMA_REL,
                            "schema_sha256": schema_sha, "raw_records": len(records),
                            "dropped_empty_response": count_empty_responses(c_src)}, **common)
    for profile in PROFILES:
        conv = conv_by_profile[profile]
        written += _emit(build_rows(conv, tokenizer, f"multiturn_{profile}"), out_dir, f"multiturn_{profile}",
                         version, extra={"workload": f"multiturn_{profile}", "profile": profile,
                                         "note": "condense-call prompts in turn order; the app never resends "
                                                 "history to the agent (HISTORY_WINDOW bounds the transcript)",
                                         "raw_records": len(records),
                                         "dropped_empty_response": count_empty_responses(conv)},
                         **common)
    return written
