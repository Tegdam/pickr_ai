"""aiohttp app for the echo server (Task 9): an OpenAI-`/v1/completions`-shaped
stand-in for a real inference engine, used to measure the client/harness's own
overhead ceiling with no GPU engine in the loop at all.

Routes:
- `GET /health` -> 200 (always; there is nothing to warm up).
- `POST /v1/completions` -> accepts the OpenAI payload (`prompt`, `max_tokens`,
  `stream`, `model`, anything else ignored); streams `max_tokens` SSE chunks
  (one `"x"`-token each) with `per_token_ms` between them when `stream: true`.
  The LAST content chunk carries `finish_reason: "length"`; a separate final
  chunk then carries `usage` with an EMPTY `choices: []`, then `data: [DONE]`.
  Fix round 1 (Important): this is deliberate, not a stray field -- vLLM's own
  client parses each SSE payload as `if choices := data.get("choices"): ...
  elif usage := data.get("usage"): ...` (contract doc line 508: real engines
  also send `"choices": []` on their usage chunk), so a non-empty `choices`
  alongside `usage` would make the client's `if` branch win and it would
  never read `usage.completion_tokens` at all -- derived output_tokens,
  throughput and TPOT would all be silently wrong. A non-streaming request
  sleeps the same total delay (`max_tokens * per_token_ms`) and returns the
  whole completion (and its `usage`) in one response.
- `POST /reset_prefix_cache` -> 200 `{"success": true}` (nothing to reset).
- `GET /metrics` -> a tiny Prometheus-style exposition with one counter,
  `echo:requests_total` (bumped once per `/v1/completions` call) -- see
  `ENGINES["echo"].metric_names` in bench/runner/engine.py for how the
  runner's scraper resolves every metric key to this one counter.
"""
from __future__ import annotations

import asyncio
import json

from aiohttp import web

# AppKey (not a bare string key) so aiohttp doesn't warn, and a mutable dict
# behind it so bumping the counter is mutating that dict's contents rather
# than re-assigning `app[STATE]` after the app has started (aiohttp forbids
# mutating `app` itself -- not the objects it holds -- once running).
STATE = web.AppKey("state", dict)


def _prompt_token_count(prompt) -> int:
    """Whitespace token count -- not a real tokenizer, just enough to fill
    `usage.prompt_tokens` with something plausible."""
    if isinstance(prompt, list):
        prompt = " ".join(str(p) for p in prompt)
    return len(str(prompt).split())


def create_app(per_token_ms: int = 5) -> web.Application:
    app = web.Application()
    app[STATE] = {"per_token_ms": per_token_ms, "requests_total": 0}

    app.router.add_get("/health", _health)
    app.router.add_post("/v1/completions", _completions)
    app.router.add_post("/reset_prefix_cache", _reset_prefix_cache)
    app.router.add_get("/metrics", _metrics)
    return app


async def _health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})


async def _reset_prefix_cache(request: web.Request) -> web.Response:
    return web.json_response({"success": True})


async def _metrics(request: web.Request) -> web.Response:
    n = request.app[STATE]["requests_total"]
    return web.Response(text=f"echo:requests_total {n}\n", content_type="text/plain")


async def _completions(request: web.Request) -> web.Response:
    payload = await request.json()
    state = request.app[STATE]
    state["requests_total"] += 1

    prompt = payload.get("prompt", "")
    max_tokens = int(payload.get("max_tokens") or 0)
    stream = bool(payload.get("stream", False))
    per_token_s = state["per_token_ms"] / 1000.0
    prompt_tokens = _prompt_token_count(prompt)  # computed once, reused below

    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": max_tokens,
        "total_tokens": prompt_tokens + max_tokens,
    }

    if not stream:
        await asyncio.sleep(per_token_s * max_tokens)
        body = {
            "id": "echo", "object": "text_completion",
            "choices": [{"index": 0, "text": "x" * max_tokens, "finish_reason": "length"}],
            "usage": usage,
        }
        return web.json_response(body)

    response = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await response.prepare(request)

    for i in range(max_tokens):
        await asyncio.sleep(per_token_s)
        # Fix round 1: finish_reason lands on this, the LAST content chunk --
        # not on a trailing chunk that also carries usage (see module docstring).
        chunk = {
            "id": "echo", "object": "text_completion",
            "choices": [{"index": 0, "text": "x", "finish_reason": "length" if i == max_tokens - 1 else None}],
        }
        await response.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))

    usage_chunk = {
        "id": "echo", "object": "text_completion",
        "choices": [],  # empty, not omitted or non-empty -- see module docstring
        "usage": usage,
    }
    await response.write(f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8"))
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response
