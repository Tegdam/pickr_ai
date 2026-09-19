"""In-process tests for the echo server (Task 9): the aiohttp app is started
via `aiohttp.test_utils.TestServer`/`TestClient` on a free port inside this
process -- no real docker, no GPU, no separate subprocess. There is no
pytest-asyncio/pytest-aiohttp plugin installed in this repo's venv, so each
test drives its own coroutine with a bare `asyncio.run(...)`.
"""
from __future__ import annotations

import asyncio
import json
import time

from aiohttp.test_utils import TestClient, TestServer

from bench.echo_server.server import create_app


def _run(coro):
    return asyncio.run(coro)


async def _post_stream(client, **payload):
    resp = await client.post("/v1/completions", json=payload)
    assert resp.status == 200
    raw = await resp.content.read()
    events = []
    for frame in raw.split(b"\n\n"):
        frame = frame.strip()
        if not frame:
            continue
        assert frame.startswith(b"data: "), frame
        events.append(frame[len(b"data: "):].decode("utf-8"))
    return events


def test_streaming_completion_paces_chunks_and_ends_with_usage_and_done():
    async def scenario():
        app = create_app(per_token_ms=20)
        async with TestServer(app) as server, TestClient(server) as client:
            t0 = time.monotonic()
            events = await _post_stream(
                client, model="m", prompt="hello world", max_tokens=5, stream=True,
            )
            elapsed = time.monotonic() - t0

        assert elapsed >= 5 * 0.020
        assert events[-1] == "[DONE]"

        payloads = [json.loads(e) for e in events[:-1]]
        content_chunks = [p for p in payloads if p["choices"][0]["finish_reason"] is None]
        final_chunks = [p for p in payloads if p["choices"][0]["finish_reason"] == "length"]

        assert len(content_chunks) == 5
        assert all(c["choices"][0]["text"] == "x" for c in content_chunks)

        assert len(final_chunks) == 1
        usage = final_chunks[0]["usage"]
        assert usage["completion_tokens"] == 5
        assert usage["prompt_tokens"] == 2  # "hello world" -> 2 whitespace tokens
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    _run(scenario())


def test_non_streaming_completion_waits_full_delay_and_returns_whole_text():
    async def scenario():
        app = create_app(per_token_ms=10)
        async with TestServer(app) as server, TestClient(server) as client:
            t0 = time.monotonic()
            resp = await client.post("/v1/completions", json={
                "model": "m", "prompt": "one two three", "max_tokens": 4, "stream": False,
            })
            elapsed = time.monotonic() - t0
            assert resp.status == 200
            body = await resp.json()

        assert elapsed >= 4 * 0.010
        choice = body["choices"][0]
        assert choice["finish_reason"] == "length"
        assert choice["text"] == "xxxx"
        assert body["usage"] == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}

    _run(scenario())


def test_health_reset_and_metrics():
    async def scenario():
        app = create_app(per_token_ms=1)
        async with TestServer(app) as server, TestClient(server) as client:
            health = await client.get("/health")
            assert health.status == 200

            await client.post("/v1/completions", json={
                "model": "m", "prompt": "hi", "max_tokens": 1, "stream": False,
            })

            metrics = await client.get("/metrics")
            assert metrics.status == 200
            text = await metrics.text()
            assert "echo:requests_total 1" in text

            reset = await client.post("/reset_prefix_cache", json={})
            assert reset.status == 200
            assert await reset.json() == {"success": True}

    _run(scenario())
