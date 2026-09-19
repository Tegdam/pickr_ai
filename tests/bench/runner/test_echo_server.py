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


def test_streaming_completion_paces_chunks_finish_reason_and_empty_choices_usage():
    """Fix round 1 (Important): `finish_reason: "length"` belongs on the LAST
    CONTENT chunk; the trailing usage chunk carries an EMPTY `choices: []`
    (not omitted, not non-empty) -- see server.py's module docstring for why
    that distinction matters to a real OpenAI-style streaming client."""
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
        content_payloads = [p for p in payloads if p["choices"]]
        usage_payloads = [p for p in payloads if not p["choices"]]

        assert len(content_payloads) == 5
        assert all(c["choices"][0]["text"] == "x" for c in content_payloads)
        assert [c["choices"][0]["finish_reason"] for c in content_payloads] == [None, None, None, None, "length"]

        assert len(usage_payloads) == 1
        assert usage_payloads[0]["choices"] == []
        usage = usage_payloads[0]["usage"]
        assert usage["completion_tokens"] == 5
        assert usage["prompt_tokens"] == 2  # "hello world" -> 2 whitespace tokens
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    _run(scenario())


def test_streaming_completion_parses_like_the_real_client():
    """Fix round 1 (Important): mirrors vLLM's own streaming-client parse
    rule -- `if choices := data.get("choices"): ... elif usage :=
    data.get("usage"): ...` -- so a regression back to a non-empty `choices`
    on the usage chunk (which would make the `if` branch win and the client
    never see `usage.completion_tokens`) fails this test, not just a
    same-shaped assertion on the raw payloads."""
    async def scenario():
        app = create_app(per_token_ms=10)
        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/v1/completions", json={
                "model": "m", "prompt": "hello world", "max_tokens": 4, "stream": True,
            })
            assert resp.status == 200

            ttft = None
            itl: list[float] = []
            last_t = None
            output_tokens = None
            t_start = time.monotonic()

            buf = b""
            async for raw in resp.content.iter_any():
                buf += raw
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    frame = frame.strip()
                    if not frame:
                        continue
                    text = frame[len(b"data: "):].decode("utf-8")
                    if text == "[DONE]":
                        continue
                    data = json.loads(text)
                    now = time.monotonic()
                    # The exact branching shape of the real client's parser.
                    if choices := data.get("choices"):
                        assert choices[0]["text"] == "x"
                        if ttft is None:
                            ttft = now - t_start
                        else:
                            itl.append(now - last_t)
                        last_t = now
                    elif usage := data.get("usage"):
                        output_tokens = usage["completion_tokens"]

        return ttft, itl, output_tokens

    ttft, itl, output_tokens = _run(scenario())

    assert output_tokens == 4  # read from usage, not re-tokenized from concatenated "x" text
    assert ttft is not None and ttft >= 0.010  # recorded at the first content chunk
    assert len(itl) == 4 - 1  # one inter-token gap per content chunk after the first


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
