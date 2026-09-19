import pytest

from bench.runner.engine import ENGINES
from bench.runner.readiness import reset_cache, wait_healthy, warmup


def test_wait_healthy_polls_until_200(fake_http):
    fake_http.healthy_after = 3
    waited = wait_healthy(fake_http, "http://localhost:8000", "/health", timeout_s=10, poll_s=0.0)
    assert fake_http.polls == 4 and waited >= 0


def test_wait_healthy_times_out_with_logs(fake_http, fake_docker):
    fake_http.healthy_after = 10**6
    fake_docker.running.add("c")  # container is alive and slow, not crashed -- a genuine timeout
    fake_docker.logs["c"] = "\n".join(f"line{i}" for i in range(100))
    with pytest.raises(TimeoutError) as e:
        wait_healthy(fake_http, "http://localhost:8000", "/health", timeout_s=0.01, poll_s=0.0, docker=fake_docker, container="c")
    assert "line99" in str(e.value) and "line10" not in str(e.value)   # last 40 lines only


def test_wait_healthy_raises_when_container_exits_during_startup(fake_http, fake_docker):
    """Fix round 1, item 6: wait_healthy cannot see a dead engine from HTTP
    alone -- a crashed launch must not cost the full readiness timeout. The
    container is running for the first 2 polls, then gone (simulating a
    crash after the process starts but before it ever answers /health)."""
    fake_http.healthy_after = 10**6  # never becomes healthy via HTTP
    fake_docker.running.add("c")
    fake_docker.logs["c"] = "\n".join(f"line{i}" for i in range(50))
    calls = {"n": 0}

    def flaky_is_running(name):
        calls["n"] += 1
        return calls["n"] <= 2

    fake_docker.is_running = flaky_is_running

    with pytest.raises(RuntimeError, match="engine container exited during startup") as e:
        wait_healthy(fake_http, "http://localhost:8000", "/health", timeout_s=600, poll_s=0.0,
                     docker=fake_docker, container="c")
    assert "line49" in str(e.value)


def test_wait_healthy_polls_ready_path_after_health(fake_http):
    """SGLang (doc §4.3, §8): poll /health, then /ready, within one overall budget."""
    fake_http.healthy_after = 0
    fake_http.ready_after = 2
    waited = wait_healthy(
        fake_http, "http://localhost:30000", "/health", timeout_s=10, poll_s=0.0, ready_path="/ready"
    )
    assert fake_http.polls == 1
    assert fake_http.ready_polls == 3
    assert waited >= 0


def test_wait_healthy_polls_health_before_ready_in_order(fake_http):
    """Fix round 1, minor 3: assert the two-phase poll ORDER, not just the counts."""
    fake_http.healthy_after = 2
    fake_http.ready_after = 1
    wait_healthy(fake_http, "http://localhost:30000", "/health", timeout_s=10, poll_s=0.0, ready_path="/ready")
    first_ready_idx = fake_http.sequence.index("/ready")
    assert fake_http.sequence[:first_ready_idx] == ["/health"] * first_ready_idx
    assert all(p == "/ready" for p in fake_http.sequence[first_ready_idx:])
    assert first_ready_idx > 0


def test_wait_healthy_ready_path_times_out_with_logs(fake_http, fake_docker):
    """Health passes immediately but /ready never does -- still times out with log tail."""
    fake_http.healthy_after = 0
    fake_http.ready_after = 10**6
    fake_docker.running.add("c")  # container is alive and slow, not crashed -- a genuine timeout
    fake_docker.logs["c"] = "\n".join(f"line{i}" for i in range(100))
    with pytest.raises(TimeoutError) as e:
        wait_healthy(
            fake_http, "http://localhost:30000", "/health", timeout_s=0.01, poll_s=0.0,
            docker=fake_docker, container="c", ready_path="/ready",
        )
    assert "line99" in str(e.value) and "line10" not in str(e.value)


def test_wait_healthy_recovers_from_connection_errors(fake_http, monkeypatch):
    """Fix round 1, minor 4: the container's port can refuse connections outright
    before the process is listening at all -- wait_healthy must swallow that and
    keep polling, not just non-200 responses."""
    real_get = fake_http.get
    state = {"n": 0}

    def flaky_get(url, timeout=5):
        state["n"] += 1
        if state["n"] <= 2:
            raise ConnectionError("not listening yet")
        return real_get(url, timeout=timeout)

    monkeypatch.setattr(fake_http, "get", flaky_get)
    waited = wait_healthy(fake_http, "http://localhost:8000", "/health", timeout_s=10, poll_s=0.0)
    assert waited >= 0
    assert state["n"] == 3


def test_warmup_posts_each_prompt_to_completions(fake_http):
    n = warmup(fake_http, "http://localhost:8000", "m", ["p1", "p2", "p3"], max_tokens=4)
    assert n == 3
    assert [u for u, _ in fake_http.posts] == ["http://localhost:8000/v1/completions"] * 3
    assert fake_http.posts[0][1] == {"model": "m", "prompt": "p1", "max_tokens": 4, "temperature": 0}


def _reset_posts(fake_http, spec):
    return [u for u, _ in fake_http.posts if u.endswith(spec.reset_cache_path)]


def test_reset_cache_uses_engine_route_and_checks_vllm_success_body(fake_http):
    fake_http.reset_responses = [{"status": 200, "json": {"success": True}, "text": ""}]
    reset_cache(fake_http, "http://localhost:8000", ENGINES["vllm"], sleep=lambda s: None)
    assert _reset_posts(fake_http, ENGINES["vllm"]) == ["http://localhost:8000" + ENGINES["vllm"].reset_cache_path]


def test_reset_cache_vllm_retries_while_refused_then_succeeds(fake_http):
    """doc §3.3/§8: {"success": false} while blocks are still held by running
    requests -- must retry, not treat the 200 as success."""
    fake_http.reset_responses = [
        {"status": 200, "json": {"success": False}, "text": ""},
        {"status": 200, "json": {"success": False}, "text": ""},
        {"status": 200, "json": {"success": True}, "text": ""},
    ]
    sleeps = []
    reset_cache(fake_http, "http://localhost:8000", ENGINES["vllm"], sleep=sleeps.append)
    assert len(_reset_posts(fake_http, ENGINES["vllm"])) == 3
    assert sleeps == [1.0, 1.0]


def test_reset_cache_vllm_raises_after_exhausting_attempts(fake_http):
    fake_http.reset_responses = [{"status": 200, "json": {"success": False}, "text": ""} for _ in range(10)]
    with pytest.raises(RuntimeError, match="refused after 5 attempts"):
        reset_cache(fake_http, "http://localhost:8000", ENGINES["vllm"], sleep=lambda s: None)
    assert len(_reset_posts(fake_http, ENGINES["vllm"])) == 5


def test_reset_cache_sglang_accepts_body_starting_with_cache_flushed(fake_http):
    fake_http.reset_responses = [
        {"status": 200, "text": "Cache flushed.\nPlease check backend logs for more details.\n", "json": {}},
    ]
    reset_cache(fake_http, "http://localhost:30000", ENGINES["sglang"], sleep=lambda s: None)
    assert len(_reset_posts(fake_http, ENGINES["sglang"])) == 1


def test_reset_cache_sglang_refused_body_retries_then_succeeds(fake_http):
    """doc §4.3/§8: status is 200 even when the flush is refused (requests
    running or queued) -- the runner must check the body, not the code."""
    fake_http.reset_responses = [
        {"status": 200, "text": "There are running or waiting requests. Cache not flushed.\n", "json": {}},
        {"status": 200, "text": "Cache flushed.\n", "json": {}},
    ]
    sleeps = []
    reset_cache(fake_http, "http://localhost:30000", ENGINES["sglang"], sleep=sleeps.append)
    assert len(_reset_posts(fake_http, ENGINES["sglang"])) == 2
    assert sleeps == [1.0]


def test_reset_cache_uses_get_when_method_is_get(fake_http):
    """SGLang's route accepts GET or POST; reset_cache must honor whichever
    reset_cache_method the EngineSpec table says (doc §4.3, §8) -- a GET reset
    goes through http.get, not http.post (FakeHTTP.get 404s on an unmodelled
    path, which is enough to prove the GET branch, not the POST one, ran)."""
    from dataclasses import replace

    get_spec = replace(ENGINES["sglang"], reset_cache_method="GET")
    with pytest.raises(RuntimeError):
        reset_cache(fake_http, "http://localhost:30000", get_spec, sleep=lambda s: None)
    assert not any(u.endswith(get_spec.reset_cache_path) for u, _ in fake_http.posts)


def test_reset_cache_raises_on_non_2xx(fake_http, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(fake_http, "post", lambda url, json=None, timeout=30: SimpleNamespace(status_code=500, text=""))
    with pytest.raises(RuntimeError):
        reset_cache(fake_http, "http://localhost:8000", ENGINES["vllm"], sleep=lambda s: None)
