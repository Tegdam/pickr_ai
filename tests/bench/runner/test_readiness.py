import pytest

from bench.runner.engine import ENGINES
from bench.runner.readiness import reset_cache, wait_healthy, warmup


def test_wait_healthy_polls_until_200(fake_http):
    fake_http.healthy_after = 3
    waited = wait_healthy(fake_http, "http://localhost:8000", "/health", timeout_s=10, poll_s=0.0)
    assert fake_http.polls == 4 and waited >= 0


def test_wait_healthy_times_out_with_logs(fake_http, fake_docker):
    fake_http.healthy_after = 10**6
    fake_docker.logs["c"] = "\n".join(f"line{i}" for i in range(100))
    with pytest.raises(TimeoutError) as e:
        wait_healthy(fake_http, "http://localhost:8000", "/health", timeout_s=0.01, poll_s=0.0, docker=fake_docker, container="c")
    assert "line99" in str(e.value) and "line10" not in str(e.value)   # last 40 lines only


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


def test_wait_healthy_ready_path_times_out_with_logs(fake_http, fake_docker):
    """Health passes immediately but /ready never does -- still times out with log tail."""
    fake_http.healthy_after = 0
    fake_http.ready_after = 10**6
    fake_docker.logs["c"] = "\n".join(f"line{i}" for i in range(100))
    with pytest.raises(TimeoutError) as e:
        wait_healthy(
            fake_http, "http://localhost:30000", "/health", timeout_s=0.01, poll_s=0.0,
            docker=fake_docker, container="c", ready_path="/ready",
        )
    assert "line99" in str(e.value) and "line10" not in str(e.value)


def test_warmup_posts_each_prompt_to_completions(fake_http):
    n = warmup(fake_http, "http://localhost:8000", "m", ["p1", "p2", "p3"], max_tokens=4)
    assert n == 3
    assert [u for u, _ in fake_http.posts] == ["http://localhost:8000/v1/completions"] * 3
    assert fake_http.posts[0][1] == {"model": "m", "prompt": "p1", "max_tokens": 4, "temperature": 0}


def test_reset_cache_uses_engine_route(fake_http):
    reset_cache(fake_http, "http://localhost:8000", ENGINES["vllm"])
    assert fake_http.posts[-1][0] == "http://localhost:8000" + ENGINES["vllm"].reset_cache_path


def test_reset_cache_uses_get_when_method_is_get(fake_http):
    """SGLang's route accepts GET or POST; reset_cache must honor whichever
    reset_cache_method the EngineSpec table says (doc §4.3, §8) -- a GET reset
    goes through http.get, not http.post (FakeHTTP.get 404s on an unmodelled
    path, which is enough to prove the GET branch, not the POST one, ran)."""
    from dataclasses import replace

    get_spec = replace(ENGINES["sglang"], reset_cache_method="GET")
    with pytest.raises(RuntimeError):
        reset_cache(fake_http, "http://localhost:30000", get_spec)
    assert not any(u.endswith(get_spec.reset_cache_path) for u, _ in fake_http.posts)


def test_reset_cache_raises_on_non_2xx(fake_http, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(fake_http, "post", lambda url, json=None, timeout=30: SimpleNamespace(status_code=500))
    with pytest.raises(RuntimeError):
        reset_cache(fake_http, "http://localhost:8000", ENGINES["vllm"])
