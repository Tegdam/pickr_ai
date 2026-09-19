"""Server readiness: 'port open' is not 'model loaded and warm' (spec §4 item 5).

wait_healthy polls health_path until 200; when the engine also exposes a
readiness route (SGLang's /ready, doc §4.3) it then polls that too, within the
same overall timeout budget -- vLLM has no separate readiness route at this
pin (doc §3.3), so ready_path stays None there (see EngineSpec.ready_path).
"""
from __future__ import annotations

import time

from .engine import EngineSpec


def wait_healthy(http, base_url, health_path, timeout_s, poll_s=2.0, docker=None, container=None,
                  ready_path=None) -> float:
    t0 = time.monotonic()

    def _poll(path: str) -> None:
        while True:
            try:
                if http.get(base_url + path, timeout=5).status_code == 200:
                    return
            except Exception:
                pass
            if time.monotonic() - t0 > timeout_s:
                tail = ""
                if docker is not None and container:
                    tail = "\n".join(docker.container_logs(container).splitlines()[-40:])
                raise TimeoutError(f"{base_url}{path} not healthy after {timeout_s}s\n{tail}")
            time.sleep(poll_s)

    _poll(health_path)
    if ready_path:
        _poll(ready_path)
    return time.monotonic() - t0


def warmup(http, base_url, model, prompts, max_tokens=8) -> int:
    n = 0
    for p in prompts:
        r = http.post(f"{base_url}/v1/completions", json={"model": model, "prompt": p, "max_tokens": max_tokens, "temperature": 0}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"warmup request failed: {r.status_code}")
        n += 1
    return n


def reset_cache(http, base_url, spec: EngineSpec) -> None:
    url = base_url + spec.reset_cache_path
    r = http.post(url, json={}, timeout=30) if spec.reset_cache_method == "POST" else http.get(url, timeout=30)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"cache reset failed: {spec.reset_cache_method} {url} -> {r.status_code}")
