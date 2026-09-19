"""Fakes at the two boundaries the runner talks through: the docker CLI and HTTP."""
import json
from types import SimpleNamespace

import pytest


class FakeDocker:
    """Records calls; returns canned outputs. `running` tracks container names."""

    def __init__(self):
        self.calls = []
        self.running = set()
        self.logs = {}
        self.digest = "sha256:" + "ab" * 32

    def run(self, image, name, args, gpus=True, network_host=True, mounts=(), env=None, entrypoint=None):
        self.calls.append(("run", image, name, list(args)))
        self.running.add(name)
        return "cid-" + name

    def stop(self, name, timeout=30):
        self.calls.append(("stop", name))
        self.running.discard(name)

    def is_running(self, name):
        return name in self.running

    def image_digest(self, image):
        return self.digest

    def pip_freeze(self, image):
        return "vllm==0.29.0\ntorch==2.9.0\n"

    def container_logs(self, name):
        return self.logs.get(name, "")


class FakeHTTP:
    """Minimal stand-in for the engine's HTTP surface."""

    def __init__(self):
        self.healthy_after = 0   # number of health polls before 200
        self.polls = 0
        self.posts = []
        self.metrics_text = ""

    def get(self, url, timeout=5):
        if url.endswith("/health"):
            self.polls += 1
            ok = self.polls > self.healthy_after
            return SimpleNamespace(status_code=200 if ok else 503, text="")
        if url.endswith("/metrics"):
            return SimpleNamespace(status_code=200, text=self.metrics_text)
        return SimpleNamespace(status_code=404, text="")

    def post(self, url, json=None, timeout=60):
        self.posts.append((url, json))
        if url.endswith("/v1/completions"):
            return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"text": "ok"}], "usage": {"prompt_tokens": 265, "completion_tokens": 4}})
        return SimpleNamespace(status_code=200, json=lambda: {})


@pytest.fixture
def fake_docker():
    return FakeDocker()


@pytest.fixture
def fake_http():
    return FakeHTTP()


@pytest.fixture
def client_json():
    """Shape of `vllm bench serve --save-result --save-detailed` output (keys verified in Task 1)."""
    return {
        "backend": "openai", "model_id": "Qwen/Qwen2.5-3B-Instruct-AWQ", "num_prompts": 4, "duration": 2.0,
        "completed": 4, "total_input_tokens": 1000, "total_output_tokens": 200,
        "request_throughput": 2.0, "output_throughput": 100.0,
        "mean_ttft_ms": 50.0, "median_ttft_ms": 48.0, "p99_ttft_ms": 60.0,
        "mean_tpot_ms": 20.0, "median_tpot_ms": 19.0, "p99_tpot_ms": 25.0,
        "mean_itl_ms": 20.0, "median_itl_ms": 19.0, "p99_itl_ms": 25.0,
        "input_lens": [250, 250, 250, 250], "output_lens": [50, 50, 50, 50],
        "ttfts": [0.048, 0.050, 0.049, 0.053], "itls": [[0.02] * 49, [0.02] * 49, [0.02] * 49, [0.02] * 49],
        "generated_texts": ["a", "b", "c", "d"], "errors": ["", "", "", ""],
    }
