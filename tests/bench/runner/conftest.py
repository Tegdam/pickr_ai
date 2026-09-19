"""Fakes at the two boundaries the runner talks through: the docker CLI and HTTP."""
from types import SimpleNamespace

import pytest


class FakeDocker:
    """Records calls; returns canned outputs. `running` tracks container names.

    Client lifecycle knobs (task 6): a `run()` container exits immediately
    (removed from `running`) unless `keep_running` is set, so a poll loop that
    waits on `is_running` finishes on the very next check by default -- set
    `keep_running = True` to simulate a container that never exits (timeout
    tests). `fail_next` makes the *next* `run()` record a non-zero exit code,
    read back via `exit_code(name)`.
    """

    def __init__(self):
        self.calls = []
        self.running = set()
        self.logs = {}
        self.digest = "sha256:" + "ab" * 32
        self.exit_codes = {}
        self.fail_next = False
        self.keep_running = False

    def run(self, image, name, args, gpus=True, network_host=True, mounts=(), env=None, entrypoint=None, extra_args=None):
        self.calls.append({
            "op": "run", "image": image, "name": name, "args": list(args),
            "gpus": gpus, "network_host": network_host, "mounts": list(mounts),
            "env": dict(env or {}), "entrypoint": entrypoint, "extra_args": list(extra_args or []),
        })
        if self.keep_running:
            self.running.add(name)
        else:
            self.running.discard(name)
        self.exit_codes[name] = 1 if self.fail_next else self.exit_codes.get(name, 0)
        self.fail_next = False
        return "cid-" + name

    def stop(self, name, timeout=30):
        self.calls.append({"op": "stop", "name": name, "timeout": timeout})
        self.running.discard(name)

    def is_running(self, name):
        return name in self.running

    def exit_code(self, name):
        return self.exit_codes.get(name, 0)

    def image_digest(self, image):
        return self.digest

    def pip_freeze(self, image):
        return "vllm==0.29.0\ntorch==2.9.0\n"

    def container_logs(self, name):
        return self.logs.get(name, "")

    def build(self, tag, dockerfile, context):
        self.calls.append({"op": "build", "tag": tag, "dockerfile": dockerfile, "context": context})
        return "build output"


class FakeHTTP:
    """Minimal stand-in for the engine's HTTP surface."""

    def __init__(self):
        self.healthy_after = 0   # number of health polls before 200
        self.polls = 0
        self.ready_after = 0     # number of /ready polls before 200
        self.ready_polls = 0
        self.posts = []
        self.metrics_text = ""
        self.sequence = []       # "/health" / "/ready" paths hit, in poll order
        self.reset_responses = []  # [{"status": 200, "json": {...}, "text": "..."}], consumed in order for reset-cache calls

    def get(self, url, timeout=5):
        if url.endswith("/ready"):
            self.sequence.append("/ready")
            self.ready_polls += 1
            ok = self.ready_polls > self.ready_after
            return SimpleNamespace(status_code=200 if ok else 503, text="")
        if url.endswith("/health"):
            self.sequence.append("/health")
            self.polls += 1
            ok = self.polls > self.healthy_after
            return SimpleNamespace(status_code=200 if ok else 503, text="")
        if url.endswith("/metrics"):
            return SimpleNamespace(status_code=200, text=self.metrics_text)
        return SimpleNamespace(status_code=404, text="")

    def post(self, url, json=None, timeout=60):
        self.posts.append((url, json))
        if url.endswith("/v1/completions"):
            return SimpleNamespace(status_code=200, text="", json=lambda: {"choices": [{"text": "ok"}], "usage": {"prompt_tokens": 265, "completion_tokens": 4}})
        if self.reset_responses:
            entry = self.reset_responses.pop(0)
            body_json = entry.get("json", {})
            return SimpleNamespace(status_code=entry.get("status", 200), json=lambda: body_json, text=entry.get("text", ""))
        return SimpleNamespace(status_code=200, json=lambda: {}, text="")


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
        "start_times": [0.0, 0.01, 0.02, 0.03],
    }
