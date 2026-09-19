"""End-to-end drive of `run_one` (spec §6 steps 1-9) and `run_sweep` (resume,
requeue-to-end) through the fakes at the docker/HTTP boundary (conftest.py).

No real docker or nvidia-smi is invoked: `gpu_reader` here stands in for
`bench.runner.gpu_monitor.read_gpu`, keyed off whether a `bench-*` container
is currently in `fake_docker.running` -- WSL-side usage is 0 before the
engine container starts (or after it is stopped) and 3000 while it runs,
Windows-side total/used/temp are fixed, matching the brief's calibration
numbers (total 6141, host used 0 -> resolved fraction 0.95; temp 45 <= the
default cooldown_temp_c 55).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

import bench.runner.env_capture as env_capture
import bench.runner.lifecycle as lifecycle
from bench.runner.engine import ENGINES
from bench.runner.gpu_monitor import WIN_SMI, WSL_SMI
from bench.runner.lifecycle import RunPaths, run_one, run_sweep
from bench.runner.state import SweepState
from tests.bench.runner.conftest import FakeDocker, FakeHTTP
from tests.bench.runner.test_engine import _cfg
from tests.bench.runner.test_metrics_scraper import PROM

WIN_TOTAL_MB = 6141
WIN_USED_MB = 0
WIN_TEMP_C = 45


def _make_gpu_reader(fake_docker):
    """Stands in for `read_gpu`: WSL-side usage tracks whether any `bench-*`
    container is currently running in the fake docker's `running` set."""

    def gpu_reader(cmd):
        running = any(n.startswith("bench-") for n in fake_docker.running)
        if list(cmd) == list(WSL_SMI):
            return {"used_mb": 3000 if running else 0, "total_mb": WIN_TOTAL_MB,
                     "sm_util": 50, "mem_util": 50, "sm_clock": 1500, "mem_clock": 5000,
                     "temp_c": WIN_TEMP_C, "power_w": 40.0, "throttle_reasons": "0x0"}
        return {"used_mb": WIN_USED_MB, "total_mb": WIN_TOTAL_MB,
                "sm_util": 50, "mem_util": 50, "sm_clock": 1500, "mem_clock": 5000,
                "temp_c": WIN_TEMP_C, "power_w": 40.0, "throttle_reasons": "0x0"}

    return gpu_reader


def _make_clock():
    """Monotonically increasing by a large fixed step every call, so any
    cooldown_min_s / vram-return timeout used in these tests is already
    satisfied on the very first check -- no real sleeping needed."""
    state = {"t": 0.0}

    def clock():
        state["t"] += 1000.0
        return state["t"]

    return clock


def _no_sleep(_seconds):
    return None


def _write_trace(path: Path, n: int = 4) -> str:
    rows = [
        {"record_id": f"q{i}", "call_role": "chat", "workload": "A", "prompt": f"hello {i}", "output_tokens": 50}
        for i in range(n)
    ]
    text = "\n".join(json.dumps(r) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(tmp_path, monkeypatch, client_json, *, cfg_over=None, reset_ok=True):
    """Common fixture-ish plumbing shared by the run_one tests below."""
    monkeypatch.setattr(env_capture, "_host_lines", lambda: {})

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    trace_file = traces_dir / "chat_v1.jsonl"
    sha = _write_trace(trace_file, n=4)

    results_root = tmp_path / "results"
    sweep_dir = results_root / "s1"
    cfg = _cfg(num_prompts=4, trace_file=str(trace_file), trace_sha256=sha, **(cfg_over or {}))
    paths = RunPaths(
        results_root=results_root, sweep_dir=sweep_dir, run_dir=sweep_dir / cfg.run_id,
        traces_dir=traces_dir, hf_cache_dir=tmp_path / "hfcache",
        compile_cache_root=tmp_path / "compile_cache",
    )

    fake_docker = FakeDocker()
    fake_docker.keep_running = True

    fake_http = FakeHTTP()
    if reset_ok:
        fake_http.reset_responses = [{"status": 200, "json": {"success": True}}]

    def fake_run_client(docker, cfg_, spec, run_dir, traces_dir_, hf_cache_dir, **kw):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "client_raw.json").write_text(json.dumps(client_json), encoding="utf-8")
        return client_json

    monkeypatch.setattr(lifecycle, "run_client", fake_run_client)

    return cfg, paths, fake_docker, fake_http


def test_run_one_happy_path_writes_every_artifact_and_is_valid(tmp_path, monkeypatch, client_json):
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)

    summary = run_one(
        cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
        gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
        opts={"schedule_index": 2, "attempt": 1, "gpu_headroom_mb": 256, "try_clock_pin": False},
    )

    # docker run call: launch args, env, --shm-size 2g, HF mount, vLLM compile-cache mount.
    run_calls = [c for c in fake_docker.calls if c["op"] == "run"]
    assert len(run_calls) == 1
    run_call = run_calls[0]
    assert run_call["name"] == f"bench-{cfg.run_id}"
    assert run_call["gpus"] is True and run_call["network_host"] is True
    assert run_call["env"] == ENGINES["vllm"].env
    assert run_call["extra_args"] == ["--shm-size", "2g"]
    assert "--gpu-memory-utilization 0.95" in " ".join(run_call["args"])
    assert (str(paths.hf_cache_dir), "/root/.cache/huggingface", "ro") in run_call["mounts"]
    assert (str(paths.compile_cache_root / "vllm"), "/root/.cache/vllm", "rw") in run_call["mounts"]

    # HTTP sequence: health poll(s), then warmup posts, then the reset route.
    assert fake_http.sequence == ["/health"]
    assert fake_http.posts[0][0].endswith("/v1/completions")
    assert fake_http.posts[-1][0].endswith("/reset_prefix_cache")
    assert all(u.endswith("/v1/completions") for u, _ in fake_http.posts[:-1])

    run_dir = paths.run_dir
    for name in ("config.yaml", "env.json", "requests.jsonl", "gpu_samples.jsonl",
                 "engine_metrics.jsonl", "client_raw.json", "summary.json", "meta.json", "log.txt"):
        assert (run_dir / name).exists(), name

    assert summary["valid"] is True

    config_yaml = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config_yaml["gpu_memory_utilization"] == pytest.approx(0.95)
    assert config_yaml["free_vram_mb_at_start"] == WIN_TOTAL_MB - WIN_USED_MB

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["schedule_index"] == 2
    assert meta["attempt"] == 1
    assert meta["cooldown_observed_s"] is not None
    assert meta["cooldown_start_temp_c"] == WIN_TEMP_C
    assert meta["cooldown_end_temp_c"] == WIN_TEMP_C

    # VRAM-return check: container stopped -> WSL usage back to the pre-run
    # baseline immediately -> no leak recorded.
    assert summary["timing"]["vram_leak_mb"] == 0

    # engine container was stopped exactly once.
    stop_calls = [c for c in fake_docker.calls if c["op"] == "stop"]
    assert stop_calls == [{"op": "stop", "name": f"bench-{cfg.run_id}", "timeout": 30}]


def test_run_one_warm_cache_skips_reset(tmp_path, monkeypatch, client_json):
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json, cfg_over={"cache_state": "warm"})

    run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
            gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep)

    assert not any(u.endswith("/reset_prefix_cache") for u, _ in fake_http.posts)


def test_run_one_draft_spec_never_advances_is_invalid_with_acceptance_reason(tmp_path, monkeypatch, client_json):
    cfg, paths, fake_docker, fake_http = _setup(
        tmp_path, monkeypatch, client_json,
        cfg_over={"spec_method": "draft", "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
                  "draft_revision": "def", "spec_k": 3},
    )
    # PROM's spec_decode counters never move across scrapes (same text every
    # poll) -- acceptance never advances.
    fake_http.metrics_text = PROM

    summary = run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                       gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep)

    assert summary["valid"] is False
    assert "acceptance" in summary["invalid_reason"]


def _base_sweep_dict(traces_dir: Path) -> dict:
    return dict(
        phase="p0b", model="Qwen/Qwen2.5-3B-Instruct-AWQ", model_revision="abc", quantization="awq",
        draft_model=None, draft_revision=None, draft_quantization=None,
        spec_method="off", spec_k=None, ngram_lookup_max=4,
        load_mode="concurrency", request_rate=None, burstiness=1.0, num_prompts=4,
        cache_state="cold", prefix_caching=True, max_model_len=2048, max_num_seqs=64,
        chunked_prefill_tokens=1024, cudagraph_capture_sizes=[1, 2, 4, 8],
        sampling={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0},
        ignore_eos=True, max_tokens_cap=None, warmup_requests=2, cooldown_temp_c=55, cooldown_min_s=1,
        gpu_headroom_mb=256, seed=0, extra_body={}, traces_dir=str(traces_dir), try_clock_pin=False,
    )


def _write_sweep_files(tmp_path, sweep_id="sw1", max_retries_total=2):
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    trace_file = traces_dir / "chat_v1.jsonl"
    sha = _write_trace(trace_file, n=4)
    (traces_dir / "chat_v1.meta.json").write_text(json.dumps({"trace_sha256": sha}), encoding="utf-8")

    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_base_sweep_dict(traces_dir)), encoding="utf-8")

    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump({
        "base": str(base_path), "sweep_id": sweep_id, "rq_tag": "x", "schedule_seed": 3,
        "max_retries_total": max_retries_total, "reps": 1,
        "axes": {"engine": ["vllm"], "workload": ["A"], "concurrency": [1, 4, 8]},
    }), encoding="utf-8")
    return sweep_path


def _sweep_kwargs(fake_docker, http_factory, tmp_path):
    return dict(
        docker=fake_docker, http_factory=http_factory, spec_for=lambda cfg: ENGINES["vllm"],
        gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
        hf_cache_dir=tmp_path / "hfcache", compile_cache_root=tmp_path / "compile_cache",
    )


def test_run_sweep_requeues_invalid_run_to_end_and_finishes_all_done(tmp_path, monkeypatch, client_json):
    monkeypatch.setattr(env_capture, "_host_lines", lambda: {})
    sweep_path = _write_sweep_files(tmp_path)
    results_root = tmp_path / "results"

    fake_docker = FakeDocker()
    fake_docker.keep_running = True

    flaky_run_id = "sw1-0001-r0"  # concurrency=4 point; independent of schedule shuffle order
    attempts: dict[str, int] = {}

    def fake_run_client(docker, cfg, spec, run_dir, traces_dir, hf_cache_dir, **kw):
        n = attempts.get(cfg.run_id, 0)
        attempts[cfg.run_id] = n + 1
        data = json.loads(json.dumps(client_json))
        if cfg.run_id == flaky_run_id and n == 0:
            # Short by one request -> attempted (len(requests)) != num_prompts -> invalid.
            data["completed"] = 3
            for key in ("ttfts", "itls", "input_lens", "output_lens", "generated_texts",
                        "errors", "start_times"):
                data[key] = data[key][:3]
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "client_raw.json").write_text(json.dumps(data), encoding="utf-8")
        return data

    monkeypatch.setattr(lifecycle, "run_client", fake_run_client)

    def http_factory():
        h = FakeHTTP()
        h.reset_responses = [{"status": 200, "json": {"success": True}}]
        return h

    report = run_sweep(sweep_path, results_root, resume=False, **_sweep_kwargs(fake_docker, http_factory, tmp_path))

    assert attempts[flaky_run_id] == 2  # failed once, then re-run
    state_json = json.loads((report.sweep_dir / "state.json").read_text(encoding="utf-8"))
    statuses = [r["status"] for r in state_json["runs"].values()]
    assert statuses.count("done") == 3
    assert state_json["runs"][flaky_run_id]["attempts"] == 2


def test_run_sweep_resume_resets_stale_running_run_and_completes_it(tmp_path, monkeypatch, client_json):
    monkeypatch.setattr(env_capture, "_host_lines", lambda: {})
    sweep_path = _write_sweep_files(tmp_path)
    results_root = tmp_path / "results"

    fake_docker = FakeDocker()
    fake_docker.keep_running = True

    def fake_run_client(docker, cfg, spec, run_dir, traces_dir, hf_cache_dir, **kw):
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "client_raw.json").write_text(json.dumps(client_json), encoding="utf-8")
        return client_json

    monkeypatch.setattr(lifecycle, "run_client", fake_run_client)

    def http_factory():
        h = FakeHTTP()
        h.reset_responses = [{"status": 200, "json": {"success": True}}]
        return h

    first = run_sweep(sweep_path, results_root, resume=False, **_sweep_kwargs(fake_docker, http_factory, tmp_path))
    state_json = json.loads((first.sweep_dir / "state.json").read_text(encoding="utf-8"))
    assert all(r["status"] == "done" for r in state_json["runs"].values())

    # Simulate a crash: one run left "running" in state.json.
    target = next(iter(state_json["runs"]))
    state = SweepState.load(first.sweep_dir / "state.json")
    state.mark(target, "running")

    resumed = run_sweep(sweep_path, results_root, resume=True, **_sweep_kwargs(fake_docker, http_factory, tmp_path))
    final_state = json.loads((resumed.sweep_dir / "state.json").read_text(encoding="utf-8"))
    assert final_state["runs"][target]["status"] == "done"
    assert all(r["status"] == "done" for r in final_state["runs"].values())
