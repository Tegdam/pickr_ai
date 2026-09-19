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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import bench.runner.env_capture as env_capture
import bench.runner.lifecycle as lifecycle
from bench.runner.client import CLIENT_IMAGE
from bench.runner.engine import ENGINES
from bench.runner.gpu_monitor import GpuSampler, WIN_SMI, WSL_SMI
from bench.runner.lifecycle import PreflightError, RunPaths, run_one, run_sweep
from bench.runner.metrics_scraper import MetricsScraper
from bench.runner.state import SweepState
from tests.bench.runner.conftest import FakeDocker, FakeHTTP
from tests.bench.runner.test_engine import _cfg

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

    # Important 2: start_engine defensively stops any leaked container of the
    # same name BEFORE docker.run -- so the very first two calls are stop, run.
    assert fake_docker.calls[0]["op"] == "stop" and fake_docker.calls[0]["name"] == f"bench-{cfg.run_id}"
    assert fake_docker.calls[1]["op"] == "run"

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
    assert summary["timing"]["wsl_pre_used_mb"] == WIN_USED_MB
    assert summary["timing"]["win_pre_used_mb"] == WIN_USED_MB
    assert "clocks_pinned" not in summary["timing"]  # moved to env.json only

    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["client_image_digest"] == fake_docker.digest

    # engine container was stopped twice: the defensive pre-run stop
    # (Important 2) and the real end-of-run cleanup.
    stop_calls = [c for c in fake_docker.calls if c["op"] == "stop"]
    assert stop_calls == [{"op": "stop", "name": f"bench-{cfg.run_id}", "timeout": 30}] * 2


def test_run_one_warm_cache_skips_reset(tmp_path, monkeypatch, client_json):
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json, cfg_over={"cache_state": "warm"})

    run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
            gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep)

    assert not any(u.endswith("/reset_prefix_cache") for u, _ in fake_http.posts)


class _FakePopen:
    """Stands in for `subprocess.Popen` (Task 9): `poll()` is None until
    `terminate()`/`kill()` sets a returncode, exactly like a real live process."""

    def __init__(self, args, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        self.returncode = None
        self.terminate_called = False
        self.wait_called = False
        self.kill_called = False
        self.stderr = SimpleNamespace(read=lambda: "")

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_called = True
        self.returncode = 0

    def wait(self, timeout=None):
        self.wait_called = True
        return self.returncode

    def kill(self):
        self.kill_called = True
        self.returncode = -9


def test_run_one_echo_engine_is_a_subprocess_not_a_docker_container(tmp_path, monkeypatch, client_json):
    """Task 9: ENGINES["echo"] has spec.image=None -- start_engine must launch
    a local subprocess instead of a docker container, while the *client*
    container is unaffected (still runs on --network host so it can reach
    localhost:8000)."""
    monkeypatch.setattr(env_capture, "_host_lines", lambda: {})

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    trace_file = traces_dir / "chat_v1.jsonl"
    sha = _write_trace(trace_file, n=4)

    results_root = tmp_path / "results"
    sweep_dir = results_root / "s1"
    cfg = _cfg(engine="echo", image=None, num_prompts=4, trace_file=str(trace_file), trace_sha256=sha)
    paths = RunPaths(
        results_root=results_root, sweep_dir=sweep_dir, run_dir=sweep_dir / cfg.run_id,
        traces_dir=traces_dir, hf_cache_dir=tmp_path / "hfcache",
        compile_cache_root=tmp_path / "compile_cache",
    )

    fake_docker = FakeDocker()
    fake_http = FakeHTTP()
    fake_http.reset_responses = [{"status": 200, "json": {"success": True}}]

    # Real (non-monkeypatched) run_client is exercised here so the client
    # container's own docker.run call can be asserted directly -- seed the
    # result file docker.run would otherwise produce inside the container.
    real_docker_run = fake_docker.run

    def run_and_seed_client_result(image, name, args, **kw):
        cid = real_docker_run(image, name, args, **kw)
        if image == CLIENT_IMAGE:
            paths.run_dir.mkdir(parents=True, exist_ok=True)
            (paths.run_dir / "client_raw.json").write_text(json.dumps(client_json), encoding="utf-8")
        return cid

    fake_docker.run = run_and_seed_client_result

    created_popens: list[_FakePopen] = []

    def fake_popen(args, **kwargs):
        p = _FakePopen(args, **kwargs)
        created_popens.append(p)
        return p

    summary = run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["echo"],
                       gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
                       popen=fake_popen)

    # No docker `run` for the engine -- the only docker.run call is the client.
    run_calls = [c for c in fake_docker.calls if c["op"] == "run"]
    assert len(run_calls) == 1
    assert run_calls[0]["image"] == CLIENT_IMAGE
    assert run_calls[0]["network_host"] is True and run_calls[0]["gpus"] is False

    assert len(created_popens) == 1
    proc = created_popens[0]
    assert proc.args[1:] == ["-m", "bench.echo_server", "--port", "8000", "--per-token-ms", "5"]
    assert proc.terminate_called and proc.wait_called, "stop_engine must terminate the subprocess"

    assert summary["valid"] is True
    assert summary["cooldown_skipped"] == "echo"

    meta = json.loads((paths.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["cooldown_skipped"] == "echo"

    env = json.loads((paths.run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["image"] is None and env["image_digest"] is None and env["pip_freeze"] is None
    # The client image's own digest is still captured (unaffected by the engine having none).
    assert env["client_image_digest"] == fake_docker.digest


def test_run_one_echo_engine_crash_before_healthy_raises_with_stderr(tmp_path, monkeypatch, client_json):
    """Task 9: a dead subprocess (poll() is not None) must fail fast with its
    captured stderr, rather than burning the full readiness timeout polling a
    port nothing is listening on any more."""
    cfg, paths, fake_docker, fake_http = _setup(
        tmp_path, monkeypatch, client_json, cfg_over={"engine": "echo", "image": None},
    )

    class _DeadPopen(_FakePopen):
        def __init__(self, args, **kwargs):
            super().__init__(args, **kwargs)
            self.returncode = 1
            self.stderr = SimpleNamespace(read=lambda: "Traceback: boom")

    with pytest.raises(RuntimeError, match="Traceback: boom"):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["echo"],
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
                popen=_DeadPopen)


class _NoopGpuSampler:
    """Deterministic stand-in for `GpuSampler` (Task 9's wrapped-reader test
    below): the point of that test is to control exactly which `gpu_reader`
    call fails, which a real background-thread sampler's timing would make
    nondeterministic."""

    def __init__(self, out_path, interval_s=1.0, reader=None, **kw):
        self.out_path = Path(out_path)

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text("", encoding="utf-8")

    def stop(self) -> None:
        pass


def test_run_one_cooldown_reader_failure_is_recorded_not_raised(tmp_path, monkeypatch, client_json):
    """Carried from Task 8 re-review: a gpu_reader flake during the post-run
    cooldown wait must never raise out of run_one's `finally` -- that would
    mask an otherwise-valid run's own result. `cooldown_error` is recorded on
    meta.json instead and the run stays valid."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    monkeypatch.setattr(lifecycle, "GpuSampler", _NoopGpuSampler)

    base_reader = _make_gpu_reader(fake_docker)
    calls = {"n": 0}

    def flaky_reader(cmd):
        calls["n"] += 1
        # Calls 1-2: pre-flight. Call 3: the VRAM-return check (succeeds).
        # Call 4 onward: _cooldown's own reads -- fail those.
        if calls["n"] >= 4:
            raise RuntimeError("nvidia-smi flaked")
        return base_reader(cmd)

    summary = run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                       gpu_reader=flaky_reader, clock=_make_clock(), sleep=_no_sleep)

    assert summary["valid"] is True
    assert summary["timing"]["vram_leak_mb"] == 0  # the VRAM-return check itself never saw the flake

    meta = json.loads((paths.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert "cooldown_error" in meta and "nvidia-smi flaked" in meta["cooldown_error"]
    assert "cooldown_observed_s" not in meta  # _cooldown raised before returning its normal dict


def test_run_one_vram_return_reader_failure_is_recorded_not_raised(tmp_path, monkeypatch, client_json):
    """Same ruling as above, for the VRAM-return check specifically."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    monkeypatch.setattr(lifecycle, "GpuSampler", _NoopGpuSampler)

    calls = {"n": 0}

    def flaky_reader(cmd):
        calls["n"] += 1
        if calls["n"] == 3:  # the VRAM-return check's own read
            raise RuntimeError("nvidia-smi flaked")
        return _make_gpu_reader(fake_docker)(cmd)

    summary = run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                       gpu_reader=flaky_reader, clock=_make_clock(), sleep=_no_sleep)

    assert summary["valid"] is True
    assert summary["timing"]["vram_returned_mb"] is None
    assert summary["timing"]["vram_leak_mb"] is None
    assert "nvidia-smi flaked" in summary["timing"]["vram_return_error"]
    # cooldown still ran normally (it is a separate wrapped call).
    meta = json.loads((paths.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["cooldown_observed_s"] is not None


def test_run_one_preflight_failure_writes_preflight_error_txt(tmp_path, monkeypatch, client_json):
    """Task 8 re-review, carried into Task 9: a step-1 PreflightError leaves a
    human-readable breadcrumb in the run's own directory."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    fake_docker.running.add("bench-leaked")

    with pytest.raises(PreflightError):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep)

    text = (paths.run_dir / "preflight_error.txt").read_text(encoding="utf-8")
    assert "bench-leaked" in text


class _StaticMetricsScraper:
    """Fix round 1, minor: a deterministic stand-in for `MetricsScraper` that
    writes >=2 IDENTICAL rows synchronously instead of racing a real
    background-thread scrape against the mocked, near-instant client phase --
    the previous version of this test could get 0 or 1 real samples depending
    on thread scheduling, which happened to still prove the point (both
    `None` and a zero delta invalidate) but never actually exercised "the
    counters moved across two samples and the delta is zero"."""

    def __init__(self, http, base_url, spec, out_path, interval_s=1.0):
        self.out_path = Path(out_path)

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        row = {"t": 0.0, "wall": "x", "engine": "vllm", "raw": {},
               "kv_usage": None, "running": None, "waiting": None,
               "spec_accepted": 120.0, "spec_draft": 200.0, "spec_drafts": 40.0,
               "prefix_hits": None, "prefix_queries": None}
        with self.out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.write(json.dumps(row) + "\n")

    def stop(self) -> None:
        pass


def test_run_one_draft_spec_never_advances_is_invalid_with_acceptance_reason(tmp_path, monkeypatch, client_json):
    cfg, paths, fake_docker, fake_http = _setup(
        tmp_path, monkeypatch, client_json,
        cfg_over={"spec_method": "draft", "draft_model": "Qwen/Qwen2.5-0.5B-Instruct",
                  "draft_revision": "def", "spec_k": 3},
    )
    monkeypatch.setattr(lifecycle, "MetricsScraper", _StaticMetricsScraper)

    summary = run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                       gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep)

    assert summary["valid"] is False
    assert "acceptance" in summary["invalid_reason"]
    metric_rows = [json.loads(l) for l in
                   (paths.run_dir / "engine_metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(metric_rows) >= 2


def test_run_one_isolates_previous_attempt_artifacts(tmp_path, monkeypatch, client_json):
    """Critical 1: a retried run reuses `run_dir`; anything already there from
    a prior attempt must be moved aside (never deleted, never left in place
    to bleed into this attempt's append-mode sampler files)."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    run_dir = paths.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    stale_row = {"used_ours_mb": 999999, "used_host_mb": 1, "sm_clock": 1000,
                 "power_w": 1.0, "throttle_reasons": "0x0"}
    (run_dir / "gpu_samples.jsonl").write_text(json.dumps(stale_row) + "\n", encoding="utf-8")
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")

    summary = run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                       gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep)

    assert (run_dir / "attempt-1" / "gpu_samples.jsonl").exists()
    assert (run_dir / "attempt-1" / "summary.json").exists()
    moved = json.loads((run_dir / "attempt-1" / "gpu_samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert moved["used_ours_mb"] == 999999

    new_rows = [json.loads(l) for l in
                (run_dir / "gpu_samples.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all(r.get("used_ours_mb") != 999999 for r in new_rows)
    assert summary["peak_used_ours_mb"] != 999999


def test_run_one_preflight_error_on_none_gpu_totals(tmp_path, monkeypatch, client_json):
    """Important 3 / minor: a reader returning None must raise PreflightError,
    never a bare TypeError from `total_mb - host_used_mb`."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)

    def bad_reader(cmd):
        return {"used_mb": None, "total_mb": None}

    with pytest.raises(PreflightError):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                gpu_reader=bad_reader, clock=_make_clock(), sleep=_no_sleep)


def test_run_one_wait_healthy_timeout_stops_engine_and_writes_log(tmp_path, monkeypatch, client_json):
    """Important 7(a): a readiness timeout stops the engine, never starts the
    samplers, still writes log.txt with the engine's own logs, and propagates."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    fake_http.healthy_after = 10**6  # never becomes healthy
    fake_docker.logs[f"bench-{cfg.run_id}"] = "engine boot log line"
    tiny_timeout_spec = replace(ENGINES["vllm"], readiness_timeout_s=0.01)

    with pytest.raises(TimeoutError):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=tiny_timeout_spec,
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep)

    assert f"bench-{cfg.run_id}" not in fake_docker.running
    stop_calls = [c for c in fake_docker.calls if c["op"] == "stop" and c["name"] == f"bench-{cfg.run_id}"]
    # one defensive pre-run stop (Important 2) + one real cleanup after the failure.
    assert len(stop_calls) == 2

    log_text = (paths.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "engine boot log line" in log_text
    assert not (paths.run_dir / "gpu_samples.jsonl").exists()
    assert not (paths.run_dir / "engine_metrics.jsonl").exists()

    meta = json.loads((paths.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert "not healthy" in meta["error"]


def test_run_one_client_exception_stops_engine_and_samplers(tmp_path, monkeypatch, client_json):
    """Important 7(b): a run_client failure still stops the engine and the
    sampler/scraper background threads, then propagates."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)

    created = []

    class TrackingSampler(GpuSampler):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.append(self)

    class TrackingScraper(MetricsScraper):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created.append(self)

    monkeypatch.setattr(lifecycle, "GpuSampler", TrackingSampler)
    monkeypatch.setattr(lifecycle, "MetricsScraper", TrackingScraper)

    def failing_run_client(*a, **kw):
        raise RuntimeError("client blew up")

    monkeypatch.setattr(lifecycle, "run_client", failing_run_client)

    with pytest.raises(RuntimeError, match="client blew up"):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep)

    assert f"bench-{cfg.run_id}" not in fake_docker.running
    assert created, "expected sampler/scraper instances to have been created"
    for obj in created:
        assert obj._thread is None or not obj._thread.is_alive()


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

    # Important 7(c): the flaky run must land LAST in state.order right after
    # its requeue -- spy on SweepState.requeue to capture that snapshot.
    orders_after_requeue: list[list[str]] = []
    orig_requeue = SweepState.requeue

    def spy_requeue(self, run_id):
        result = orig_requeue(self, run_id)
        orders_after_requeue.append(list(self.order))
        return result

    monkeypatch.setattr(SweepState, "requeue", spy_requeue)

    report = run_sweep(sweep_path, results_root, resume=False, **_sweep_kwargs(fake_docker, http_factory, tmp_path))

    assert attempts[flaky_run_id] == 2  # failed once, then re-run
    assert orders_after_requeue and orders_after_requeue[0][-1] == flaky_run_id
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


def test_run_sweep_preflight_failure_raises_and_leaves_run_running(tmp_path, monkeypatch, client_json):
    """Important 3: a bench-* container already present (e.g. a leaked
    container from an unrelated crash) must not be treated as a per-run
    invalidity -- run_sweep re-raises PreflightError, and the run stays
    'running' (never marked invalid/requeued) so a resume's reset_stale()
    recovers it."""
    monkeypatch.setattr(env_capture, "_host_lines", lambda: {})
    sweep_path = _write_sweep_files(tmp_path)
    results_root = tmp_path / "results"

    fake_docker = FakeDocker()
    fake_docker.keep_running = True
    fake_docker.running.add("bench-leaked-from-elsewhere")

    def http_factory():
        h = FakeHTTP()
        h.reset_responses = [{"status": 200, "json": {"success": True}}]
        return h

    with pytest.raises(PreflightError):
        run_sweep(sweep_path, results_root, resume=False, **_sweep_kwargs(fake_docker, http_factory, tmp_path))

    state_json = json.loads((results_root / "sw1" / "state.json").read_text(encoding="utf-8"))
    statuses = {rid: r["status"] for rid, r in state_json["runs"].items()}
    assert list(statuses.values()).count("running") == 1
    assert "invalid" not in statuses.values() and "requeued" not in statuses.values()


def test_run_sweep_fresh_start_refuses_to_clobber_existing_sweep(tmp_path, monkeypatch, client_json):
    """Important 4: `run` (resume=False) on an existing sweep id must not
    silently clobber state.json/schedule.json/sweep.yaml -- it should point
    the caller at `resume` instead."""
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

    run_sweep(sweep_path, results_root, resume=False, **_sweep_kwargs(fake_docker, http_factory, tmp_path))

    with pytest.raises(FileExistsError, match="resume"):
        run_sweep(sweep_path, results_root, resume=False, **_sweep_kwargs(fake_docker, http_factory, tmp_path))
