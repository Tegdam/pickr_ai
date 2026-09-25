"""End-to-end drive of `run_one` (spec §6 steps 1-9) and `run_sweep` (resume,
requeue-to-end) through the fakes at the docker/HTTP boundary (conftest.py).

No real docker or nvidia-smi is invoked: `gpu_reader` here stands in for
`bench.runner.gpu_monitor.read_gpu`, keyed off whether a `bench-*` container
is currently in `fake_docker.running` -- WSL-side usage is 0 before the
engine container starts (or after it is stopped) and 3000 while it runs,
Windows-side total/used/temp are fixed, matching the brief's calibration
numbers (total 6141, host used 0 -> resolved fraction 0.79 for vLLM: P36's
per-engine headroom is 256 (sweep default) + 1024 (vLLM) = 1280, so
(6141-0-1280)*100 // 6141 == 79; temp 45 <= the default cooldown_temp_c 55).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

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


def _free_port(_port):
    """I4/P33: `run_one`'s port-free pre-flight check, faked so tests never
    touch a real socket -- every port is "free" unless a test says otherwise."""
    return True


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
        gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port,
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
    assert "--gpu-memory-utilization 0.79" in " ".join(run_call["args"])  # P36: vLLM headroom 256+1024=1280
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
    assert config_yaml["gpu_memory_utilization"] == pytest.approx(0.79)
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
            gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

    assert not any(u.endswith("/reset_prefix_cache") for u, _ in fake_http.posts)


class _FakePopen:
    """Stands in for `subprocess.Popen` (Task 9): `poll()` is None until
    `terminate()`/`kill()` sets a returncode, exactly like a real live
    process. `kwargs["stderr"]` is the real, already-open file object
    `start_engine` redirects the subprocess's stderr to (fix round 1: a file,
    not PIPE) -- a subclass simulating a crash writes into it directly, the
    same way a real crashing subprocess's own stderr would land there."""

    def __init__(self, args, **kwargs):
        self.args = list(args)
        self.kwargs = kwargs
        self.returncode = None
        self.terminate_called = False
        self.wait_called = False
        self.kill_called = False

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
                       gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port,
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
    # Fix round 1, minors: repo root as cwd (so `-m bench.echo_server`
    # resolves regardless of the runner's own invocation directory), and
    # stderr redirected to a real file (not PIPE).
    assert proc.kwargs["cwd"] == str(lifecycle._REPO_ROOT)
    assert hasattr(proc.kwargs["stderr"], "write")  # a real file object, not subprocess.PIPE
    assert (paths.run_dir / "engine_stderr.txt").exists()

    assert summary["valid"] is True
    assert summary["cooldown_skipped"] == "echo"

    meta = json.loads((paths.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["cooldown_skipped"] == "echo"

    env = json.loads((paths.run_dir / "env.json").read_text(encoding="utf-8"))
    assert env["image"] is None and env["image_digest"] is None and env["pip_freeze"] is None
    # The client image's own digest is still captured (unaffected by the engine having none).
    assert env["client_image_digest"] == fake_docker.digest


def test_run_one_echo_engine_crash_before_healthy_raises_with_stderr(tmp_path, monkeypatch, client_json):
    """Fix round 1 (Important): the readiness probe's per-poll liveness check
    (readiness.py's existing `docker.is_running`/`container_logs` logic,
    unmodified) must catch a subprocess that dies mid-poll well before the
    readiness timeout, and surface its real captured stderr -- not a single
    poll()-right-after-Popen check that would otherwise burn the full budget."""
    import bench.runner.readiness as readiness_mod
    monkeypatch.setattr(readiness_mod.time, "sleep", lambda s: None)  # no real waits between polls

    cfg, paths, fake_docker, fake_http = _setup(
        tmp_path, monkeypatch, client_json, cfg_over={"engine": "echo", "image": None},
    )
    fake_http.healthy_after = 10**6  # never becomes healthy on its own -- only the dead-process check should fire

    class _DiesOnThirdPoll(_FakePopen):
        """poll() -> None, None, 1, ... : alive for the first two liveness
        checks, dead on the third -- and writes its "crash" into the real
        stderr file `start_engine` opened and handed it, exactly as a real
        crashing subprocess would."""

        def __init__(self, args, **kwargs):
            super().__init__(args, **kwargs)
            self._polls = 0
            stderr_file = kwargs.get("stderr")
            if stderr_file is not None:
                stderr_file.write("Traceback: boom")
                stderr_file.flush()

        def poll(self):
            self._polls += 1
            return None if self._polls <= 2 else 1

    with pytest.raises(RuntimeError) as excinfo:
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["echo"],
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port,
                popen=_DiesOnThirdPoll)

    assert "exited during startup" in str(excinfo.value)
    assert "Traceback: boom" in str(excinfo.value)
    # And log.txt gets the same stderr text (read from the same file).
    log_text = (paths.run_dir / "log.txt").read_text(encoding="utf-8")
    assert "Traceback: boom" in log_text


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
    """I5/P34: a gpu_reader flake during the post-run cooldown wait is caught
    and counted (`cooldown_reader_errors`) INSIDE `_cooldown` itself, which
    keeps waiting rather than aborting the whole cooldown measurement -- the
    run stays valid and still gets a normal `cooldown_observed_s`."""
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
                       gpu_reader=flaky_reader, clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

    assert summary["valid"] is True
    assert summary["timing"]["vram_leak_mb"] == 0  # the VRAM-return check itself never saw the flake

    meta = json.loads((paths.run_dir / "meta.json").read_text(encoding="utf-8"))
    assert "cooldown_error" not in meta
    assert meta["cooldown_reader_errors"] >= 1
    assert meta["cooldown_observed_s"] is not None  # _cooldown returned normally despite the flakes


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
                       gpu_reader=flaky_reader, clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

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
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

    text = (paths.run_dir / "preflight_error.txt").read_text(encoding="utf-8")
    assert "bench-leaked" in text


def test_isolate_previous_attempt_moves_preflight_error_txt(tmp_path):
    """Fix round 1, minor: a stale `preflight_error.txt` left over from a
    prior failed attempt must move aside with everything else -- it is just
    another plain file in `run_dir`, so `_isolate_previous_attempt`'s
    existing "everything not an attempt-*/ dir" scan already covers it; this
    locks that down explicitly rather than leaving it implicit."""
    run_dir = tmp_path / "r1"
    run_dir.mkdir()
    (run_dir / "preflight_error.txt").write_text("boom", encoding="utf-8")

    moved = lifecycle._isolate_previous_attempt(run_dir)

    assert moved is not None and "preflight_error.txt" in moved
    assert (run_dir / "attempt-1" / "preflight_error.txt").read_text(encoding="utf-8") == "boom"
    assert not (run_dir / "preflight_error.txt").exists()


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
                       gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

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
                       gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

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
                gpu_reader=bad_reader, clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)


def test_run_one_preflight_fails_when_gpu_not_idle(tmp_path, monkeypatch, client_json):
    """I3/P32: a non-idle WSL-side reading at the very start of a run (>150
    MiB used, no bench-* container running yet) means a previous run's
    context never actually tore down -- refuse to start rather than layer a
    fresh engine on top of a leaked one."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)

    def not_idle_reader(cmd):
        if list(cmd) == list(WSL_SMI):
            return {"used_mb": 500, "total_mb": WIN_TOTAL_MB}
        return {"used_mb": WIN_USED_MB, "total_mb": WIN_TOTAL_MB}

    with pytest.raises(PreflightError, match="GPU not idle"):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                gpu_reader=not_idle_reader, clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

    # No engine container was ever launched for this preflight failure.
    assert not any(c["op"] == "run" for c in fake_docker.calls)


def test_run_one_preflight_fails_when_engine_image_missing(tmp_path, monkeypatch, client_json):
    """I4/P33: a missing engine image is an environmental failure -- caught
    before anything is launched, not surfaced as a measurement failure."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    fake_docker.images_present[ENGINES["vllm"].image] = False

    with pytest.raises(PreflightError, match="image not present"):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
                port_free=_free_port)


def test_run_one_preflight_fails_when_client_image_missing(tmp_path, monkeypatch, client_json):
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    fake_docker.images_present[CLIENT_IMAGE] = False

    with pytest.raises(PreflightError, match="client image not present"):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
                port_free=_free_port)


def test_run_one_preflight_skips_engine_image_check_for_echo(tmp_path, monkeypatch, client_json):
    """I4/P33: the echo engine has no image at all (Task 9) -- must not call
    docker.image_present(None)."""
    cfg, paths, fake_docker, fake_http = _setup(
        tmp_path, monkeypatch, client_json, cfg_over={"engine": "echo", "image": None},
    )

    image_present_calls = []
    real_image_present = fake_docker.image_present

    def tracking_image_present(image):
        image_present_calls.append(image)
        assert image is not None, "must not be called for image=None"
        return real_image_present(image)

    fake_docker.image_present = tracking_image_present

    run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["echo"],
            gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
            port_free=_free_port, popen=_FakePopen)

    # Only the client image was checked -- echo's own spec.image is None.
    assert image_present_calls == [CLIENT_IMAGE]


def test_run_one_preflight_fails_when_port_not_free(tmp_path, monkeypatch, client_json):
    """I4/P33: something already listening on the engine's port must refuse
    the run before anything is launched."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)

    with pytest.raises(PreflightError, match="already in use"):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=ENGINES["vllm"],
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
                port_free=lambda p: False)

    assert not any(c["op"] == "run" for c in fake_docker.calls)


def test_run_one_writes_config_yaml_early_right_after_validate(tmp_path, monkeypatch, client_json):
    """Minor: config.yaml is written as soon as validate() succeeds (step 1),
    not only after the full run -- so a run that fails before _record_run
    still leaves a config.yaml with the resolved fraction behind."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    fake_http.healthy_after = 10**6  # never becomes healthy -- fails after step 1

    with pytest.raises(TimeoutError):
        run_one(cfg, paths, docker=fake_docker, http=fake_http,
                spec=replace(ENGINES["vllm"], readiness_timeout_s=0.01),
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep,
                port_free=_free_port)

    config_yaml = yaml.safe_load((paths.run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert config_yaml["gpu_memory_utilization"] == pytest.approx(0.79)


def test_run_one_wait_healthy_timeout_stops_engine_and_writes_log(tmp_path, monkeypatch, client_json):
    """Important 7(a): a readiness timeout stops the engine, never starts the
    samplers, still writes log.txt with the engine's own logs, and propagates."""
    cfg, paths, fake_docker, fake_http = _setup(tmp_path, monkeypatch, client_json)
    fake_http.healthy_after = 10**6  # never becomes healthy
    fake_docker.logs[f"bench-{cfg.run_id}"] = "engine boot log line"
    tiny_timeout_spec = replace(ENGINES["vllm"], readiness_timeout_s=0.01)

    with pytest.raises(TimeoutError):
        run_one(cfg, paths, docker=fake_docker, http=fake_http, spec=tiny_timeout_spec,
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

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
                gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port)

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
        gpu_reader=_make_gpu_reader(fake_docker), clock=_make_clock(), sleep=_no_sleep, port_free=_free_port,
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


def test_resume_loads_frozen_sweep_yaml_not_the_edited_live_base(tmp_path, monkeypatch, client_json):
    """I2/P31: the sweep.yaml a fresh start freezes into sweep_dir carries no
    `base:` key -- resuming must load it directly, never re-merging against
    whatever the live base.yaml says by the time resume runs. Edits the base
    after the first run, then resumes a run that got reset to `running`
    (simulating a crash), and asserts its config comes back unchanged."""
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
    target = next(iter(first.state.order))
    original_max_num_seqs = yaml.safe_load(
        (first.sweep_dir / target / "config.yaml").read_text(encoding="utf-8"))["max_num_seqs"]
    assert original_max_num_seqs == 64

    # The frozen sweep.yaml carries the merged fields directly -- no `base:`.
    frozen_sweep_path = results_root / first.sweep_id / "sweep.yaml"
    frozen = yaml.safe_load(frozen_sweep_path.read_text(encoding="utf-8"))
    assert "base" not in frozen and frozen["max_num_seqs"] == 64

    # Edit the LIVE base.yaml after the fact -- must have zero effect on resume.
    base_path = tmp_path / "base.yaml"
    base_dict = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base_dict["max_num_seqs"] = 999
    base_path.write_text(yaml.safe_dump(base_dict), encoding="utf-8")

    # Simulate a crash: one run left "running" for resume to redo.
    state = SweepState.load(first.sweep_dir / "state.json")
    state.mark(target, "running")

    resumed = run_sweep(frozen_sweep_path, results_root, resume=True,
                         **_sweep_kwargs(fake_docker, http_factory, tmp_path))

    resumed_cfg = yaml.safe_load((resumed.sweep_dir / target / "config.yaml").read_text(encoding="utf-8"))
    assert resumed_cfg["max_num_seqs"] == 64  # unchanged despite the live base.yaml edit


def test_run_sweep_halts_on_a_vram_leak_but_still_marks_the_run_done(tmp_path, monkeypatch, client_json):
    """I3/P32: a run reporting a >512 MiB VRAM leak is still marked `done`
    (its own measurement is fine) but the sweep halts -- every later run's
    "free VRAM" math would otherwise be built on a false premise."""
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

    # A WSL-side reader that looks idle (0) until the engine has launched at
    # least once, then reports high usage EVEN AFTER the container stops --
    # i.e. the VRAM-return check at the end of the run sees a leak, while the
    # run's own preflight (before anything launches) still sees an idle GPU.
    base_reader = _make_gpu_reader(fake_docker)

    def leaking_reader(cmd):
        if list(cmd) != list(WSL_SMI):
            return base_reader(cmd)
        engine_ever_launched = any(
            c["op"] == "run" and c["name"].startswith("bench-") and not c["name"].endswith("-client")
            for c in fake_docker.calls
        )
        engine_running_now = any(
            n.startswith("bench-") and not n.endswith("-client") for n in fake_docker.running
        )
        if engine_ever_launched and not engine_running_now:
            return {**base_reader(cmd), "used_mb": 1000}
        return base_reader(cmd)

    kwargs = _sweep_kwargs(fake_docker, http_factory, tmp_path)
    kwargs["gpu_reader"] = leaking_reader

    with pytest.raises(PreflightError, match="VRAM leak"):
        run_sweep(sweep_path, results_root, resume=False, **kwargs)

    state_json = json.loads((results_root / "sw1" / "state.json").read_text(encoding="utf-8"))
    statuses = [r["status"] for r in state_json["runs"].values()]
    assert statuses.count("done") == 1  # the leaking run's own data is fine
    assert statuses.count("running") == 0 and statuses.count("invalid") == 0


def test_run_sweep_halts_after_three_consecutive_first_attempt_failures(tmp_path, monkeypatch, client_json):
    """I5/P34: 3 consecutive first-attempt failures (no success in between)
    signal a systematic problem -- the sweep halts rather than grinding
    through every remaining run toward the same failure."""
    monkeypatch.setattr(env_capture, "_host_lines", lambda: {})
    # No retries: every failure below is a distinct run's first (and only) attempt.
    sweep_path = _write_sweep_files(tmp_path, max_retries_total=0)
    results_root = tmp_path / "results"

    fake_docker = FakeDocker()
    fake_docker.keep_running = True

    def always_short_run_client(docker, cfg, spec, run_dir, traces_dir, hf_cache_dir, **kw):
        # Short by 3 of num_prompts=4 -> attempted != num_prompts -> invalid.
        data = json.loads(json.dumps(client_json))
        data["completed"] = 1
        for key in ("ttfts", "itls", "input_lens", "output_lens", "generated_texts", "errors", "start_times"):
            data[key] = data[key][:1]
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        (Path(run_dir) / "client_raw.json").write_text(json.dumps(data), encoding="utf-8")
        return data

    monkeypatch.setattr(lifecycle, "run_client", always_short_run_client)

    def http_factory():
        h = FakeHTTP()
        h.reset_responses = [{"status": 200, "json": {"success": True}}]
        return h

    with pytest.raises(PreflightError, match="3 consecutive first-attempt failures"):
        run_sweep(sweep_path, results_root, resume=False, **_sweep_kwargs(fake_docker, http_factory, tmp_path))

    state_json = json.loads((results_root / "sw1" / "state.json").read_text(encoding="utf-8"))
    attempted_runs = [r for r in state_json["runs"].values() if r["attempts"] > 0]
    assert len(attempted_runs) == 3  # the sweep's 3 configs, never a requeued 4th attempt
    assert all(r["status"] == "invalid" for r in attempted_runs)
