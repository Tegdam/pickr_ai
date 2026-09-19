"""One run, start to cooldown (spec §6 steps 1-9) and the sweep loop that
drives many runs through `run_one` with resume (spec §7 sweep-level files).

Every collaborator here is the real Task 2-7 module, not the brief's sketch --
see the module docstrings this file imports for the contract each one keeps.
`opts` (sweep-level runtime knobs: `gpu_headroom_mb`, `try_clock_pin`,
`schedule_index`, `attempt`, `requeued_from`) is not part of `RunConfig` or
`RunPaths` (ruling: engine-launch config vs. sweep/runtime execution state are
different things, sweep.py's own `_NON_RUNCONFIG_KEYS` docstring says the
same) -- it travels to `run_one` as a plain dict, the way `sweep_options()`
already hands it to callers.

Fix round 1 (verbatim rulings from review): attempt isolation (Critical 1),
a `PreflightError` that keeps environmental failures out of the retry budget
(Important 3), artifacts recorded right after the client phase rather than
after cooldown (Important 5), and a defensively-cleaned engine container
name (Important 2). See each helper's docstring for the specific ruling it
implements.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .client import CLIENT_IMAGE, run_client
from .config import RunConfig, resolve_gpu_memory_fraction, validate
from .engine import ENGINES, EngineSpec
from .env_capture import capture_env
from .gpu_monitor import GpuSampler, WIN_SMI, WSL_SMI, read_gpu
from .metrics_scraper import MetricsScraper
from .readiness import reset_cache, wait_healthy, warmup
from .schema import CLIENT_OUTPUT_SCHEMA_VERSION
from .state import SweepState
from .summary import build_summary, write_requests
from .sweep import expand, load_sweep, schedule as schedule_configs, sweep_options

_CLOCK_PIN_NOTE_DEFAULT = "not attempted (requires Administrator; deferred to the user)"
# Fix round 1, minor: match the doc's literal log lines (P10) -- vLLM's
# "Using MarlinLinearKernel for AutoAWQMarlinLinearMethod" and SGLang's
# "Using awq_marlin kernel." -- not our own launch-arg echo, which merely
# repeats the CLI flag "--quantization awq" back and would false-positive.
_KERNEL_RE = re.compile(r"Using awq_marlin kernel\.|MarlinLinearKernel")

VRAM_RETURN_TOLERANCE_MB = 200
VRAM_RETURN_TIMEOUT_S = 60.0
VRAM_RETURN_POLL_S = 2.0
COOLDOWN_POLL_S = 5.0
COOLDOWN_CAP_S = 15 * 60.0


class PreflightError(RuntimeError):
    """Important 3: an environment-level failure -- step 1's bench-* guard, a
    `None`/failing nvidia-smi reader, or a failed `docker run` in
    `start_engine` -- as opposed to a per-run measurement failure. These are
    not requeued (a leaked container or one flaky nvidia-smi call must not
    silently burn the sweep's retry budget): `run_sweep` leaves the run
    `running` for `reset_stale()` to recover on resume, and re-raises so the
    process exits non-zero instead of ploughing on.
    """


@dataclass
class RunPaths:
    """Filesystem layout one run/sweep needs (ruling: `RunPaths` carries no
    sweep-runtime options -- those travel through `opts`, see module docstring)."""
    results_root: Path
    sweep_dir: Path
    run_dir: Path
    traces_dir: Path
    hf_cache_dir: Path
    compile_cache_root: Path


@dataclass
class EngineHandle:
    """Task 9 seam (ruling): engine start/stop is isolated behind this pair so
    a second implementation (a local-process echo engine) can be added
    without touching `run_one`'s body."""
    name: str
    base_url: str
    launch_args: list[str]
    compile_cache_mounted: bool


def _quantization_kernel(log_text: str) -> str | None:
    return "marlin" if _KERNEL_RE.search(log_text or "") else None


def _isolate_previous_attempt(run_dir: Path) -> list[str] | None:
    """Critical 1: samplers open `gpu_samples.jsonl`/`engine_metrics.jsonl` in
    append mode, and a retried run reuses the same `run_dir` -- without this,
    attempt 2's summary (drift, peaks, clock_cv, delta-based acceptance) would
    be computed over two engine processes' samples. If `run_dir` already
    holds anything from a prior attempt, move all of it into
    `run_dir/attempt-<N>/` (forensics kept, never deleted) before proceeding
    with a clean directory. Returns the moved file names, or None if nothing
    needed moving."""
    if not run_dir.exists():
        return None
    existing = [p for p in run_dir.iterdir() if not (p.is_dir() and p.name.startswith("attempt-"))]
    if not existing:
        return None
    n = 1
    while (run_dir / f"attempt-{n}").exists():
        n += 1
    dest = run_dir / f"attempt-{n}"
    dest.mkdir(parents=True)
    moved = []
    for p in existing:
        p.rename(dest / p.name)
        moved.append(p.name)
    return moved


def _compile_cache_mount(spec: EngineSpec, paths: RunPaths):
    """Ruling P13: vLLM's torch.compile cache is mounted read-write so repeat
    launches of the same shapes don't pay compile time again; SGLang names no
    such cache dir in the contract doc, so nothing is mounted for it (recorded
    via `compile_cache_mounted` in env.json either way)."""
    if spec.name != "vllm":
        return None, False
    host_dir = Path(paths.compile_cache_root) / "vllm"
    host_dir.mkdir(parents=True, exist_ok=True)
    return (str(host_dir), "/root/.cache/vllm", "rw"), True


def start_engine(cfg: RunConfig, spec: EngineSpec, docker, paths: RunPaths) -> EngineHandle:
    """Step 2 (spec §6): launch the engine container. Does not block for
    readiness -- that is the caller's job (`wait_healthy`).

    Important 2: a failed `docker run` can leave a `Created` container behind
    that no other path removes, so every retry after that fails on a name
    conflict -- `docker.stop(name)` (rm -f, best-effort) runs defensively
    BEFORE `docker.run`, and any exception from `docker.run` itself triggers
    the same cleanup before re-raising as a `PreflightError` (Important 3:
    a failed launch is environmental, not a measurement failure).
    """
    name = f"bench-{cfg.run_id}"
    docker.stop(name)  # best-effort: clear any leaked Created/Exited container from a prior failed attempt

    launch_args = spec.build_launch_args(cfg, cfg.gpu_memory_utilization)
    mounts = [(str(paths.hf_cache_dir), "/root/.cache/huggingface", "ro")]
    compile_mount, compile_cache_mounted = _compile_cache_mount(spec, paths)
    if compile_mount:
        mounts.append(compile_mount)

    try:
        docker.run(spec.image, name, launch_args, gpus=True, network_host=True,
                   mounts=mounts, env=spec.env, extra_args=spec.docker_extra_args)
    except Exception as e:
        docker.stop(name)
        raise PreflightError(f"failed to start engine container {name!r}: {e}") from e

    return EngineHandle(
        name=name, base_url=f"http://localhost:{spec.port}",
        launch_args=launch_args, compile_cache_mounted=compile_cache_mounted,
    )


def stop_engine(handle: EngineHandle, docker) -> None:
    docker.stop(handle.name)


def _read_trace_rows(cfg: RunConfig) -> list[dict]:
    rows = []
    with open(cfg.trace_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[: min(cfg.num_prompts, len(rows))]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _wait_vram_return(gpu_reader, pre_used_mb, clock, sleep,
                       tolerance_mb: int = VRAM_RETURN_TOLERANCE_MB,
                       timeout_s: float = VRAM_RETURN_TIMEOUT_S,
                       poll_s: float = VRAM_RETURN_POLL_S) -> tuple[int | None, int]:
    """Step 8's VRAM-return check: poll the WSL-side reading until it is back
    within `tolerance_mb` of the pre-run baseline, or `timeout_s` elapses.
    Returns `(vram_returned_mb, vram_leak_mb)` -- leak is 0 on success."""
    t0 = clock()
    last_mb = None
    while True:
        last_mb = gpu_reader(WSL_SMI).get("used_mb")
        if last_mb is not None and pre_used_mb is not None and abs(last_mb - pre_used_mb) <= tolerance_mb:
            return last_mb, 0
        if clock() - t0 > timeout_s:
            leak = max(0, (last_mb or 0) - (pre_used_mb or 0))
            return last_mb, leak
        sleep(poll_s)


def _cooldown(gpu_reader, cooldown_temp_c: int, cooldown_min_s: int, clock, sleep,
              poll_s: float = COOLDOWN_POLL_S, cap_s: float = COOLDOWN_CAP_S) -> dict:
    """Step 9: sleep until Windows-side temp <= threshold AND the minimum
    wall gap has elapsed, capped at `cap_s` (spec §6 step 9)."""
    t0 = clock()
    start_temp = gpu_reader(WIN_SMI).get("temp_c")
    end_temp = start_temp
    while True:
        win = gpu_reader(WIN_SMI)
        end_temp = win.get("temp_c")
        elapsed = clock() - t0
        cool_enough = end_temp is not None and end_temp <= cooldown_temp_c
        long_enough = elapsed >= cooldown_min_s
        if cool_enough and long_enough:
            return {"cooldown_observed_s": elapsed, "cooldown_start_temp_c": start_temp,
                    "cooldown_end_temp_c": end_temp, "cooldown_capped": False}
        if elapsed >= cap_s:
            return {"cooldown_observed_s": elapsed, "cooldown_start_temp_c": start_temp,
                    "cooldown_end_temp_c": end_temp, "cooldown_capped": True}
        sleep(poll_s)


def _record_run(cfg, spec, docker, handle, client, trace_rows, run_dir, kernel, clock_pin_note,
                 clocks_pinned, timing, opts, wall_start, mono_anchor) -> tuple[dict, dict]:
    """Important 5: `write_requests`/`build_summary`/`capture_env`/
    `config.yaml`/`meta.json` are written immediately after the samplers stop
    -- NOT after `docker.stop` + the VRAM-return check + cooldown, which used
    to mean the artifacts didn't exist until minutes later and `wall_end`
    silently included cooldown. Returns `(summary, meta)` so `run_one` can
    later amend both in place with the post-cooldown/VRAM fields once those
    steps finish, without recomputing anything here.
    """
    requests = write_requests(client, trace_rows, cfg, run_dir / "requests.jsonl")
    gpu_rows = _read_jsonl(run_dir / "gpu_samples.jsonl")
    metric_rows = _read_jsonl(run_dir / "engine_metrics.jsonl")
    summary = build_summary(client, requests, gpu_rows, metric_rows, cfg, timing)

    env = capture_env(cfg, docker, spec, handle.launch_args, CLIENT_IMAGE, extra={
        "quantization_kernel": kernel,
        "clocks_pinned": clocks_pinned,
        "clock_pin_note": clock_pin_note,
        "compile_cache_mounted": handle.compile_cache_mounted,
        "client_output_schema_version": CLIENT_OUTPUT_SCHEMA_VERSION,
        # Fix round 1, minor: the client image's own digest, distinct from
        # the engine image's (already carried as env["image_digest"]).
        "client_image_digest": docker.image_digest(CLIENT_IMAGE),
    })

    wall_end = datetime.now(timezone.utc).isoformat()
    meta = {
        "run_id": cfg.run_id, "sweep_id": cfg.sweep_id, "phase": cfg.phase, "rq_tag": cfg.rq_tag,
        "schedule_index": opts.get("schedule_index"), "attempt": opts.get("attempt"),
        "requeued_from": opts.get("requeued_from"),
        "wall_start": wall_start, "wall_end": wall_end,
        "mono_to_wall_anchor": {"mono": mono_anchor, "wall": wall_start},
    }

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    (run_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    # client_raw.json is NOT rewritten here (spec §7: "the tool's own output,
    # untouched") -- run_client already wrote it via the /results bind mount.
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return summary, meta


def run_one(cfg: RunConfig, paths: RunPaths, *, docker, http, spec: EngineSpec,
            gpu_reader=read_gpu, clock=time.monotonic, sleep=time.sleep,
            opts: dict | None = None) -> dict:
    """Executes spec §6 steps 1-9 end to end and writes every artifact spec §7
    names into `paths.run_dir`. Returns the `summary.json` dict.

    `opts` (see module docstring) carries `gpu_headroom_mb`, `try_clock_pin`,
    and the sweep-loop's per-attempt bookkeeping (`schedule_index`, `attempt`,
    `requeued_from`) that lands in `meta.json`. Raises `PreflightError` for
    environmental failures (see its docstring) -- callers must not treat
    those like an ordinary measurement failure.
    """
    opts = dict(opts or {})
    gpu_headroom_mb = opts.get("gpu_headroom_mb", 256)
    try_clock_pin = opts.get("try_clock_pin", False)

    run_dir = Path(paths.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _isolate_previous_attempt(run_dir)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    wall_start = datetime.now(timezone.utc).isoformat()
    mono_anchor = clock()

    # --- 1. Pre-flight (spec §6 step 1) ---------------------------------
    try:
        already_running = [n for n in docker.ps_names() if n.startswith("bench-")]
    except Exception as e:
        raise PreflightError(f"docker ps failed: {e}") from e
    if already_running:
        raise PreflightError(f"refusing to start: bench-* container(s) already running: {already_running}")

    try:
        win_pre = gpu_reader(WIN_SMI)
        wsl_pre = gpu_reader(WSL_SMI)
    except Exception as e:
        raise PreflightError(f"gpu reader failed: {e}") from e

    total_mb = win_pre.get("total_mb")
    host_used_mb = win_pre.get("used_mb")
    if total_mb is None or host_used_mb is None:
        raise PreflightError(f"gpu reader returned no usable total_mb/used_mb: {win_pre}")

    cfg.free_vram_mb_at_start = total_mb - host_used_mb
    cfg.gpu_memory_utilization = resolve_gpu_memory_fraction(total_mb, host_used_mb, gpu_headroom_mb)

    clocks_pinned = False
    if try_clock_pin:
        # Ruling P8 pins try_clock_pin False for every current sweep; when a
        # future sweep does opt in, pinning still needs an Administrator
        # session this runner cannot obtain for itself -- record the attempt
        # honestly rather than pretending to have pinned anything.
        clock_pin_note = "attempted (try_clock_pin=True) but clock pinning is not implemented under WSL2"
    else:
        clock_pin_note = _CLOCK_PIN_NOTE_DEFAULT
    log(f"pre-flight: total={total_mb}MiB host_used={host_used_mb}MiB "
        f"free={cfg.free_vram_mb_at_start}MiB frac={cfg.gpu_memory_utilization} "
        f"clocks_pinned={clocks_pinned}")

    validate(cfg)

    # --- 2. Launch the engine (PreflightError on a failed launch) --------
    handle = start_engine(cfg, spec, docker, paths)
    log(f"launched {handle.name}: {' '.join(handle.launch_args)}")

    timing: dict = {}
    kernel = None
    exc: Exception | None = None
    base_exc: BaseException | None = None
    summary = meta = None

    try:
        # --- 3. Readiness -> warmup -> cache reset -----------------------
        ready_s = wait_healthy(http, handle.base_url, spec.health_path, spec.readiness_timeout_s,
                                docker=docker, container=handle.name, ready_path=spec.ready_path)
        timing["ready_s"] = ready_s
        kernel = _quantization_kernel(docker.container_logs(handle.name))

        trace_rows = _read_trace_rows(cfg)
        warmup_prompts = [r["prompt"] for r in trace_rows[: cfg.warmup_requests]]
        t0 = clock()
        warmup(http, handle.base_url, spec.served_model_name(cfg), warmup_prompts)
        timing["warmup_s"] = clock() - t0

        if cfg.cache_state != "warm":
            reset_cache(http, handle.base_url, spec, sleep=sleep)

        # --- 4. Start samplers --------------------------------------------
        sampler = GpuSampler(run_dir / "gpu_samples.jsonl", reader=gpu_reader)
        scraper = MetricsScraper(http, handle.base_url, spec, run_dir / "engine_metrics.jsonl")
        sampler.start()
        scraper.start()

        # --- 5/6. Run the client, then stop the samplers -------------------
        t0 = clock()
        try:
            client = run_client(docker, cfg, spec, run_dir, paths.traces_dir, paths.hf_cache_dir)
        finally:
            sampler.stop()
            scraper.stop()
        timing["client_s"] = clock() - t0

        # --- 7. Record artifacts NOW (Important 5) --------------------------
        summary, meta = _record_run(cfg, spec, docker, handle, client, trace_rows, run_dir, kernel,
                                     clock_pin_note, clocks_pinned, timing, opts, wall_start, mono_anchor)
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure still needs cleanup below
        exc = e
        log(f"run failed: {e!r}")
    except BaseException as e:  # KeyboardInterrupt/SystemExit: clean up, then let it propagate
        base_exc = e
        log(f"run interrupted: {e!r}")
        raise
    finally:
        # Fix round 1, minor: wrapped so a container_logs failure can never
        # skip stop_engine below.
        try:
            engine_logs = docker.container_logs(handle.name)
        except Exception:
            engine_logs = ""

        # --- 8. Stop the engine; verify VRAM returned -----------------------
        stop_engine(handle, docker)

        # Fix round 1, minor: a Ctrl-C mid-run should not be delayed by a
        # VRAM-settle poll or a full cooldown wait -- clean the container and
        # get out.
        if base_exc is None:
            vram_returned_mb, vram_leak_mb = _wait_vram_return(gpu_reader, wsl_pre.get("used_mb"), clock, sleep)
            cooldown_info = _cooldown(gpu_reader, cfg.cooldown_temp_c, cfg.cooldown_min_s, clock, sleep)

    (run_dir / "log.txt").write_text(engine_logs + "\n" + "\n".join(log_lines), encoding="utf-8")

    if exc is not None:
        minimal_meta = {
            "run_id": cfg.run_id, "sweep_id": cfg.sweep_id, "phase": cfg.phase, "rq_tag": cfg.rq_tag,
            "schedule_index": opts.get("schedule_index"), "attempt": opts.get("attempt"),
            "requeued_from": opts.get("requeued_from"),
            "wall_start": wall_start, "wall_end": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }
        (run_dir / "meta.json").write_text(json.dumps(minimal_meta, indent=2), encoding="utf-8")
        raise exc

    # --- 9. Amend meta.json/summary.json with the post-cooldown/VRAM fields
    meta.update(cooldown_info)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary["timing"]["vram_returned_mb"] = vram_returned_mb
    summary["timing"]["vram_leak_mb"] = vram_leak_mb
    summary["timing"]["wsl_pre_used_mb"] = wsl_pre.get("used_mb")
    summary["timing"]["win_pre_used_mb"] = win_pre.get("used_mb")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if vram_leak_mb:
        print(f"[{cfg.run_id}] VRAM LEAK {vram_leak_mb} MB")

    return summary


@dataclass
class SweepReport:
    sweep_id: str
    sweep_dir: Path
    state: SweepState
    summaries: dict[str, dict]


def _load_sweep_state(sweep_path, results_root: Path, *, resume: bool) -> tuple[dict, str, Path, SweepState, list[str]]:
    """Shared setup for `run_sweep`'s resume and fresh-start paths: load the
    sweep dict, resolve `sweep_dir`, and produce a `SweepState` plus the
    schedule's original run-id order. Fresh starts also write `sweep.yaml`/
    `schedule.json` and refuse to clobber an existing sweep (Important 4)."""
    sweep_dict = load_sweep(sweep_path)
    sweep_id = sweep_dict["sweep_id"]
    sweep_dir = results_root / sweep_id
    state_path = sweep_dir / "state.json"

    if resume:
        state = SweepState.load(state_path)
        reset = state.reset_stale()
        if reset:
            print(f"resume: reset {len(reset)} stale running run(s) to pending: {reset}")
        original_order = json.loads((sweep_dir / "schedule.json").read_text(encoding="utf-8"))["order"]
        return sweep_dict, sweep_id, sweep_dir, state, original_order

    if state_path.exists():
        raise FileExistsError(
            f"sweep {sweep_id!r} already has state at {state_path} -- pass resume=True "
            f"(or `python -m bench.runner resume {sweep_id}`) instead of starting it fresh"
        )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    opts = sweep_options(sweep_dict)
    ordered = schedule_configs(expand(sweep_dict, sweep_id), opts["schedule_seed"])
    original_order = [c.run_id for c in ordered]
    state = SweepState(state_path, original_order, max_retries_total=opts["max_retries_total"])
    (sweep_dir / "sweep.yaml").write_text(Path(sweep_path).read_text(encoding="utf-8"), encoding="utf-8")
    (sweep_dir / "schedule.json").write_text(
        json.dumps({"order": original_order, "seed": opts["schedule_seed"]}, indent=2), encoding="utf-8")
    return sweep_dict, sweep_id, sweep_dir, state, original_order


def run_sweep(sweep_path, results_root, *, resume: bool = False,
              docker, http_factory, spec_for=lambda cfg: ENGINES[cfg.engine],
              gpu_reader=read_gpu, clock=time.monotonic, sleep=time.sleep,
              hf_cache_dir=None, compile_cache_root=None) -> SweepReport:
    """Drives every run in a sweep through `run_one`, resuming from
    `state.json` when `resume=True` (spec §6/§7, ruling: sweep loop).

    `http_factory()` builds a fresh HTTP client per run (each run gets its own
    engine container/port lifecycle, so nothing about the client should be
    shared across runs). `spec_for(cfg)` resolves the `EngineSpec` for a
    config's engine -- injectable so tests need not depend on `ENGINES`
    carrying every engine they exercise.

    A `PreflightError` from `run_one` (Important 3) is not requeued: the run
    is left `running` (so `reset_stale()` recovers it on the next `resume`)
    and the error propagates out of `run_sweep` so the process exits non-zero.
    """
    results_root = Path(results_root)
    sweep_dict, sweep_id, sweep_dir, state, original_order = _load_sweep_state(
        sweep_path, results_root, resume=resume)

    opts = sweep_options(sweep_dict)
    configs_by_id = {c.run_id: c for c in expand(sweep_dict, sweep_id)}
    traces_dir = Path(opts["traces_dir"])
    hf_cache_dir = Path(hf_cache_dir) if hf_cache_dir else Path.home() / ".cache" / "huggingface"
    compile_cache_root = Path(compile_cache_root) if compile_cache_root else (results_root / ".cache")

    summaries: dict[str, dict] = {}
    # Fixed at schedule-creation time -- a run's position in the original,
    # seeded permutation never changes even after it is requeued to the end
    # of the live queue (state.order mutates on requeue; this does not).
    schedule_index_by_id = {rid: i for i, rid in enumerate(original_order)}

    while state.pending():
        run_id = state.pending()[0]
        cfg = configs_by_id[run_id]
        paths = RunPaths(
            results_root=results_root, sweep_dir=sweep_dir, run_dir=sweep_dir / run_id,
            traces_dir=traces_dir, hf_cache_dir=hf_cache_dir, compile_cache_root=compile_cache_root,
        )
        run_opts = dict(opts)
        run_opts["schedule_index"] = schedule_index_by_id.get(run_id)
        run_opts["attempt"] = state.runs[run_id]["attempts"] + 1
        run_opts["requeued_from"] = state.runs[run_id].get("reason") if state.runs[run_id]["status"] == "requeued" else None

        state.mark(run_id, "running")
        spec = spec_for(cfg)
        http = http_factory()
        try:
            summary = run_one(cfg, paths, docker=docker, http=http, spec=spec,
                               gpu_reader=gpu_reader, clock=clock, sleep=sleep, opts=run_opts)
        except PreflightError as e:
            # Important 3: not a measurement failure -- leave the run
            # `running` (reset_stale() on the next resume recovers it) and do
            # not touch the retry budget. Re-raise so the caller/process
            # notices instead of silently limping through the rest of the sweep.
            print(f"[{run_id}] PREFLIGHT FAILURE: {e} -- left 'running' for the next resume to recover")
            raise
        except Exception as e:  # noqa: BLE001 - a failed run must be requeued, not crash the sweep
            state.mark(run_id, "invalid", reason=str(e))
            requeued = state.requeue(run_id)
            print(f"[{run_id}] EXCEPTION: {e} -> {'requeued to end' if requeued else 'retry budget exhausted'}")
            continue

        summaries[run_id] = summary
        if summary.get("valid"):
            state.mark(run_id, "done")
            print(f"[{run_id}] done")
        else:
            reason = summary.get("invalid_reason")
            state.mark(run_id, "invalid", reason=reason)
            requeued = state.requeue(run_id)
            print(f"[{run_id}] invalid ({reason}) -> {'requeued to end' if requeued else 'retry budget exhausted'}")

    return SweepReport(sweep_id=sweep_id, sweep_dir=sweep_dir, state=state, summaries=summaries)
