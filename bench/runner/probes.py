"""P0b hardware/harness calibration probes (spec §3.1, §9 P0b exit; doc
bench/docs/p0b-engine-verification.md §7-8).

Four pure functions, unit-tested directly against hand-built inputs:

- `ceiling_from_rows`: reduces the echo-server request-rate ladder
  (`p0b_ceiling.yaml`'s `summary.json`s) to the harness ceiling (spec §9:
  ceiling >= 3x the study's peak rate).
- `parity_verdict`: reduces the chat-template parity sweep
  (`p0b_parity.yaml`'s `summary.json`s, one `usage.prompt_tokens` per engine)
  to a pass/fail against the trace row's own token count (spec §4).
- `acceptance_delta`: the pre-registered `ignore_eos` x acceptance-rate rule
  (spec §3.1): a >10% relative divergence between the with/without
  `--ignore-eos` acceptance rates means P2 reports acceptance over the
  natural-length prefix only.
- `throttle_baseline`: per-bit `clocks_throttle_reasons.active` fractions
  over a set of GPU samples (the `host_reservation` probe's idle phase),
  reusing `summary._throttle_bits_fraction`/`_throttle_reason_values` when
  importable so the bit semantics never drift from `summary.py`'s.

`parity` and `ignore_eos_acceptance` are **not** probes run by this module --
they are the `p0b_parity.yaml` / `p0b_ignore_eos.yaml` sweeps (ordinary
`run_sweep` sweeps over real engines), whose `summary.json` results the P0b
analysis feeds through `parity_verdict`/`acceptance_delta` above. Only the
four hardware/harness probes below (`clock_pin`, `host_reservation`,
`cudagraph_cost`, `oom_signal`) are launched by `run_probe`/the `probe` CLI
subcommand.

The orchestration below (`run_probe` and the four `_probe_*` functions) is a
thin, injectable seam over the same collaborators `lifecycle.run_one` uses
(`start_engine`/`stop_engine`/`wait_healthy`/`readiness.warmup`) so probe
launches get the same isolation, env and mounts as every real run. It is not
unit-tested beyond argument plumbing (`run_probe("clock_pin", ...)`) -- the
other three probes launch real engine containers and are exercised on the
target machine, never against a fake docker/GPU in CI.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import RunConfig, resolve_gpu_memory_fraction, validate
from .engine import ENGINES
from .gpu_monitor import WIN_SMI, WSL_SMI, read_gpu
from .lifecycle import (
    PreflightError,  # noqa: F401 -- re-exported for callers that catch it around run_probe
    RunPaths,
    _read_trace_rows,
    _wait_vram_return,
    start_engine,
    stop_engine,
)
from .readiness import wait_healthy, warmup
from .sweep import _trace_for

# ---------------------------------------------------------------------------
# Pure functions (unit-tested directly).
# ---------------------------------------------------------------------------


def ceiling_from_rows(rows: list[dict]) -> dict:
    """Spec §9 P0b exit: the harness ceiling is the highest `request_rate` R
    (from `p0b_ceiling.yaml`'s per-rate `summary.json`s, one row per rate:
    `{"request_rate", "req_s", "ttft_p99_ms"}`) at which the achieved
    `req_s >= 0.95 * R` AND `ttft_p99_ms` stays within 2x of the lowest
    rate's `ttft_p99_ms` (the harness/client is not itself the bottleneck).
    Returns `{"ceiling_req_s": R or None, "table": rows sorted by rate, each
    with a "passes" bool}`."""
    sorted_rows = sorted(rows, key=lambda r: r["request_rate"])
    if not sorted_rows:
        return {"ceiling_req_s": None, "table": []}

    baseline_ttft = sorted_rows[0]["ttft_p99_ms"]
    table: list[dict] = []
    ceiling: float | None = None
    for row in sorted_rows:
        passes = (
            row["req_s"] >= 0.95 * row["request_rate"]
            and row["ttft_p99_ms"] <= 2 * baseline_ttft
        )
        table.append({**row, "passes": passes})
        if passes:
            ceiling = row["request_rate"]
    return {"ceiling_req_s": ceiling, "table": table}


def parity_verdict(prompt_tokens_by_engine: dict[str, int], expected: int) -> dict:
    """Spec §4 chat-template parity: pass iff every engine's reported
    `usage.prompt_tokens` equals `expected` (the trace row's own
    `prompt_tokens_qwen`)."""
    return {
        "pass": all(v == expected for v in prompt_tokens_by_engine.values()),
        "values": dict(prompt_tokens_by_engine),
        "expected": expected,
    }


def acceptance_delta(with_ignore_eos: float | None, without: float | None) -> dict:
    """Spec §3.1 pre-registered rule: if the with/without `--ignore-eos`
    acceptance rates differ by more than 10% relative, P2 reports acceptance
    over the natural-length prefix only. `rel_diff` is `None` (and the rule
    does not apply) when either rate is missing or `without` is zero."""
    if with_ignore_eos is None or without is None or without == 0:
        return {"rel_diff": None, "prefix_rule_applies": False}
    rel_diff = (with_ignore_eos - without) / without
    return {"rel_diff": rel_diff, "prefix_rule_applies": abs(rel_diff) > 0.10}


# nvidia-smi clocks_throttle_reasons.active bits (summary.py's own constant,
# duplicated here only for the import-failure fallback below).
_THROTTLE_BITS = (0x4, 0x8, 0x20, 0x40, 0x80)


def throttle_baseline(gpu_rows: list[dict]) -> dict:
    """Per-bit fraction of samples (with a non-None `throttle_reasons`)
    having each `clocks_throttle_reasons.active` bit set, over `gpu_rows`
    (the `host_reservation` probe's idle-phase samples). Reuses
    `summary._throttle_bits_fraction`/`_throttle_reason_values` -- the exact
    computation `build_summary` uses for a real run -- when importable, so
    the bit semantics never drift from `summary.py`'s; falls back to the
    same logic inline if that private API ever moves."""
    try:
        from .summary import _throttle_bits_fraction, _throttle_reason_values

        return _throttle_bits_fraction(_throttle_reason_values(gpu_rows))
    except ImportError:
        values: list[int] = []
        for row in gpu_rows:
            reasons = row.get("throttle_reasons")
            if reasons is None:
                continue
            try:
                values.append(int(reasons, 16))
            except (TypeError, ValueError):
                continue
        if not values:
            return {f"0x{b:x}": None for b in _THROTTLE_BITS}
        n = len(values)
        return {f"0x{b:x}": sum(1 for v in values if v & b) / n for b in _THROTTLE_BITS}


# ---------------------------------------------------------------------------
# Orchestration (thin, injectable; see module docstring).
# ---------------------------------------------------------------------------

_OOM_RE = re.compile(r"OutOfMemory|CUDA out of memory|OOM", re.IGNORECASE)

# T1-verified target-only vLLM fields (doc §8 vLLM table) shared by every
# engine-launching probe below -- these are the same values base.yaml pins,
# duplicated here because probes don't run through a sweep YAML.
_TARGET_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"
_TARGET_REVISION = "3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd"


def _minimal_vllm_config(paths: RunPaths, run_id: str, *, num_prompts: int = 8,
                          concurrency: int = 8) -> RunConfig:
    """A target-only vLLM `RunConfig` good enough for a probe launch (T1
    fields from doc §8; readiness/warmup only, never a real load test).
    `gpu_memory_utilization`/`free_vram_mb_at_start` are placeholders the
    caller overwrites from a fresh `read_gpu(WIN_SMI)` reading, exactly as
    `lifecycle.run_one` step 1 does."""
    trace_file, trace_sha, trace_version = _trace_for("A", Path(paths.traces_dir))
    return RunConfig(
        run_id=run_id, sweep_id="p0b-probes", phase="p0b", rq_tag="calibration",
        engine="vllm", image=ENGINES["vllm"].image,
        model=_TARGET_MODEL, model_revision=_TARGET_REVISION, quantization="awq",
        draft_model=None, draft_revision=None, draft_quantization=None,
        spec_method="off", spec_k=None, ngram_lookup_max=4,
        workload="A", trace_file=trace_file, trace_version=trace_version, trace_sha256=trace_sha,
        load_mode="concurrency", concurrency=concurrency, request_rate=None, burstiness=1.0,
        num_prompts=num_prompts, cache_state="cold", prefix_caching=True,
        max_model_len=2048, max_num_seqs=32, chunked_prefill_tokens=1024,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
        sampling={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0},
        ignore_eos=True, max_tokens_cap=None, warmup_requests=8,
        cooldown_temp_c=50, cooldown_min_s=0,
        gpu_memory_utilization=0.0, free_vram_mb_at_start=0, seed=0, extra_body={},
    )


def _resolve_mem_fraction_and_free(gpu_reader, headroom_mb: int = 256) -> tuple[int, float]:
    """Same measurement `lifecycle.run_one` step 1 makes: fraction/free VRAM
    from a fresh Windows-side reading, never 0.9 by reflex (spec §3.1)."""
    win = gpu_reader(WIN_SMI)
    total_mb, used_mb = win.get("total_mb"), win.get("used_mb")
    if total_mb is None or used_mb is None:
        raise RuntimeError(f"gpu reader returned no usable total_mb/used_mb: {win}")
    free_mb = total_mb - used_mb
    return free_mb, resolve_gpu_memory_fraction(total_mb, used_mb, headroom_mb)


def _gpu_pair_sample(gpu_reader) -> dict:
    """One Windows+WSL nvidia-smi pair, shaped like `gpu_monitor.GpuSampler`'s
    own sample (`used_host_mb` = Windows-side minus WSL-side)."""
    win = gpu_reader(WIN_SMI)
    wsl = gpu_reader(WSL_SMI)
    used_host_mb = None
    if win.get("used_mb") is not None and wsl.get("used_mb") is not None:
        used_host_mb = max(0, win["used_mb"] - wsl["used_mb"])
    return {"used_host_mb": used_host_mb, "throttle_reasons": win.get("throttle_reasons"),
            "win": win, "wsl": wsl}


def _sample_for(gpu_reader, duration_s: float, sleep, clock, interval_s: float = 2.0) -> list[dict]:
    rows = []
    t0 = clock()
    while clock() - t0 < duration_s:
        rows.append(_gpu_pair_sample(gpu_reader))
        sleep(interval_s)
    return rows


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "min": None, "max": None, "n": 0}
    return {"mean": statistics.fmean(values), "min": min(values), "max": max(values), "n": len(values)}


def _probe_clock_pin(params: dict, paths: RunPaths, *, docker, http, gpu_reader, popen, sleep, clock) -> dict:
    """Doc §7 / spec §3.1 P8/P14: `nvidia-smi.exe -lgc <sm>,<sm>` under
    load, `-rgc` after -- record exit code + stdout/stderr only (no
    elevated-Administrator retry loop; ruling P14: "deferred to the user").
    Sets `try_clock_pin` in `base.yaml` / `clocks_pinned` in every later
    `env.json` (doc §7-8)."""
    sm = params.get("sm_mhz", 2055)
    pin = subprocess.run([*WIN_SMI, "-lgc", f"{sm},{sm}"], capture_output=True, text=True, check=False)
    reset = subprocess.run([*WIN_SMI, "-rgc"], capture_output=True, text=True, check=False)
    return {
        "pinned": pin.returncode == 0,
        "output": {
            "pin_returncode": pin.returncode, "pin_stdout": pin.stdout, "pin_stderr": pin.stderr,
            "reset_returncode": reset.returncode, "reset_stdout": reset.stdout, "reset_stderr": reset.stderr,
        },
    }


def _probe_host_reservation(params: dict, paths: RunPaths, *, docker, http, gpu_reader, popen, sleep, clock) -> dict:
    """Spec §3.1 "host reservation measured at idle and tracked during runs":
    idle-phase samples with no container running, then load-phase samples
    while a light stream of completions (the first 8 trace prompts, via
    `readiness.warmup`) keeps a target-only vLLM engine busy. Reports both
    `used_host_mb` distributions and their drift."""
    idle_seconds = params.get("idle_seconds", 60)
    load_seconds = params.get("load_seconds", 60)
    load_concurrency = params.get("load_concurrency", 8)
    interval_s = 2.0

    idle_rows = _sample_for(gpu_reader, idle_seconds, sleep, clock, interval_s)
    idle_used = [r["used_host_mb"] for r in idle_rows if r.get("used_host_mb") is not None]

    cfg = _minimal_vllm_config(paths, run_id=f"probe-host-reservation-{int(clock())}",
                                concurrency=load_concurrency)
    free_mb, frac = _resolve_mem_fraction_and_free(gpu_reader)
    cfg.free_vram_mb_at_start, cfg.gpu_memory_utilization = free_mb, frac
    validate(cfg)

    spec = ENGINES["vllm"]
    load_rows: list[dict] = []
    handle = start_engine(cfg, spec, docker, paths, popen=popen)
    try:
        wait_healthy(http, handle.base_url, spec.health_path, spec.readiness_timeout_s,
                     docker=docker, container=handle.name, ready_path=spec.ready_path)
        trace_rows = _read_trace_rows(cfg)
        prompts = [r["prompt"] for r in trace_rows[:8]]
        t0 = clock()
        while clock() - t0 < load_seconds:
            warmup(http, handle.base_url, spec.served_model_name(cfg), prompts)
            load_rows.append(_gpu_pair_sample(gpu_reader))
            sleep(interval_s)
    finally:
        stop_engine(handle, docker)

    load_used = [r["used_host_mb"] for r in load_rows if r.get("used_host_mb") is not None]
    idle_stats, load_stats = _stats(idle_used), _stats(load_used)
    drift_mb = (
        load_stats["mean"] - idle_stats["mean"]
        if idle_stats["mean"] is not None and load_stats["mean"] is not None else None
    )
    return {
        "idle": {"used_host_mb": idle_stats, "throttle_baseline": throttle_baseline(idle_rows)},
        "load": {"used_host_mb": load_stats},
        "drift_mb": drift_mb,
    }


def _probe_cudagraph_cost(params: dict, paths: RunPaths, *, docker, http, gpu_reader, popen, sleep, clock) -> dict:
    """Spec §3.1 "CUDA-graph and workspace cost measured for the pinned
    capture list": one target-only vLLM launch per capture list, WSL-side
    `used_mb` after readiness is the graph line of the budget table (doc
    §3.1 budget)."""
    capture_lists = params.get("capture_lists", [[1], [1, 2, 4, 8], [1, 2, 4, 8, 16, 32]])
    spec = ENGINES["vllm"]
    results: list[dict] = []
    baseline_used = None

    for i, sizes in enumerate(capture_lists):
        free_mb, frac = _resolve_mem_fraction_and_free(gpu_reader)
        cfg = _minimal_vllm_config(paths, run_id=f"probe-cudagraph-{i}")
        cfg.cudagraph_capture_sizes = list(sizes)
        cfg.free_vram_mb_at_start, cfg.gpu_memory_utilization = free_mb, frac
        validate(cfg)

        pre_used = gpu_reader(WSL_SMI).get("used_mb")
        used_mb = None
        handle = start_engine(cfg, spec, docker, paths, popen=popen)
        try:
            wait_healthy(http, handle.base_url, spec.health_path, spec.readiness_timeout_s,
                         docker=docker, container=handle.name, ready_path=spec.ready_path)
            used_mb = gpu_reader(WSL_SMI).get("used_mb")
        finally:
            stop_engine(handle, docker)
            _wait_vram_return(gpu_reader, pre_used, clock, sleep)

        results.append({"capture_sizes": list(sizes), "used_mb": used_mb})
        if list(sizes) == [1]:
            baseline_used = used_mb

    if baseline_used is None and results:
        baseline_used = results[0]["used_mb"]
    for r in results:
        r["delta_vs_min_mb"] = (
            r["used_mb"] - baseline_used if r["used_mb"] is not None and baseline_used is not None else None
        )
    return {"capture_lists": results, "baseline_used_mb": baseline_used}


def _measure_rate(http, base_url: str, model: str, prompts: list[str], n: int, clock) -> float | None:
    """A simple requests/s proxy from `n` timed warmup-style completions
    (round-robin over `prompts`) -- good enough for the fits/spill
    comparison the OOM probe needs; not a load-test throughput number."""
    if n <= 0 or not prompts:
        return None
    t0 = clock()
    for i in range(n):
        warmup(http, base_url, model, [prompts[i % len(prompts)]])
    elapsed = clock() - t0
    return (n / elapsed) if elapsed > 0 else None


def _probe_oom_signal(params: dict, paths: RunPaths, *, docker, http, gpu_reader, popen, sleep, clock) -> dict:
    """Spec §3.1 OOM-signal probe: WSL2's WDDM driver model can page GPU
    allocations to host RAM instead of failing cleanly. For each fraction
    (ascending): does the launch fail cleanly (readiness never succeeds,
    logs grepped for an OOM signature) or does it serve? If it serves,
    compare a small timed-request rate against the `baseline_fraction` run's
    rate -- a collapse without an OOM error (`rate < 0.5x baseline`) is the
    WDDM-paging signature."""
    fractions = sorted(params.get("fractions", [0.90, 0.95, 0.98, 1.00]))
    baseline_fraction = params.get("baseline_fraction", 0.85)
    n_requests = params.get("requests", 20)
    spec = ENGINES["vllm"]

    def _run_fraction(frac: float, tag: str) -> dict:
        cfg = _minimal_vllm_config(paths, run_id=f"probe-oom-{tag}")
        free_mb, _ = _resolve_mem_fraction_and_free(gpu_reader)
        cfg.free_vram_mb_at_start, cfg.gpu_memory_utilization = free_mb, frac
        validate(cfg)

        pre_used = gpu_reader(WSL_SMI).get("used_mb")
        result: dict = {"fraction": frac}
        handle = None
        try:
            handle = start_engine(cfg, spec, docker, paths, popen=popen)
            wait_healthy(http, handle.base_url, spec.health_path, spec.readiness_timeout_s,
                         docker=docker, container=handle.name, ready_path=spec.ready_path)
        except Exception as e:  # noqa: BLE001 - a dead/timed-out launch is a probe outcome, not a crash
            logs = docker.container_logs(handle.name) if handle is not None else ""
            oom_lines = "\n".join(line for line in logs.splitlines() if _OOM_RE.search(line))
            result.update(outcome="clean_fail", detail=str(e), oom_lines=oom_lines)
        else:
            trace_rows = _read_trace_rows(cfg)
            prompts = [r["prompt"] for r in trace_rows]
            rate = _measure_rate(http, handle.base_url, spec.served_model_name(cfg), prompts, n_requests, clock)
            result.update(outcome="served", rate=rate)
        finally:
            if handle is not None:
                stop_engine(handle, docker)
                try:
                    _wait_vram_return(gpu_reader, pre_used, clock, sleep)
                except Exception:  # noqa: BLE001 - never let a settle-poll flake mask this fraction's result
                    pass
        return result

    baseline = _run_fraction(baseline_fraction, "baseline")
    baseline_rate = baseline.get("rate") if baseline.get("outcome") == "served" else None

    runs = [baseline]
    for frac in fractions:
        r = _run_fraction(frac, f"{frac:.2f}")
        if r.get("outcome") == "served":
            if baseline_rate:
                ratio = (r["rate"] / baseline_rate) if r.get("rate") is not None else None
                r["throughput_ratio"] = ratio
                r["outcome"] = "fits" if (ratio is not None and ratio >= 0.5) else "spill"
            else:
                r["throughput_ratio"] = None
        runs.append(r)

    return {"baseline_fraction": baseline_fraction, "fractions": fractions, "runs": runs}


PROBE_ORDER = ("clock_pin", "host_reservation", "cudagraph_cost", "oom_signal")

_PROBE_FUNCS = {
    "clock_pin": _probe_clock_pin,
    "host_reservation": _probe_host_reservation,
    "cudagraph_cost": _probe_cudagraph_cost,
    "oom_signal": _probe_oom_signal,
}


def run_probe(name: str, params: dict, paths: RunPaths, *, docker, http,
              gpu_reader=read_gpu, popen=subprocess.Popen, sleep=time.sleep,
              clock=time.monotonic) -> dict:
    """Runs probe `name` (or, when `name == "all"`, every probe in
    `PROBE_ORDER`, each writing its own file) and writes
    `bench/results/probes/<name>-<UTC timestamp>.json` under
    `paths.results_root`. Always writes the file, even on failure -- a
    failure is recorded as `result["error"]`, never raised, so a probe run
    never loses whatever the run before it in `probe all` already wrote.
    Every engine launch goes through `start_engine`/`stop_engine` +
    `wait_healthy`, so probes get the same isolation, env and mounts as a
    real sweep run (never stopped skipped, even on failure -- see the
    `finally` blocks in each `_probe_*`)."""
    if name == "all":
        return {
            n: run_probe(n, params, paths, docker=docker, http=http, gpu_reader=gpu_reader,
                         popen=popen, sleep=sleep, clock=clock)
            for n in PROBE_ORDER
        }

    fn = _PROBE_FUNCS.get(name)
    if fn is None:
        raise ValueError(f"unknown probe {name!r}; choose one of {sorted(_PROBE_FUNCS)} or 'all'")

    started = datetime.now(timezone.utc)
    result: dict = {"probe": name, "started_at": started.isoformat()}
    try:
        result.update(fn(params.get(name, {}), paths, docker=docker, http=http, gpu_reader=gpu_reader,
                          popen=popen, sleep=sleep, clock=clock))
    except Exception as e:  # noqa: BLE001 - the probe's own JSON must exist even when it fails outright
        result["error"] = str(e)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()

    out_dir = Path(paths.results_root) / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["_written_to"] = str(out_path)
    return result
