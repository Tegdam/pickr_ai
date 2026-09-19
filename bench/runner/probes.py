"""P0b hardware/harness calibration probes (spec §3.1, §9 P0b exit; doc
bench/docs/p0b-engine-verification.md §7-8).

Five pure functions, unit-tested directly against hand-built inputs:

- `ceiling_from_rows`: reduces the echo-server request-rate ladder
  (`p0b_ceiling.yaml`'s `summary.json`s) to the harness ceiling (spec §9:
  ceiling >= 3x the study's peak rate).
- `ceiling_rows_from_sweep`: reads a real (or fixture) `p0b-ceiling` sweep
  directory's per-run `config.yaml`/`summary.json` files and averages reps
  per `request_rate`, producing `ceiling_from_rows`'s input directly.
- `parity_verdict`: reduces the chat-template parity sweep
  (`p0b_parity.yaml`'s `summary.json`s, one `usage.prompt_tokens` per engine)
  to a pass/fail against the trace row's own token count (spec §4).
- `acceptance_delta`: the pre-registered `ignore_eos` x acceptance-rate rule
  (spec §3.1): a >10% relative divergence between the with/without
  `--ignore-eos` acceptance rates means P2 reports acceptance over the
  natural-length prefix only.
- `throttle_baseline`: per-bit `clocks_throttle_reasons.active` fractions
  over a set of GPU samples (the `host_reservation` probe's idle phase),
  reusing `summary._throttle_bits_fraction`/`_throttle_reason_values` so the
  bit semantics never drift from `summary.py`'s.

`parity` and `ignore_eos_acceptance` are **not** probes run by this module --
they are the `p0b_parity.yaml` / `p0b_ignore_eos.yaml` sweeps (ordinary
`run_sweep` sweeps over real engines), whose `summary.json` results the P0b
analysis feeds through `parity_verdict`/`acceptance_delta` above. Only the
four hardware/harness probes below (`clock_pin`, `host_reservation`,
`cudagraph_cost`, `oom_signal`) are launched by `run_probe`/the `probe` CLI
subcommand.

The orchestration below (`run_probe`, the four `_probe_*` functions, and
`_run_oom_fraction`) is a thin, injectable seam over the same collaborators
`lifecycle.run_one` uses (`start_engine`/`stop_engine`/`wait_healthy`/
`readiness.warmup`) so probe launches get the same isolation, env, mounts
and pre-flight guard as every real run. Probe `RunConfig`s are built from
`base.yaml` itself (`_probe_run_config`, via `load_sweep`/`expand` -- the
same path a real sweep takes), never hand-duplicated, so a field like
`cudagraph_capture_sizes` can never drift from the pinned value. Every
engine-launching probe accumulates partial results (one entry per fraction/
capture-list) that are always returned even when a later iteration fails, and
`run_probe` always writes the probe's JSON, recording a top-level `error`/
`error_type` on total failure instead of raising. `_run_oom_fraction` is unit-
tested directly (the three outcome paths + "engine always stopped"); the rest
of the orchestration is exercised on the target machine, never against a
fake docker/GPU in CI.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import RunConfig, resolve_gpu_memory_fraction, validate
from .engine import ENGINES
from .gpu_monitor import WIN_SMI, WSL_SMI, read_gpu
from .lifecycle import PreflightError, RunPaths, _read_trace_rows, _wait_vram_return, start_engine, stop_engine
from .readiness import wait_healthy, warmup
from .summary import _throttle_bits_fraction, _throttle_reason_values
from .sweep import expand, load_sweep

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


def ceiling_rows_from_sweep(sweep_dir) -> list[dict]:
    """Reads a `p0b-ceiling` sweep directory's per-run `config.yaml`
    (`request_rate`) and `summary.json` (`req_s`, `ttft_p99_client`) and
    averages reps per rate, producing `ceiling_from_rows`'s input directly
    from a real (or fixture) sweep directory. Runs missing either file, or
    whose `config.yaml` carries no `request_rate` (not a poisson-mode run),
    are skipped."""
    sweep_dir = Path(sweep_dir)
    by_rate: dict[float, list[dict]] = {}
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        config_path, summary_path = run_dir / "config.yaml", run_dir / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rate = cfg.get("request_rate")
        if rate is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        by_rate.setdefault(rate, []).append({
            "req_s": summary.get("req_s"),
            "ttft_p99_ms": summary.get("ttft_p99_client"),
        })

    rows = []
    for rate, entries in by_rate.items():
        req_s_vals = [e["req_s"] for e in entries if e["req_s"] is not None]
        ttft_vals = [e["ttft_p99_ms"] for e in entries if e["ttft_p99_ms"] is not None]
        rows.append({
            "request_rate": rate,
            "req_s": statistics.fmean(req_s_vals) if req_s_vals else None,
            "ttft_p99_ms": statistics.fmean(ttft_vals) if ttft_vals else None,
        })
    return rows


def parity_verdict(prompt_tokens_by_engine: dict[str, int], expected: int) -> dict:
    """Spec §4 chat-template parity: pass iff `prompt_tokens_by_engine` is
    non-empty and every engine's reported `usage.prompt_tokens` equals
    `expected` (the trace row's own `prompt_tokens_qwen`) -- an empty dict
    never passes vacuously."""
    return {
        "pass": bool(prompt_tokens_by_engine) and all(v == expected for v in prompt_tokens_by_engine.values()),
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


def throttle_baseline(gpu_rows: list[dict]) -> dict:
    """Per-bit fraction of samples (with a non-None `throttle_reasons`)
    having each `clocks_throttle_reasons.active` bit set, over `gpu_rows`
    (the `host_reservation` probe's idle-phase samples). Reuses
    `summary._throttle_bits_fraction`/`_throttle_reason_values` directly --
    the exact computation `build_summary` uses for a real run -- so the bit
    semantics never drift from `summary.py`'s."""
    return _throttle_bits_fraction(_throttle_reason_values(gpu_rows))


# ---------------------------------------------------------------------------
# Orchestration (thin, injectable; see module docstring).
# ---------------------------------------------------------------------------

_OOM_RE = re.compile(
    r"out of memory|OutOfMemory|OOM|less than desired GPU memory utilization|cudaErrorMemoryAllocation",
    re.IGNORECASE,
)
_GRAPH_GIB_RE = re.compile(r"graph capturing finished in.*?took\s+([0-9.]+)\s*gi?b", re.IGNORECASE)
_KV_TOKENS_RE = re.compile(r"gpu kv cache size:\s*([0-9,]+)\s*tokens", re.IGNORECASE)

_BASE_YAML = Path("bench/configs/base.yaml")


def _preflight_guard(docker) -> None:
    """Same step-1 guard `lifecycle.run_one` applies before launching an
    engine (fix round 1 minor): refuse to start a probe while a leaked
    `bench-*` container from a previous run/probe is still up."""
    running = [n for n in docker.ps_names() if n.startswith("bench-")]
    if running:
        raise PreflightError(f"refusing to start probe: bench-* container(s) already running: {running}")


def _probe_run_config(paths: RunPaths, run_id: str, *, num_prompts: int = 8) -> RunConfig:
    """Fix round 1, item 7: build the probe's `RunConfig` the same way a real
    sweep does -- `load_sweep` + `expand` over `base.yaml` -- so every
    T1-verified field (`cudagraph_capture_sizes` included) tracks
    `base.yaml` instead of being hand-duplicated and drifting from it.
    `gpu_memory_utilization`/`free_vram_mb_at_start` are placeholders the
    caller overwrites from a fresh `read_gpu(WIN_SMI)` reading, exactly as
    `lifecycle.run_one` step 1 does."""
    base = load_sweep(str(_BASE_YAML))
    sweep_dict = {
        **base, "sweep_id": "p0b-probes", "rq_tag": "probe",
        "axes": {"engine": ["vllm"], "workload": ["A"]},
        "traces_dir": str(paths.traces_dir),
    }
    cfg = expand(sweep_dict, "p0b-probes")[0]
    cfg.run_id = run_id
    cfg.num_prompts = num_prompts
    return cfg


def _load_base_capture_sizes() -> list[int]:
    base = yaml.safe_load(_BASE_YAML.read_text(encoding="utf-8"))
    return list(base["cudagraph_capture_sizes"])


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
    load, `-rgc` after. Fix round 1, item 6: a zero exit code alone does not
    mean the pin held -- sample `sm_clock` every 2s for 20s (injectable
    `sleep`/`clock`) and record whether it stayed within 50 MHz of its own
    range (`held`). Sets `try_clock_pin` in `base.yaml` / `clocks_pinned` in
    every later `env.json` (doc §7-8)."""
    sm = params.get("sm_mhz", 2055)
    pin = subprocess.run([*WIN_SMI, "-lgc", f"{sm},{sm}"], capture_output=True, text=True, check=False)
    result: dict = {
        "pinned": pin.returncode == 0,
        "output": {"pin_returncode": pin.returncode, "pin_stdout": pin.stdout, "pin_stderr": pin.stderr},
    }

    if pin.returncode == 0:
        samples: list[float] = []
        t0 = clock()
        while clock() - t0 < 20:
            v = gpu_reader(WIN_SMI).get("sm_clock")
            if v is not None:
                samples.append(v)
            sleep(2.0)
        sm_min = min(samples) if samples else None
        sm_max = max(samples) if samples else None
        result["sm_clock_min"] = sm_min
        result["sm_clock_max"] = sm_max
        result["held"] = (sm_max - sm_min) <= 50 if (sm_min is not None and sm_max is not None) else False

    reset = subprocess.run([*WIN_SMI, "-rgc"], capture_output=True, text=True, check=False)
    result["output"].update(reset_returncode=reset.returncode, reset_stdout=reset.stdout, reset_stderr=reset.stderr)
    return result


def _probe_host_reservation(params: dict, paths: RunPaths, *, docker, http, gpu_reader, popen, sleep, clock) -> dict:
    """Spec §3.1 "host reservation measured at idle and tracked during runs":
    idle-phase samples with no container running, then load-phase samples
    while a light stream of completions (the first 8 trace prompts, via
    `readiness.warmup`) keeps a target-only vLLM engine busy. Reports both
    `used_host_mb` distributions and their drift."""
    _preflight_guard(docker)
    idle_seconds = params.get("idle_seconds", 60)
    load_seconds = params.get("load_seconds", 60)
    interval_s = 2.0

    idle_rows = _sample_for(gpu_reader, idle_seconds, sleep, clock, interval_s)
    idle_used = [r["used_host_mb"] for r in idle_rows if r.get("used_host_mb") is not None]

    cfg = _probe_run_config(paths, run_id=f"probe-host-reservation-{int(clock())}")
    free_mb, frac = _resolve_mem_fraction_and_free(gpu_reader)
    cfg.free_vram_mb_at_start, cfg.gpu_memory_utilization = free_mb, frac
    validate(cfg)

    spec = ENGINES["vllm"]
    pre_used = gpu_reader(WSL_SMI).get("used_mb")
    load_rows: list[dict] = []
    vram_returned_mb = vram_leak_mb = None
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
        # Fix round 1 minor: the VRAM-return check was missing here entirely.
        try:
            vram_returned_mb, vram_leak_mb = _wait_vram_return(gpu_reader, pre_used, clock, sleep)
        except Exception:  # noqa: BLE001 - a settle-poll flake must never mask this probe's own result
            pass

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
        "vram_returned_mb": vram_returned_mb,
        "vram_leak_mb": vram_leak_mb,
    }


def _parse_graph_gib(logs: str) -> float | None:
    m = _GRAPH_GIB_RE.search(logs or "")
    return float(m.group(1)) if m else None


def _parse_kv_tokens(logs: str) -> int | None:
    m = _KV_TOKENS_RE.search(logs or "")
    return int(m.group(1).replace(",", "")) if m else None


def _probe_cudagraph_cost(params: dict, paths: RunPaths, *, docker, http, gpu_reader, popen, sleep, clock) -> dict:
    """Spec §3.1 "CUDA-graph and workspace cost measured for the pinned
    capture list": one target-only vLLM launch per capture list. Fix round
    1, item 5: `used_mb` alone is the wrong quantity -- vLLM sizes KV to
    fill whatever fraction it is given, so `used_mb` is fraction-driven, not
    graph-driven. The delta of interest is `kv_tokens` (fewer KV tokens at
    the same fraction = more graph cost); `graph_gib` and `kv_tokens` are
    grepped from the launch log alongside `used_mb`. The capture lists are
    the params YAML's small lists plus `base.yaml`'s own pinned list, read
    live so this can never silently drift from it (item 7)."""
    _preflight_guard(docker)
    capture_lists = [list(x) for x in params.get("capture_lists", [[1], [1, 2, 4, 8]])]
    base_list = _load_base_capture_sizes()
    if base_list not in capture_lists:
        capture_lists.append(base_list)

    spec = ENGINES["vllm"]
    results: list[dict] = []

    for i, sizes in enumerate(capture_lists):
        try:
            free_mb, frac = _resolve_mem_fraction_and_free(gpu_reader)
            cfg = _probe_run_config(paths, run_id=f"probe-cudagraph-{i}")
            cfg.cudagraph_capture_sizes = list(sizes)
            cfg.free_vram_mb_at_start, cfg.gpu_memory_utilization = free_mb, frac
            validate(cfg)
        except Exception as e:  # noqa: BLE001 - never lose earlier capture-lists' results
            results.append({"capture_sizes": list(sizes), "outcome": "probe_error",
                             "detail": str(e), "error_type": type(e).__name__})
            continue

        entry: dict = {"capture_sizes": list(sizes)}
        pre_used = gpu_reader(WSL_SMI).get("used_mb")
        handle = None
        try:
            handle = start_engine(cfg, spec, docker, paths, popen=popen)
            wait_healthy(http, handle.base_url, spec.health_path, spec.readiness_timeout_s,
                         docker=docker, container=handle.name, ready_path=spec.ready_path)
            logs = docker.container_logs(handle.name)
            entry.update(outcome="served", used_mb=gpu_reader(WSL_SMI).get("used_mb"),
                         graph_gib=_parse_graph_gib(logs), kv_tokens=_parse_kv_tokens(logs))
        except Exception as e:  # noqa: BLE001 - a failed list is a probe outcome, not a crash
            entry.update(outcome="launch_failed", detail=str(e), error_type=type(e).__name__)
        finally:
            if handle is not None:
                stop_engine(handle, docker)
                try:
                    _wait_vram_return(gpu_reader, pre_used, clock, sleep)
                except Exception:  # noqa: BLE001 - never mask this list's result over a settle-poll flake
                    pass
        results.append(entry)

    served = [r for r in results if r.get("outcome") == "served"]
    baseline_kv = next((r["kv_tokens"] for r in served if r["capture_sizes"] == [1]), None)
    if baseline_kv is None and served:
        baseline_kv = served[0].get("kv_tokens")
    for r in results:
        r["kv_tokens_delta_vs_min"] = (
            r["kv_tokens"] - baseline_kv
            if r.get("outcome") == "served" and r.get("kv_tokens") is not None and baseline_kv is not None
            else None
        )

    return {"capture_lists": results}


def _measure_rate(http, base_url: str, model: str, prompts: list[str], n: int, clock, max_tokens: int = 64) -> float | None:
    """A tokens/s proxy from `n` timed completions (round-robin over
    `prompts`, `max_tokens` tokens each) -- good enough for the OOM probe's
    fits/spill comparison; not a load-test throughput number. Fix round 1,
    item 4: `max_tokens=64` (not the 8-token warmup default), tok/s =
    `n * max_tokens / elapsed`."""
    if n <= 0 or not prompts:
        return None
    t0 = clock()
    for i in range(n):
        warmup(http, base_url, model, [prompts[i % len(prompts)]], max_tokens=max_tokens)
    elapsed = clock() - t0
    return (n * max_tokens / elapsed) if elapsed > 0 else None


def _run_oom_fraction(cfg: RunConfig, paths: RunPaths, frac: float, timeout_s: float,
                       baseline_rate: float | None, baseline_ready_s: float | None, *,
                       spec, docker, http, gpu_reader, popen, sleep, clock, n_requests: int) -> dict:
    """One fraction of the OOM-signal ladder (fix round 1, items 1-4), pulled
    out of `_probe_oom_signal` so it is directly unit-testable: classifies a
    readiness failure as `slow_or_hung` (container still running -- a live-
    but-glacial/paging container, item 1) vs an exited container, which is
    then `clean_oom` (log tail matches `_OOM_RE`) or `clean_fail_no_oom_text`
    (item 2); a failure during serving (after a successful launch) is
    `served_then_failed`, never loses the launch having worked (item 3). The
    engine is always started through `start_engine`/stopped through
    `stop_engine` -- `finally` guarantees the stop (and the VRAM-return
    check, tuple kept this time -- item 4) runs no matter which branch fired."""
    result: dict = {"fraction": frac}
    pre_used = gpu_reader(WSL_SMI).get("used_mb")
    handle = None
    try:
        handle = start_engine(cfg, spec, docker, paths, popen=popen)
        ready_s = wait_healthy(http, handle.base_url, spec.health_path, timeout_s,
                                docker=docker, container=handle.name, ready_path=spec.ready_path)
    except Exception as e:  # noqa: BLE001 - a dead/hung/timed-out launch is a probe outcome, not a crash
        still_running = docker.is_running(handle.name) if handle is not None else False
        logs = docker.container_logs(handle.name) if handle is not None else ""
        if still_running:
            # Item 1: a live-but-glacial (paging) container must not be
            # mislabelled clean_fail -- it is still up, just very slow.
            result.update(outcome="slow_or_hung", detail=str(e), error_type=type(e).__name__)
        else:
            oom_lines = "\n".join(line for line in logs.splitlines() if _OOM_RE.search(line))
            if oom_lines:
                result.update(outcome="clean_oom", detail=str(e), error_type=type(e).__name__, oom_lines=oom_lines)
            else:
                result.update(outcome="clean_fail_no_oom_text", detail=str(e), error_type=type(e).__name__,
                               log_tail="\n".join(logs.splitlines()[-40:]))
    else:
        result["ready_s"] = ready_s
        result["wsl_used_mb_after_ready"] = gpu_reader(WSL_SMI).get("used_mb")
        try:
            trace_rows = _read_trace_rows(cfg)
            prompts = [r["prompt"] for r in trace_rows]
            rate = _measure_rate(http, handle.base_url, spec.served_model_name(cfg), prompts, n_requests, clock)
            after = _gpu_pair_sample(gpu_reader)
            result.update(outcome="served", rate_tok_s=rate,
                           used_host_mb_during_serving=after.get("used_host_mb"))
        except Exception as e:  # noqa: BLE001 - item 3: a request failure after a successful launch must not
            # abort the whole ladder / lose earlier fractions -- it is this fraction's own outcome.
            logs = docker.container_logs(handle.name)
            result.update(outcome="served_then_failed", detail=str(e), error_type=type(e).__name__,
                           log_tail="\n".join(logs.splitlines()[-40:]))
    finally:
        if handle is not None:
            stop_engine(handle, docker)
            vram_returned_mb = vram_leak_mb = None
            try:
                vram_returned_mb, vram_leak_mb = _wait_vram_return(gpu_reader, pre_used, clock, sleep)
            except Exception:  # noqa: BLE001 - never let a settle-poll flake mask this fraction's result
                pass
            result["vram_returned_mb"] = vram_returned_mb
            result["vram_leak_mb"] = vram_leak_mb

    if result.get("outcome") == "served":
        if baseline_rate and result.get("rate_tok_s") is not None:
            result["rate_ratio_vs_baseline"] = result["rate_tok_s"] / baseline_rate
        if baseline_ready_s and result.get("ready_s") is not None:
            result["ready_ratio_vs_baseline"] = result["ready_s"] / baseline_ready_s
    return result


def _classify_oom_outcome(ladder_results: list[dict]) -> str:
    """Fix round 1, item 4's top-level verdict: `clean_oom_boundary` if the
    first fraction (ascending) that didn't serve failed with an OOM
    signature; `spill_suspected` if any served fraction shows a >2x
    throughput collapse or a >3x readiness slowdown relative to the
    baseline (the WDDM-paging signature, spec §3.1) with no OOM error;
    otherwise `fits_all`."""
    first_failure = next((r for r in ladder_results if r.get("outcome") != "served"), None)
    if first_failure is not None and first_failure.get("outcome") == "clean_oom":
        return "clean_oom_boundary"
    for r in ladder_results:
        if r.get("outcome") != "served":
            continue
        rate_ratio, ready_ratio = r.get("rate_ratio_vs_baseline"), r.get("ready_ratio_vs_baseline")
        if (rate_ratio is not None and rate_ratio < 0.5) or (ready_ratio is not None and ready_ratio > 3):
            return "spill_suspected"
    return "fits_all"


def _probe_oom_signal(params: dict, paths: RunPaths, *, docker, http, gpu_reader, popen, sleep, clock) -> dict:
    """Spec §3.1 OOM-signal probe: WSL2's WDDM driver model can page GPU
    allocations to host RAM instead of failing cleanly. For each fraction
    (ascending): does the launch fail cleanly, hang/page, or serve? If it
    serves, compare a small timed-request rate and readiness time against
    the `baseline_fraction` run's -- a throughput collapse or readiness
    slowdown without an OOM error is the WDDM-paging signature (fix round 1,
    items 1-4; see `_run_oom_fraction`/`_classify_oom_outcome`)."""
    _preflight_guard(docker)
    fractions = sorted(params.get("fractions", [0.90, 0.95, 0.98, 1.00]))
    baseline_fraction = params.get("baseline_fraction", 0.85)
    n_requests = params.get("requests", 20)
    spec = ENGINES["vllm"]

    def _build_and_run(frac: float, tag: str, timeout_s: float,
                        baseline_rate: float | None, baseline_ready_s: float | None) -> dict:
        win = gpu_reader(WIN_SMI)
        total_mb, host_used_mb_before = win.get("total_mb"), win.get("used_mb")
        free_mb = total_mb - host_used_mb_before if total_mb is not None and host_used_mb_before is not None else None
        requested_mb = frac * total_mb if total_mb is not None else None
        try:
            cfg = _probe_run_config(paths, run_id=f"probe-oom-{tag}")
            cfg.free_vram_mb_at_start = free_mb if free_mb is not None else 0
            cfg.gpu_memory_utilization = frac
            validate(cfg)
        except Exception as e:  # noqa: BLE001 - item 3: never lose earlier fractions' results
            return {"fraction": frac, "outcome": "probe_error", "detail": str(e), "error_type": type(e).__name__,
                    "total_mb": total_mb, "host_used_mb_before": host_used_mb_before,
                    "free_mb": free_mb, "requested_mb": requested_mb}

        r = _run_oom_fraction(cfg, paths, frac, timeout_s, baseline_rate, baseline_ready_s,
                               spec=spec, docker=docker, http=http, gpu_reader=gpu_reader,
                               popen=popen, sleep=sleep, clock=clock, n_requests=n_requests)
        r.update(total_mb=total_mb, host_used_mb_before=host_used_mb_before,
                  free_mb=free_mb, requested_mb=requested_mb)
        return r

    try:
        baseline = _build_and_run(baseline_fraction, "baseline", spec.readiness_timeout_s, None, None)
    except Exception as e:  # noqa: BLE001 - item 3: a totally unexpected failure still yields a JSON-able row
        baseline = {"fraction": baseline_fraction, "outcome": "probe_error",
                     "detail": str(e), "error_type": type(e).__name__}
    baseline_rate = baseline.get("rate_tok_s") if baseline.get("outcome") == "served" else None
    baseline_ready_s = baseline.get("ready_s") if baseline.get("outcome") == "served" else None
    # Item 1: the ladder's own readiness budget is derived from how long the
    # baseline actually took to become healthy, not the full 900s engine
    # timeout -- a 3x-or-more readiness slowdown is itself the paging
    # tripwire, not something the timeout should swallow.
    ladder_timeout = max(3 * baseline_ready_s, 300) if baseline_ready_s else spec.readiness_timeout_s

    runs = [baseline]
    ladder_results: list[dict] = []
    for frac in fractions:
        try:
            r = _build_and_run(frac, f"{frac:.2f}", ladder_timeout, baseline_rate, baseline_ready_s)
        except Exception as e:  # noqa: BLE001 - item 3: never lose earlier fractions' results
            r = {"fraction": frac, "outcome": "probe_error", "detail": str(e), "error_type": type(e).__name__}
        ladder_results.append(r)
        runs.append(r)

    return {
        "baseline_fraction": baseline_fraction, "fractions": fractions,
        "runs": runs, "outcome": _classify_oom_outcome(ladder_results),
    }


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
    failure is recorded as `result["error"]`/`result["error_type"]`, never
    raised, so a probe run never loses whatever the run before it in
    `probe all` already wrote. Every engine launch goes through
    `start_engine`/`stop_engine` + `wait_healthy`, so probes get the same
    isolation, env and mounts as a real sweep run."""
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
        result["error_type"] = type(e).__name__
    result["finished_at"] = datetime.now(timezone.utc).isoformat()

    out_dir = Path(paths.results_root) / "probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}-{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["_written_to"] = str(out_path)
    return result
