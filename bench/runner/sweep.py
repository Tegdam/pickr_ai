"""Sweep YAML -> resolved RunConfigs -> seeded schedule. Configs are data; nothing here launches anything.

`expand()` deliberately does NOT call `bench.runner.config.validate()` -- expansion must
stay pure (no filesystem trace-hash verification at YAML-load time), and validate()'s
trace sha256 check re-hashes the real trace file, which unit tests here exercise against
synthetic tmp-path fixtures whose recorded sha won't match a re-derived hash for an
arbitrary file. Task 8's lifecycle calls validate() on each expanded config before launch.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import yaml

from .config import RunConfig
from .engine import ENGINES

WORKLOAD_TRACES = {
    "A": "chat",
    "B": "summarization",
    "C": "structured",
    "MT-shallow": "multiturn_shallow",
    "MT-medium": "multiturn_medium",
    "MT-deep": "multiturn_deep",
}

# Sweep-level/runtime keys that describe the sweep or its execution, not a single
# run's engine-launch config -- excluded from the RunConfig kwargs built in expand().
# `gpu_headroom_mb` and `try_clock_pin` travel to the lifecycle via the sweep dict
# itself (see sweep_options), not through RunConfig (ruling P2).
_NON_RUNCONFIG_KEYS = {
    "axes", "reps", "base", "sweep_id", "schedule_seed", "max_retries_total",
    "traces_dir", "gpu_headroom_mb", "try_clock_pin", "_source",
}


def load_sweep(path: str | Path) -> dict:
    """Load a sweep YAML, merging its `base:` file (if any) underneath the sweep's own keys."""
    sweep = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(sweep["base"]).read_text(encoding="utf-8")) if "base" in sweep else {}
    merged = {**base, **{k: v for k, v in sweep.items() if k != "base"}}
    merged["_source"] = str(path)
    return merged


def sweep_options(sweep: dict) -> dict:
    """Sweep-level/runtime options the lifecycle (Task 8) reads directly from the
    sweep dict rather than from any RunConfig -- one place to keep their defaults."""
    return {
        "gpu_headroom_mb": sweep.get("gpu_headroom_mb", 256),
        "try_clock_pin": sweep.get("try_clock_pin", False),
        "max_retries_total": sweep.get("max_retries_total", 0),
        "schedule_seed": sweep.get("schedule_seed", 0),
        "traces_dir": sweep.get("traces_dir", "bench/traces"),
    }


def _trace_for(workload: str, traces_dir: Path, version: int = 1) -> tuple[str, str, int]:
    name = WORKLOAD_TRACES[workload]
    trace = traces_dir / f"{name}_v{version}.jsonl"
    meta = json.loads((traces_dir / f"{name}_v{version}.meta.json").read_text(encoding="utf-8"))
    return str(trace), meta["trace_sha256"], version


def expand(sweep: dict, sweep_id: str) -> list[RunConfig]:
    """Cross every list-valued key under `axes:` (order-preserving), apply `reps:`
    (each config repeated with `rep` 0..N-1 folded into run_id), and resolve each
    point's trace file/sha/version from its `workload` axis value."""
    axes = sweep.get("axes", {})
    keys = list(axes)
    reps = int(sweep.get("reps", 1))
    traces_dir = Path(sweep.get("traces_dir", "bench/traces"))
    fixed = {k: v for k, v in sweep.items() if k not in _NON_RUNCONFIG_KEYS}

    out: list[RunConfig] = []
    for index, combo in enumerate(itertools.product(*[axes[k] for k in keys])):
        point = {**fixed, **dict(zip(keys, combo))}
        trace_version = int(point.pop("trace_version", 1))
        trace_file, sha, ver = _trace_for(point["workload"], traces_dir, trace_version)
        image = ENGINES[point["engine"]].image  # known at expansion time (Task 1's table)
        for rep in range(reps):
            out.append(RunConfig(
                run_id=f"{sweep_id}-{index:04d}-r{rep}",
                sweep_id=sweep_id,
                image=image,
                trace_file=trace_file, trace_sha256=sha, trace_version=ver,
                gpu_memory_utilization=0.0, free_vram_mb_at_start=0,  # resolved at run time
                **point,
            ))
    return out


def schedule(configs: list[RunConfig], seed: int) -> list[RunConfig]:
    """A seeded permutation of `configs` (thermal randomization across the sweep)."""
    order = list(configs)
    random.Random(seed).shuffle(order)
    return order
