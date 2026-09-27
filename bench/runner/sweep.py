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
from .paths import REPO_ROOT as _REPO_ROOT

# C1/P28: single source of truth for the traces-dir default, anchored to the
# repo root rather than left relative -- resolved regardless of the process's
# invocation cwd (a relative "bench/traces" only ever worked when launched
# from the repo root).
_DEFAULT_TRACES_DIR = str(_REPO_ROOT / "bench" / "traces")

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
    "traces_dir", "gpu_headroom_mb", "try_clock_pin", "block_axis", "_source",
}


def _resolve_relative(path_str: str, sweep_path: Path) -> Path:
    """C1/P28: resolve a sweep YAML's `base:` (or any such relative
    reference) against the sweep file's own directory first, then the repo
    root, then cwd -- the first candidate that exists on disk. An already-
    absolute `path_str` (every tmp-path fixture's `base:`) collapses the
    first candidate back to itself, so this is a no-op for those. Falls back
    to the first candidate (sweep-dir-relative) when none exist, so the
    caller's own `read_text()`/`FileNotFoundError` still names something
    sensible."""
    candidates = [sweep_path.parent / path_str, _REPO_ROOT / path_str, Path.cwd() / path_str]
    return next((c for c in candidates if c.exists()), candidates[0])


def load_sweep(path: str | Path) -> dict:
    """Load a sweep YAML, merging its `base:` file (if any) underneath the
    sweep's own keys. C1/P28: `base:` is resolved via `_resolve_relative`
    (sweep-file dir -> repo root -> cwd) rather than taken as a literal
    cwd-relative path, so `base: bench/configs/base.yaml` keeps working
    regardless of the process's invocation directory. A frozen sweep.yaml
    written by `run_sweep` (Important 2/P31) carries no `base:` key at all,
    so resuming from it is a straight load with no re-merge against the
    live base.yaml."""
    path = Path(path)
    sweep = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = {}
    if "base" in sweep:
        base = yaml.safe_load(_resolve_relative(sweep["base"], path).read_text(encoding="utf-8"))
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
        # Name a RunConfig field here to turn the free shuffle into a randomised
        # block design on that axis (see schedule()).
        "block_axis": sweep.get("block_axis"),
        "traces_dir": sweep.get("traces_dir", _DEFAULT_TRACES_DIR),
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
    traces_dir = Path(sweep.get("traces_dir", _DEFAULT_TRACES_DIR))
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


def schedule(configs: list[RunConfig], seed: int, block_axis: str | None = None) -> list[RunConfig]:
    """A seeded permutation of `configs` (thermal randomization across the sweep).

    With `block_axis` set, this becomes a **randomised block design** instead of
    a free shuffle: runs are grouped so each consecutive block holds one run per
    value of that axis, with the order inside each block randomised. A free
    shuffle balances the arms only on average and can still cluster badly --
    seed 20260927 over 10+10 power-overlay runs produced a 5-run streak and left
    one arm's mean position 1.6 slots earlier than the other's, which is exactly
    the drift confound a paired comparison exists to remove. Blocking caps any
    streak at 2 and equalises mean position by construction, so anything
    drifting over the session (ambient temperature, background load, driver
    state) hits both arms alike.

    Unequal groups are allowed: the longer ones simply fill the later blocks on
    their own, so nothing is dropped.
    """
    rng = random.Random(seed)
    if block_axis is None:
        order = list(configs)
        rng.shuffle(order)
        return order

    groups: dict[object, list[RunConfig]] = {}
    for cfg in configs:
        if not hasattr(cfg, block_axis):
            raise ValueError(f"block_axis {block_axis!r} is not a RunConfig field")
        groups.setdefault(getattr(cfg, block_axis), []).append(cfg)
    for members in groups.values():
        rng.shuffle(members)

    out: list[RunConfig] = []
    for i in range(max(len(m) for m in groups.values())):
        block = [m[i] for m in groups.values() if i < len(m)]
        rng.shuffle(block)
        out.extend(block)
    return out
