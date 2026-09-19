"""Environment snapshot for one run: image + digest, pip freeze, host GPU/OS
telemetry, engine env/launch composition, and the git sha of bench's own
tracked code (spec %7 env.json).

`_host_lines()` is the only place that shells out, and every call inside it
is wrapped in try/except -- a missing or failing tool (most likely the
Windows-side nvidia-smi/powercfg, reached through the Windows filesystem
mount) degrades to `None` rather than aborting env capture. The Windows-side
nvidia-smi driver/power query and its power-value parse are two *separate*
try/except blocks: `--query-gpu=...,power.max_limit` without `nounits` comes
back unit-suffixed (`"100.00 W"`), and a driver line must not be lost just
because the power value failed to parse.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from bench.capture.cli import git_sha

from .engine import EngineSpec
from .gpu_monitor import WIN_SMI, WSL_SMI

_CLOCK_PIN_NOTE_DEFAULT = "not attempted (requires Administrator; deferred to the user)"
_POWERCFG = "/mnt/c/Windows/System32/powercfg.exe"

# pip_freeze shells into the image once and is identical for every run that
# shares an image within a sweep -- cache it here rather than re-running it.
_PIP_FREEZE_CACHE: dict[str, str] = {}


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, timeout=10).strip()


def _parse_power_w(raw: str | None) -> float | None:
    if not raw or raw in ("[N/A]", "N/A"):
        return None
    return float(raw.rstrip("W").strip())


def _host_lines() -> dict:
    out: dict = {}

    try:
        out["wsl_nvidia_smi"] = _run(WSL_SMI)
    except Exception:
        out["wsl_nvidia_smi"] = None

    driver_raw = power_raw = None
    try:
        csv = _run(WIN_SMI + ["--query-gpu=driver_version,power.max_limit", "--format=csv,noheader,nounits"])
        parts = [p.strip() for p in csv.splitlines()[0].split(",")]
        if len(parts) >= 2:
            driver_raw, power_raw = parts[0], parts[1]
    except Exception:
        pass

    try:
        out["win_nvidia_smi"] = f"Driver {driver_raw}" if driver_raw else None
    except Exception:
        out["win_nvidia_smi"] = None

    try:
        out["power_max_limit_w"] = _parse_power_w(power_raw)
    except Exception:
        out["power_max_limit_w"] = None

    try:
        out["uname"] = _run(["uname", "-r"])
    except Exception:
        out["uname"] = None

    try:
        out["docker_version"] = _run(["docker", "--version"])
    except Exception:
        out["docker_version"] = None

    try:
        out["windows_power_mode"] = _run([_POWERCFG, "/getactivescheme"])
    except Exception:
        out["windows_power_mode"] = None

    return out


def _cached_pip_freeze(docker, image: str) -> str:
    if image not in _PIP_FREEZE_CACHE:
        _PIP_FREEZE_CACHE[image] = docker.pip_freeze(image)
    return _PIP_FREEZE_CACHE[image]


def capture_env(cfg, docker, spec: EngineSpec, launch_args: list[str], client_image: str,
                 extra: dict | None = None) -> dict:
    extra = extra or {}

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None

    env: dict = {
        "image": spec.image,
        "image_digest": docker.image_digest(spec.image),
        "pip_freeze": _cached_pip_freeze(docker, spec.image),
        "served_model_name": spec.served_model_name(cfg),
        "engine_env": dict(spec.env),
        "docker_extra_args": list(spec.docker_extra_args),
        "launch_cmd": list(launch_args),
        "client_image": client_image,
        "bench_git_sha": git_sha(),
        "transformers_version": transformers_version,
        # Task 8 greps the container log for Marlin/awq_marlin to fill this in.
        "quantization_kernel": extra.get("quantization_kernel"),
        "compile_cache_mounted": extra.get("compile_cache_mounted", False),
        "clock_pin_note": extra.get("clock_pin_note", _CLOCK_PIN_NOTE_DEFAULT),
        # Spec %7 fields: seeded with defaults so they always land in env.json,
        # not only when the caller happens to pass them in `extra`.
        "clocks_pinned": extra.get("clocks_pinned", False),
        "client_output_schema_version": extra.get("client_output_schema_version"),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    env.update(_host_lines())
    for k, v in extra.items():
        env.setdefault(k, v)
    return env
