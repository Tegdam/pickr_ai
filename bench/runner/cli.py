"""python -m bench.runner <check-env|run|resume|probe> (spec §6 "check-env";
ruling: CLI surface for the sweep loop). `probe` is Task 10's territory."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .client import CLIENT_IMAGE, build_client_image
from .docker import Docker
from .engine import ENGINES
from .gpu_monitor import WIN_SMI, WSL_SMI, read_gpu
from .lifecycle import run_sweep
from .sweep import WORKLOAD_TRACES, load_sweep, sweep_options

DEFAULT_RESULTS_ROOT = "bench/results"
DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
DEFAULT_COMPILE_CACHE = str(Path(DEFAULT_RESULTS_ROOT) / ".cache")


def _image_present(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, text=True, check=False).returncode == 0


def _docker_nvidia_ok() -> tuple[bool, str]:
    try:
        out = subprocess.check_output(["docker", "info"], text=True, timeout=15)
    except Exception as e:  # noqa: BLE001
        return False, f"docker info failed: {e}"
    if "nvidia" not in out.lower():
        return False, "nvidia runtime not listed in `docker info` output"
    return True, "docker reachable, nvidia runtime present"


def _hf_snapshot_dir(hf_cache_dir: Path, repo_id: str, revision: str) -> Path:
    return hf_cache_dir / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision


def check_env(sweep_path: str | None, results_root: Path, hf_cache_dir: Path) -> dict:
    """Spec §6 "check-env (before any sweep)". Writes `check_env.json` under
    `results_root` and returns the same dict; `ok: False` means the caller
    (`main`) should exit 1."""
    result: dict = {"ok": True, "checks": {}}

    def record(name: str, ok: bool, detail=None) -> None:
        result["checks"][name] = {"ok": ok, "detail": detail}
        if not ok:
            result["ok"] = False

    ok, detail = _docker_nvidia_ok()
    record("docker_nvidia_runtime", ok, detail)

    for name, cmd in (("wsl_nvidia_smi", WSL_SMI), ("win_nvidia_smi", WIN_SMI)):
        try:
            row = read_gpu(cmd)
            record(name, True, row)
        except Exception as e:  # noqa: BLE001
            record(name, False, str(e))

    sweep_dict = load_sweep(sweep_path) if sweep_path else None

    engines_needed = set(ENGINES)
    if sweep_dict is not None:
        axes_engines = sweep_dict.get("axes", {}).get("engine")
        if axes_engines:
            engines_needed = set(axes_engines)
    for name in sorted(engines_needed):
        spec = ENGINES[name]
        if spec.image is None:
            # Task 9 fix round 1: the echo engine is a local subprocess, not
            # a docker image -- there is nothing to `docker image inspect`
            # (and passing None into it raises TypeError).
            record(f"image_present:{name}", True, "local subprocess, no image")
            continue
        record(f"image_present:{name}", _image_present(spec.image), spec.image)

    if sweep_dict is not None:
        model, model_revision = sweep_dict.get("model"), sweep_dict.get("model_revision")
        if model and model_revision:
            path = _hf_snapshot_dir(hf_cache_dir, model, model_revision)
            record("model_revision_on_disk:target", path.exists(), str(path))
        draft_model, draft_revision = sweep_dict.get("draft_model"), sweep_dict.get("draft_revision")
        if draft_model and draft_revision:
            path = _hf_snapshot_dir(hf_cache_dir, draft_model, draft_revision)
            record("model_revision_on_disk:draft", path.exists(), str(path))

        traces_dir = Path(sweep_options(sweep_dict)["traces_dir"])
        for workload, base_name in WORKLOAD_TRACES.items():
            trace = traces_dir / f"{base_name}_v1.jsonl"
            meta_path = traces_dir / f"{base_name}_v1.meta.json"
            if not trace.exists() or not meta_path.exists():
                continue  # not every workload is used by every sweep
            actual = hashlib.sha256(trace.read_bytes()).hexdigest()
            expected = json.loads(meta_path.read_text(encoding="utf-8")).get("trace_sha256")
            record(f"trace_sha:{workload}", actual == expected, base_name)

    if _image_present(CLIENT_IMAGE):
        record("client_image_present", True, CLIENT_IMAGE)
    else:
        try:
            build_client_image(Docker())
            record("client_image_present", True, f"built {CLIENT_IMAGE}")
        except Exception as e:  # noqa: BLE001
            record("client_image_present", False, str(e))

    # Fix round 1, minor: land check_env.json next to the sweep it checked
    # (results/<sweep_id>/) when a --sweep was given, not always the results root.
    out_dir = results_root / sweep_dict["sweep_id"] if sweep_dict is not None else results_root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "check_env.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _http_module():
    import requests
    return requests


def _cmd_check_env(a) -> int:
    result = check_env(a.sweep, Path(a.results), Path(a.hf_cache))
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def _run_sweep_cmd(sweep_path, a, *, resume: bool) -> int:
    report = run_sweep(
        sweep_path, a.results, resume=resume, docker=Docker(), http_factory=_http_module,
        hf_cache_dir=a.hf_cache, compile_cache_root=a.compile_cache,
    )
    verb = "resumed sweep" if resume else "sweep"
    print(f"{verb} {report.sweep_id}: {len(report.summaries)} run(s) recorded this session")
    return 0


def _cmd_run(a) -> int:
    return _run_sweep_cmd(a.sweep, a, resume=False)


def _cmd_resume(a) -> int:
    return _run_sweep_cmd(Path(a.results) / a.sweep_id / "sweep.yaml", a, resume=True)


def _cmd_probe(a) -> int:
    print("see Task 10")
    return 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench.runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    ce = sub.add_parser("check-env")
    ce.add_argument("--sweep", default=None, help="sweep YAML to check images/models/traces against")
    ce.add_argument("--results", default=DEFAULT_RESULTS_ROOT)
    ce.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    ce.set_defaults(fn=_cmd_check_env)

    r = sub.add_parser("run")
    r.add_argument("sweep")
    r.add_argument("--results", default=DEFAULT_RESULTS_ROOT)
    r.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    r.add_argument("--compile-cache", default=DEFAULT_COMPILE_CACHE)
    r.set_defaults(fn=_cmd_run)

    rs = sub.add_parser("resume")
    rs.add_argument("sweep_id")
    rs.add_argument("--results", default=DEFAULT_RESULTS_ROOT)
    rs.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    rs.add_argument("--compile-cache", default=DEFAULT_COMPILE_CACHE)
    rs.set_defaults(fn=_cmd_resume)

    pr = sub.add_parser("probe")
    pr.add_argument("name", nargs="?", default=None)
    pr.set_defaults(fn=_cmd_probe)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
