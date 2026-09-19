"""`bench.runner.cli.check_env` (Task 9 fix round 1, item 3): a sweep whose
`axes.engine` lists the echo engine (`spec.image is None`) must not crash --
before this fix, `check_env` called `_image_present(spec.image)` for every
engine unconditionally, and `_image_present(None)` raises TypeError inside
the real `subprocess.run([..., None], ...)` call. No real docker/GPU probes
are invoked here; every boundary function `check_env` reaches out to is
monkeypatched.
"""
from __future__ import annotations

from pathlib import Path

import yaml

import bench.runner.cli as cli
from tests.bench.runner.test_lifecycle import _base_sweep_dict, _write_trace


def _write_echo_sweep(tmp_path: Path) -> Path:
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir / "chat_v1.jsonl", n=2)

    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_base_sweep_dict(traces_dir)), encoding="utf-8")

    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump({
        "base": str(base_path), "sweep_id": "sw-echo", "rq_tag": "x", "schedule_seed": 0,
        "max_retries_total": 0, "reps": 1,
        "axes": {"engine": ["echo"], "workload": ["A"], "concurrency": [1]},
    }), encoding="utf-8")
    return sweep_path


def test_check_env_skips_image_presence_for_engines_with_no_image(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_docker_nvidia_ok", lambda: (True, "docker reachable"))
    monkeypatch.setattr(cli, "read_gpu", lambda cmd: {"used_mb": 0, "total_mb": 1000, "temp_c": 40})

    def guarded_image_present(image):
        assert image is not None, "_image_present must not be called with spec.image=None"
        return True

    monkeypatch.setattr(cli, "_image_present", guarded_image_present)
    monkeypatch.setattr(cli, "build_client_image", lambda docker: "sha256:fake")

    sweep_path = _write_echo_sweep(tmp_path)
    result = cli.check_env(str(sweep_path), tmp_path / "results", tmp_path / "hfcache")

    assert result["checks"]["image_present:echo"] == {"ok": True, "detail": "local subprocess, no image"}
    assert result["checks"]["client_image_present"]["ok"] is True


def test_check_env_still_checks_a_real_image_for_a_docker_engine(tmp_path, monkeypatch):
    """Companion test: the skip is specific to `spec.image is None` -- a
    docker-based engine in the same sweep still goes through `_image_present`
    normally."""
    monkeypatch.setattr(cli, "_docker_nvidia_ok", lambda: (True, "docker reachable"))
    monkeypatch.setattr(cli, "read_gpu", lambda cmd: {"used_mb": 0, "total_mb": 1000, "temp_c": 40})

    calls = []

    def fake_image_present(image):
        calls.append(image)
        return image is not None

    monkeypatch.setattr(cli, "_image_present", fake_image_present)
    monkeypatch.setattr(cli, "build_client_image", lambda docker: "sha256:fake")

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    _write_trace(traces_dir / "chat_v1.jsonl", n=2)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(_base_sweep_dict(traces_dir)), encoding="utf-8")
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump({
        "base": str(base_path), "sweep_id": "sw-vllm", "rq_tag": "x", "schedule_seed": 0,
        "max_retries_total": 0, "reps": 1,
        "axes": {"engine": ["vllm"], "workload": ["A"], "concurrency": [1]},
    }), encoding="utf-8")

    result = cli.check_env(str(sweep_path), tmp_path / "results", tmp_path / "hfcache")

    assert "vllm/vllm-openai:v0.29.0" in calls
    assert result["checks"]["image_present:vllm"]["detail"] == "vllm/vllm-openai:v0.29.0"
