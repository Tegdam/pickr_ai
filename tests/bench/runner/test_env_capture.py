from bench.runner.engine import ENGINES
from bench.runner.env_capture import capture_env
from tests.bench.runner.test_engine import _cfg


def test_capture_env_has_the_spec_fields(fake_docker, monkeypatch):
    import bench.runner.env_capture as ec
    monkeypatch.setattr(ec, "_host_lines", lambda: {"wsl_nvidia_smi": "Driver 610", "win_nvidia_smi": "Driver 610.88", "power_max_limit_w": 100.0,
                                                    "uname": "6.6", "docker_version": "28.0.4", "windows_power_mode": "High performance"})
    env = capture_env(_cfg(), fake_docker, ENGINES["vllm"], ["--model", "m"], client_image="vllm/vllm-openai:v0.29.0",
                      extra={"clocks_pinned": False, "client_output_schema_version": "v0.29.0-save-detailed"})
    for k in ("image", "image_digest", "pip_freeze", "wsl_nvidia_smi", "win_nvidia_smi", "power_max_limit_w", "uname",
              "docker_version", "windows_power_mode", "clocks_pinned", "client_image", "client_output_schema_version",
              "bench_git_sha", "launch_cmd", "captured_at", "engine_env", "docker_extra_args", "quantization_kernel",
              "compile_cache_mounted", "clock_pin_note", "transformers_version"):
        assert k in env, k
    assert env["image_digest"].startswith("sha256:") and env["launch_cmd"] == ["--model", "m"]
    assert env["engine_env"] == ENGINES["vllm"].env and env["docker_extra_args"] == ENGINES["vllm"].docker_extra_args
    assert env["quantization_kernel"] is None and env["compile_cache_mounted"] is False
    assert env["clock_pin_note"] == "not attempted (requires Administrator; deferred to the user)"


def test_capture_env_extra_overrides_new_keys(fake_docker, monkeypatch):
    import bench.runner.env_capture as ec
    monkeypatch.setattr(ec, "_host_lines", lambda: {})
    env = capture_env(_cfg(), fake_docker, ENGINES["sglang"], ["--model", "m"], client_image="c",
                      extra={"quantization_kernel": "awq_marlin", "compile_cache_mounted": True, "clock_pin_note": "pinned by hand"})
    assert env["quantization_kernel"] == "awq_marlin" and env["compile_cache_mounted"] is True
    assert env["clock_pin_note"] == "pinned by hand"
    assert env["engine_env"] == ENGINES["sglang"].env


def test_host_lines_survives_missing_tools(monkeypatch):
    import bench.runner.env_capture as ec

    def fake_check_output(cmd, text=True, timeout=10):
        if cmd[0] == "uname":
            return "6.6.0\n"
        if cmd[0] == "docker":
            return "Docker version 28.0.4\n"
        if "nvidia-smi.exe" in cmd[0]:
            return "610.88, 100.00\n"
        if cmd[0] == "nvidia-smi":
            return "Driver Version: 610.88\n"
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(ec.subprocess, "check_output", fake_check_output)
    lines = ec._host_lines()
    assert lines["uname"] == "6.6.0"
    assert lines["docker_version"] == "Docker version 28.0.4"
    assert lines["win_nvidia_smi"] == "Driver 610.88"
    assert lines["power_max_limit_w"] == 100.0
    assert lines["wsl_nvidia_smi"] == "Driver Version: 610.88"
    assert lines["windows_power_mode"] is None  # powercfg.exe path raises above -> caught, best effort
