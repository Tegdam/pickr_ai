import json, time

from bench.runner.gpu_monitor import GpuSampler, read_gpu


def test_read_gpu_parses_csv(monkeypatch):
    import bench.runner.gpu_monitor as gm
    monkeypatch.setattr(gm.subprocess, "check_output", lambda cmd, text=True, timeout=5: "512, 6141, 87, 45, 2055, 7001, 61, 63.20, 0x0000000000000004\n")
    d = read_gpu(["nvidia-smi"])
    assert d == {"used_mb": 512, "total_mb": 6141, "sm_util": 87, "mem_util": 45, "sm_clock": 2055, "mem_clock": 7001,
                 "temp_c": 61, "power_w": 63.2, "throttle_reasons": "0x0000000000000004"}


def test_read_gpu_handles_na(monkeypatch):
    import bench.runner.gpu_monitor as gm
    monkeypatch.setattr(gm.subprocess, "check_output", lambda cmd, text=True, timeout=5: "0, 6141, 0, 0, 2055, 7001, 49, [N/A], 0x0\n")
    assert read_gpu(["x"])["power_w"] is None


def test_sampler_writes_host_and_ours(tmp_path):
    calls = {"n": 0}
    def reader(cmd):
        calls["n"] += 1
        return {"used_mb": 1000 if "exe" in cmd[0] else 700, "total_mb": 6141, "sm_util": 1, "mem_util": 1, "sm_clock": 1,
                "mem_clock": 1, "temp_c": 50, "power_w": 20.0, "throttle_reasons": "0x0"}
    s = GpuSampler(tmp_path / "g.jsonl", interval_s=0.01, wsl_cmd=["nvidia-smi"], win_cmd=["nvidia-smi.exe"], reader=reader)
    s.start(); time.sleep(0.1); s.stop()
    rows = [json.loads(l) for l in (tmp_path / "g.jsonl").read_text().splitlines()]
    assert len(rows) >= 3
    assert rows[0]["used_total_mb"] == 1000 and rows[0]["used_ours_mb"] == 700 and rows[0]["used_host_mb"] == 300
    assert "t" in rows[0] and "wall" in rows[0] and rows[0]["power_w"] == 20.0


def test_sampler_records_reader_errors_instead_of_dying(tmp_path):
    def reader(cmd):
        if "exe" in cmd[0]:
            raise RuntimeError("no windows smi")
        return {"used_mb": 1, "total_mb": 2, "sm_util": 0, "mem_util": 0, "sm_clock": 0, "mem_clock": 0, "temp_c": 0, "power_w": None, "throttle_reasons": "0x0"}
    s = GpuSampler(tmp_path / "g.jsonl", interval_s=0.01, wsl_cmd=["a"], win_cmd=["b.exe"], reader=reader)
    s.start(); time.sleep(0.05); s.stop()
    row = json.loads((tmp_path / "g.jsonl").read_text().splitlines()[0])
    assert row["used_ours_mb"] == 1 and row["used_total_mb"] is None and "no windows smi" in row["win_error"]


def test_sampler_stop_is_prompt(tmp_path):
    """Thread stop must not block on a full interval (Event.wait, not time.sleep)."""
    def reader(cmd):
        return {"used_mb": 1, "total_mb": 2, "sm_util": 0, "mem_util": 0, "sm_clock": 0, "mem_clock": 0, "temp_c": 0, "power_w": None, "throttle_reasons": "0x0"}
    s = GpuSampler(tmp_path / "g.jsonl", interval_s=5.0, wsl_cmd=["a"], win_cmd=["b"], reader=reader)
    s.start(); time.sleep(0.05)
    t0 = time.monotonic()
    s.stop()
    assert time.monotonic() - t0 < 1.0
