"""P39: host CPU alongside the two GPU views. The load generator is co-located
with the engine, so CPU contention can starve the client and inflate TTFT with
no trace in the GPU samples -- and the user asked the right question ("does me
using the laptop skew the measurements?"), which this makes answerable from the
artifacts instead of by assumption."""
import json
import time

import pytest

from bench.runner.gpu_monitor import GpuSampler, read_host_cpu, reset_host_cpu_counter

# /proc/stat line shape: cpu user nice system idle iowait irq softirq steal ...
STAT_1 = "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 1 2 3 4\n"
STAT_2 = "cpu  200 0 200 1200 0 0 0 0 0 0\ncpu0 1 2 3 4\n"
LOADAVG = "0.42 0.31 0.20 1/900 12345\n"


def _write(tmp_path, stat_text):
    stat = tmp_path / "stat"
    stat.write_text(stat_text)
    load = tmp_path / "loadavg"
    load.write_text(LOADAVG)
    return str(stat), str(load)


def test_first_sample_has_no_interval_and_the_second_computes_busy_pct(tmp_path):
    reset_host_cpu_counter()
    s1, l1 = _write(tmp_path, STAT_1)
    first = read_host_cpu(s1, l1)
    assert first["cpu_busy_pct"] is None            # nothing to difference against yet
    assert first["loadavg_1m"] == 0.42 and first["loadavg_15m"] == 0.20
    assert first["cpu_total_jiffies"] == 1000 and first["cpu_idle_jiffies"] == 800

    s2, l2 = _write(tmp_path, STAT_2)
    second = read_host_cpu(s2, l2)
    # total 1000 -> 1600 (+600), idle 800 -> 1200 (+400): 200 of 600 jiffies busy
    assert second["cpu_busy_pct"] == pytest.approx(33.3)


def test_counter_reset_makes_the_next_sample_a_first_sample(tmp_path):
    s1, l1 = _write(tmp_path, STAT_1)
    read_host_cpu(s1, l1)
    reset_host_cpu_counter()
    assert read_host_cpu(s1, l1)["cpu_busy_pct"] is None


def test_sampler_rows_carry_host_cpu_and_survive_a_failing_host_reader(tmp_path):
    def gpu_reader(cmd):
        return {"used_mb": 10, "total_mb": 6141, "sm_util": 1, "mem_util": 1, "sm_clock": 2055,
                "mem_clock": 8001, "temp_c": 50, "power_w": 20.0, "throttle_reasons": "0x0"}

    calls = {"n": 0}

    def host_reader():
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("proc read failed")
        return {"cpu_busy_pct": 12.5, "loadavg_1m": 0.4}

    out = tmp_path / "g.jsonl"
    s = GpuSampler(out, interval_s=0.01, reader=gpu_reader, host_reader=host_reader)
    s.start()
    deadline = time.monotonic() + 5
    while calls["n"] < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    s.stop()

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["cpu_busy_pct"] == 12.5 and rows[0]["loadavg_1m"] == 0.4
    # a failing host reader is recorded, never fatal, and the GPU fields survive
    failed = [r for r in rows if "host_cpu_error" in r]
    assert failed and "proc read failed" in failed[0]["host_cpu_error"]
    assert failed[0]["used_ours_mb"] == 10
