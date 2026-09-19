"""Task 10 (+ fix round 1): the pure P0b calibration functions (hand-built
inputs), `run_probe`/`_run_oom_fraction` argument-plumbing tests (fakes, no
real hardware), `_probe_run_config` tracking `base.yaml`, and the three
calibration sweep YAMLs expanding/validating against the real `bench/traces`
metas. The bulk of `host_reservation`/`cudagraph_cost`/`oom_signal`
orchestration launches real engine containers and is exercised on the target
machine, never here (see `bench/runner/probes.py`'s module docstring)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import bench.runner.probes as probes
from bench.runner.config import validate
from bench.runner.engine import ENGINES
from bench.runner.lifecycle import RunPaths
from bench.runner.probes import acceptance_delta, ceiling_from_rows, ceiling_rows_from_sweep, parity_verdict, throttle_baseline
from bench.runner.sweep import expand, load_sweep
from tests.bench.runner.conftest import FakeDocker, FakeHTTP
from tests.bench.runner.test_engine import _cfg

# ---------------------------------------------------------------------------
# ceiling_from_rows
# ---------------------------------------------------------------------------


def test_ceiling_from_rows_finds_knee_where_throughput_collapses():
    # req_s tracks ~0.98x the offered rate through 64 req/s, then collapses;
    # ttft_p99 stays under 2x the rate-8 baseline (100ms) through 64, then
    # doubles past it at 128 -- both signals agree the ceiling is 64.
    rows = [
        {"request_rate": 8.0, "req_s": 7.9, "ttft_p99_ms": 100.0},
        {"request_rate": 16.0, "req_s": 15.8, "ttft_p99_ms": 105.0},
        {"request_rate": 32.0, "req_s": 31.5, "ttft_p99_ms": 110.0},
        {"request_rate": 64.0, "req_s": 62.9, "ttft_p99_ms": 150.0},
        {"request_rate": 128.0, "req_s": 90.0, "ttft_p99_ms": 250.0},
        {"request_rate": 256.0, "req_s": 100.0, "ttft_p99_ms": 400.0},
    ]
    result = ceiling_from_rows(rows)
    assert result["ceiling_req_s"] == 64.0
    assert [r["passes"] for r in result["table"]] == [True, True, True, True, False, False]
    assert [r["request_rate"] for r in result["table"]] == [8.0, 16.0, 32.0, 64.0, 128.0, 256.0]


def test_ceiling_from_rows_out_of_order_input_is_sorted():
    rows = [
        {"request_rate": 16.0, "req_s": 15.8, "ttft_p99_ms": 105.0},
        {"request_rate": 8.0, "req_s": 7.9, "ttft_p99_ms": 100.0},
    ]
    result = ceiling_from_rows(rows)
    assert [r["request_rate"] for r in result["table"]] == [8.0, 16.0]
    assert result["ceiling_req_s"] == 16.0


def test_ceiling_from_rows_empty_input():
    assert ceiling_from_rows([]) == {"ceiling_req_s": None, "table": []}


def test_ceiling_from_rows_no_rate_ever_passes():
    rows = [{"request_rate": 8.0, "req_s": 1.0, "ttft_p99_ms": 100.0}]
    result = ceiling_from_rows(rows)
    assert result["ceiling_req_s"] is None
    assert result["table"][0]["passes"] is False


# ---------------------------------------------------------------------------
# ceiling_rows_from_sweep
# ---------------------------------------------------------------------------


def _write_ceiling_run(sweep_dir: Path, run_id: str, rate: float, req_s: float, ttft_p99: float) -> None:
    run_dir = sweep_dir / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"request_rate": rate}), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps({"req_s": req_s, "ttft_p99_client": ttft_p99}), encoding="utf-8")


def test_ceiling_rows_from_sweep_averages_reps_per_rate(tmp_path):
    sweep_dir = tmp_path / "p0b-ceiling"
    _write_ceiling_run(sweep_dir, "p0b-ceiling-0000-r0", 8.0, 7.9, 100.0)
    _write_ceiling_run(sweep_dir, "p0b-ceiling-0000-r1", 8.0, 7.7, 110.0)
    _write_ceiling_run(sweep_dir, "p0b-ceiling-0001-r0", 16.0, 15.8, 105.0)

    rows = {r["request_rate"]: r for r in ceiling_rows_from_sweep(sweep_dir)}

    assert rows[8.0]["req_s"] == pytest.approx((7.9 + 7.7) / 2)
    assert rows[8.0]["ttft_p99_ms"] == pytest.approx((100.0 + 110.0) / 2)
    assert rows[16.0]["req_s"] == pytest.approx(15.8)
    assert rows[16.0]["ttft_p99_ms"] == pytest.approx(105.0)


def test_ceiling_rows_from_sweep_skips_incomplete_runs(tmp_path):
    sweep_dir = tmp_path / "p0b-ceiling"
    missing_summary = sweep_dir / "run-missing-summary"
    missing_summary.mkdir(parents=True)
    (missing_summary / "config.yaml").write_text(yaml.safe_dump({"request_rate": 8.0}), encoding="utf-8")

    no_rate = sweep_dir / "run-no-rate"
    no_rate.mkdir(parents=True)
    (no_rate / "config.yaml").write_text(yaml.safe_dump({"concurrency": 8}), encoding="utf-8")
    (no_rate / "summary.json").write_text(json.dumps({"req_s": 1.0, "ttft_p99_client": 1.0}), encoding="utf-8")

    assert ceiling_rows_from_sweep(sweep_dir) == []


# ---------------------------------------------------------------------------
# parity_verdict
# ---------------------------------------------------------------------------


def test_parity_verdict_pass_when_all_engines_match_expected():
    result = parity_verdict({"vllm": 265, "sglang": 265}, expected=265)
    assert result == {"pass": True, "values": {"vllm": 265, "sglang": 265}, "expected": 265}


def test_parity_verdict_fail_when_one_engine_diverges():
    result = parity_verdict({"vllm": 265, "sglang": 264}, expected=265)
    assert result["pass"] is False
    assert result["values"] == {"vllm": 265, "sglang": 264}
    assert result["expected"] == 265


def test_parity_verdict_empty_dict_never_passes_vacuously():
    result = parity_verdict({}, 265)
    assert result["pass"] is False
    assert result["values"] == {}


# ---------------------------------------------------------------------------
# acceptance_delta
# ---------------------------------------------------------------------------


def test_acceptance_delta_divergence_over_10_percent_applies_the_prefix_rule():
    result = acceptance_delta(0.79, 0.70)
    assert result["rel_diff"] == pytest.approx(0.1286, abs=1e-3)
    assert result["prefix_rule_applies"] is True


def test_acceptance_delta_small_divergence_does_not_apply_the_rule():
    result = acceptance_delta(0.79, 0.78)
    assert result["rel_diff"] == pytest.approx(0.0128, abs=1e-3)
    assert result["prefix_rule_applies"] is False


def test_acceptance_delta_missing_values_never_apply_the_rule():
    assert acceptance_delta(None, 0.5) == {"rel_diff": None, "prefix_rule_applies": False}
    assert acceptance_delta(0.5, None) == {"rel_diff": None, "prefix_rule_applies": False}
    assert acceptance_delta(0.5, 0.0) == {"rel_diff": None, "prefix_rule_applies": False}


# ---------------------------------------------------------------------------
# throttle_baseline
# ---------------------------------------------------------------------------


def test_throttle_baseline_computes_per_bit_fractions():
    # 0x24 = SW Power Cap (0x4) | SW Thermal Slowdown (0x20); half the
    # samples carry it, half carry no bits at all.
    rows = [{"throttle_reasons": "0x24"}, {"throttle_reasons": "0x24"},
            {"throttle_reasons": "0x0"}, {"throttle_reasons": "0x0"}]
    result = throttle_baseline(rows)
    assert result["0x4"] == 0.5
    assert result["0x20"] == 0.5
    assert result["0x8"] == 0.0
    assert result["0x40"] == 0.0
    assert result["0x80"] == 0.0


def test_throttle_baseline_ignores_none_reasons_and_empty_input():
    assert throttle_baseline([{"throttle_reasons": None}]) == {
        "0x4": None, "0x8": None, "0x20": None, "0x40": None, "0x80": None,
    }
    assert throttle_baseline([]) == {"0x4": None, "0x8": None, "0x20": None, "0x40": None, "0x80": None}


# ---------------------------------------------------------------------------
# _probe_run_config tracks base.yaml (fix round 1, item 7)
# ---------------------------------------------------------------------------


def _scratch_paths(tmp_path) -> RunPaths:
    return RunPaths(
        results_root=tmp_path / "results", sweep_dir=tmp_path / "results" / "probes",
        run_dir=tmp_path / "results" / "probes" / "_scratch",
        traces_dir=Path("bench/traces"), hf_cache_dir=tmp_path / "hfcache",
        compile_cache_root=tmp_path / "compile_cache",
    )


def test_probe_run_config_tracks_base_yaml_cudagraph_capture_sizes(tmp_path):
    cfg = probes._probe_run_config(_scratch_paths(tmp_path), run_id="probe-test")
    base = yaml.safe_load(Path("bench/configs/base.yaml").read_text(encoding="utf-8"))
    assert cfg.cudagraph_capture_sizes == base["cudagraph_capture_sizes"]
    assert cfg.run_id == "probe-test" and cfg.engine == "vllm" and cfg.workload == "A"
    assert cfg.rq_tag == "probe" and cfg.sweep_id == "p0b-probes"


def test_probe_run_config_validates(tmp_path):
    cfg = probes._probe_run_config(_scratch_paths(tmp_path), run_id="probe-test")
    cfg.free_vram_mb_at_start, cfg.gpu_memory_utilization = 1000, 0.5
    validate(cfg)  # must not raise


# ---------------------------------------------------------------------------
# run_probe("clock_pin", ...) argument plumbing
# ---------------------------------------------------------------------------


class _FakeCompleted:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_probe_clock_pin_records_pinned_false_on_nonzero_exit(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, capture_output=True, text=True, check=False):
        calls.append(cmd)
        return _FakeCompleted(returncode=1, stdout="", stderr="permission denied")

    monkeypatch.setattr(probes.subprocess, "run", fake_run)

    paths = _scratch_paths(tmp_path)
    result = probes.run_probe(
        "clock_pin", {"clock_pin": {"sm_mhz": 2055}}, paths,
        docker=None, http=None, gpu_reader=lambda cmd: {}, popen=None,
        sleep=lambda s: None, clock=lambda: 0.0,
    )

    assert result["pinned"] is False
    assert result["output"]["pin_returncode"] == 1
    assert "sm_clock_min" not in result  # never sampled -- the pin itself failed
    assert "error" not in result

    files = list((tmp_path / "results" / "probes").glob("clock_pin-*.json"))
    assert len(files) == 1
    written = json.loads(files[0].read_text(encoding="utf-8"))
    assert written["pinned"] is False
    assert written["probe"] == "clock_pin"

    # Pin, then reset -- both against the Windows-side nvidia-smi binary.
    assert len(calls) == 2
    assert "-lgc" in calls[0] and "2055,2055" in calls[0]
    assert "-rgc" in calls[1]


def test_run_probe_clock_pin_verifies_the_pin_holds(tmp_path, monkeypatch):
    """Fix round 1, item 6: a zero exit code alone doesn't mean the pin
    held -- sample sm_clock for 20s and record whether it stayed within
    50 MHz of its own range."""
    calls = []

    def fake_run(cmd, capture_output=True, text=True, check=False):
        calls.append(cmd)
        return _FakeCompleted(returncode=0)

    monkeypatch.setattr(probes.subprocess, "run", fake_run)

    clock_readings = iter([2050, 2060, 2055])

    def fake_gpu_reader(cmd):
        return {"sm_clock": next(clock_readings, 2055)}

    state = {"t": 0.0}

    def fake_clock():
        state["t"] += 5.0
        return state["t"]

    result = probes.run_probe(
        "clock_pin", {"clock_pin": {"sm_mhz": 2055}}, _scratch_paths(tmp_path),
        docker=None, http=None, gpu_reader=fake_gpu_reader, popen=None,
        sleep=lambda s: None, clock=fake_clock,
    )

    assert result["pinned"] is True
    assert result["sm_clock_min"] == 2050 and result["sm_clock_max"] == 2060
    assert result["held"] is True
    assert "-rgc" in calls[-1]  # reset still runs after sampling


def test_run_probe_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown probe"):
        probes.run_probe("nonsense", {}, RunPaths(
            results_root="x", sweep_dir="x", run_dir="x",
            traces_dir="x", hf_cache_dir="x", compile_cache_root="x",
        ), docker=None, http=None)


def test_run_probe_with_raising_probe_writes_json_with_error(tmp_path, monkeypatch):
    def raising_run(cmd, **kwargs):
        raise RuntimeError("smi not found")

    monkeypatch.setattr(probes.subprocess, "run", raising_run)

    result = probes.run_probe(
        "clock_pin", {}, _scratch_paths(tmp_path), docker=None, http=None,
        gpu_reader=lambda cmd: {}, popen=None, sleep=lambda s: None, clock=lambda: 0.0,
    )

    assert result["error"] == "smi not found"
    assert result["error_type"] == "RuntimeError"

    files = list((tmp_path / "results" / "probes").glob("clock_pin-*.json"))
    assert len(files) == 1
    written = json.loads(files[0].read_text(encoding="utf-8"))
    assert written["error"] == "smi not found" and written["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# _run_oom_fraction: the three outcome paths, engine always stopped
# (fix round 1, items 1-4).
# ---------------------------------------------------------------------------


def _counting_clock():
    state = {"t": 0.0}

    def clock():
        state["t"] += 1.0
        return state["t"]

    return clock


def _constant_gpu_reader(cmd):
    return {"used_mb": 0, "total_mb": 6141}


def test_run_oom_fraction_still_running_is_slow_or_hung_and_always_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "wait_healthy", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("nope")))
    cfg = _cfg(run_id="probe-oom-t1")
    fake_docker = FakeDocker()
    fake_docker.keep_running = True  # container never exits on its own -- still up when we check

    result = probes._run_oom_fraction(
        cfg, _scratch_paths(tmp_path), 0.90, 900, None, None,
        spec=ENGINES["vllm"], docker=fake_docker, http=FakeHTTP(),
        gpu_reader=_constant_gpu_reader, popen=None, sleep=lambda s: None,
        clock=_counting_clock(), n_requests=5,
    )

    assert result["outcome"] == "slow_or_hung"
    assert result["error_type"] == "TimeoutError"
    stop_calls = [c for c in fake_docker.calls if c["op"] == "stop" and c["name"] == f"bench-{cfg.run_id}"]
    assert stop_calls, "engine must always be stopped, even on a hang"


def test_run_oom_fraction_exited_with_oom_text_is_clean_oom_and_always_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "wait_healthy", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dead")))
    cfg = _cfg(run_id="probe-oom-t2")
    fake_docker = FakeDocker()  # keep_running=False -> container "exits" the instant it's launched
    fake_docker.logs[f"bench-{cfg.run_id}"] = "boom: CUDA out of memory. Tried to allocate ..."

    result = probes._run_oom_fraction(
        cfg, _scratch_paths(tmp_path), 0.98, 900, None, None,
        spec=ENGINES["vllm"], docker=fake_docker, http=FakeHTTP(),
        gpu_reader=_constant_gpu_reader, popen=None, sleep=lambda s: None,
        clock=_counting_clock(), n_requests=5,
    )

    assert result["outcome"] == "clean_oom"
    assert "out of memory" in result["oom_lines"].lower()
    stop_calls = [c for c in fake_docker.calls if c["op"] == "stop" and c["name"] == f"bench-{cfg.run_id}"]
    assert stop_calls, "engine must always be stopped, even on a clean failure"


def test_run_oom_fraction_exited_without_oom_text_is_clean_fail_no_oom_text(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "wait_healthy", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("dead")))
    cfg = _cfg(run_id="probe-oom-t2b")
    fake_docker = FakeDocker()
    fake_docker.logs[f"bench-{cfg.run_id}"] = "some unrelated startup error, nothing about memory"

    result = probes._run_oom_fraction(
        cfg, _scratch_paths(tmp_path), 0.95, 900, None, None,
        spec=ENGINES["vllm"], docker=fake_docker, http=FakeHTTP(),
        gpu_reader=_constant_gpu_reader, popen=None, sleep=lambda s: None,
        clock=_counting_clock(), n_requests=5,
    )

    assert result["outcome"] == "clean_fail_no_oom_text"
    assert "log_tail" in result


def test_run_oom_fraction_success_is_served_and_always_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "wait_healthy", lambda *a, **k: 5.0)
    cfg = _cfg(run_id="probe-oom-t3", num_prompts=10)
    fake_docker = FakeDocker()
    fake_docker.keep_running = True

    result = probes._run_oom_fraction(
        cfg, _scratch_paths(tmp_path), 0.85, 900, None, None,
        spec=ENGINES["vllm"], docker=fake_docker, http=FakeHTTP(),
        gpu_reader=_constant_gpu_reader, popen=None, sleep=lambda s: None,
        clock=_counting_clock(), n_requests=3,
    )

    assert result["outcome"] == "served"
    assert result["rate_tok_s"] is not None and result["rate_tok_s"] > 0
    stop_calls = [c for c in fake_docker.calls if c["op"] == "stop" and c["name"] == f"bench-{cfg.run_id}"]
    assert stop_calls, "engine must always be stopped after serving"


def test_run_oom_fraction_computes_ratios_vs_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "wait_healthy", lambda *a, **k: 10.0)
    cfg = _cfg(run_id="probe-oom-t4", num_prompts=10)
    fake_docker = FakeDocker()
    fake_docker.keep_running = True

    result = probes._run_oom_fraction(
        cfg, _scratch_paths(tmp_path), 0.98, 900, 100.0, 5.0,
        spec=ENGINES["vllm"], docker=fake_docker, http=FakeHTTP(),
        gpu_reader=_constant_gpu_reader, popen=None, sleep=lambda s: None,
        clock=_counting_clock(), n_requests=3,
    )

    assert result["outcome"] == "served"
    assert result["ready_ratio_vs_baseline"] == pytest.approx(10.0 / 5.0)
    assert result["rate_ratio_vs_baseline"] == pytest.approx(result["rate_tok_s"] / 100.0)


# ---------------------------------------------------------------------------
# The three calibration sweep YAMLs expand/validate against real traces.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path,expected_count", [
    ("bench/configs/p0b_ceiling.yaml", 12),
    ("bench/configs/p0b_parity.yaml", 2),
    ("bench/configs/p0b_ignore_eos.yaml", 4),
])
def test_p0b_sweep_yaml_expands_to_expected_count_and_every_config_validates(path, expected_count):
    sweep_dict = load_sweep(path)
    cfgs = expand(sweep_dict, sweep_dict["sweep_id"])
    assert len(cfgs) == expected_count
    for cfg in cfgs:
        validate(cfg)  # raises on failure; real bench/traces files back the trace-sha check


def test_p0b_ceiling_axes_and_fixed_fields():
    cfgs = expand(load_sweep("bench/configs/p0b_ceiling.yaml"), "p0b-ceiling")
    assert {c.engine for c in cfgs} == {"echo"}
    assert {c.request_rate for c in cfgs} == {8.0, 16.0, 32.0, 64.0, 128.0, 256.0}
    assert all(c.load_mode == "poisson" and c.concurrency is None for c in cfgs)
    assert all(c.num_prompts == 500 and c.cooldown_min_s == 0 and c.cooldown_temp_c == 99 for c in cfgs)
    # reps=2 folded into run_id
    assert sum(1 for c in cfgs if c.request_rate == 8.0) == 2


def test_p0b_parity_axes_and_fixed_fields():
    cfgs = expand(load_sweep("bench/configs/p0b_parity.yaml"), "p0b-parity")
    assert {c.engine for c in cfgs} == {"vllm", "sglang"}
    assert all(c.concurrency == 1 and c.num_prompts == 1 and c.cooldown_min_s == 30 for c in cfgs)


def test_p0b_ignore_eos_axes_and_fixed_fields():
    cfgs = expand(load_sweep("bench/configs/p0b_ignore_eos.yaml"), "p0b-ignore-eos")
    assert {c.engine for c in cfgs} == {"vllm"}
    assert {c.spec_method for c in cfgs} == {"draft", "ngram"}
    assert {c.ignore_eos for c in cfgs} == {True, False}
    assert all(c.spec_k == 3 and c.concurrency == 1 and c.num_prompts == 100 and c.max_tokens_cap == 512
               for c in cfgs)


# ---------------------------------------------------------------------------
# p0b_probes.yaml still parses to the shape run_probe expects.
# ---------------------------------------------------------------------------


def test_p0b_probes_yaml_shape():
    params = yaml.safe_load(Path("bench/configs/p0b_probes.yaml").read_text(encoding="utf-8"))
    assert set(params) == {"oom_signal", "cudagraph_cost", "host_reservation", "clock_pin"}
    assert "load_concurrency" not in params["host_reservation"]  # dead key, dropped (fix round 1)
    assert params["cudagraph_cost"]["capture_lists"] == [[1], [1, 2, 4, 8]]
