"""Task 10: the pure P0b calibration functions (hand-built inputs), the
`run_probe("clock_pin", ...)` argument-plumbing test (a monkeypatched
`subprocess.run`, no real hardware), and the three calibration sweep YAMLs
expanding/validating against the real `bench/traces` metas. The other three
probes (`host_reservation`, `cudagraph_cost`, `oom_signal`) launch real
engine containers and are exercised on the target machine, never here (see
`bench/runner/probes.py`'s module docstring)."""
from __future__ import annotations

import json

import pytest

import bench.runner.probes as probes
from bench.runner.config import validate
from bench.runner.lifecycle import RunPaths
from bench.runner.probes import acceptance_delta, ceiling_from_rows, parity_verdict, throttle_baseline
from bench.runner.sweep import expand, load_sweep

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

    paths = RunPaths(
        results_root=tmp_path / "results", sweep_dir=tmp_path / "results" / "probes",
        run_dir=tmp_path / "results" / "probes" / "_scratch",
        traces_dir=tmp_path / "traces", hf_cache_dir=tmp_path / "hfcache",
        compile_cache_root=tmp_path / "compile_cache",
    )

    result = probes.run_probe(
        "clock_pin", {"clock_pin": {"sm_mhz": 2055}}, paths,
        docker=None, http=None, gpu_reader=lambda cmd: {}, popen=None,
        sleep=lambda s: None, clock=lambda: 0.0,
    )

    assert result["pinned"] is False
    assert result["output"]["pin_returncode"] == 1
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


def test_run_probe_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown probe"):
        probes.run_probe("nonsense", {}, RunPaths(
            results_root="x", sweep_dir="x", run_dir="x",
            traces_dir="x", hf_cache_dir="x", compile_cache_root="x",
        ), docker=None, http=None)


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
