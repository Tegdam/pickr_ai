import hashlib
import json

import pytest

from bench.runner.summary import build_summary, goodput_grid, write_requests
from tests.bench.runner.test_engine import _cfg

TRACE_ROWS = [
    {"record_id": "q000000-c0", "call_role": "chat", "workload": "A"},
    {"record_id": "q000001-c0", "call_role": "chat", "workload": "A"},
    {"record_id": "q000002-c0", "call_role": "condense", "workload": "A"},
    {"record_id": "q000003-c0", "call_role": "chat", "workload": "B"},
]


def test_write_requests_maps_trace_row_by_index_and_converts_seconds_to_ms(client_json, tmp_path):
    out = tmp_path / "requests.jsonl"
    requests = write_requests(client_json, TRACE_ROWS, _cfg(), out)

    assert len(requests) == 4
    r0 = requests[0]
    assert r0["request_id"] == 0
    assert r0["trace_record_id"] == "q000000-c0"
    assert r0["call_role"] == "chat"
    assert r0["workload"] == "A"
    assert r0["ttft_ms_client"] == pytest.approx(48.0)
    assert r0["tpot_ms_client"] == pytest.approx(20.0)
    assert r0["itl_ms"] == pytest.approx([20.0] * 49)
    assert r0["itl_is_per_chunk"] is False  # spec_method="off" in _cfg()
    assert r0["e2e_ms_derived"] == pytest.approx((0.048 + 49 * 0.02) * 1000)
    # tpot_ms_derived = (e2e - ttft_client) / max(output_tokens - 1, 1)
    expected_tpot = ((0.048 + 49 * 0.02) * 1000 - 48.0) / 49
    assert r0["tpot_ms_derived"] == pytest.approx(expected_tpot)
    assert r0["prompt_tokens"] == 250
    assert r0["output_tokens"] == 50
    assert r0["error"] is None
    assert r0["finish_reason"] is None
    assert r0["schema_valid"] is None
    assert r0["start_time"] == pytest.approx(0.0)
    assert r0["generated_text_sha256"] == hashlib.sha256(b"a").hexdigest()

    # trace row order follows i <-> request i, not any grouping by role/workload.
    assert requests[2]["trace_record_id"] == "q000002-c0" and requests[2]["call_role"] == "condense"
    assert requests[3]["workload"] == "B"

    # persisted to out_path as JSONL, one record per line.
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    assert json.loads(lines[0])["trace_record_id"] == "q000000-c0"


def test_write_requests_error_field_none_when_empty_string(client_json, tmp_path):
    client_json["errors"] = ["boom", "", "", ""]
    requests = write_requests(client_json, TRACE_ROWS, _cfg(), tmp_path / "r.jsonl")
    assert requests[0]["error"] == "boom"
    assert requests[1]["error"] is None


def test_write_requests_itl_is_per_chunk_true_under_speculation(client_json, tmp_path):
    cfg = _cfg(spec_method="ngram", spec_k=3, ngram_lookup_max=4)
    requests = write_requests(client_json, TRACE_ROWS, cfg, tmp_path / "r.jsonl")
    assert all(r["itl_is_per_chunk"] is True for r in requests)


def _req(ttft, tpot, error=None):
    return {"ttft_ms_client": ttft, "tpot_ms_derived": tpot, "error": error}


def test_goodput_grid_fraction_of_completed_requests():
    requests = [
        _req(400, 40),   # meets ttft<=500 & tpot<=50
        _req(600, 40),   # fails ttft<=500
        _req(400, 60),   # fails tpot<=50
        _req(100, 10, error="timeout"),  # excluded: not completed
    ]
    grid = goodput_grid(requests)
    # only 3 completed requests; 1 of them meets the pre-registered SLO cell.
    assert grid["ttft<=500&tpot<=50"] == pytest.approx(1 / 3)
    assert grid["ttft<=1000&tpot<=100"] == pytest.approx(3 / 3)
    assert grid["ttft<=250&tpot<=25"] == pytest.approx(0.0)


def test_goodput_grid_all_keys_present_and_zero_when_no_completed_requests():
    grid = goodput_grid([_req(100, 10, error="x")])
    assert grid["ttft<=500&tpot<=50"] == 0.0
    assert len(grid) == 6 * 5


def test_build_summary_derived_percentiles_and_client_copies(client_json, tmp_path):
    cfg = _cfg(num_prompts=4)
    requests = write_requests(client_json, TRACE_ROWS, cfg, tmp_path / "r.jsonl")
    gpu_rows = [
        {"used_ours_mb": 4000, "used_host_mb": 500, "sm_clock": 1500, "power_w": 40.0, "throttle_reasons": "0x0"},
        {"used_ours_mb": 4100, "used_host_mb": 520, "sm_clock": 1500, "power_w": 42.0, "throttle_reasons": "0x0"},
    ]
    metric_rows = []
    timing = {"ready_s": 10.0, "warmup_s": 2.0, "client_s": 5.0, "cooldown_s": 3.0,
              "vram_returned_mb": 4000, "vram_leak_mb": 0, "clocks_pinned": True}

    summary = build_summary(client_json, requests, gpu_rows, metric_rows, cfg, timing)

    import statistics
    assert summary["ttft_ms_client"]["p50"] == pytest.approx(
        statistics.median([r["ttft_ms_client"] for r in requests])
    )
    assert summary["tpot_ms_derived"]["p50"] == pytest.approx(
        statistics.median([r["tpot_ms_derived"] for r in requests])
    )
    assert summary["e2e_ms_derived"]["p50"] == pytest.approx(
        statistics.median([r["e2e_ms_derived"] for r in requests])
    )

    # *_client copies come straight from the client's own reported percentiles.
    assert summary["ttft_p50_client"] == client_json["median_ttft_ms"]
    assert summary["ttft_p99_client"] == client_json["p99_ttft_ms"]
    assert summary["tpot_p50_client"] == client_json["median_tpot_ms"]
    assert summary["tpot_p99_client"] == client_json["p99_tpot_ms"]
    assert summary["itl_p50_client"] == client_json["median_itl_ms"]
    # e2e client fields are absent from this fixture -- copied as None, not a KeyError.
    assert summary["e2e_p50_client"] is None

    assert summary["output_tok_s"] == client_json["output_throughput"]
    assert summary["req_s"] == client_json["request_throughput"]
    assert summary["completed"] == 4
    assert summary["num_prompts"] == cfg.num_prompts
    assert summary["error_rate"] == 0.0
    assert summary["itl_is_per_chunk"] is False

    assert summary["peak_used_ours_mb"] == 4100
    assert summary["peak_used_host_mb"] == 520
    assert summary["host_share_drift_mb"] == 20
    assert summary["clock_cv"] == 0.0  # constant sm_clock
    assert summary["throttled"] is False
    assert summary["power_w"]["max"] == 42.0

    assert summary["goodput_pre_registered"] == summary["goodput_by_threshold"]["ttft<=500&tpot<=50"]
    assert summary["timing"] == timing

    assert summary["acceptance_source"] == "vllm_counters"
    assert summary["engine_metrics"]["acceptance_source"] == "vllm_counters"

    assert summary["valid"] is True
    assert summary["invalid_reason"] is None


def test_build_summary_throttled_true_when_reasons_include_hw_slowdown_bit(client_json, tmp_path):
    cfg = _cfg(num_prompts=4)
    requests = write_requests(client_json, TRACE_ROWS, cfg, tmp_path / "r.jsonl")
    gpu_rows = [
        {"used_ours_mb": 4000, "used_host_mb": 500, "sm_clock": 1500, "power_w": 40.0, "throttle_reasons": "0x8"},
    ]
    timing = {"ready_s": 1.0, "warmup_s": 1.0, "client_s": 1.0, "cooldown_s": 1.0,
              "vram_returned_mb": 0, "vram_leak_mb": 0, "clocks_pinned": True}
    summary = build_summary(client_json, requests, gpu_rows, [], cfg, timing)
    assert summary["throttled"] is True
    # throttled alone never invalidates the run (recorded only).
    assert summary["valid"] is True


def test_build_summary_ignores_gpu_samples_with_none_throttle_reasons(client_json, tmp_path):
    cfg = _cfg(num_prompts=4)
    requests = write_requests(client_json, TRACE_ROWS, cfg, tmp_path / "r.jsonl")
    gpu_rows = [{"used_ours_mb": 100, "used_host_mb": None, "sm_clock": None, "power_w": None, "throttle_reasons": None}]
    timing = {"ready_s": 1.0, "warmup_s": 1.0, "client_s": 1.0, "cooldown_s": 1.0,
              "vram_returned_mb": 0, "vram_leak_mb": 0, "clocks_pinned": True}
    summary = build_summary(client_json, requests, gpu_rows, [], cfg, timing)
    assert summary["throttled"] is False
    assert summary["clock_cv"] is None
    assert summary["host_share_drift_mb"] is None
    assert summary["power_w"] == {"p50": None, "p90": None, "max": None}


def test_build_summary_marks_run_invalid_when_completed_short(client_json, tmp_path):
    cfg = _cfg(num_prompts=5)  # client only completed 4
    requests = write_requests(client_json, TRACE_ROWS, cfg, tmp_path / "r.jsonl")
    timing = {"ready_s": 1.0, "warmup_s": 1.0, "client_s": 1.0, "cooldown_s": 1.0,
              "vram_returned_mb": 0, "vram_leak_mb": 0, "clocks_pinned": True}
    summary = build_summary(client_json, requests, [], [], cfg, timing)
    assert summary["valid"] is False
    assert "completed" in summary["invalid_reason"]


def test_build_summary_invalid_when_spec_on_and_acceptance_zero(client_json, tmp_path):
    cfg = _cfg(num_prompts=4, spec_method="ngram", spec_k=3, ngram_lookup_max=4)
    requests = write_requests(client_json, TRACE_ROWS, cfg, tmp_path / "r.jsonl")
    # no spec_drafts progress at all -> engine_metrics.drafts_delta is None -> acceptance tripwire fires.
    metric_rows = [{"engine": "vllm", "spec_accepted": 0, "spec_draft": 0, "spec_drafts": 0}]
    timing = {"ready_s": 1.0, "warmup_s": 1.0, "client_s": 1.0, "cooldown_s": 1.0,
              "vram_returned_mb": 0, "vram_leak_mb": 0, "clocks_pinned": True}
    summary = build_summary(client_json, requests, [], metric_rows, cfg, timing)
    assert summary["valid"] is False
    assert "acceptance" in summary["invalid_reason"]
