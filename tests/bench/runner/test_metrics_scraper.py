import json, threading, time
from types import SimpleNamespace

import pytest

from bench.runner.engine import ENGINES
from bench.runner.metrics_scraper import MetricsScraper, _resolve, parse_prometheus, summarise

# T1: kv_usage resolves through vllm:kv_cache_usage_perc (doc %3.4 -- NOT
# vllm:gpu_cache_usage_perc, which does not exist at 0.29.0), and
# spec_decode_num_drafts_total is included (there is no spec_emitted metric).
PROM = """# HELP vllm:kv_cache_usage_perc GPU KV-cache usage.
vllm:kv_cache_usage_perc{model_name="m"} 0.42
vllm:num_requests_running{model_name="m"} 8.0
vllm:num_requests_waiting{model_name="m"} 3.0
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 120.0
vllm:spec_decode_num_draft_tokens_total{model_name="m"} 200.0
vllm:spec_decode_num_drafts_total{model_name="m"} 40.0
vllm:prefix_cache_hits_total{model_name="m"} 50.0
vllm:prefix_cache_queries_total{model_name="m"} 100.0
"""

# doc %4.4/%6.3: cached_tokens_total is multi-labelled by cache_source (only
# device is the prefix-hit proxy) and prompt_tokens_total is split by
# is_streaming with no unlabelled aggregate series (must be summed).
SGLANG_PROM = """sglang:token_usage{model_name="m"} 0.33
sglang:num_running_reqs{model_name="m"} 2.0
sglang:num_queue_reqs{model_name="m"} 1.0
sglang:spec_accept_length{model_name="m"} 3.05
sglang:spec_accept_rate{model_name="m"} 0.6
sglang:spec_verify_calls_total{model_name="m"} 21.0
sglang:cached_tokens_total{model_name="m",cache_source="device"} 400.0
sglang:cached_tokens_total{model_name="m",cache_source="host"} 999.0
sglang:prompt_tokens_total{model_name="m",is_streaming="True"} 30.0
sglang:prompt_tokens_total{model_name="m",is_streaming="False"} 6900.0
"""


def test_parse_prometheus_keeps_labels_and_values():
    d = parse_prometheus(PROM)
    assert d['vllm:kv_cache_usage_perc{model_name="m"}'] == 0.42 and len(d) == 8


def test_parse_prometheus_drops_non_finite_and_tolerates_trailing_timestamp():
    text = "vllm:a 1.0 1700000000000\nvllm:b inf\nvllm:c nan\nvllm:d -inf\nvllm:e 2.5\n"
    assert parse_prometheus(text) == {"vllm:a": 1.0, "vllm:e": 2.5}


def test_resolve_does_not_collide_with_similarly_prefixed_metrics():
    raw = {"vllm:foo": 1.0, "vllm:foo_total": 99.0, "vllm:foo_bucket": 5.0}
    assert _resolve(raw, "vllm:foo") == 1.0
    raw2 = {'vllm:foo{a="1"}': 2.0, "vllm:foo_total": 99.0}
    assert _resolve(raw2, "vllm:foo") == 2.0


def test_resolve_sums_configured_multi_series_metrics():
    raw = {'sglang:prompt_tokens_total{is_streaming="True"}': 30.0,
           'sglang:prompt_tokens_total{is_streaming="False"}': 6900.0}
    assert _resolve(raw, "sglang:prompt_tokens_total") == 6930.0


def test_scraper_resolves_engine_names_and_summarises(fake_http, tmp_path):
    fake_http.metrics_text = PROM
    s = MetricsScraper(fake_http, "http://localhost:8000", ENGINES["vllm"], tmp_path / "m.jsonl", interval_s=0.01)
    s.start(); time.sleep(0.05); s.stop()
    rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert rows[0]["engine"] == "vllm"
    assert rows[0]["kv_usage"] == 0.42 and rows[0]["running"] == 8 and rows[0]["spec_accepted"] == 120
    assert rows[0]["spec_drafts"] == 40 and rows[0]["prefix_hits"] == 50 and rows[0]["prefix_queries"] == 100
    rows[-1]["spec_accepted"], rows[-1]["spec_draft"] = 320.0, 400.0     # simulate progress over the run
    rows[-1]["spec_drafts"], rows[-1]["prefix_hits"], rows[-1]["prefix_queries"] = 70.0, 90.0, 150.0
    summ = summarise(rows)
    assert summ["acceptance_rate_mean"] == (320 - 120) / (400 - 200) and summ["kv_usage_peak"] == 0.42 and summ["waiting_max"] == 3
    assert summ["acceptance_source"] == "vllm_counters"
    assert summ["running_max"] == 8
    assert summ["drafts_delta"] == 70.0 - 40.0
    assert summ["prefix_hit_rate"] == (90.0 - 50.0) / (150.0 - 100.0)


def test_scraper_resolves_sglang_names_including_summed_prefix_queries(fake_http, tmp_path):
    fake_http.metrics_text = SGLANG_PROM
    s = MetricsScraper(fake_http, "http://localhost:30000", ENGINES["sglang"], tmp_path / "m.jsonl", interval_s=0.01)
    s.start(); time.sleep(0.05); s.stop()
    rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert rows[0]["engine"] == "sglang"
    assert rows[0]["prefix_hits"] == 400.0            # device label only, not the 999.0 host series
    assert rows[0]["prefix_queries"] == 30.0 + 6900.0  # summed across is_streaming series
    assert rows[0]["kv_usage"] == 0.33 and rows[0]["running"] == 2 and rows[0]["waiting"] == 1
    assert rows[0]["spec_accepted"] == 3.05            # the gauge, not a counter


def test_summarise_vllm_branch_tags_source_and_has_no_sglang_keys():
    rows = [
        {"engine": "vllm", "kv_usage": 0.1, "running": 1, "waiting": 0, "spec_accepted": 10.0, "spec_draft": 20.0,
         "spec_drafts": 5.0, "prefix_hits": 1.0, "prefix_queries": 2.0},
        {"engine": "vllm", "kv_usage": 0.2, "running": 2, "waiting": 1, "spec_accepted": 30.0, "spec_draft": 40.0,
         "spec_drafts": 9.0, "prefix_hits": 3.0, "prefix_queries": 6.0},
    ]
    summ = summarise(rows)
    assert summ["acceptance_source"] == "vllm_counters"
    assert summ["acceptance_rate_mean"] == (30.0 - 10.0) / (40.0 - 20.0)
    assert "spec_accept_length_last" not in summ and "spec_accept_rate_mean" not in summ


def test_summarise_sglang_branch_uses_gauges_not_fabricated_counters():
    rows = [
        {"engine": "sglang", "kv_usage": 0.1, "running": 0, "waiting": 0, "spec_accepted": 2.5, "spec_draft": 5.0,
         "spec_drafts": 5.0, "prefix_hits": 10.0, "prefix_queries": 20.0,
         "raw": {'sglang:spec_accept_rate{model_name="m"}': 0.5}},
        {"engine": "sglang", "kv_usage": 0.2, "running": 3, "waiting": 1, "spec_accepted": 3.1, "spec_draft": 8.0,
         "spec_drafts": 8.0, "prefix_hits": 12.0, "prefix_queries": 24.0,
         "raw": {'sglang:spec_accept_rate{model_name="m"}': 0.7}},
    ]
    summ = summarise(rows)
    assert summ["acceptance_rate_mean"] is None       # never a fabricated real number for sglang
    assert summ["acceptance_source"] == "sglang_gauge"
    assert summ["spec_accept_length_last"] == 3.1
    assert summ["spec_accept_rate_mean"] == 0.7        # only the running > 0 row counts


def test_summarise_without_spec_counters_gives_none():
    rows = [{"kv_usage": 0.1, "running": 1, "waiting": 0, "spec_accepted": None, "spec_draft": None,
             "spec_drafts": None, "prefix_hits": None, "prefix_queries": None}]
    summ = summarise(rows)
    assert summ["acceptance_rate_mean"] is None
    assert summ["drafts_delta"] is None
    assert summ["prefix_hit_rate"] is None


def test_scraper_records_error_on_non_200_status(tmp_path):
    class BadHTTP:
        def get(self, url, timeout=5):
            return SimpleNamespace(status_code=500, text="")

    s = MetricsScraper(BadHTTP(), "http://localhost:8000", ENGINES["vllm"], tmp_path / "m.jsonl", interval_s=0.01)
    s.start(); time.sleep(0.05); s.stop()
    row = json.loads((tmp_path / "m.jsonl").read_text().splitlines()[0])
    assert "error" in row and row["kv_usage"] is None


def test_double_start_raises(fake_http, tmp_path):
    s = MetricsScraper(fake_http, "http://localhost:8000", ENGINES["vllm"], tmp_path / "m.jsonl", interval_s=0.05)
    s.start()
    try:
        with pytest.raises(RuntimeError):
            s.start()
    finally:
        s.stop()


def test_stop_blocks_past_short_join_timeout_until_thread_truly_exits(tmp_path, monkeypatch):
    monkeypatch.setattr(MetricsScraper, "JOIN_TIMEOUT_S", 0.02)
    started = threading.Event()
    release = threading.Event()

    class SlowHTTP:
        def get(self, url, timeout=5):
            started.set()
            release.wait(2)
            return SimpleNamespace(status_code=200, text=PROM)

    s = MetricsScraper(SlowHTTP(), "http://localhost:8000", ENGINES["vllm"], tmp_path / "m.jsonl", interval_s=0.01)
    s.start()
    assert started.wait(2)

    def _release_soon():
        time.sleep(0.1)
        release.set()

    threading.Thread(target=_release_soon, daemon=True).start()
    t0 = time.monotonic()
    s.stop()
    assert time.monotonic() - t0 >= 0.09
    assert not s._thread.is_alive()
