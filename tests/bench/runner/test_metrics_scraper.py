import json, time

from bench.runner.engine import ENGINES
from bench.runner.metrics_scraper import MetricsScraper, parse_prometheus, summarise

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

# doc %4.4/%6.3: cached_tokens_total is multi-labelled by cache_source; only
# the device label is the prefix-hit proxy.
SGLANG_PROM = """sglang:token_usage{model_name="m"} 0.33
sglang:num_running_reqs{model_name="m"} 2.0
sglang:num_queue_reqs{model_name="m"} 1.0
sglang:spec_accept_length{model_name="m"} 3.05
sglang:spec_verify_calls_total{model_name="m"} 21.0
sglang:cached_tokens_total{model_name="m",cache_source="device"} 400.0
sglang:cached_tokens_total{model_name="m",cache_source="host"} 999.0
sglang:prompt_tokens_total{model_name="m",is_streaming="False"} 6930.0
"""


def test_parse_prometheus_keeps_labels_and_values():
    d = parse_prometheus(PROM)
    assert d['vllm:kv_cache_usage_perc{model_name="m"}'] == 0.42 and len(d) == 8


def test_scraper_resolves_engine_names_and_summarises(fake_http, tmp_path):
    fake_http.metrics_text = PROM
    s = MetricsScraper(fake_http, "http://localhost:8000", ENGINES["vllm"], tmp_path / "m.jsonl", interval_s=0.01)
    s.start(); time.sleep(0.05); s.stop()
    rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert rows[0]["kv_usage"] == 0.42 and rows[0]["running"] == 8 and rows[0]["spec_accepted"] == 120
    assert rows[0]["spec_drafts"] == 40 and rows[0]["prefix_hits"] == 50 and rows[0]["prefix_queries"] == 100
    rows[-1]["spec_accepted"], rows[-1]["spec_draft"] = 320.0, 400.0     # simulate progress over the run
    rows[-1]["spec_drafts"], rows[-1]["prefix_hits"], rows[-1]["prefix_queries"] = 70.0, 90.0, 150.0
    summ = summarise(rows)
    assert summ["acceptance_rate_mean"] == (320 - 120) / (400 - 200) and summ["kv_usage_peak"] == 0.42 and summ["waiting_max"] == 3
    assert summ["running_max"] == 8
    assert summ["drafts_delta"] == 70.0 - 40.0
    assert summ["prefix_hit_rate"] == (90.0 - 50.0) / (150.0 - 100.0)


def test_scraper_prefers_device_cache_source_for_sglang_prefix_hits(fake_http, tmp_path):
    fake_http.metrics_text = SGLANG_PROM
    s = MetricsScraper(fake_http, "http://localhost:30000", ENGINES["sglang"], tmp_path / "m.jsonl", interval_s=0.01)
    s.start(); time.sleep(0.05); s.stop()
    rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert rows[0]["prefix_hits"] == 400.0
    assert rows[0]["kv_usage"] == 0.33 and rows[0]["running"] == 2 and rows[0]["waiting"] == 1


def test_summarise_without_spec_counters_gives_none():
    rows = [{"kv_usage": 0.1, "running": 1, "waiting": 0, "spec_accepted": None, "spec_draft": None,
             "spec_drafts": None, "prefix_hits": None, "prefix_queries": None}]
    summ = summarise(rows)
    assert summ["acceptance_rate_mean"] is None
    assert summ["drafts_delta"] is None
    assert summ["prefix_hit_rate"] is None
