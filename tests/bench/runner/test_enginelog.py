"""P37/P38: the engine's own startup log is the authoritative memory source on
this platform. These parsers run against the verbatim lines two real vLLM
v0.29.0 containers printed on 2026-09-25 -- one with a warm torch.compile
cache, one cold -- because the difference between them is what makes a cold
first run an outlier in a sweep of otherwise identical runs."""
from bench.runner.enginelog import parse_graph_gib, parse_kv_tokens, parse_memory_breakdown

WARM = """
(EngineCore pid=114) INFO 09-25 02:25:45 [gpu_model_runner.py:5471] Model loading took 1.95 GiB memory and 2.566028 seconds
(EngineCore pid=114) INFO 09-25 02:25:49 [gpu_worker.py:625] Available KV cache memory: 2.41 GiB
(EngineCore pid=114) INFO 09-25 02:25:49 [kv_cache_utils.py:2032] GPU KV cache size: 70,240 tokens, Maximum concurrency for 2,048 tokens per request: 34.30x
(EngineCore pid=114) INFO 09-25 02:25:51 [gpu_model_runner.py:7001] Graph capturing finished in 1 secs, took 0.07 GiB
(EngineCore pid=114) INFO 09-25 02:25:51 [gpu_worker.py:860] Free memory on device (4.95/6.0 GiB) on startup. Desired GPU memory utilization is (0.79, 4.74 GiB). Actual usage is 2.1 GiB for consumed memory (weights + non-torch), 0.22 GiB for peak activation, and 0.07 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=2358995846` (2.2 GiB) to fit into requested memory, or `--kv-cache-memory=2590736384` (2.41 GiB) to fully utilize gpu memory. Current kv cache memory in use is 2.41 GiB.
(EngineCore pid=114) INFO 09-25 02:25:53 [core.py:361] init engine (profile, create kv cache, warmup model) took 7.32 s (compilation: 0.84 s)
"""

COLD = """
(EngineCore pid=114) INFO 09-25 01:58:00 [gpu_worker.py:625] Available KV cache memory: 1.5 GiB
(EngineCore pid=114) INFO 09-25 01:58:00 [kv_cache_utils.py:2032] GPU KV cache size: 43,664 tokens, Maximum concurrency for 2,048 tokens per request: 21.32x
(EngineCore pid=114) INFO 09-25 01:58:01 [gpu_model_runner.py:7001] Graph capturing finished in 1 secs, took 0.07 GiB
(EngineCore pid=114) INFO 09-25 01:58:01 [gpu_worker.py:860] Free memory on device (4.95/6.0 GiB) on startup. Desired GPU memory utilization is (0.8, 4.8 GiB). Actual usage is 2.63 GiB for consumed memory (weights + non-torch), 0.67 GiB for peak activation, and 0.07 GiB for CUDAGraph memory. Replace gpu_memory_utilization config with `--kv-cache-memory=1378989671` (1.28 GiB) to fit into requested memory, or `--kv-cache-memory=1546342400` (1.44 GiB) to fully utilize gpu memory. Current kv cache memory in use is 1.5 GiB.
"""


def test_parses_the_warm_cache_breakdown():
    b = parse_memory_breakdown(WARM)
    assert b["cuda_free_gib_at_startup"] == 4.95 and b["cuda_total_gib"] == 6.0
    assert b["requested_fraction"] == 0.79 and b["requested_gib"] == 4.74
    assert b["weights_plus_non_torch_gib"] == 2.1 and b["peak_activation_gib"] == 0.22
    assert b["cudagraph_gib"] == 0.07 and b["kv_cache_gib"] == 2.41
    assert b["kv_cache_tokens"] == 70240
    assert b["model_load_gib"] == 1.95 and b["compilation_s"] == 0.84
    assert b["max_concurrency_at_tokens"] == 2048 and b["max_concurrency_x"] == 34.30


def test_parses_the_cold_cache_breakdown_and_shows_the_kv_penalty():
    warm, cold = parse_memory_breakdown(WARM), parse_memory_breakdown(COLD)
    assert cold["weights_plus_non_torch_gib"] == 2.63 and cold["peak_activation_gib"] == 0.67
    assert cold["kv_cache_tokens"] == 43664
    # The whole reason this module exists: a cold cache costs ~26k KV tokens.
    assert warm["kv_cache_tokens"] - cold["kv_cache_tokens"] > 20000
    assert cold["compilation_s"] is None            # the cold log did not print it


def test_scalar_parsers_and_missing_lines():
    assert parse_graph_gib(WARM) == 0.07 and parse_kv_tokens(COLD) == 43664
    empty = parse_memory_breakdown("no engine lines here")
    assert set(empty) == set(parse_memory_breakdown(WARM))     # same shape, all None
    assert all(v is None for v in empty.values())
