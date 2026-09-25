import pytest

from bench.runner.engine import ENGINES, EngineSpec
from bench.runner.config import RunConfig


def _cfg(**over):
    base = dict(run_id="r1", sweep_id="s1", phase="p0b", rq_tag="cal", engine="vllm", image="vllm/vllm-openai:v0.29.0",
                model="Qwen/Qwen2.5-3B-Instruct-AWQ", model_revision="abc", quantization="awq",
                draft_model=None, draft_revision=None, draft_quantization=None, spec_method="off", spec_k=None,
                ngram_lookup_max=None, workload="A", trace_file="bench/traces/chat_v1.jsonl", trace_version=1,
                trace_sha256="0" * 64, load_mode="concurrency", concurrency=8, request_rate=None, burstiness=1.0,
                num_prompts=200, cache_state="cold", prefix_caching=True, max_model_len=2048, max_num_seqs=64,
                chunked_prefill_tokens=1024, cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
                sampling={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0},
                ignore_eos=True, max_tokens_cap=None, warmup_requests=8, cooldown_temp_c=55, cooldown_min_s=60,
                gpu_memory_utilization=0.85, free_vram_mb_at_start=6000, seed=0, extra_body={})
    base.update(over)
    return RunConfig(**base)


def test_engines_are_verified_and_complete():
    for name, e in ENGINES.items():
        assert isinstance(e, EngineSpec) and e.name == name
        assert e.verified_against, f"{name} table must carry the image tag it was verified against (Task 1)"
        for k in ("kv_usage", "running", "waiting", "spec_accepted", "spec_draft", "spec_drafts", "prefix_hits", "prefix_queries"):
            assert k in e.metric_names, (name, k)
        assert e.health_path.startswith("/") and e.reset_cache_path.startswith("/") and e.metrics_path == "/metrics"
        assert e.mem_headroom_mb >= 0 and 0 < e.max_mem_fraction <= 1.0


def test_per_engine_mem_headroom_and_cap():
    """C2/P29: SGLang needs materially more headroom than vLLM (doc §7 smoke
    footprints), and each engine's fraction is capped until an OOM probe says
    otherwise; the echo engine has no GPU engine in the loop at all."""
    assert ENGINES["vllm"].mem_headroom_mb == 1024 and ENGINES["vllm"].max_mem_fraction == 0.80
    assert ENGINES["sglang"].mem_headroom_mb == 1280 and ENGINES["sglang"].max_mem_fraction == 0.80
    assert ENGINES["echo"].mem_headroom_mb == 0 and ENGINES["echo"].max_mem_fraction == 1.0


def test_readiness_route_and_timeout():
    """Fix round 1, item 3: SGLang's /ready route and both engines' 900s
    readiness budget (ruling P4; doc §4.3, §6.3) must be represented on EngineSpec."""
    assert ENGINES["vllm"].ready_path is None
    assert ENGINES["sglang"].ready_path == "/ready"
    assert ENGINES["vllm"].readiness_timeout_s == 900
    assert ENGINES["sglang"].readiness_timeout_s == 900


def test_vllm_launch_args_off_and_draft():
    e = ENGINES["vllm"]
    args = e.build_launch_args(_cfg(), mem_fraction=0.83)
    joined = " ".join(args)
    assert "--model Qwen/Qwen2.5-3B-Instruct-AWQ" in joined and "--revision abc" in joined
    assert "--gpu-memory-utilization 0.83" in joined and "--max-model-len 2048" in joined
    assert "--enforce-eager" not in joined and "--seed 0" in joined
    assert "speculative" not in joined
    args = e.build_launch_args(_cfg(spec_method="draft", draft_model="Qwen/Qwen2.5-0.5B-Instruct", draft_revision="def", spec_k=3), mem_fraction=0.83)
    joined = " ".join(args)
    assert "draft_model" in joined and "Qwen2.5-0.5B-Instruct" in joined and '"num_speculative_tokens": 3' in joined.replace("'", '"')


def test_vllm_prefix_caching_off_and_ngram():
    e = ENGINES["vllm"]
    joined = " ".join(e.build_launch_args(_cfg(prefix_caching=False, spec_method="ngram", spec_k=5, ngram_lookup_max=4), 0.8))
    assert "prefix-caching" in joined          # exact flag spelled per Task 1 doc; test only checks presence
    assert "ngram" in joined and "prompt_lookup_max" in joined.replace("-", "_")


def test_sglang_launch_args_standalone():
    e = ENGINES["sglang"]
    joined = " ".join(e.build_launch_args(_cfg(engine="sglang", spec_method="draft", draft_model="Qwen/Qwen2.5-0.5B-Instruct", draft_revision="def", spec_k=3), 0.8))
    assert "--model-path Qwen/Qwen2.5-3B-Instruct-AWQ" in joined and "--mem-fraction-static 0.8" in joined
    assert "STANDALONE" in joined and "Qwen2.5-0.5B-Instruct" in joined
    assert "--context-length 2048" in joined


def test_sglang_awq_maps_to_awq_marlin():
    """T1 P4/P10: --quantization awq_marlin is forced regardless of cfg.quantization=='awq'
    (doc §6.3: plain awq forces the unoptimised kernel, 5x slower decode; §8 SGLang table)."""
    e = ENGINES["sglang"]
    joined = " ".join(e.build_launch_args(_cfg(engine="sglang", quantization="awq"), 0.8))
    assert "--quantization awq_marlin" in joined
    assert "--quantization awq " not in joined and not joined.endswith("--quantization awq")


def test_sglang_draft_requires_draft_model():
    """Fix round 1, item 2: a missing draft_model must raise ValueError, not
    AttributeError, out of hf_snapshot_dir's cfg.draft_model.replace(...)."""
    e = ENGINES["sglang"]
    with pytest.raises(ValueError, match="draft_model"):
        e.build_launch_args(_cfg(engine="sglang", spec_method="draft", draft_revision="def", spec_k=3), 0.8)


def test_unknown_spec_method_raises():
    with pytest.raises(ValueError):
        ENGINES["vllm"].build_launch_args(_cfg(spec_method="eagle"), 0.8)


def test_vllm_exact_args_off():
    """Exact-args regression guard (fix round 1): the argument spellings pinned
    here are the main regression guard for the whole study."""
    e = ENGINES["vllm"]
    args = e.build_launch_args(_cfg(), mem_fraction=0.83)
    assert args == [
        "--model", "Qwen/Qwen2.5-3B-Instruct-AWQ", "--revision", "abc",
        "--served-model-name", "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "--max-model-len", "2048", "--max-num-seqs", "64",
        "--enable-chunked-prefill", "--max-num-batched-tokens", "1024",
        "--gpu-memory-utilization", "0.83", "--seed", "0", "--port", "8000",
        "--cudagraph-capture-sizes", "1", "2", "4", "8", "16", "32",
        "--generation-config", "vllm",
        "--quantization", "awq",
        "--enable-prefix-caching",
    ]


def test_vllm_exact_args_draft():
    e = ENGINES["vllm"]
    args = e.build_launch_args(
        _cfg(spec_method="draft", draft_model="Qwen/Qwen2.5-0.5B-Instruct", draft_revision="def", spec_k=3),
        mem_fraction=0.83,
    )
    assert args == [
        "--model", "Qwen/Qwen2.5-3B-Instruct-AWQ", "--revision", "abc",
        "--served-model-name", "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "--max-model-len", "2048", "--max-num-seqs", "64",
        "--enable-chunked-prefill", "--max-num-batched-tokens", "1024",
        "--gpu-memory-utilization", "0.83", "--seed", "0", "--port", "8000",
        "--cudagraph-capture-sizes", "1", "2", "4", "8", "16", "32",
        "--generation-config", "vllm",
        "--quantization", "awq",
        "--enable-prefix-caching",
        "--speculative-config",
        '{"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 3, "revision": "def"}',
    ]


def test_vllm_exact_args_ngram():
    e = ENGINES["vllm"]
    args = e.build_launch_args(_cfg(spec_method="ngram", spec_k=5, ngram_lookup_max=4), mem_fraction=0.83)
    assert args == [
        "--model", "Qwen/Qwen2.5-3B-Instruct-AWQ", "--revision", "abc",
        "--served-model-name", "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "--max-model-len", "2048", "--max-num-seqs", "64",
        "--enable-chunked-prefill", "--max-num-batched-tokens", "1024",
        "--gpu-memory-utilization", "0.83", "--seed", "0", "--port", "8000",
        "--cudagraph-capture-sizes", "1", "2", "4", "8", "16", "32",
        "--generation-config", "vllm",
        "--quantization", "awq",
        "--enable-prefix-caching",
        "--speculative-config",
        '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}',
    ]


def test_sglang_exact_args_off():
    e = ENGINES["sglang"]
    args = e.build_launch_args(_cfg(engine="sglang"), mem_fraction=0.8)
    assert args == [
        "python", "-m", "sglang.launch_server",
        "--model-path", "Qwen/Qwen2.5-3B-Instruct-AWQ", "--revision", "abc",
        "--served-model-name", "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "--context-length", "2048", "--max-running-requests", "64",
        "--chunked-prefill-size", "1024", "--max-prefill-tokens", "1024",
        "--mem-fraction-static", "0.80",
        "--random-seed", "0", "--port", "30000", "--host", "0.0.0.0",
        "--cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32",
        # P40: prefill graphs pinned to the same sizes as decode -- SGLang's default
        # captures 42 of them for 0.47 GB; pinning leaves 0.08 GB. (KV is unchanged
        # either way: SGLang sizes its static pool before capture. See engine.py.)
        "--cuda-graph-bs-prefill", "1", "2", "4", "8", "16", "32",
        "--sampling-defaults", "openai",
        "--enable-metrics", "--enable-cache-report",
        "--quantization", "awq_marlin",
    ]


def test_sglang_exact_args_standalone():
    e = ENGINES["sglang"]
    args = e.build_launch_args(
        _cfg(engine="sglang", spec_method="draft", draft_model="Qwen/Qwen2.5-0.5B-Instruct", draft_revision="def", spec_k=3),
        mem_fraction=0.8,
    )
    assert args == [
        "python", "-m", "sglang.launch_server",
        "--model-path", "Qwen/Qwen2.5-3B-Instruct-AWQ", "--revision", "abc",
        "--served-model-name", "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "--context-length", "2048", "--max-running-requests", "64",
        "--chunked-prefill-size", "1024", "--max-prefill-tokens", "1024",
        "--mem-fraction-static", "0.80",
        "--random-seed", "0", "--port", "30000", "--host", "0.0.0.0",
        "--cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32",
        # P40: prefill graphs pinned to the same sizes as decode -- SGLang's default
        # captures 42 of them for 0.47 GB; pinning leaves 0.08 GB. (KV is unchanged
        # either way: SGLang sizes its static pool before capture. See engine.py.)
        "--cuda-graph-bs-prefill", "1", "2", "4", "8", "16", "32",
        "--sampling-defaults", "openai",
        "--enable-metrics", "--enable-cache-report",
        "--quantization", "awq_marlin",
        "--speculative-algorithm", "STANDALONE",
        "--speculative-draft-model-path",
        "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/def",
        "--speculative-draft-model-quantization", "unquant",
        "--speculative-num-steps", "3",
        "--speculative-eagle-topk", "1",
        "--speculative-num-draft-tokens", "4",
    ]


def test_echo_engine_is_a_local_subprocess_with_a_complete_spec():
    """Task 9: the echo engine carries image=None (a local `python -m
    bench.echo_server` subprocess, not a docker image) but is otherwise a
    normal, complete EngineSpec -- test_engines_are_verified_and_complete
    already asserts this for every engine in the table; this locks down the
    echo-specific values the brief calls out."""
    e = ENGINES["echo"]
    assert e.image is None
    assert e.verified_against == "local"
    assert e.port == 8000
    assert e.health_path == "/health" and e.ready_path is None
    assert e.reset_cache_path == "/reset_prefix_cache" and e.reset_cache_method == "POST"
    assert e.metrics_path == "/metrics"
    assert e.env == {} and e.docker_extra_args == []
    assert set(e.metric_names.values()) == {"echo:requests_total"}
    assert e.served_model_name(_cfg(engine="echo")) == "Qwen/Qwen2.5-3B-Instruct-AWQ"


def test_echo_launch_args():
    e = ENGINES["echo"]
    assert e.build_launch_args(_cfg(engine="echo"), mem_fraction=0.83) == [
        "--port", "8000", "--per-token-ms", "5",
    ]


def test_sglang_exact_args_ngram():
    e = ENGINES["sglang"]
    args = e.build_launch_args(_cfg(engine="sglang", spec_method="ngram", spec_k=5, ngram_lookup_max=4), mem_fraction=0.8)
    assert args == [
        "python", "-m", "sglang.launch_server",
        "--model-path", "Qwen/Qwen2.5-3B-Instruct-AWQ", "--revision", "abc",
        "--served-model-name", "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "--context-length", "2048", "--max-running-requests", "64",
        "--chunked-prefill-size", "1024", "--max-prefill-tokens", "1024",
        "--mem-fraction-static", "0.80",
        "--random-seed", "0", "--port", "30000", "--host", "0.0.0.0",
        "--cuda-graph-bs-decode", "1", "2", "4", "8", "16", "32",
        # P40: prefill graphs pinned to the same sizes as decode -- SGLang's default
        # captures 42 of them for 0.47 GB; pinning leaves 0.08 GB. (KV is unchanged
        # either way: SGLang sizes its static pool before capture. See engine.py.)
        "--cuda-graph-bs-prefill", "1", "2", "4", "8", "16", "32",
        "--sampling-defaults", "openai",
        "--enable-metrics", "--enable-cache-report",
        "--quantization", "awq_marlin",
        "--speculative-algorithm", "NGRAM",
        "--speculative-num-steps", "5",
        "--speculative-num-draft-tokens", "6",
        "--speculative-ngram-max-bfs-breadth", "1",
        "--speculative-ngram-max-trie-depth", "4",
    ]
