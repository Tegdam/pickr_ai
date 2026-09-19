"""Per-engine data: image, launch-arg builder, HTTP routes, metric names, env.

Everything marked T1 is copied verbatim (flag spelling, routes, metric names,
env vars, revisions) from bench/docs/p0b-engine-verification.md's Decisions
section (§8) -- the contract Task 1 verified against the pinned images. Do not
edit these values from memory; if a pin or flag ever changes, re-verify first
and update the doc, then this table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .config import RunConfig


@dataclass(frozen=True)
class EngineSpec:
    name: str
    image: str
    verified_against: str | None
    port: int
    health_path: str
    ready_path: str | None
    readiness_timeout_s: int
    reset_cache_path: str
    reset_cache_method: str
    metrics_path: str
    metric_names: dict[str, str]
    env: dict[str, str]
    docker_extra_args: list[str]
    _launch: Callable[[RunConfig, float], list[str]] = field(repr=False)

    def build_launch_args(self, cfg: RunConfig, mem_fraction: float) -> list[str]:
        return self._launch(cfg, mem_fraction)

    def served_model_name(self, cfg: RunConfig) -> str:
        return _served_name(cfg)


def _served_name(cfg: RunConfig) -> str:
    """T1 (doc §8): served name must equal the model repo id so that under
    HF_HUB_OFFLINE=1 the /metrics `model_name` label and the client's own
    `--model` stay in agreement (doc §3.4). Single source of truth used by
    both launch-arg builders and EngineSpec.served_model_name -- the builders
    don't receive the EngineSpec, so this lives at module level rather than
    calling back into the method."""
    return cfg.model


def _vllm_args(cfg: RunConfig, mem: float) -> list[str]:
    args = [
        "--model", cfg.model, "--revision", cfg.model_revision,
        "--served-model-name", _served_name(cfg),                            # T1 (doc §8)
        "--max-model-len", str(cfg.max_model_len), "--max-num-seqs", str(cfg.max_num_seqs),
        # T1 (doc §8 "chunked prefill size" row, spelled exactly): chunked
        # prefill is already the V1 runner's default, so --enable-chunked-prefill
        # is a no-op here, but it is still passed to match the doc's row.
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(cfg.chunked_prefill_tokens),
        "--gpu-memory-utilization", f"{mem:.2f}", "--seed", str(cfg.seed), "--port", "8000",
        # T1: the CUDA-graph capture list is passed via --cudagraph-capture-sizes, not
        # --compilation-config (doc §3.1 flag list; §8 vLLM table row).
        "--cudagraph-capture-sizes", *[str(s) for s in cfg.cudagraph_capture_sizes],
        "--generation-config", "vllm",                                       # T1 P9: no model generation_config.json leakage
    ]
    if cfg.quantization:
        args += ["--quantization", cfg.quantization]                          # T1 P10: literal stays "awq" on vLLM (resolves to Marlin)
    args += ["--enable-prefix-caching"] if cfg.prefix_caching else ["--no-enable-prefix-caching"]  # T1 (doc §3.1, §8)
    if cfg.spec_method == "draft":
        # T1 (doc §8 "spec: draft model" row): repo-id "model" + "revision" is
        # the launched/verified form on vLLM. cfg.draft_quantization is
        # intentionally ignored here: the doc's vLLM speculative-config JSON
        # has no quantization key (only SGLang's CLI needs one, doc §4.2).
        spec = {"method": "draft_model", "model": cfg.draft_model, "num_speculative_tokens": cfg.spec_k}
        if cfg.draft_revision:
            spec["revision"] = cfg.draft_revision
        args += ["--speculative-config", json.dumps(spec)]
    elif cfg.spec_method == "ngram":
        args += ["--speculative-config", json.dumps({
            "method": "ngram", "num_speculative_tokens": cfg.spec_k,
            "prompt_lookup_max": cfg.ngram_lookup_max,                        # T1 (doc §8): no `or 4` fallback -- the recorded config must equal the launched one; validate() enforces this is set
        })]
    elif cfg.spec_method != "off":
        raise ValueError(f"unsupported spec_method {cfg.spec_method!r} for vllm; see p0b-engine-verification.md")
    return args


# T1 P4/P10 (doc §6.3, §8 SGLang table): plain `awq` forces SGLang's unoptimised
# (non-Marlin) kernel -- observed 5x slower decode at identical memory. Every
# SGLang arm must run the Marlin kernel, so `awq` is always rewritten to
# `awq_marlin` regardless of what the run config carries for vLLM parity.
_SGLANG_QUANT_MAP = {"awq": "awq_marlin"}


def _hf_snapshot_dir(repo_id: str, revision: str) -> str:
    """HF hub cache layout inside every container (doc §2, §4.2): SGLang's draft
    model must be resolved as its snapshot directory -- passing the repo id fails
    offline (`get_config` is called without a revision, and the cache has no
    refs/main), so this is the only launched/verified form for the SGLang draft."""
    return f"/root/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}/snapshots/{revision}"


def _sglang_args(cfg: RunConfig, mem: float) -> list[str]:
    args = [
        "python", "-m", "sglang.launch_server",
        "--model-path", cfg.model, "--revision", cfg.model_revision,
        "--served-model-name", _served_name(cfg),                            # T1 (doc §8)
        "--context-length", str(cfg.max_model_len),
        "--max-running-requests", str(cfg.max_num_seqs),
        "--chunked-prefill-size", str(cfg.chunked_prefill_tokens),           # T1 P11
        "--max-prefill-tokens", str(cfg.chunked_prefill_tokens),             # T1 P11 (vLLM's --max-num-batched-tokens counterpart)
        "--mem-fraction-static", f"{mem:.2f}",
        "--random-seed", str(cfg.seed), "--port", "30000", "--host", "0.0.0.0",
        # T1: the real flag at v0.5.20 is --cuda-graph-bs-decode; --cuda-graph-bs /
        # --cuda-graph-max-bs do not exist (doc §4.1: "the brief's ... do not exist").
        "--cuda-graph-bs-decode", *[str(b) for b in cfg.cudagraph_capture_sizes],
        "--sampling-defaults", "openai",                                    # T1 P9
        "--enable-metrics",                                                 # T1: required for /metrics to exist (doc §4.1, §8)
        "--enable-cache-report",                                            # T1 (doc §8 "metrics names" row): per-request cached tokens for the runner's own prefix pass
    ]
    if cfg.quantization:
        args += ["--quantization", _SGLANG_QUANT_MAP.get(cfg.quantization, cfg.quantization)]  # T1 P4/P10
    if not cfg.prefix_caching:
        args += ["--disable-radix-cache"]                                   # T1 (doc §8: on by default otherwise)
    if cfg.spec_method == "draft":
        if not cfg.draft_model:
            raise ValueError("draft_model is required for spec_method=draft (sglang)")
        # doc §4.2: an unset draft quantization must resolve to "unquant" -- the
        # draft otherwise silently inherits the target's `awq` and fails to load.
        draft_quant = cfg.draft_quantization or "unquant"
        args += [
            "--speculative-algorithm", "STANDALONE",                        # T1 (doc §8)
            "--speculative-draft-model-path", _hf_snapshot_dir(cfg.draft_model, cfg.draft_revision),  # T1 (doc §4.2)
            "--speculative-draft-model-quantization", draft_quant,          # T1 (doc §4.2)
            "--speculative-num-steps", str(cfg.spec_k),
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", str(cfg.spec_k + 1),
        ]
    elif cfg.spec_method == "ngram":
        args += [
            "--speculative-algorithm", "NGRAM",                             # T1 (doc §8)
            "--speculative-num-steps", str(cfg.spec_k),
            "--speculative-num-draft-tokens", str(cfg.spec_k + 1),
            "--speculative-ngram-max-bfs-breadth", "1",                     # T1 P15: linear draft, matches vLLM's ngram
            "--speculative-ngram-max-trie-depth", str(cfg.ngram_lookup_max),  # T1 P15: no `or 4` fallback -- must equal the launched config; validate() enforces this is set
        ]
    elif cfg.spec_method != "off":
        raise ValueError(f"unsupported spec_method {cfg.spec_method!r} for sglang; see p0b-engine-verification.md")
    return args


ENGINES: dict[str, EngineSpec] = {
    "vllm": EngineSpec(
        name="vllm", image="vllm/vllm-openai:v0.29.0", verified_against="vllm/vllm-openai:v0.29.0",  # T1 (doc §1, §8)
        port=8000,
        health_path="/health",                                              # T1 (doc §3.3)
        ready_path=None,                                                    # T1: vLLM has no separate readiness route at this pin (doc §3.3 routes list)
        readiness_timeout_s=900,                                            # ruling P4 (doc §6.3, §8): same generous budget applied to both engines
        reset_cache_path="/reset_prefix_cache", reset_cache_method="POST",  # T1 (doc §3.3, §8: 405 on GET, 404 without dev mode)
        metrics_path="/metrics",                                           # T1 (doc §3.3)
        metric_names={  # T1 -- exact names verified live on /metrics at the pin (doc §3.4, §8)
            "kv_usage": "vllm:kv_cache_usage_perc",                         # T1: NOT vllm:gpu_cache_usage_perc (doc §3.4: "does not exist at 0.29.0")
            "running": "vllm:num_requests_running",                        # T1
            "waiting": "vllm:num_requests_waiting",                        # T1
            "spec_accepted": "vllm:spec_decode_num_accepted_tokens_total", # T1
            "spec_draft": "vllm:spec_decode_num_draft_tokens_total",       # T1: cumulative drafted-token count (used for acceptance rate, doc §3.4)
            "spec_drafts": "vllm:spec_decode_num_drafts_total",            # T1: cumulative draft-round count. Renamed from the earlier, semantically-wrong
                                                                            # "spec_emitted" -- there is no true "emitted" counter (doc §3.4); consumers
                                                                            # derive emitted tokens downstream as spec_accepted + spec_drafts (one bonus
                                                                            # token is always emitted per draft round, doc §3.4/§8).
            "prefix_hits": "vllm:prefix_cache_hits_total",                 # T1
            "prefix_queries": "vllm:prefix_cache_queries_total",           # T1
        },
        env={
            "HF_HUB_OFFLINE": "1",                                          # T1 (doc §8 "Common to every container")
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",                             # T1 (doc §3.5, §8 "required env": WSL2 pinned-memory gate, else "UVA is not available")
            "VLLM_USE_V2_MODEL_RUNNER": "0",                                # T1 P3 (doc §8: all vLLM arms share the V1 runner)
            "VLLM_SERVER_DEV_MODE": "1",                                    # T1 P6 (doc §8: mounts /reset_prefix_cache)
        },
        docker_extra_args=["--shm-size", "2g"],                             # controller decision (doc §8 intro): both engines get it
        _launch=_vllm_args,
    ),
    "sglang": EngineSpec(
        name="sglang", image="lmsysorg/sglang:v0.5.20-runtime", verified_against="lmsysorg/sglang:v0.5.20-runtime",  # T1 (doc §1, §8)
        port=30000,
        health_path="/health",                                              # T1 (doc §4.3, §8: no generation without SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION)
        ready_path="/ready",                                                # T1 (doc §4.3): GET /ready -> 200 once tokenizer_manager.is_ready(), else 503
        readiness_timeout_s=900,                                            # T1 ruling P4 (doc §6.3: awq_marlin graph capture observed 288-348s; "readiness timeouts must allow well over 6 min for SGLang")
        reset_cache_path="/flush_cache", reset_cache_method="POST",         # T1 (doc §4.3, §8: GET or POST both 200; body must be checked, not status)
        metrics_path="/metrics",                                          # T1 (doc §4.3: added only with --enable-metrics)
        metric_names={  # T1 -- doc §4.4/§6.3/§8; SGLang has no exact vLLM-equivalent for every
            # key, so the closest available metric is mapped in with a comment naming the proxy.
            "kv_usage": "sglang:token_usage",                               # T1: KV occupancy as a fraction, same shape as vllm:kv_cache_usage_perc
            "running": "sglang:num_running_reqs",                          # T1
            "waiting": "sglang:num_queue_reqs",                            # T1
            "spec_accepted": "sglang:spec_accept_length",                  # PROXY: no cumulative accepted-tokens counter exists on SGLang (doc §4.4: "spec_accept_* are most-recent-interval gauges"); mean acceptance length is the closest available signal
            "spec_draft": "sglang:spec_verify_calls_total",                # PROXY: the only cumulative spec counter on SGLang (doc §4.4). This raw value is
                                                                            # verify-call (round) count; to get drafted-*token* volume (vLLM's spec_draft
                                                                            # meaning) apply the doc §4.2 conversion: num_proposed_drafts = spec_verify_ct
                                                                            # * (speculative_num_draft_tokens - 1).
            "spec_drafts": "sglang:spec_verify_calls_total",               # PROXY (doc §4.2, §4.4): same raw counter as spec_draft above, used here as the
                                                                            # direct round-count analogue of vLLM's spec_decode_num_drafts_total (no
                                                                            # conversion needed for this key -- one verify call is one draft round).
            "prefix_hits": "sglang:cached_tokens_total",                   # T1: label cache_source="device" selects device hits (doc §6.3); pair with prefix_queries below
            "prefix_queries": "sglang:prompt_tokens_total",                # T1: doc-stated proxy for queries (doc §4.4); slightly overcounts vs vLLM's true counter -- compare hit ratios only, per doc caveat
        },
        env={
            "HF_HUB_OFFLINE": "1",                                          # T1 (doc §8 "Common to every container")
        },
        docker_extra_args=["--shm-size", "2g"],                             # T1 (doc §8: used in all SGLang smokes)
        _launch=_sglang_args,
    ),
}
