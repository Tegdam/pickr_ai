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
        # T1 (doc §8 vLLM table): served name must equal the model repo id so
        # that under HF_HUB_OFFLINE=1 the /metrics `model_name` label and the
        # client's own `--model` stay in agreement (doc §3.4).
        return cfg.model


def _vllm_args(cfg: RunConfig, mem: float) -> list[str]:
    args = [
        "--model", cfg.model, "--revision", cfg.model_revision,
        "--served-model-name", cfg.model,                                    # T1 (doc §8)
        "--max-model-len", str(cfg.max_model_len), "--max-num-seqs", str(cfg.max_num_seqs),
        # T1: chunked prefill is the V1 model runner's default at v0.29.0 -- only the
        # batched-token budget is passed; --enable-chunked-prefill is not (doc §3.1, §8).
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
        # T1: repo-id "model" + "revision" is the launched/verified form on vLLM
        # (doc §8: "the launched form"); the snapshot-dir form is untested there.
        spec = {"method": "draft_model", "model": cfg.draft_model, "num_speculative_tokens": cfg.spec_k}
        if cfg.draft_revision:
            spec["revision"] = cfg.draft_revision
        args += ["--speculative-config", json.dumps(spec)]
    elif cfg.spec_method == "ngram":
        args += ["--speculative-config", json.dumps({
            "method": "ngram", "num_speculative_tokens": cfg.spec_k,
            "prompt_lookup_max": cfg.ngram_lookup_max or 4,                    # T1 (doc §8: prompt_lookup_min copies max when omitted)
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
        "--served-model-name", cfg.model,                                    # T1 (doc §8)
        "--context-length", str(cfg.max_model_len),
        "--max-running-requests", str(cfg.max_num_seqs),
        "--chunked-prefill-size", str(cfg.chunked_prefill_tokens),            # T1 P11
        "--max-prefill-tokens", str(cfg.chunked_prefill_tokens),              # T1 P11 (vLLM's --max-num-batched-tokens counterpart)
        "--mem-fraction-static", f"{mem:.2f}",
        "--random-seed", str(cfg.seed), "--port", "30000", "--host", "0.0.0.0",
        # T1: the real flag at v0.5.20 is --cuda-graph-bs-decode; --cuda-graph-bs /
        # --cuda-graph-max-bs do not exist (doc §4.1: "the brief's ... do not exist").
        "--cuda-graph-bs-decode", *[str(b) for b in cfg.cudagraph_capture_sizes],
        "--sampling-defaults", "openai",                                     # T1 P9
        "--enable-metrics",                                                  # T1: required for /metrics to exist (doc §4.1, §8)
    ]
    if cfg.quantization:
        args += ["--quantization", _SGLANG_QUANT_MAP.get(cfg.quantization, cfg.quantization)]  # T1 P4/P10
    if not cfg.prefix_caching:
        args += ["--disable-radix-cache"]                                    # T1 (doc §8: on by default otherwise)
    if cfg.spec_method == "draft":
        args += [
            "--speculative-algorithm", "STANDALONE",                         # T1 (doc §8)
            "--speculative-draft-model-path", _hf_snapshot_dir(cfg.draft_model, cfg.draft_revision),  # T1 (doc §4.2)
            "--speculative-draft-model-quantization", "unquant",             # T1 (doc §4.2: draft otherwise inherits target's awq and fails)
            "--speculative-num-steps", str(cfg.spec_k),
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", str(cfg.spec_k + 1),
        ]
    elif cfg.spec_method == "ngram":
        args += [
            "--speculative-algorithm", "NGRAM",                              # T1 (doc §8)
            "--speculative-num-steps", str(cfg.spec_k),
            "--speculative-num-draft-tokens", str(cfg.spec_k + 1),
            "--speculative-ngram-max-bfs-breadth", "1",                      # T1 P15: linear draft, matches vLLM's ngram
            "--speculative-ngram-max-trie-depth", str(cfg.ngram_lookup_max or 4),  # T1 P15: mirrors vLLM's prompt_lookup_max
        ]
    elif cfg.spec_method != "off":
        raise ValueError(f"unsupported spec_method {cfg.spec_method!r} for sglang; see p0b-engine-verification.md")
    return args


ENGINES: dict[str, EngineSpec] = {
    "vllm": EngineSpec(
        name="vllm", image="vllm/vllm-openai:v0.29.0", verified_against="vllm/vllm-openai:v0.29.0",  # T1 (doc §1, §8)
        port=8000,
        health_path="/health",                                               # T1 (doc §3.3)
        reset_cache_path="/reset_prefix_cache", reset_cache_method="POST",    # T1 (doc §3.3, §8: 405 on GET, 404 without dev mode)
        metrics_path="/metrics",                                             # T1 (doc §3.3)
        metric_names={  # T1 -- exact names verified live on /metrics at the pin (doc §3.4, §8)
            "kv_usage": "vllm:kv_cache_usage_perc",                          # T1: NOT vllm:gpu_cache_usage_perc (doc §3.4: "does not exist at 0.29.0")
            "running": "vllm:num_requests_running",                          # T1
            "waiting": "vllm:num_requests_waiting",                          # T1
            "spec_accepted": "vllm:spec_decode_num_accepted_tokens_total",   # T1
            "spec_draft": "vllm:spec_decode_num_draft_tokens_total",         # T1
            "spec_emitted": "vllm:spec_decode_num_drafts_total",             # T1: no "emitted" counter exists (doc §3.4); drafts_total is the closest cumulative counterpart (emitted = accepted + drafts, doc §8)
            "prefix_hits": "vllm:prefix_cache_hits_total",                   # T1
            "prefix_queries": "vllm:prefix_cache_queries_total",             # T1
        },
        env={
            "HF_HUB_OFFLINE": "1",                                           # T1 (doc §8 "Common to every container")
            "VLLM_WSL2_ENABLE_PIN_MEMORY": "1",                              # T1 (doc §3.5, §8 "required env": WSL2 pinned-memory gate, else "UVA is not available")
            "VLLM_USE_V2_MODEL_RUNNER": "0",                                 # T1 P3 (doc §8: all vLLM arms share the V1 runner)
            "VLLM_SERVER_DEV_MODE": "1",                                     # T1 P6 (doc §8: mounts /reset_prefix_cache)
        },
        docker_extra_args=["--shm-size", "2g"],                              # controller decision (doc §8 intro): both engines get it
        _launch=_vllm_args,
    ),
    "sglang": EngineSpec(
        name="sglang", image="lmsysorg/sglang:v0.5.20-runtime", verified_against="lmsysorg/sglang:v0.5.20-runtime",  # T1 (doc §1, §8)
        port=30000,
        health_path="/health",                                               # T1 (doc §4.3, §8: no generation without SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION)
        reset_cache_path="/flush_cache", reset_cache_method="POST",           # T1 (doc §4.3, §8: GET or POST both 200; body must be checked, not status)
        metrics_path="/metrics",                                             # T1 (doc §4.3: added only with --enable-metrics)
        metric_names={  # T1 -- doc §4.4/§6.3/§8; SGLang has no exact vLLM-equivalent for every
            # key, so the closest available metric is mapped in with a comment naming the proxy.
            "kv_usage": "sglang:token_usage",                                 # T1: KV occupancy as a fraction, same shape as vllm:kv_cache_usage_perc
            "running": "sglang:num_running_reqs",                            # T1
            "waiting": "sglang:num_queue_reqs",                              # T1
            "spec_accepted": "sglang:spec_accept_length",                    # PROXY: no cumulative accepted-tokens counter exists on SGLang (doc §4.4: "spec_accept_* are most-recent-interval gauges"); mean acceptance length is the closest available signal
            "spec_draft": "sglang:spec_verify_calls_total",                  # PROXY: the only cumulative spec counter on SGLang (doc §4.4); one verify call corresponds to one draft round, so it stands in for cumulative draft volume
            "spec_emitted": "sglang:spec_accept_rate",                       # PROXY: neither engine has a true "emitted" counter (doc §3.4); the interval acceptance-rate gauge is the closest rate-based analogue
            "prefix_hits": "sglang:cached_tokens_total",                     # T1: label cache_source="device" selects device hits (doc §6.3); pair with prefix_queries below
            "prefix_queries": "sglang:prompt_tokens_total",                  # T1: doc-stated proxy for queries (doc §4.4); slightly overcounts vs vLLM's true counter -- compare hit ratios only, per doc caveat
        },
        env={
            "HF_HUB_OFFLINE": "1",                                           # T1 (doc §8 "Common to every container")
        },
        docker_extra_args=["--shm-size", "2g"],                              # T1 (doc §8: used in all SGLang smokes)
        _launch=_sglang_args,
    ),
}
