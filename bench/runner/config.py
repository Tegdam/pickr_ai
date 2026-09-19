"""One run's fully resolved configuration -- every field lands in results/<run>/config.yaml."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPEC_METHODS = ("off", "ngram", "draft")

# P9 (doc §3.6, §4.5): every arm strips the model's own generation_config.json
# (vLLM: --generation-config vllm; SGLang: --sampling-defaults openai), so the
# client is the *only* source of sampling parameters and must always send all four.
REQUIRED_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty")


@dataclass
class RunConfig:
    run_id: str
    sweep_id: str
    phase: str
    rq_tag: str
    engine: str
    image: str
    model: str
    model_revision: str
    quantization: str | None
    draft_model: str | None
    draft_revision: str | None
    draft_quantization: str | None
    spec_method: str
    spec_k: int | None
    ngram_lookup_max: int | None
    workload: str
    trace_file: str
    trace_version: int
    trace_sha256: str
    load_mode: str
    concurrency: int | None
    request_rate: float | None
    burstiness: float
    num_prompts: int
    cache_state: str
    prefix_caching: bool
    max_model_len: int
    max_num_seqs: int
    chunked_prefill_tokens: int
    cudagraph_capture_sizes: list[int]
    sampling: dict
    ignore_eos: bool
    max_tokens_cap: int | None
    warmup_requests: int
    cooldown_temp_c: int
    cooldown_min_s: int
    gpu_memory_utilization: float
    free_vram_mb_at_start: int
    seed: int
    extra_body: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_gpu_memory_fraction(total_mb: int, host_used_mb: int, headroom_mb: int = 256) -> float:
    """Fraction of TOTAL memory the engine may claim, derived from what is
    actually free (spec §3.1: never 0.9 by reflex on a shared GPU)."""
    usable = max(0, total_mb - host_used_mb - headroom_mb)
    return (usable * 100 // total_mb) / 100


def validate(cfg: RunConfig) -> None:
    from .engine import ENGINES  # local import: engine.py imports RunConfig for typing

    if cfg.engine not in ENGINES:
        raise ValueError(f"unknown engine {cfg.engine!r}")
    if not ENGINES[cfg.engine].verified_against:
        raise ValueError(f"engine table for {cfg.engine} is unverified; run Task 1 first")
    if cfg.spec_method not in SPEC_METHODS:
        raise ValueError(f"spec_method must be one of {SPEC_METHODS}")
    if cfg.spec_method != "off" and not cfg.spec_k:
        raise ValueError("spec_k is required when speculation is on")
    if cfg.spec_method == "draft" and not cfg.draft_model:
        raise ValueError("draft_model is required for spec_method=draft")
    if cfg.spec_method == "draft" and not cfg.draft_revision:
        raise ValueError("draft_revision is required for spec_method=draft")
    if cfg.spec_method == "ngram" and not cfg.ngram_lookup_max:
        raise ValueError("ngram_lookup_max is required for spec_method=ngram")
    if not cfg.cudagraph_capture_sizes:
        raise ValueError("cudagraph_capture_sizes must be explicit (enforce-eager is prohibited)")
    if cfg.load_mode == "concurrency" and not cfg.concurrency:
        raise ValueError("concurrency required for load_mode=concurrency")
    if cfg.load_mode == "poisson" and not cfg.request_rate:
        raise ValueError("request_rate required for load_mode=poisson")
    missing_sampling = [k for k in REQUIRED_SAMPLING_KEYS if k not in cfg.sampling]
    if missing_sampling:
        raise ValueError(
            f"sampling config missing required keys {missing_sampling} "
            "(P9: the client is the sole source of sampling params, so all four must always be sent)"
        )
    p = Path(cfg.trace_file)
    if not p.exists():
        raise ValueError(f"trace file missing: {p}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != cfg.trace_sha256:
        raise ValueError(f"trace sha256 mismatch for {p}: config {cfg.trace_sha256[:12]} vs file {actual[:12]}")
