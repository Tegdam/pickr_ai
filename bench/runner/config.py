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
    image: str | None  # Task 9: None for the echo engine (a local subprocess, no docker image)
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
    # Unused until P3 (natural termination) -- the client always passes
    # per-row output_tokens (--custom-output-len -1), so nothing in P0b reads
    # this yet.
    max_tokens_cap: int | None
    warmup_requests: int
    cooldown_temp_c: int
    cooldown_min_s: int
    gpu_memory_utilization: float
    free_vram_mb_at_start: int
    seed: int
    extra_body: dict = field(default_factory=dict)
    # C2/P29: gpu_headroom_mb (sweep-level) + spec.mem_headroom_mb
    # (per-engine) -- recorded next to gpu_memory_utilization so config.yaml
    # shows the full headroom actually applied, not just the sweep's own
    # knob. Default 0 so callers/tests that never set it (pre-C2) still
    # round-trip.
    mem_headroom_mb_total: int = 0
    # Windows 11 power-mode overlay to force before the engine launches, or
    # None to leave whatever the machine is already in (every sweep before
    # 2026-09-27). Making it a RunConfig field is what lets it be a sweep
    # AXIS, so the existing seeded shuffle interleaves the arms instead of
    # running one power mode's runs after the other's -- the confound that
    # made the first Balanced-vs-Performance comparison uninterpretable.
    # Lifecycle applies it in pre-flight and fails the run if it did not take
    # effect (power_overlay.py).
    power_overlay: str | None = None
    # Equal-KV comparison (P1 decision, 2026-09-28). At equal memory-fraction
    # flags the engines do NOT get equal KV -- vLLM 70,240 tokens vs SGLang
    # 48,154 -- because SGLang allocates its static pool and KV before CUDA-graph
    # capture while vLLM sizes KV from what remains after. Worse, ~40% of that gap
    # is our own per-engine headroom rather than the engines. Setting this pins
    # the KV pool to the SAME number of token slots on both, so RQ1's knee
    # compares engines instead of our configuration.
    #
    # Expressed in TOKENS, not bytes, because tokens are the unit that drives
    # concurrency and the unit both engines report (vLLM `kv_cache_tokens`,
    # SGLang `max_total_num_tokens`). The engine builders convert: vLLM takes
    # `--kv-cache-memory-bytes`, SGLang takes `--max-total-tokens` directly.
    # Both page KV at 36 KiB/token for this model, measured independently from
    # each engine's own reporting (writeup §7), so the conversion is exact.
    # None keeps each engine's own fraction-derived sizing.
    kv_cache_tokens: int | None = None

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
    if cfg.kv_cache_tokens is not None:
        if cfg.kv_cache_tokens <= 0:
            raise ValueError(f"kv_cache_tokens must be positive, got {cfg.kv_cache_tokens}")
        # A pool smaller than one full-length request cannot serve that request at
        # all; a pool smaller than max_model_len x max_num_seqs merely queues,
        # which is legitimate (and is the case for the equal-KV pin, since the
        # realistic worst case is far below the synthetic one -- writeup §7).
        if cfg.kv_cache_tokens < cfg.max_model_len:
            raise ValueError(
                f"kv_cache_tokens {cfg.kv_cache_tokens} is below max_model_len "
                f"{cfg.max_model_len}: no single full-length request could be served"
            )
        if cfg.spec_method != "off":
            # The draft model adds 12,288 B/token of its own KV, and whether
            # vLLM's --kv-cache-memory-bytes covers the draft pool as well as the
            # target's is not verified at our pin. Converting tokens to bytes here
            # would be guessing, and an equal-KV claim built on a guess is worse
            # than an unequal-KV comparison stated honestly.
            raise ValueError(
                "kv_cache_tokens is only supported with spec_method='off' until the draft "
                "model's KV accounting under --kv-cache-memory-bytes is verified (engine.py "
                "KV_BYTES_PER_TOKEN_DRAFT); P2 verifies it before the equal-KV spec arms run"
            )
    if cfg.power_overlay is not None:
        from .power_overlay import OVERLAYS  # local import: keeps the Windows shell-out off this path

        if cfg.power_overlay not in OVERLAYS:
            raise ValueError(
                f"unknown power_overlay {cfg.power_overlay!r}; expected one of {sorted(OVERLAYS)} or null"
            )
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
