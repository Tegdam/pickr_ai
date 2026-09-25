"""Parsers for the numbers only the engine itself reports.

P37/P38 (measured 2026-09-25): on this platform `nvidia-smi` cannot supply the
memory facts the study needs -- both views report the same device-wide figure
and neither sees the ~1.05 GiB WSL2/WDDM reservation that CUDA does. The
engine's own startup log is therefore the authoritative source for the RQ4
memory table, and it is also the only place the compile-cache effect shows up:
with a cold torch.compile cache vLLM profiles 2.63 GiB consumed + 0.67 GiB
peak activation and sizes KV at 1.5 GiB (43,664 tokens); with the cache warm
the same config profiles 2.1 GiB + 0.22 GiB and sizes KV at 2.41 GiB (70,240
tokens). A cold first run in a sweep of "identical" runs would therefore be an
outlier, which is why `lifecycle` records `compile_cache_warm` alongside these.
"""
from __future__ import annotations

import re

# vLLM v0.29.0 (verified against real container logs, see bench/docs/p0b-engine-verification.md)
_GRAPH_GIB = re.compile(r"Graph capturing finished in \d+ secs, took ([0-9.]+) GiB", re.I)
_KV_TOKENS = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", re.I)
_KV_GIB = re.compile(r"Available KV cache memory:\s*([0-9.]+) GiB", re.I)
_MAX_CONC = re.compile(r"Maximum concurrency for ([0-9,]+) tokens per request:\s*([0-9.]+)x", re.I)
_BREAKDOWN = re.compile(
    r"Free memory on device \(([0-9.]+)/([0-9.]+) GiB\) on startup\. "
    r"Desired GPU memory utilization is \(([0-9.]+), ([0-9.]+) GiB\)\. "
    r"Actual usage is ([0-9.]+) GiB for consumed memory \(weights \+ non-torch\), "
    r"([0-9.]+) GiB for peak activation, and ([0-9.]+) GiB for CUDAGraph memory",
    re.I,
)
_MODEL_LOAD = re.compile(r"Model loading took ([0-9.]+) GiB memory and ([0-9.]+) seconds", re.I)
_COMPILATION_S = re.compile(r"compilation:\s*([0-9.]+) s", re.I)


def _f(m, i=1):
    return float(m.group(i)) if m else None


def parse_graph_gib(logs: str) -> float | None:
    return _f(_GRAPH_GIB.search(logs))


def parse_kv_tokens(logs: str) -> int | None:
    m = _KV_TOKENS.search(logs)
    return int(m.group(1).replace(",", "")) if m else None


def parse_memory_breakdown(logs: str) -> dict:
    """Everything the engine says about where the 6 GB went. Keys are None when
    the engine did not print them (a failed launch, or another engine)."""
    b = _BREAKDOWN.search(logs)
    c = _MAX_CONC.search(logs)
    out = {
        "cuda_free_gib_at_startup": _f(b, 1),
        "cuda_total_gib": _f(b, 2),
        "requested_fraction": _f(b, 3),
        "requested_gib": _f(b, 4),
        "weights_plus_non_torch_gib": _f(b, 5),
        "peak_activation_gib": _f(b, 6),
        "cudagraph_gib": _f(b, 7),
        "kv_cache_gib": _f(_KV_GIB.search(logs)),
        "kv_cache_tokens": parse_kv_tokens(logs),
        "model_load_gib": _f(_MODEL_LOAD.search(logs), 1),
        "compilation_s": _f(_COMPILATION_S.search(logs)),
        "max_concurrency_at_tokens": int(c.group(1).replace(",", "")) if c else None,
        "max_concurrency_x": float(c.group(2)) if c else None,
    }
    return out
