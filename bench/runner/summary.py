"""Per-request records, the run summary (with the goodput grid), and the
final validity stamp (spec §3.3, §7; controller ruling P7 on Task 7).

TPOT under speculation (ruling P7): a chunk can carry several tokens once
speculation is on, so `mean(itls)` is not tokens/time -- `tpot_ms_derived` is
computed instead from `output_lens[i]` and the wall-clock span, per request:
`tpot_ms_derived = (e2e_ms_derived - ttft_ms_client) / max(output_tokens - 1, 1)`.
The client's own `mean(itls)*1000` is kept alongside as `tpot_ms_client` (its
view, not the reported one) and `itl_is_per_chunk` records whether `itl_ms`
entries are per-token or per-chunk. `summary.json` reports the derived TPOT
everywhere; the client's percentiles are kept too, suffixed `_client`.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

from bench.capture.export import quantiles as _quantiles

from . import metrics_scraper
from .assertions import Thresholds, check as check_validity

# nvidia-smi clocks_throttle_reasons.active bits (doc §7 gpu_samples.jsonl):
# HW Slowdown | SW Thermal Slowdown | HW Thermal Slowdown | HW Power Brake.
THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80

# summary.json "<metric>_p50_client"/"<metric>_p99_client" <- client's own
# "median_<prefix>_ms"/"p99_<prefix>_ms" (client's key for e2e is "e2el").
_CLIENT_METRIC_PREFIX = {"ttft": "ttft", "tpot": "tpot", "itl": "itl", "e2e": "e2el"}


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_requests(client: dict, trace_rows: list[dict], cfg, out_path) -> list[dict]:
    """One record per request, index i <-> trace row i (`--disable-shuffle`,
    doc §5). Writes `out_path` as JSONL and returns the same list of dicts."""
    ttfts = client["ttfts"]
    itls = client["itls"]
    input_lens = client["input_lens"]
    output_lens = client["output_lens"]
    errors = client["errors"]
    generated_texts = client["generated_texts"]
    start_times = client.get("start_times")
    itl_is_per_chunk = cfg.spec_method != "off"

    requests: list[dict] = []
    for i, ttft in enumerate(ttfts):
        trace_row = trace_rows[i]
        itl_list = itls[i]
        ttft_ms_client = ttft * 1000.0
        itl_ms = [x * 1000.0 for x in itl_list]
        tpot_ms_client = (sum(itl_list) / len(itl_list) * 1000.0) if itl_list else None
        e2e_ms_derived = (ttft + sum(itl_list)) * 1000.0
        output_tokens = output_lens[i]
        tpot_ms_derived = (e2e_ms_derived - ttft_ms_client) / max(output_tokens - 1, 1)

        record = {
            "request_id": i,
            "trace_record_id": trace_row.get("record_id"),
            "call_role": trace_row.get("call_role"),
            "workload": trace_row.get("workload"),
            "ttft_ms_client": ttft_ms_client,
            "tpot_ms_client": tpot_ms_client,
            "tpot_ms_derived": tpot_ms_derived,
            "itl_ms": itl_ms,
            "itl_is_per_chunk": itl_is_per_chunk,
            "e2e_ms_derived": e2e_ms_derived,
            "prompt_tokens": input_lens[i],
            "output_tokens": output_tokens,
            "finish_reason": None,
            "schema_valid": None,
            "error": errors[i] or None,
            "generated_text_sha256": _sha256_hex(generated_texts[i]),
        }
        if start_times is not None:
            record["start_time"] = start_times[i]
        requests.append(record)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in requests:
            f.write(json.dumps(r) + "\n")

    return requests


def goodput_grid(
    requests: list[dict],
    ttft_grid_ms: tuple[float, ...] = (250, 500, 750, 1000, 1500, 2000),
    tpot_grid_ms: tuple[float, ...] = (25, 50, 75, 100, 150),
) -> dict[str, float]:
    """Fraction of COMPLETED requests (no error) meeting each (TTFT, TPOT)
    bound, keyed `"ttft<=T&tpot<=P"` (spec §3.3 sensitivity grid)."""
    completed = [r for r in requests if not r.get("error")]
    total = len(completed)
    grid: dict[str, float] = {}
    for t in ttft_grid_ms:
        for p in tpot_grid_ms:
            key = f"ttft<={t}&tpot<={p}"
            if total == 0:
                grid[key] = 0.0
                continue
            hits = sum(1 for r in completed if r["ttft_ms_client"] <= t and r["tpot_ms_derived"] <= p)
            grid[key] = hits / total
    return grid


def _peak(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return max(vals) if vals else None


def _host_share_drift(gpu_rows: list[dict]):
    vals = [r["used_host_mb"] for r in gpu_rows if r.get("used_host_mb") is not None]
    return (max(vals) - min(vals)) if vals else None


def _power_stats(gpu_rows: list[dict]) -> dict:
    vals = [r["power_w"] for r in gpu_rows if r.get("power_w") is not None]
    if not vals:
        return {"p50": None, "p90": None, "max": None}
    q = _quantiles(vals)
    return {"p50": q["p50"], "p90": q["p90"], "max": max(vals)}


def _clock_cv(gpu_rows: list[dict]):
    vals = [r["sm_clock"] for r in gpu_rows if r.get("sm_clock") is not None]
    if len(vals) < 2:
        return None
    mean = statistics.fmean(vals)
    if mean == 0:
        return None
    return statistics.pstdev(vals) / mean


def _throttled(gpu_rows: list[dict]) -> bool:
    for row in gpu_rows:
        reasons = row.get("throttle_reasons")
        if reasons is None:
            continue
        try:
            value = int(reasons, 16)
        except (TypeError, ValueError):
            continue
        if value & THROTTLE_MASK:
            return True
    return False


def build_summary(client: dict, requests: list[dict], gpu_rows: list[dict], metric_rows: list[dict],
                   cfg, timing: dict) -> dict:
    """Computed once by the runner from the run's artifacts (spec §7
    summary.json). `valid`/`invalid_reason` are filled in-place by
    `assertions.check`, using `cfg.spec_method != "off"` for `spec_on` and
    the default `Thresholds`."""
    completed_requests = [r for r in requests if not r.get("error")]

    summary: dict = {
        "completed": client.get("completed"),
        "num_prompts": cfg.num_prompts,
        "output_tok_s": client.get("output_throughput"),
        "req_s": client.get("request_throughput"),
    }

    summary["ttft_ms_client"] = _quantiles([r["ttft_ms_client"] for r in completed_requests])
    summary["tpot_ms_derived"] = _quantiles([r["tpot_ms_derived"] for r in completed_requests])
    summary["e2e_ms_derived"] = _quantiles([r["e2e_ms_derived"] for r in completed_requests])
    flat_itl_ms = [x for r in completed_requests for x in r["itl_ms"]]
    summary["itl_ms"] = _quantiles(flat_itl_ms)
    summary["itl_is_per_chunk"] = cfg.spec_method != "off"

    for key, prefix in _CLIENT_METRIC_PREFIX.items():
        summary[f"{key}_p50_client"] = client.get(f"median_{prefix}_ms")
        summary[f"{key}_p99_client"] = client.get(f"p99_{prefix}_ms")

    engine_metrics = metrics_scraper.summarise(metric_rows)
    summary["engine_metrics"] = engine_metrics
    summary["acceptance_rate_mean"] = engine_metrics.get("acceptance_rate_mean")
    summary["acceptance_source"] = engine_metrics.get("acceptance_source")

    summary["peak_used_ours_mb"] = _peak(gpu_rows, "used_ours_mb")
    summary["peak_used_host_mb"] = _peak(gpu_rows, "used_host_mb")
    summary["host_share_drift_mb"] = _host_share_drift(gpu_rows)
    summary["power_w"] = _power_stats(gpu_rows)
    summary["clock_cv"] = _clock_cv(gpu_rows)
    summary["throttled"] = _throttled(gpu_rows)

    total = len(requests)
    summary["error_rate"] = (sum(1 for r in requests if r.get("error")) / total) if total else 0.0

    grid = goodput_grid(requests)
    summary["goodput_by_threshold"] = grid
    summary["goodput_pre_registered"] = grid.get("ttft<=500&tpot<=50")

    summary["timing"] = dict(timing)

    spec_on = cfg.spec_method != "off"
    valid, invalid_reason = check_validity(cfg, summary, spec_on, Thresholds())
    summary["valid"] = valid
    summary["invalid_reason"] = invalid_reason

    return summary
