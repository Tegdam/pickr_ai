"""Run validity assertions (spec §6 step 7; controller rulings P7/P22 on Task 7).

`throttled` is recorded in `summary.json` but never invalidates a run here --
runs are excluded for it at analysis time, by policy, not by the runner.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Thresholds:
    max_error_rate: float = 0.01
    max_host_drift_mb: int = 256


def check(cfg, summary: dict, spec_on: bool, thresholds: Thresholds) -> tuple[bool, str | None]:
    """Returns `(valid, invalid_reason)`. Rule order matches the controller
    ruling: attempted count, error rate, host-memory drift, then (when
    speculation is on) the acceptance tripwire appropriate to the engine's
    `acceptance_source` (vLLM has real cumulative counters; SGLang's
    spec_accept_length is a most-recent-interval gauge, so its tripwire is
    "at least one token accepted on average", not a nonzero delta).

    Ruling P22: rule (1) checks `attempted` (completed + failed), not
    `completed` -- the client's `completed` field counts successes only, so
    checking it against `num_prompts` would make rule (2)'s error-rate check
    dead code (any single failure would already fail rule (1)). `attempted`
    is computed once in `build_summary` and stored on `summary`; failures are
    then judged by the error-rate threshold instead.

    Ruling P23: rule (3) reads the pre-computed `host_drift_flag` off
    `summary` (also set by `build_summary` from the same `thresholds`)
    rather than recomputing the comparison here.
    """
    attempted = summary.get("attempted")
    if attempted != cfg.num_prompts:
        return False, f"attempted {attempted} of {cfg.num_prompts}"

    error_rate = summary.get("error_rate")
    if error_rate is not None and error_rate > thresholds.max_error_rate:
        return False, f"error_rate {error_rate} exceeds max_error_rate {thresholds.max_error_rate}"

    if summary.get("host_drift_flag"):
        drift = summary.get("host_share_drift_mb")
        return False, f"host_share_drift_mb {drift} exceeds max_host_drift_mb {thresholds.max_host_drift_mb}"

    if spec_on:
        engine_metrics = summary.get("engine_metrics") or {}
        source = summary.get("acceptance_source")
        if source == "vllm_counters":
            rate = summary.get("acceptance_rate_mean")
            drafts_delta = engine_metrics.get("drafts_delta")
            if rate is None or rate <= 0 or not drafts_delta:
                return False, "acceptance_rate_mean is zero/None or drafts_delta did not advance (vllm_counters)"
        elif source == "sglang_gauge":
            accept_length = engine_metrics.get("spec_accept_length_last")
            if accept_length is None or accept_length <= 1.0:
                return False, (
                    "spec_accept_length_last missing or <= 1.0 -- silent spec-decode "
                    "disablement tripwire (sglang_gauge)"
                )

    return True, None
