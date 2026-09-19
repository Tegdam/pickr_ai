import pytest

from bench.runner.assertions import Thresholds, check
from tests.bench.runner.test_engine import _cfg


def _summary(**over):
    base = {
        "attempted": 200,
        "error_rate": 0.0,
        "host_drift_flag": False,
        "host_share_drift_mb": 10,
        "acceptance_rate_mean": 0.7,
        "acceptance_source": "vllm_counters",
        "engine_metrics": {"drafts_delta": 40.0, "spec_accept_length_last": None},
    }
    base.update(over)
    return base


def test_clean_run_is_valid():
    cfg = _cfg(num_prompts=200, spec_method="off")
    valid, reason = check(cfg, _summary(), spec_on=False, thresholds=Thresholds())
    assert valid is True and reason is None


def test_attempted_short_of_num_prompts_is_invalid():
    cfg = _cfg(num_prompts=200)
    valid, reason = check(cfg, _summary(attempted=190), spec_on=False, thresholds=Thresholds())
    assert valid is False
    assert "attempted" in reason and "190" in reason and "200" in reason


def test_error_rate_above_threshold_is_invalid():
    cfg = _cfg(num_prompts=200)
    valid, reason = check(cfg, _summary(error_rate=0.05), spec_on=False, thresholds=Thresholds())
    assert valid is False
    assert "error_rate" in reason


def test_error_rate_at_threshold_is_valid():
    cfg = _cfg(num_prompts=200)
    valid, reason = check(cfg, _summary(error_rate=0.01), spec_on=False, thresholds=Thresholds())
    assert valid is True and reason is None


# Ruling P22: a single failed request (attempted == num_prompts, one of them
# an error) no longer fails rule (1) outright -- it is judged by error rate.
def test_one_failure_at_low_error_rate_is_valid():
    cfg = _cfg(num_prompts=200)
    valid, reason = check(cfg, _summary(attempted=200, error_rate=0.005), spec_on=False, thresholds=Thresholds())
    assert valid is True and reason is None


def test_several_failures_above_error_rate_threshold_is_invalid():
    cfg = _cfg(num_prompts=200)
    valid, reason = check(cfg, _summary(attempted=200, error_rate=0.015), spec_on=False, thresholds=Thresholds())
    assert valid is False
    assert "error_rate" in reason


def test_attempted_short_with_zero_failures_is_invalid_attempted():
    # e.g. completed = N-1, failed = 0: the client never even attempted one
    # of the prompts (a crash/hang, not a request-level failure).
    cfg = _cfg(num_prompts=200)
    valid, reason = check(cfg, _summary(attempted=199, error_rate=0.0), spec_on=False, thresholds=Thresholds())
    assert valid is False
    assert "attempted" in reason


def test_host_drift_flag_true_is_invalid():
    cfg = _cfg(num_prompts=200)
    valid, reason = check(cfg, _summary(host_drift_flag=True, host_share_drift_mb=300),
                           spec_on=False, thresholds=Thresholds())
    assert valid is False
    assert "host_share_drift_mb" in reason


def test_host_drift_flag_false_does_not_invalidate_even_with_large_value():
    # assertions trusts the pre-computed flag rather than recomputing it.
    cfg = _cfg(num_prompts=200)
    valid, reason = check(cfg, _summary(host_drift_flag=False, host_share_drift_mb=9999),
                           spec_on=False, thresholds=Thresholds())
    assert valid is True


def test_throttled_alone_does_not_invalidate():
    cfg = _cfg(num_prompts=200)
    summary = _summary()
    summary["throttled"] = True
    valid, reason = check(cfg, summary, spec_on=False, thresholds=Thresholds())
    assert valid is True and reason is None


def test_spec_on_vllm_counters_acceptance_zero_is_invalid():
    cfg = _cfg(num_prompts=200, spec_method="ngram", spec_k=3, ngram_lookup_max=4)
    summary = _summary(acceptance_rate_mean=0.0, acceptance_source="vllm_counters")
    valid, reason = check(cfg, summary, spec_on=True, thresholds=Thresholds())
    assert valid is False
    assert "acceptance" in reason


def test_spec_on_vllm_counters_drafts_not_advanced_is_invalid():
    cfg = _cfg(num_prompts=200, spec_method="ngram", spec_k=3, ngram_lookup_max=4)
    summary = _summary(acceptance_rate_mean=0.7, acceptance_source="vllm_counters",
                        engine_metrics={"drafts_delta": 0.0})
    valid, reason = check(cfg, summary, spec_on=True, thresholds=Thresholds())
    assert valid is False
    assert "acceptance" in reason


def test_spec_on_vllm_counters_healthy_acceptance_is_valid():
    cfg = _cfg(num_prompts=200, spec_method="ngram", spec_k=3, ngram_lookup_max=4)
    summary = _summary(acceptance_rate_mean=0.7, acceptance_source="vllm_counters",
                        engine_metrics={"drafts_delta": 40.0})
    valid, reason = check(cfg, summary, spec_on=True, thresholds=Thresholds())
    assert valid is True and reason is None


def test_spec_on_sglang_gauge_accept_length_at_or_below_one_is_invalid():
    cfg = _cfg(num_prompts=200, engine="sglang", spec_method="ngram", spec_k=3, ngram_lookup_max=4)
    summary = _summary(acceptance_source="sglang_gauge", engine_metrics={"spec_accept_length_last": 1.0})
    valid, reason = check(cfg, summary, spec_on=True, thresholds=Thresholds())
    assert valid is False
    assert "spec_accept_length_last" in reason


def test_spec_on_sglang_gauge_accept_length_none_is_invalid():
    cfg = _cfg(num_prompts=200, engine="sglang", spec_method="ngram", spec_k=3, ngram_lookup_max=4)
    summary = _summary(acceptance_source="sglang_gauge", engine_metrics={"spec_accept_length_last": None})
    valid, reason = check(cfg, summary, spec_on=True, thresholds=Thresholds())
    assert valid is False


def test_spec_on_sglang_gauge_healthy_accept_length_is_valid():
    cfg = _cfg(num_prompts=200, engine="sglang", spec_method="ngram", spec_k=3, ngram_lookup_max=4)
    summary = _summary(acceptance_source="sglang_gauge", engine_metrics={"spec_accept_length_last": 2.8})
    valid, reason = check(cfg, summary, spec_on=True, thresholds=Thresholds())
    assert valid is True and reason is None
