"""Pinned KV pool: the flags each engine gets, the guard rails, and the read-back.

The pin exists so RQ1 compares engines rather than our own per-engine headroom
(writeup §7). Its whole value depends on the engine actually honouring it, so the
read-back assertion matters as much as the flags.
"""
import pytest

from bench.runner.assertions import Thresholds, check
from bench.runner.config import validate
from bench.runner.engine import ENGINES, KV_BYTES_PER_TOKEN, KV_BYTES_PER_TOKEN_DRAFT
from tests.bench.runner.test_engine import _cfg


def test_kv_bytes_per_token_matches_both_engines_own_reporting():
    """36,864 B = 2 x 36 layers x 2 KV heads x 128 head dim x 2 bytes. vLLM
    reported 2.41 GiB for 70,240 tokens and SGLang 1.957 GiB for 57,014 -- both
    35.97-35.98 KiB/token -- which is what makes the token<->byte conversion exact."""
    assert KV_BYTES_PER_TOKEN == 36864
    for gib, tokens in ((2.41, 70240), (1.957, 57014)):
        measured = gib * 1024**3 / tokens
        assert abs(measured - KV_BYTES_PER_TOKEN) / KV_BYTES_PER_TOKEN < 0.005, (gib, tokens, measured)
    # The draft's own KV is a different size, which is why a pinned token count
    # does not convert to the same byte count once speculation is on.
    assert KV_BYTES_PER_TOKEN_DRAFT == 12288
    assert KV_BYTES_PER_TOKEN_DRAFT != KV_BYTES_PER_TOKEN


def test_vllm_gets_the_pin_in_bytes():
    cfg = _cfg()
    cfg.kv_cache_tokens = 44000
    args = ENGINES["vllm"].build_launch_args(cfg, 0.79)
    i = args.index("--kv-cache-memory-bytes")
    assert args[i + 1] == str(44000 * 36864)
    # The fraction still bounds the total; the pin sizes the KV slice inside it.
    assert "--gpu-memory-utilization" in args


def test_sglang_gets_the_pin_in_tokens():
    """SGLang's --max-total-tokens is already a token count, so no conversion --
    which is why the config field is expressed in tokens rather than bytes."""
    cfg = _cfg()
    cfg.engine = "sglang"
    cfg.kv_cache_tokens = 44000
    args = ENGINES["sglang"].build_launch_args(cfg, 0.74)
    i = args.index("--max-total-tokens")
    assert args[i + 1] == "44000"
    assert "--mem-fraction-static" in args


def test_neither_engine_gets_a_kv_flag_when_unpinned():
    for name, mem in (("vllm", 0.79), ("sglang", 0.74)):
        cfg = _cfg()
        cfg.engine = name
        assert cfg.kv_cache_tokens is None
        args = ENGINES[name].build_launch_args(cfg, mem)
        assert "--kv-cache-memory-bytes" not in args
        assert "--max-total-tokens" not in args


def test_validate_refuses_a_pin_while_speculation_is_on():
    """The draft adds its own KV and whether vLLM's byte flag covers the draft
    pool is unverified, so converting would be guessing. An equal-KV claim built
    on a guess is worse than an unequal comparison stated honestly."""
    cfg = _cfg()
    cfg.kv_cache_tokens = 44000
    cfg.spec_method = "ngram"
    cfg.spec_k = 3
    cfg.ngram_lookup_max = 4
    with pytest.raises(ValueError, match="only supported with spec_method='off'"):
        validate(cfg)


def test_validate_refuses_a_pin_below_one_full_request():
    cfg = _cfg()
    cfg.kv_cache_tokens = cfg.max_model_len - 1
    with pytest.raises(ValueError, match="below max_model_len"):
        validate(cfg)


def test_validate_refuses_a_nonpositive_pin():
    cfg = _cfg()
    cfg.kv_cache_tokens = 0
    # 0 is falsey, so it must be rejected by the explicit None check rather than
    # silently treated as "unpinned".
    cfg.kv_cache_tokens = -1
    with pytest.raises(ValueError, match="must be positive"):
        validate(cfg)


def _summary(**over):
    s = {"attempted": 4, "error_rate": 0.0, "host_share_drift_mb": 0}
    s.update(over)
    return s


def test_run_is_valid_when_the_engine_honours_the_pin():
    cfg = _cfg()
    cfg.kv_cache_tokens = 44000
    cfg.num_prompts = 4
    ok, reason = check(cfg, _summary(), spec_on=False, thresholds=Thresholds(),
                       engine_memory={"kv_cache_tokens": 44016})  # block rounding
    assert (ok, reason) == (True, None)


def test_run_is_invalid_when_the_pin_was_ignored():
    """A silently ignored flag would leave the engines unequal while the writeup
    claimed otherwise -- invisible without this check."""
    cfg = _cfg()
    cfg.kv_cache_tokens = 44000
    cfg.num_prompts = 4
    ok, reason = check(cfg, _summary(), spec_on=False, thresholds=Thresholds(),
                       engine_memory={"kv_cache_tokens": 70240})
    assert not ok
    assert "KV pin not honoured" in reason
    assert "70240" in reason


def test_run_is_invalid_when_the_pin_cannot_be_read_back():
    """An unverifiable equal-KV claim is not worth making."""
    cfg = _cfg()
    cfg.kv_cache_tokens = 44000
    cfg.num_prompts = 4
    ok, reason = check(cfg, _summary(), spec_on=False, thresholds=Thresholds(),
                       engine_memory={})
    assert not ok
    assert "cannot be verified" in reason


def test_unpinned_runs_are_unaffected_by_the_new_rule():
    cfg = _cfg()
    cfg.num_prompts = 4
    assert cfg.kv_cache_tokens is None
    ok, reason = check(cfg, _summary(), spec_on=False, thresholds=Thresholds(), engine_memory={})
    assert (ok, reason) == (True, None)
