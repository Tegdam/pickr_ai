import json

import pytest
import yaml

from bench.runner.engine import ENGINES
from bench.runner.paths import REPO_ROOT
from bench.runner.sweep import expand, load_sweep, schedule, sweep_options


def _traces(tmp_path):
    (tmp_path / "chat_v1.jsonl").write_text('{"prompt":"a","output_tokens":1}\n')
    (tmp_path / "chat_v1.meta.json").write_text(json.dumps({"trace_sha256": "f" * 64}))
    (tmp_path / "summarization_v1.jsonl").write_text('{"prompt":"b","output_tokens":1}\n')
    (tmp_path / "summarization_v1.meta.json").write_text(json.dumps({"trace_sha256": "e" * 64}))


def test_expand_crosses_axes_and_reps_and_resolves_traces(tmp_path):
    _traces(tmp_path)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"phase": "p0b", "model": "m", "model_revision": "r", "quantization": "awq",
        "spec_method": "off", "spec_k": None, "ngram_lookup_max": 4, "load_mode": "concurrency", "request_rate": None,
        "burstiness": 1.0, "num_prompts": 10, "cache_state": "cold", "prefix_caching": True, "max_model_len": 2048,
        "max_num_seqs": 64, "chunked_prefill_tokens": 1024, "cudagraph_capture_sizes": [1, 2], "sampling": {},
        "ignore_eos": True, "max_tokens_cap": None, "warmup_requests": 2, "cooldown_temp_c": 55, "cooldown_min_s": 1,
        "gpu_headroom_mb": 256, "seed": 0, "extra_body": {}, "draft_model": None, "draft_revision": None,
        "draft_quantization": None, "traces_dir": str(tmp_path), "try_clock_pin": False}))
    sweep = tmp_path / "s.yaml"
    sweep.write_text(yaml.safe_dump({"base": str(base), "sweep_id": "t", "rq_tag": "x", "schedule_seed": 1,
        "max_retries_total": 2, "reps": 2, "axes": {"engine": ["vllm", "sglang"], "workload": ["A", "B"], "concurrency": [1, 8]}}))
    cfgs = expand(load_sweep(sweep), "t")
    assert len(cfgs) == 2 * 2 * 2 * 2
    assert [c.run_id for c in cfgs][:3] == ["t-0000-r0", "t-0000-r1", "t-0001-r0"]
    a = next(c for c in cfgs if c.workload == "A")
    assert a.trace_file.endswith("chat_v1.jsonl") and a.trace_sha256 == "f" * 64 and a.trace_version == 1
    b = next(c for c in cfgs if c.workload == "B")
    assert b.trace_sha256 == "e" * 64
    assert {c.engine for c in cfgs} == {"vllm", "sglang"} and {c.concurrency for c in cfgs} == {1, 8}
    assert all(c.gpu_memory_utilization == 0.0 and c.free_vram_mb_at_start == 0 for c in cfgs)  # resolved at run time
    assert cfgs[0].image == ENGINES["vllm"].image


def test_load_sweep_resolves_repo_root_relative_base_regardless_of_cwd(tmp_path, monkeypatch):
    """C1/P28: a sweep file that lives somewhere other than the repo (a
    tmp-path fixture here) with a `base:` spelled the way every real
    bench/configs/p0b_*.yaml spells it ("bench/configs/base.yaml", relative
    to the repo root, not to the sweep file's own directory) must still
    resolve -- even when the process's cwd is somewhere else entirely."""
    monkeypatch.chdir(tmp_path)  # cwd is neither the sweep's dir nor the repo root
    sweep = tmp_path / "s.yaml"
    sweep.write_text(yaml.safe_dump({"base": "bench/configs/base.yaml", "sweep_id": "t"}))
    merged = load_sweep(sweep)
    assert merged["model"] == "Qwen/Qwen2.5-3B-Instruct-AWQ"  # a real base.yaml field
    assert merged["sweep_id"] == "t"
    assert "base" not in merged


def test_schedule_is_a_seeded_permutation(tmp_path):
    _traces(tmp_path)
    from tests.bench.runner.test_engine import _cfg
    cfgs = [_cfg(run_id=f"r{i}") for i in range(20)]
    s1 = schedule(cfgs, seed=7); s2 = schedule(cfgs, seed=7); s3 = schedule(cfgs, seed=8)
    assert [c.run_id for c in s1] == [c.run_id for c in s2] != [c.run_id for c in s3]
    assert sorted(c.run_id for c in s1) == sorted(c.run_id for c in cfgs)


def test_sweep_options_defaults_and_overrides():
    # C1/P28: the traces_dir default is anchored to the repo root, not left
    # relative -- it must resolve regardless of the process's invocation cwd.
    assert sweep_options({}) == {
        "gpu_headroom_mb": 256,
        "try_clock_pin": False,
        "max_retries_total": 0,
        "schedule_seed": 0,
        "block_axis": None,
        "traces_dir": str(REPO_ROOT / "bench" / "traces"),
    }
    overrides = {
        "gpu_headroom_mb": 512,
        "try_clock_pin": True,
        "max_retries_total": 5,
        "schedule_seed": 42,
        "traces_dir": "/tmp/traces",
        "axes": {"engine": ["vllm"]},  # unrelated sweep keys must be ignored
    }
    assert sweep_options(overrides) == {
        "gpu_headroom_mb": 512,
        "try_clock_pin": True,
        "max_retries_total": 5,
        "schedule_seed": 42,
        "block_axis": None,
        "traces_dir": "/tmp/traces",
    }


def _arm_cfgs(per_arm=10):
    from tests.bench.runner.test_engine import _cfg
    out = []
    for arm in ("balanced", "performance"):
        for i in range(per_arm):
            c = _cfg(run_id=f"{arm}-{i}")
            c.power_overlay = arm
            out.append(c)
    return out


def test_block_schedule_caps_streaks_and_equalises_mean_position():
    """A free shuffle balances two arms only on average and can still cluster --
    seed 20260927 over 10+10 gave a 5-run streak with the arms' mean positions
    1.6 slots apart. A randomised block design caps any streak at 2 and puts the
    mean positions within half a slot, so drift over the session cannot land on
    one arm."""
    import statistics as st
    cfgs = _arm_cfgs()

    ordered = schedule(cfgs, seed=20260927, block_axis="power_overlay")
    arms = [c.power_overlay for c in ordered]
    assert len(ordered) == 20
    assert sorted(c.run_id for c in ordered) == sorted(c.run_id for c in cfgs)

    longest = cur = 1
    for i in range(1, len(arms)):
        cur = cur + 1 if arms[i] == arms[i - 1] else 1
        longest = max(longest, cur)
    assert longest <= 2, arms

    means = {a: st.mean([i for i, x in enumerate(arms) if x == a]) for a in set(arms)}
    assert abs(means["balanced"] - means["performance"]) <= 0.5, means
    # Every consecutive pair holds one of each arm -- that is what blocked means.
    for i in range(0, 20, 2):
        assert set(arms[i:i + 2]) == {"balanced", "performance"}, (i, arms)


def test_free_shuffle_is_still_the_default_and_can_cluster():
    """Guards the contrast the block design was added for: with no block_axis the
    same seed and inputs produce a streak longer than a blocked schedule allows."""
    cfgs = _arm_cfgs()
    arms = [c.power_overlay for c in schedule(cfgs, seed=20260927)]
    longest = cur = 1
    for i in range(1, len(arms)):
        cur = cur + 1 if arms[i] == arms[i - 1] else 1
        longest = max(longest, cur)
    assert longest > 2, arms


def test_block_schedule_is_deterministic_per_seed():
    cfgs = _arm_cfgs()
    a = schedule(cfgs, seed=5, block_axis="power_overlay")
    b = schedule(cfgs, seed=5, block_axis="power_overlay")
    c = schedule(cfgs, seed=6, block_axis="power_overlay")
    assert [x.run_id for x in a] == [x.run_id for x in b]
    assert [x.run_id for x in a] != [x.run_id for x in c]


def test_block_schedule_keeps_unequal_groups():
    """Unequal arms must not lose runs: the longer one fills later blocks alone."""
    cfgs = [c for c in _arm_cfgs(per_arm=3)
            if not (c.power_overlay == "balanced" and c.run_id.endswith("-2"))]
    ordered = schedule(cfgs, seed=1, block_axis="power_overlay")
    assert len(ordered) == 5
    assert sorted(c.run_id for c in ordered) == sorted(c.run_id for c in cfgs)


def test_block_schedule_rejects_a_non_field_axis():
    with pytest.raises(ValueError, match="not a RunConfig field"):
        schedule(_arm_cfgs(per_arm=2), seed=1, block_axis="no_such_field")


def test_power_overlay_axis_expands_with_no_special_casing(tmp_path):
    """power_overlay is a RunConfig field, so it works as an axis through the
    ordinary expander -- that is what makes the interleave a normal sweep rather
    than a bespoke script."""
    _traces(tmp_path)
    sweep_path = tmp_path / "ab.yaml"
    sweep_path.write_text(yaml.safe_dump({
        "base": str(REPO_ROOT / "bench" / "configs" / "base.yaml"),
        "sweep_id": "ab", "rq_tag": "calibration", "traces_dir": str(tmp_path),
        "axes": {"engine": ["vllm"], "workload": ["A"],
                 "power_overlay": ["balanced", "performance"]},
        "reps": 2,
    }))
    cfgs = expand(load_sweep(sweep_path), "ab")
    assert len(cfgs) == 4
    assert sorted({c.power_overlay for c in cfgs}) == ["balanced", "performance"]
    assert sweep_options(load_sweep(sweep_path))["block_axis"] is None
