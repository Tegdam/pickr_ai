import json

import yaml

from bench.runner.sweep import expand, load_sweep, schedule


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
        "draft_quantization": None, "traces_dir": str(tmp_path)}))
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


def test_schedule_is_a_seeded_permutation(tmp_path):
    _traces(tmp_path)
    from tests.bench.runner.test_engine import _cfg
    cfgs = [_cfg(run_id=f"r{i}") for i in range(20)]
    s1 = schedule(cfgs, seed=7); s2 = schedule(cfgs, seed=7); s3 = schedule(cfgs, seed=8)
    assert [c.run_id for c in s1] == [c.run_id for c in s2] != [c.run_id for c in s3]
    assert sorted(c.run_id for c in s1) == sorted(c.run_id for c in cfgs)
