import hashlib

import pytest

from bench.runner.config import RunConfig, resolve_gpu_memory_fraction, validate
from tests.bench.runner.test_engine import _cfg


def test_resolve_gpu_memory_fraction_uses_measured_free_vram():
    assert resolve_gpu_memory_fraction(total_mb=6141, host_used_mb=0, headroom_mb=256) == 0.95
    assert resolve_gpu_memory_fraction(total_mb=6141, host_used_mb=800, headroom_mb=256) == 0.82
    assert resolve_gpu_memory_fraction(total_mb=6141, host_used_mb=6000) == 0.0


def test_validate_checks_trace_sha(tmp_path):
    trace = tmp_path / "t.jsonl"
    trace.write_text('{"prompt": "x", "output_tokens": 1}\n')
    sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    validate(_cfg(trace_file=str(trace), trace_sha256=sha))
    with pytest.raises(ValueError, match="sha256"):
        validate(_cfg(trace_file=str(trace), trace_sha256="0" * 64))


def test_validate_rejects_missing_capture_sizes_and_bad_engine(tmp_path):
    trace = tmp_path / "t.jsonl"; trace.write_text("{}\n")
    sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="cudagraph"):
        validate(_cfg(trace_file=str(trace), trace_sha256=sha, cudagraph_capture_sizes=[]))
    with pytest.raises(ValueError, match="engine"):
        validate(_cfg(trace_file=str(trace), trace_sha256=sha, engine="tgi"))


def test_validate_requires_draft_revision_for_draft_spec(tmp_path):
    """Fix round 1, item 2: an unset draft_revision must be rejected here,
    rather than silently producing a `.../snapshots/None` SGLang launch path."""
    trace = tmp_path / "t.jsonl"; trace.write_text("{}\n")
    sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="draft_revision"):
        validate(_cfg(trace_file=str(trace), trace_sha256=sha, spec_method="draft",
                       draft_model="Qwen/Qwen2.5-0.5B-Instruct", draft_revision=None, spec_k=3))


def test_validate_requires_ngram_lookup_max_for_ngram_spec(tmp_path):
    """Fix round 1 minor: the recorded config must equal the launched one --
    no silent `or 4` fallback in the builders, so validate() must enforce this."""
    trace = tmp_path / "t.jsonl"; trace.write_text("{}\n")
    sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="ngram_lookup_max"):
        validate(_cfg(trace_file=str(trace), trace_sha256=sha, spec_method="ngram", spec_k=5, ngram_lookup_max=None))


def test_validate_requires_full_sampling_dict(tmp_path):
    """P9: the client must always send temperature/top_p/top_k/repetition_penalty
    (doc §3.6, §4.5: --generation-config vllm / --sampling-defaults openai make the
    client the sole source of sampling params, so the config may never omit one)."""
    trace = tmp_path / "t.jsonl"; trace.write_text("{}\n")
    sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sampling"):
        validate(_cfg(trace_file=str(trace), trace_sha256=sha,
                       sampling={"temperature": 1.0, "top_p": 1.0}))


def test_config_round_trips_through_dict():
    c = _cfg()
    assert RunConfig(**c.to_dict()) == c
