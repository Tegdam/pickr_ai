import json

import pytest

from bench.runner.client import CLIENT_IMAGE, build_client_command, build_client_image, run_client
from bench.runner.engine import ENGINES
from tests.bench.runner.test_engine import _cfg


def _command(**over):
    cfg = _cfg(**over)
    return build_client_command(cfg, "http://localhost:8000", "/results", "client_raw.json")


def test_concurrency_mode_command():
    args = _command()
    joined = " ".join(args)
    assert "--max-concurrency 8" in joined
    assert "--request-rate inf" in joined
    assert "--dataset-name custom" in joined
    assert "--skip-chat-template" in joined
    assert "--ignore-eos" in joined
    assert "--seed 0" in joined
    assert "--temperature 1.0" in joined
    assert "--disable-shuffle" in joined
    assert "--custom-output-len -1" in joined
    assert "--base-url http://localhost:8000" in joined
    assert "--result-dir /results --result-filename client_raw.json" in joined


def test_poisson_mode_command_omits_max_concurrency():
    args = _command(load_mode="poisson", concurrency=None, request_rate=4.0, burstiness=1.0)
    joined = " ".join(args)
    assert "--request-rate 4.0 --burstiness 1.0" in joined
    assert "--max-concurrency" not in joined


def test_ignore_eos_omitted_when_false():
    args = _command(ignore_eos=False)
    assert "--ignore-eos" not in args


def test_extra_body_renders_as_one_json_arg():
    args = _command(extra_body={"response_format": {"type": "json_object"}})
    assert "--extra-body" in args
    idx = args.index("--extra-body")
    assert args[idx + 1] == json.dumps({"response_format": {"type": "json_object"}})
    # not split across multiple argv entries
    assert args[idx + 2] != "json_object"


def test_extra_body_omitted_when_empty():
    args = _command(extra_body={})
    assert "--extra-body" not in args


def test_dataset_path_uses_trace_basename():
    args = _command(trace_file="bench/traces/chat_v1.jsonl")
    assert "--dataset-path" in args
    assert args[args.index("--dataset-path") + 1] == "/traces/chat_v1.jsonl"


def test_sampling_always_sent_from_cfg():
    args = _command(sampling={"temperature": 0.0, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1})
    joined = " ".join(args)
    assert "--temperature 0.0" in joined
    assert "--top-p 0.9" in joined
    assert "--top-k 40" in joined
    assert "--repetition-penalty 1.1" in joined


def test_percentile_flags_always_present():
    joined = " ".join(_command())
    assert "--percentile-metrics ttft,tpot,itl,e2el" in joined
    assert "--metric-percentiles 50,90,99" in joined


def test_unsupported_load_mode_raises():
    with pytest.raises(ValueError, match="load_mode"):
        _command(load_mode="ramp")


def test_build_client_image_builds_and_returns_digest(fake_docker):
    digest = build_client_image(fake_docker)
    assert digest == fake_docker.digest
    build_calls = [c for c in fake_docker.calls if c["op"] == "build"]
    assert build_calls == [{
        "op": "build", "tag": CLIENT_IMAGE,
        "dockerfile": "bench/runner/client.Dockerfile", "context": "bench/runner",
    }]


def test_run_client_returns_validated_dict_and_uses_gpuless_container(tmp_path, fake_docker, client_json):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    (run_dir / "client_raw.json").write_text(json.dumps(client_json), encoding="utf-8")

    cfg = _cfg()
    spec = ENGINES["vllm"]
    result = run_client(fake_docker, cfg, spec, run_dir, traces_dir, sleep=lambda s: None)

    assert result == client_json
    call = fake_docker.calls[0]
    assert call["op"] == "run"
    assert call["gpus"] is False
    assert call["network_host"] is True
    assert call["entrypoint"] == "vllm"
    assert call["args"][:2] == ["bench", "serve"]
    assert (str(traces_dir), "/traces", "ro") in call["mounts"]
    assert (str(run_dir), "/results", "rw") in call["mounts"]


def test_run_client_raises_with_logs_on_nonzero_exit(tmp_path, fake_docker):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    cfg = _cfg()
    spec = ENGINES["vllm"]
    name = f"{cfg.run_id}-client"
    fake_docker.logs[name] = "\n".join(f"line{i}" for i in range(50))
    fake_docker.fail_next = True

    with pytest.raises(RuntimeError) as e:
        run_client(fake_docker, cfg, spec, run_dir, traces_dir, sleep=lambda s: None)
    assert "line49" in str(e.value)
    assert "line0" not in str(e.value)  # last 40 lines only


def test_run_client_raises_when_result_file_missing(tmp_path, fake_docker):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    cfg = _cfg()
    spec = ENGINES["vllm"]

    with pytest.raises(RuntimeError, match="missing result file"):
        run_client(fake_docker, cfg, spec, run_dir, traces_dir, sleep=lambda s: None)


def test_run_client_times_out_when_container_never_exits(tmp_path, fake_docker):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    cfg = _cfg()
    spec = ENGINES["vllm"]
    name = f"{cfg.run_id}-client"
    fake_docker.logs[name] = "\n".join(f"line{i}" for i in range(50))
    fake_docker.keep_running = True

    with pytest.raises(TimeoutError) as e:
        run_client(fake_docker, cfg, spec, run_dir, traces_dir, client_timeout_s=0.01, poll=0.0, sleep=lambda s: None)
    assert "line49" in str(e.value)
