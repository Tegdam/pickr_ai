"""`vllm bench serve` client wrapper (doc §5, §8 "Client" table).

Every flag name/spelling below is copied from bench/docs/p0b-engine-verification.md
(§5's `--help=all` transcript, §8's Client table) -- not from the brief's guess.
Do not edit these from memory; re-verify against the doc first.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import RunConfig
from .engine import EngineSpec
from .schema import validate_client_output

# Ruling P5 (doc §8): a derived image, not a per-run `pip install` (network +
# nondeterminism). Built from bench/runner/client.Dockerfile.
CLIENT_IMAGE = "bench-client:v0.29.0"


def build_client_image(docker) -> str:
    """Builds the derived client image and returns its digest (Task 4's
    Docker.image_digest falls back to `.Id` for locally built images that carry
    no RepoDigests). Never invoked against a real docker daemon in tests."""
    docker.build(CLIENT_IMAGE, "bench/runner/client.Dockerfile", "bench/runner")
    return docker.image_digest(CLIENT_IMAGE)


def build_client_command(cfg: RunConfig, base_url: str, result_dir_in_container: str, result_filename: str) -> list[str]:
    """Builds the `vllm bench serve ...` argument list (everything after the
    `bench serve` subcommand). doc §5/§8 flag-by-flag:

    - --backend openai --endpoint /v1/completions --base-url <base_url>       (doc §5 default backend/endpoint; §8 "Backend")
    - --model / --tokenizer <cfg.model>                                       (doc §8 "Model/tokenizer": served name == cfg.model, same value engine.py's
                                                                                 _served_name uses, so the client's --model always matches the /metrics label)
    - --dataset-name custom --dataset-path /traces/<basename>                 (doc §8 "Dataset")
    - --custom-output-len -1                                                  (doc §5: default is 256; -1 uses each row's output_tokens -- mandatory, never omit)
    - --skip-chat-template                                                    (doc §6.1: trace rows are pre-templated; server-side template must not re-apply)
    - --disable-shuffle --no-oversample                                      (doc §8 "Ordering guarantee"/"Dataset": row i <-> request i)
    - --num-prompts / --seed
    - --temperature/--top-p/--top-k/--repetition-penalty always sent          (P9: the client is the sole source of sampling params, config.REQUIRED_SAMPLING_KEYS)
    - --ignore-eos when cfg.ignore_eos                                        (doc §8 "Termination")
    - --max-concurrency C --request-rate inf (concurrency) or
      --request-rate R --burstiness B (poisson)                              (doc §8 "Load")
    - --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99   (doc §8 "Output")
    - --extra-body '<json>' when non-empty                                   (doc §8 "Sampling")
    - --save-result --save-detailed --result-dir --result-filename           (doc §8 "Output")
    """
    args = [
        "--backend", "openai",
        "--endpoint", "/v1/completions",
        "--base-url", base_url,
        "--model", cfg.model,
        "--tokenizer", cfg.model,
        "--dataset-name", "custom",
        "--dataset-path", f"/traces/{Path(cfg.trace_file).name}",
        "--custom-output-len", "-1",
        "--skip-chat-template",
        "--disable-shuffle",
        "--no-oversample",
        "--num-prompts", str(cfg.num_prompts),
        "--seed", str(cfg.seed),
        "--temperature", str(cfg.sampling["temperature"]),
        "--top-p", str(cfg.sampling["top_p"]),
        "--top-k", str(cfg.sampling["top_k"]),
        "--repetition-penalty", str(cfg.sampling["repetition_penalty"]),
    ]
    if cfg.ignore_eos:
        args += ["--ignore-eos"]

    if cfg.load_mode == "concurrency":
        args += ["--max-concurrency", str(cfg.concurrency), "--request-rate", "inf"]
    elif cfg.load_mode == "poisson":
        args += ["--request-rate", str(cfg.request_rate), "--burstiness", str(cfg.burstiness)]
    else:
        raise ValueError(f"unsupported load_mode {cfg.load_mode!r}; see p0b-engine-verification.md §8")

    args += ["--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99"]

    if cfg.extra_body:
        args += ["--extra-body", json.dumps(cfg.extra_body)]

    args += [
        "--save-result", "--save-detailed",
        "--result-dir", result_dir_in_container,
        "--result-filename", result_filename,
    ]
    return args


def run_client(docker, cfg: RunConfig, spec: EngineSpec, run_dir: Path, traces_dir: Path,
                client_image: str = CLIENT_IMAGE, poll: float = 2.0, sleep=time.sleep,
                client_timeout_s: float = 3600, result_filename: str = "client_raw.json",
                container_name: str | None = None) -> dict:
    """Runs `vllm bench serve` in a GPU-less container of `client_image` against
    the already-healthy engine at `spec.port`, waits for it to exit, then reads
    and validates `run_dir/result_filename`.

    doc §8 "Client" (Container): `--network host`, no `--gpus`, entrypoint
    `vllm`, `traces_dir:/traces:ro` and `run_dir:/results:rw` mounts. Results
    land on the bind mount -- nothing is copied out of the container.
    """
    run_dir = Path(run_dir)
    traces_dir = Path(traces_dir)
    base_url = f"http://localhost:{spec.port}"
    command = build_client_command(cfg, base_url, "/results", result_filename)
    name = container_name or f"{cfg.run_id}-client"

    docker.run(
        client_image, name, ["bench", "serve", *command],
        gpus=False, network_host=True,
        mounts=[(str(traces_dir), "/traces", "ro"), (str(run_dir), "/results", "rw")],
        entrypoint="vllm", extra_args=[],
    )

    t0 = time.monotonic()
    while docker.is_running(name):
        if time.monotonic() - t0 > client_timeout_s:
            tail = "\n".join(docker.container_logs(name).splitlines()[-40:])
            raise TimeoutError(f"client container {name!r} did not exit after {client_timeout_s}s\n{tail}")
        sleep(poll)

    exit_code = docker.exit_code(name)
    result_path = run_dir / result_filename
    if exit_code != 0 or not result_path.exists():
        tail = "\n".join(docker.container_logs(name).splitlines()[-40:])
        reason = f"exit code {exit_code}" if exit_code != 0 else f"missing result file {result_path}"
        raise RuntimeError(f"client container {name!r} failed ({reason})\n{tail}")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    validate_client_output(data)
    return data
