# Bench P0b — Runner and Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the sweep runner (container lifecycle, readiness, collection, environment capture, resume) around `vllm bench serve`, verify both engines and the client at pinned versions, and produce P0b's two calibration numbers — the variance floor and the harness ceiling — plus the pre-registered hardware probes.

**Architecture:** `bench/runner/` is a plain Python package driven by declarative sweep YAML: `sweep.py` expands a sweep into resolved run-configs and shuffles them with a logged seed; `lifecycle.py` executes one run (pre-flight → `docker run` → readiness → samplers → client → assertions → stop → cooldown) and writes the spec §7 artifact; `engine.py` holds per-engine *data tables* (image, launch args, routes, metric names) that Task 1 verifies against the installed versions before any code depends on them. GPU telemetry comes from two sources: WSL-side `nvidia-smi` (the container's view) and the Windows-side `nvidia-smi.exe` (host VRAM, power, throttle reasons), both callable from WSL. A tiny echo server stands in for an engine when measuring the client's own ceiling.

**Tech Stack:** Python 3.12 in the app venv (`env/bin/python`), `pyyaml`, `requests`, `nvidia-ml-py` (NVML), `aiohttp` (echo server), `pytest`; Docker 28 native in WSL2 with the NVIDIA runtime (verified 2026-09-18: `docker run --gpus all nvidia/cuda:12.4.1-base` sees the RTX 4050); engine images **`vllm/vllm-openai:v0.29.0`** and **`lmsysorg/sglang:v0.5.20-runtime`** (pins as of 2026-09-18; Task 1 may step one release back with the reason logged).

**Spec:** `docs/superpowers/specs/2026-09-17-pickr-inference-benchmark-design-v2.md` — §3.1 (budget + P0b probes), §4 (runtime, client, readiness), §6 (lifecycle, monitoring, invariants), §7 (artifact schema), §8 risk 6 (verify methods first), §9 P0b exit criteria. P0a's measured facts are in `bench/docs/p0a-writeup.md`.

## Global Constraints

- **`app/` is read-only.** Nothing in this plan touches it. (spec §1)
- **Engine flags, routes and metric names are verified against the installed version, never assumed** (brief §10). `engine.py` tables carry `verified_against: <image tag>`; the runner refuses to start a sweep whose engine table is unverified. Task 1 is the verification.
- **Engine images are pinned by tag *and* recorded by digest** in every `env.json`. (spec §7)
- **One engine container per run, fresh process; the client runs in a GPU-less container of the pinned vLLM image on `--network host`.** (spec §4)
- **The HF cache lives on WSL ext4** (`~/.cache/huggingface`, bind-mounted read-only into containers as `/root/.cache/huggingface`), never `/mnt/c`. (spec §8 risk 9)
- **Readiness = health route 200 → N warmup requests → cache reset (unless `cache_state: warm`) → clock starts.** (spec §4 item 5)
- **Assertions before recording** (spec §6 step 7): spec-decode on ⇒ acceptance counters non-zero; throttle flag; host-share drift below threshold; client error rate below threshold; client output validated against the expected schema. A failed assertion marks the run `valid: false`, re-queues it to the end of the sweep, and counts against a per-sweep retry cap.
- **`--enforce-eager` is prohibited**; CUDA-graph capture sizes and chunked-prefill size are explicit, recorded config. `--gpu-memory-utilization` (and SGLang's `--mem-fraction-static`) are set from measured free VRAM at run start, recorded with that measurement. (spec §3.1)
- **Sampling parameters are pinned explicitly on every run** (temperature, top_p, top_k, repetition_penalty) — vLLM would otherwise apply Qwen's `generation_config.json`. (P0a plan, open decision 1)
- **The runner never recomputes a metric it takes from the engine**; derived values are stored alongside with a `_derived` suffix. **`analysis/` never imports `runner/`.** A run is reproducible from its results dir + the trace file it names by version and SHA-256. (spec §6 invariants)
- **All commands run from the repo root inside WSL2.** Git hygiene: stage by name, never `git add -A`; **no trailers in commit messages**; commit on `benchmarking` only.
- **`bench/results/` and `bench/figures/` are git-ignored**; calibration summaries that the writeup cites are copied into `bench/docs/` by name.

## What P0a fixed for P0b (read `bench/docs/p0a-writeup.md` §8)

- Trace files exist: `bench/traces/{chat,summarization,structured,multiturn_*}_v1.jsonl`; prompts are pre-rendered ChatML → runs use the **completions** endpoint with `--skip-chat-template` (`--backend openai` in `vllm bench serve` terms; `vllm bench serve` names its backends `openai` = `/v1/completions` and `openai-chat` = `/v1/chat/completions`; Task 1 confirms the flag values at the pin).
- Chat prompt p50 = 265 tokens, p99 877 → `--max-model-len 2048` is safe for A/B/C/multiturn; verified in Task 1 against the longest row in every trace.
- A turn is a serial chain of ~2.8 calls; P0b calibrates on `chat_v1` at `--max-concurrency 8` as one config.

## File Structure

```
bench/
  runner/
    __init__.py
    engine.py            # EngineSpec tables: vllm / sglang (image, launch-arg builder, routes, metric names, verified_against)
    config.py            # RunConfig dataclass; resolve_gpu_memory_fraction(); validate()
    sweep.py             # load sweep YAML → expand axes → resolved RunConfigs → shuffle(seed) → schedule.json
    docker.py            # thin wrappers: run/stop/inspect/logs, image digest, pip freeze via `docker run --entrypoint`
    readiness.py         # wait_healthy(), warmup(), reset_cache()
    gpu_monitor.py       # Sampler thread: WSL nvidia-smi (ours) + Windows nvidia-smi.exe (host) → gpu_samples.jsonl
    metrics_scraper.py   # Scraper thread: engine /metrics → engine_metrics.jsonl (raw names) + acceptance/kv summary
    env_capture.py       # env.json builder
    client.py            # build `vllm bench serve` command; run in client container; validate + parse its JSON
    schema.py            # CLIENT_OUTPUT_SCHEMA (required keys/types) + ARTIFACT_KEYS; validate_client_output()
    summary.py           # requests.jsonl + summary.json from client JSON + samples + scrapes (goodput grid included)
    assertions.py        # post-run checks → (valid: bool, invalid_reason: str | None)
    state.py             # SweepState: pending/done/invalid/requeued, retry budget, resume
    lifecycle.py         # run_one(config, paths) — the spec §6 sequence
    cli.py / __main__.py # bench-run: check-env | run <sweep.yaml> | resume <sweep_id> | probe <name>
  echo_server/
    __init__.py
    server.py            # aiohttp OpenAI-compatible /v1/completions streaming N tokens with fixed per-token delay; /health; /metrics stub
  configs/
    base.yaml            # shared defaults (models, sampling, max-model-len, capture sizes, chunk size, warmup, cooldown)
    p0b_variance.yaml    # 10× identical runs: vllm, chat_v1, c=8, spec off
    p0b_ceiling.yaml     # echo server, request-rate ladder
    p0b_parity.yaml      # one row → both engines → prompt_tokens equality
    p0b_ignore_eos.yaml  # acceptance with/without ignore_eos (draft + ngram), c=1
    p0b_probes.yaml      # OOM-signal, cuda-graph cost, host reservation, clock pin
  analysis/
    __init__.py
    load.py              # read results dirs → pandas frames (valid filter default on)
    p0b_calibration.py   # variance floor, harness ceiling, probe tables → bench/docs/p0b-calibration.md + json
  docs/
    p0b-engine-verification.md   # Task 1 output
    p0b-writeup.md               # Task 12 output
tests/bench/runner/
  __init__.py
  conftest.py            # FakeDocker, FakeHTTP (health/metrics/completions), fixed client JSON fixture
  test_engine.py test_config.py test_sweep.py test_readiness.py test_gpu_monitor.py test_metrics_scraper.py
  test_env_capture.py test_client.py test_schema.py test_summary.py test_assertions.py test_state.py
  test_lifecycle.py test_echo_server.py test_analysis.py
```

Every module has one responsibility; `lifecycle.py` is the only one that composes them. Tests inject fakes at the `docker.py` and HTTP boundaries so the whole lifecycle runs in CI without a GPU; the calibration tasks (9–11) are runbooks that exercise the real thing.

---

### Task 1: Engine, client and model verification at the pins (runbook + verification doc)

This is spec §8 risk 6: verify *first*, build nothing on assumptions. Output is a document, plus the `verified_against` values that Task 2's tables carry.

**Files:**
- Create: `bench/docs/p0b-engine-verification.md`
- Modify: `bench/requirements.txt` (add `pyyaml`, `requests`, `nvidia-ml-py`, `aiohttp`, `jsonschema`)

- [ ] **Step 1: Pull the pinned images and record digests**

```bash
docker pull vllm/vllm-openai:v0.29.0 && docker pull lmsysorg/sglang:v0.5.20-runtime
docker image inspect vllm/vllm-openai:v0.29.0 --format '{{index .RepoDigests 0}}'
docker image inspect lmsysorg/sglang:v0.5.20-runtime --format '{{index .RepoDigests 0}}'
```
Record both digests in the doc. (~45 GB total; ext4 has 879 GB free.)

- [ ] **Step 2: Download the models into the WSL HF cache and pin revisions**

```bash
env/bin/python - <<'EOF'
from huggingface_hub import snapshot_download, model_info
for m in ["Qwen/Qwen2.5-3B-Instruct-AWQ", "Qwen/Qwen2.5-0.5B-Instruct", "BAAI/bge-small-en-v1.5"]:
    sha = model_info(m).sha
    p = snapshot_download(m, revision=sha)
    print(m, sha, p)
EOF
```
Record the three revision shas. Confirm from each `config.json`: target `num_hidden_layers=36, num_key_value_heads=2, hidden_size=2048` (head_dim 128); draft `24, 2, 896` (head_dim 64); AWQ `quantization_config.bits=4, group_size=128`. If any differ from spec §3.1, the budget table gets corrected in the doc **before** any run.

- [ ] **Step 3: Verify vLLM flags and speculative methods**

```bash
docker run --rm --entrypoint bash vllm/vllm-openai:v0.29.0 -c "vllm serve --help 2>/dev/null | grep -A3 -E 'speculative-config|gpu-memory-utilization|max-num-batched-tokens|compilation-config|enable-prefix-caching|max-num-seqs|seed|generation-config'"
docker run --rm --entrypoint bash vllm/vllm-openai:v0.29.0 -c "python -c \"import vllm, inspect; from vllm.config import SpeculativeConfig; print(vllm.__version__); src=inspect.getsource(SpeculativeConfig); import re; print(sorted(set(re.findall(r'\\\"(ngram|draft_model|eagle3?|medusa|mtp|suffix|dflash)\\\"', src))))\""
docker run --rm --entrypoint bash vllm/vllm-openai:v0.29.0 -c "vllm bench serve --help | grep -E -A2 'backend|dataset-name|dataset-path|custom-output-len|skip-chat-template|save-result|save-detailed|result-dir|result-filename|max-concurrency|request-rate|burstiness|num-prompts|seed|ignore-eos|temperature|top-p|top-k|extra-body|disable-shuffle|goodput|percentile-metrics|metric-percentiles'"
docker run --rm --entrypoint bash vllm/vllm-openai:v0.29.0 -c "python -c \"import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)\"; pip list 2>/dev/null | grep -iE '^(vllm|torch|flashinfer|xformers|triton|transformers|tokenizers) '"
```
Record: exact flag names; whether `draft_model` and `ngram` are listed methods; whether `--custom-output-len` defaults to "use per-row `output_tokens`" (`None`/`-1`) or must be passed explicitly; the client's `--save-detailed` output keys (run `vllm bench serve --help` and inspect `vllm/benchmarks/serve.py::save_to_pytorch_benchmark_format`/`main` for the JSON keys `ttfts`, `itls`, `input_lens`, `output_lens`, `generated_texts`, `errors`); CUDA version of the image.

- [ ] **Step 4: Verify SGLang flags and speculative methods**

```bash
docker run --rm --entrypoint bash lmsysorg/sglang:v0.5.20-runtime -c "python -m sglang.launch_server --help 2>/dev/null | grep -E -A2 'speculative-algorithm|speculative-draft-model|speculative-num-steps|speculative-eagle-topk|speculative-num-draft-tokens|mem-fraction-static|chunked-prefill-size|cuda-graph-bs|cuda-graph-max-bs|disable-radix-cache|enable-metrics|max-running-requests|context-length|random-seed|quantization|schedule-policy|json-model-override'"
docker run --rm --entrypoint bash lmsysorg/sglang:v0.5.20-runtime -c "python -c \"import sglang; print(sglang.__version__)\"; python -m sglang.launch_server --help 2>/dev/null | grep -A6 'speculative-algorithm' | head -12; pip list 2>/dev/null | grep -iE '^(sglang|sgl-kernel|torch|flashinfer|transformers) '"
```
Record whether `STANDALONE` and `NGRAM` are accepted values of `--speculative-algorithm`; the exact names of the flush-cache route (`grep -rn 'flush_cache' $(python -c 'import sglang,os;print(os.path.dirname(sglang.__file__))')/srt/entrypoints/http_server.py`) and the metrics names (`grep -rn 'Gauge\|Counter\|Histogram' .../srt/metrics/collector.py | grep -oE 'sglang:[a-z_]+' | sort -u`).

- [ ] **Step 5: Verify vLLM's routes and metric names the same way**

```bash
docker run --rm --entrypoint bash vllm/vllm-openai:v0.29.0 -c "d=\$(python -c 'import vllm,os;print(os.path.dirname(vllm.__file__))'); grep -n 'reset_prefix_cache\|/health\|/metrics' \$d/entrypoints/openai/api_server.py | head; grep -rhoE 'vllm:[a-z_]+' \$d/v1/metrics/loggers.py | sort -u"
```
Record: health route, cache-reset route + method, and the exact names for: accepted/draft/emitted speculative token counters, `gpu_cache_usage_perc` (KV usage), `num_requests_running`/`waiting`, `prefix_cache_hits`/`queries`.

- [ ] **Step 6: Smoke-launch each engine and send one request from one trace row**

vLLM:
```bash
docker run -d --name smoke-vllm --gpus all --network host -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
  vllm/vllm-openai:v0.29.0 --model Qwen/Qwen2.5-3B-Instruct-AWQ --revision <sha> --quantization awq \
  --max-model-len 2048 --gpu-memory-utilization 0.80 --port 8000 --seed 0
until curl -sf localhost:8000/health; do sleep 2; done
head -1 bench/traces/chat_v1.jsonl | env/bin/python -c "import json,sys,requests; r=json.load(sys.stdin); resp=requests.post('http://localhost:8000/v1/completions', json={'model':'Qwen/Qwen2.5-3B-Instruct-AWQ','prompt':r['prompt'],'max_tokens':16,'temperature':0}).json(); print(resp['usage'], '|', resp['choices'][0]['text'][:60]); print('prompt_tokens_qwen in trace:', r['prompt_tokens_qwen'])"
curl -s localhost:8000/metrics | grep -E 'vllm:(gpu_cache_usage_perc|num_requests_running|prefix_cache)' | head
docker stop smoke-vllm && docker rm smoke-vllm
```
SGLang (same model, `--port 30000`, `python -m sglang.launch_server --model-path ... --quantization awq --context-length 2048 --mem-fraction-static 0.80 --enable-metrics`), same request. **Record the `prompt_tokens` each engine reports for the same row — this is the chat-template parity check (spec §4); both must equal the trace's `prompt_tokens_qwen`.** If they differ, record the difference and stop: the discrepancy is investigated before Task 2.

Then, per engine, one spec-decode smoke: vLLM `--speculative-config '{"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","num_speculative_tokens":3}'` and `'{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}'`; SGLang `--speculative-algorithm STANDALONE --speculative-draft-model-path Qwen/Qwen2.5-0.5B-Instruct --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` and `--speculative-algorithm NGRAM ...`. After ~20 requests, scrape `/metrics` and record the acceptance counters' names and that they are non-zero. Any launch that fails is recorded with the error text; the arm is dropped or the pin stepped back with the reason.

- [ ] **Step 7: Clock-pin attempt, host telemetry, and the reservation baseline**

```bash
/mnt/c/Windows/System32/nvidia-smi.exe -lgc 2055,2055 ; echo "lgc exit=$?"     # expect failure/deprecated under WSL2
/mnt/c/Windows/System32/nvidia-smi.exe -rgc 2>/dev/null
/mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=memory.used,power.draw,power.max_limit,clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv
nvidia-smi --query-gpu=memory.used --format=csv   # WSL side, for comparison
```
Record: whether pinning works (if not, spec §6 step 1's contingency is in force: raised cooldown, `clock_cv` covariate); the idle host VRAM as seen from Windows (2026-09-18: 0 MiB — hybrid graphics, desktop on the iGPU); TGP max (100 W).

- [ ] **Step 8: Write the verification doc and commit**

`bench/docs/p0b-engine-verification.md` sections: pins + digests; model revisions + config facts; vLLM flags/methods/routes/metrics (verbatim grep output); SGLang same; client flags + output keys; parity result (three `prompt_tokens` numbers); spec-decode smoke results per engine/method; clock-pin result; host telemetry; **decisions**: final pins, any dropped arm, the exact launch-arg names Task 2's tables will use.

```bash
git add bench/docs/p0b-engine-verification.md bench/requirements.txt
git commit -m "bench: verify engine, client and model pins for P0b"
```

---

### Task 2: Engine tables and run config

**Files:**
- Create: `bench/runner/__init__.py`, `bench/runner/engine.py`, `bench/runner/config.py`
- Test: `tests/bench/runner/__init__.py`, `tests/bench/runner/conftest.py`, `tests/bench/runner/test_engine.py`, `tests/bench/runner/test_config.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) EngineSpec` with `name`, `image`, `verified_against: str | None`, `port`, `health_path`, `reset_cache_path`, `reset_cache_method`, `metrics_path`, `metric_names: dict[str, str]` (keys: `kv_usage`, `running`, `waiting`, `spec_accepted`, `spec_draft`, `spec_emitted`, `prefix_hits`, `prefix_queries`), `build_launch_args(cfg: RunConfig, mem_fraction: float) -> list[str]`, `served_model_name(cfg) -> str`.
- `ENGINES: dict[str, EngineSpec]` with `"vllm"` and `"sglang"`; values from Task 1's doc — the implementer copies flag names from `bench/docs/p0b-engine-verification.md`, not from this plan.
- `@dataclass RunConfig` fields (all recorded to `config.yaml`): `run_id`, `sweep_id`, `phase`, `rq_tag`, `engine`, `image`, `model`, `model_revision`, `quantization`, `draft_model`, `draft_revision`, `draft_quantization`, `spec_method` (`off|ngram|draft`), `spec_k`, `ngram_lookup_max`, `workload`, `trace_file`, `trace_version`, `trace_sha256`, `load_mode` (`concurrency|poisson`), `concurrency`, `request_rate`, `burstiness`, `num_prompts`, `cache_state` (`cold|warm`), `prefix_caching` (bool), `max_model_len`, `max_num_seqs`, `chunked_prefill_tokens`, `cudagraph_capture_sizes: list[int]`, `sampling: dict` (`temperature`, `top_p`, `top_k`, `repetition_penalty`), `ignore_eos`, `max_tokens_cap`, `warmup_requests`, `cooldown_temp_c`, `cooldown_min_s`, `gpu_memory_utilization` (float, resolved), `free_vram_mb_at_start` (int, measured), `seed`, `extra_body: dict`.
- `resolve_gpu_memory_fraction(total_mb, host_used_mb, headroom_mb=256) -> float` = `(total - host_used - headroom) / total` rounded down to 0.01; `validate(cfg)` raises on `enforce_eager`-style flags, missing capture sizes, unknown engine, unverified engine table, or a trace file whose SHA-256 differs from `trace_sha256`.

- [ ] **Step 1: Write the failing tests**

`tests/bench/runner/conftest.py` (shared fakes — used by later tasks too):
```python
"""Fakes at the two boundaries the runner talks through: the docker CLI and HTTP."""
import json
from types import SimpleNamespace

import pytest


class FakeDocker:
    """Records calls; returns canned outputs. `running` tracks container names."""

    def __init__(self):
        self.calls = []
        self.running = set()
        self.logs = {}
        self.digest = "sha256:" + "ab" * 32

    def run(self, image, name, args, gpus=True, network_host=True, mounts=(), env=None, entrypoint=None):
        self.calls.append(("run", image, name, list(args)))
        self.running.add(name)
        return "cid-" + name

    def stop(self, name, timeout=30):
        self.calls.append(("stop", name))
        self.running.discard(name)

    def is_running(self, name):
        return name in self.running

    def image_digest(self, image):
        return self.digest

    def pip_freeze(self, image):
        return "vllm==0.29.0\ntorch==2.9.0\n"

    def container_logs(self, name):
        return self.logs.get(name, "")


class FakeHTTP:
    """Minimal stand-in for the engine's HTTP surface."""

    def __init__(self):
        self.healthy_after = 0   # number of health polls before 200
        self.polls = 0
        self.posts = []
        self.metrics_text = ""

    def get(self, url, timeout=5):
        if url.endswith("/health"):
            self.polls += 1
            ok = self.polls > self.healthy_after
            return SimpleNamespace(status_code=200 if ok else 503, text="")
        if url.endswith("/metrics"):
            return SimpleNamespace(status_code=200, text=self.metrics_text)
        return SimpleNamespace(status_code=404, text="")

    def post(self, url, json=None, timeout=60):
        self.posts.append((url, json))
        if url.endswith("/v1/completions"):
            return SimpleNamespace(status_code=200, json=lambda: {"choices": [{"text": "ok"}], "usage": {"prompt_tokens": 265, "completion_tokens": 4}})
        return SimpleNamespace(status_code=200, json=lambda: {})


@pytest.fixture
def fake_docker():
    return FakeDocker()


@pytest.fixture
def fake_http():
    return FakeHTTP()


@pytest.fixture
def client_json():
    """Shape of `vllm bench serve --save-result --save-detailed` output (keys verified in Task 1)."""
    return {
        "backend": "openai", "model_id": "Qwen/Qwen2.5-3B-Instruct-AWQ", "num_prompts": 4, "duration": 2.0,
        "completed": 4, "total_input_tokens": 1000, "total_output_tokens": 200,
        "request_throughput": 2.0, "output_throughput": 100.0,
        "mean_ttft_ms": 50.0, "median_ttft_ms": 48.0, "p99_ttft_ms": 60.0,
        "mean_tpot_ms": 20.0, "median_tpot_ms": 19.0, "p99_tpot_ms": 25.0,
        "mean_itl_ms": 20.0, "median_itl_ms": 19.0, "p99_itl_ms": 25.0,
        "input_lens": [250, 250, 250, 250], "output_lens": [50, 50, 50, 50],
        "ttfts": [0.048, 0.050, 0.049, 0.053], "itls": [[0.02] * 49, [0.02] * 49, [0.02] * 49, [0.02] * 49],
        "generated_texts": ["a", "b", "c", "d"], "errors": ["", "", "", ""],
    }
```

`tests/bench/runner/test_engine.py`:
```python
import pytest

from bench.runner.engine import ENGINES, EngineSpec
from bench.runner.config import RunConfig


def _cfg(**over):
    base = dict(run_id="r1", sweep_id="s1", phase="p0b", rq_tag="cal", engine="vllm", image="vllm/vllm-openai:v0.29.0",
                model="Qwen/Qwen2.5-3B-Instruct-AWQ", model_revision="abc", quantization="awq",
                draft_model=None, draft_revision=None, draft_quantization=None, spec_method="off", spec_k=None,
                ngram_lookup_max=None, workload="A", trace_file="bench/traces/chat_v1.jsonl", trace_version=1,
                trace_sha256="0" * 64, load_mode="concurrency", concurrency=8, request_rate=None, burstiness=1.0,
                num_prompts=200, cache_state="cold", prefix_caching=True, max_model_len=2048, max_num_seqs=64,
                chunked_prefill_tokens=1024, cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
                sampling={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.0},
                ignore_eos=True, max_tokens_cap=None, warmup_requests=8, cooldown_temp_c=55, cooldown_min_s=60,
                gpu_memory_utilization=0.85, free_vram_mb_at_start=6000, seed=0, extra_body={})
    base.update(over)
    return RunConfig(**base)


def test_engines_are_verified_and_complete():
    for name, e in ENGINES.items():
        assert isinstance(e, EngineSpec) and e.name == name
        assert e.verified_against, f"{name} table must carry the image tag it was verified against (Task 1)"
        for k in ("kv_usage", "running", "waiting", "spec_accepted", "spec_draft", "prefix_hits", "prefix_queries"):
            assert k in e.metric_names, (name, k)
        assert e.health_path.startswith("/") and e.reset_cache_path.startswith("/") and e.metrics_path == "/metrics"


def test_vllm_launch_args_off_and_draft():
    e = ENGINES["vllm"]
    args = e.build_launch_args(_cfg(), mem_fraction=0.83)
    joined = " ".join(args)
    assert "--model Qwen/Qwen2.5-3B-Instruct-AWQ" in joined and "--revision abc" in joined
    assert "--gpu-memory-utilization 0.83" in joined and "--max-model-len 2048" in joined
    assert "--enforce-eager" not in joined and "--seed 0" in joined
    assert "speculative" not in joined
    args = e.build_launch_args(_cfg(spec_method="draft", draft_model="Qwen/Qwen2.5-0.5B-Instruct", draft_revision="def", spec_k=3), mem_fraction=0.83)
    joined = " ".join(args)
    assert "draft_model" in joined and "Qwen2.5-0.5B-Instruct" in joined and '"num_speculative_tokens": 3' in joined.replace("'", '"')


def test_vllm_prefix_caching_off_and_ngram():
    e = ENGINES["vllm"]
    joined = " ".join(e.build_launch_args(_cfg(prefix_caching=False, spec_method="ngram", spec_k=5, ngram_lookup_max=4), 0.8))
    assert "prefix-caching" in joined          # exact flag spelled per Task 1 doc; test only checks presence
    assert "ngram" in joined and "prompt_lookup_max" in joined.replace("-", "_")


def test_sglang_launch_args_standalone():
    e = ENGINES["sglang"]
    joined = " ".join(e.build_launch_args(_cfg(engine="sglang", spec_method="draft", draft_model="Qwen/Qwen2.5-0.5B-Instruct", draft_revision="def", spec_k=3), 0.8))
    assert "--model-path Qwen/Qwen2.5-3B-Instruct-AWQ" in joined and "--mem-fraction-static 0.8" in joined
    assert "STANDALONE" in joined and "Qwen2.5-0.5B-Instruct" in joined
    assert "--context-length 2048" in joined


def test_unknown_spec_method_raises():
    with pytest.raises(ValueError):
        ENGINES["vllm"].build_launch_args(_cfg(spec_method="eagle"), 0.8)
```

`tests/bench/runner/test_config.py`:
```python
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


def test_config_round_trips_through_dict():
    c = _cfg()
    assert RunConfig(**c.to_dict()) == c
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/bench/runner -q` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `config.py`**

```python
"""One run's fully resolved configuration — every field lands in results/<run>/config.yaml."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

SPEC_METHODS = ("off", "ngram", "draft")


@dataclass
class RunConfig:
    run_id: str
    sweep_id: str
    phase: str
    rq_tag: str
    engine: str
    image: str
    model: str
    model_revision: str
    quantization: str | None
    draft_model: str | None
    draft_revision: str | None
    draft_quantization: str | None
    spec_method: str
    spec_k: int | None
    ngram_lookup_max: int | None
    workload: str
    trace_file: str
    trace_version: int
    trace_sha256: str
    load_mode: str
    concurrency: int | None
    request_rate: float | None
    burstiness: float
    num_prompts: int
    cache_state: str
    prefix_caching: bool
    max_model_len: int
    max_num_seqs: int
    chunked_prefill_tokens: int
    cudagraph_capture_sizes: list[int]
    sampling: dict
    ignore_eos: bool
    max_tokens_cap: int | None
    warmup_requests: int
    cooldown_temp_c: int
    cooldown_min_s: int
    gpu_memory_utilization: float
    free_vram_mb_at_start: int
    seed: int
    extra_body: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_gpu_memory_fraction(total_mb: int, host_used_mb: int, headroom_mb: int = 256) -> float:
    """Fraction of TOTAL memory the engine may claim, derived from what is
    actually free (spec §3.1: never 0.9 by reflex on a shared GPU)."""
    usable = max(0, total_mb - host_used_mb - headroom_mb)
    return int(usable / total_mb * 100) / 100


def validate(cfg: RunConfig) -> None:
    from .engine import ENGINES  # local import: engine.py imports RunConfig for typing
    if cfg.engine not in ENGINES:
        raise ValueError(f"unknown engine {cfg.engine!r}")
    if not ENGINES[cfg.engine].verified_against:
        raise ValueError(f"engine table for {cfg.engine} is unverified; run Task 1 first")
    if cfg.spec_method not in SPEC_METHODS:
        raise ValueError(f"spec_method must be one of {SPEC_METHODS}")
    if cfg.spec_method != "off" and not cfg.spec_k:
        raise ValueError("spec_k is required when speculation is on")
    if cfg.spec_method == "draft" and not cfg.draft_model:
        raise ValueError("draft_model is required for spec_method=draft")
    if not cfg.cudagraph_capture_sizes:
        raise ValueError("cudagraph_capture_sizes must be explicit (enforce-eager is prohibited)")
    if cfg.load_mode == "concurrency" and not cfg.concurrency:
        raise ValueError("concurrency required for load_mode=concurrency")
    if cfg.load_mode == "poisson" and not cfg.request_rate:
        raise ValueError("request_rate required for load_mode=poisson")
    p = Path(cfg.trace_file)
    if not p.exists():
        raise ValueError(f"trace file missing: {p}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != cfg.trace_sha256:
        raise ValueError(f"trace sha256 mismatch for {p}: config {cfg.trace_sha256[:12]} vs file {actual[:12]}")
```

- [ ] **Step 4: Implement `engine.py`** — the values marked `# T1` are copied from `bench/docs/p0b-engine-verification.md`; the implementer must not guess them.

```python
"""Per-engine data: image, launch-arg builder, HTTP routes, metric names.
Everything marked T1 is copied verbatim from bench/docs/p0b-engine-verification.md."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .config import RunConfig


@dataclass(frozen=True)
class EngineSpec:
    name: str
    image: str
    verified_against: str | None
    port: int
    health_path: str
    reset_cache_path: str
    reset_cache_method: str
    metrics_path: str
    metric_names: dict[str, str]
    _launch: Callable[[RunConfig, float], list[str]] = field(repr=False)

    def build_launch_args(self, cfg: RunConfig, mem_fraction: float) -> list[str]:
        return self._launch(cfg, mem_fraction)

    def served_model_name(self, cfg: RunConfig) -> str:
        return cfg.model


def _vllm_args(cfg: RunConfig, mem: float) -> list[str]:
    args = ["--model", cfg.model, "--revision", cfg.model_revision,
            "--max-model-len", str(cfg.max_model_len), "--max-num-seqs", str(cfg.max_num_seqs),
            "--max-num-batched-tokens", str(cfg.chunked_prefill_tokens),           # T1: chunked prefill size
            "--gpu-memory-utilization", f"{mem:.2f}", "--seed", str(cfg.seed), "--port", "8000",
            "--compilation-config", json.dumps({"cudagraph_capture_sizes": cfg.cudagraph_capture_sizes}),  # T1
            "--generation-config", "vllm"]                                        # T1: ignore model generation_config.json
    if cfg.quantization:
        args += ["--quantization", cfg.quantization]
    args += ["--enable-prefix-caching" if cfg.prefix_caching else "--no-enable-prefix-caching"]  # T1
    if cfg.spec_method == "draft":
        spec = {"method": "draft_model", "model": cfg.draft_model, "num_speculative_tokens": cfg.spec_k}  # T1
        if cfg.draft_revision:
            spec["revision"] = cfg.draft_revision
        args += ["--speculative-config", json.dumps(spec)]
    elif cfg.spec_method == "ngram":
        args += ["--speculative-config", json.dumps({"method": "ngram", "num_speculative_tokens": cfg.spec_k,
                                                     "prompt_lookup_max": cfg.ngram_lookup_max or 4})]  # T1
    elif cfg.spec_method != "off":
        raise ValueError(f"unsupported spec_method {cfg.spec_method!r} for vllm")
    return args


def _sglang_args(cfg: RunConfig, mem: float) -> list[str]:
    args = ["python", "-m", "sglang.launch_server", "--model-path", cfg.model, "--revision", cfg.model_revision,  # T1
            "--context-length", str(cfg.max_model_len), "--max-running-requests", str(cfg.max_num_seqs),
            "--chunked-prefill-size", str(cfg.chunked_prefill_tokens),
            "--mem-fraction-static", f"{mem:.2f}", "--random-seed", str(cfg.seed), "--port", "30000",
            "--cuda-graph-bs", *[str(b) for b in cfg.cudagraph_capture_sizes],   # T1
            "--enable-metrics", "--host", "0.0.0.0"]
    if cfg.quantization:
        args += ["--quantization", cfg.quantization]
    if not cfg.prefix_caching:
        args += ["--disable-radix-cache"]                                          # T1
    if cfg.spec_method == "draft":
        args += ["--speculative-algorithm", "STANDALONE", "--speculative-draft-model-path", cfg.draft_model,  # T1
                 "--speculative-num-steps", str(cfg.spec_k), "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", str(cfg.spec_k + 1)]
    elif cfg.spec_method == "ngram":
        args += ["--speculative-algorithm", "NGRAM", "--speculative-num-draft-tokens", str(cfg.spec_k + 1)]  # T1
    elif cfg.spec_method != "off":
        raise ValueError(f"unsupported spec_method {cfg.spec_method!r} for sglang")
    return args


ENGINES: dict[str, EngineSpec] = {
    "vllm": EngineSpec(
        name="vllm", image="vllm/vllm-openai:v0.29.0", verified_against="vllm/vllm-openai:v0.29.0",  # T1
        port=8000, health_path="/health", reset_cache_path="/reset_prefix_cache", reset_cache_method="POST",  # T1
        metrics_path="/metrics",
        metric_names={  # T1 — exact names from the verification doc
            "kv_usage": "vllm:gpu_cache_usage_perc", "running": "vllm:num_requests_running",
            "waiting": "vllm:num_requests_waiting", "spec_accepted": "vllm:spec_decode_num_accepted_tokens_total",
            "spec_draft": "vllm:spec_decode_num_draft_tokens_total", "spec_emitted": "vllm:spec_decode_num_emitted_tokens_total",
            "prefix_hits": "vllm:prefix_cache_hits_total", "prefix_queries": "vllm:prefix_cache_queries_total",
        },
        _launch=_vllm_args),
    "sglang": EngineSpec(
        name="sglang", image="lmsysorg/sglang:v0.5.20-runtime", verified_against="lmsysorg/sglang:v0.5.20-runtime",  # T1
        port=30000, health_path="/health", reset_cache_path="/flush_cache", reset_cache_method="POST",  # T1
        metrics_path="/metrics",
        metric_names={  # T1
            "kv_usage": "sglang:token_usage", "running": "sglang:num_running_reqs", "waiting": "sglang:num_queue_reqs",
            "spec_accepted": "sglang:spec_accept_length", "spec_draft": "sglang:spec_num_draft_tokens",
            "spec_emitted": "sglang:spec_accept_rate", "prefix_hits": "sglang:cache_hit_rate", "prefix_queries": "sglang:cache_hit_rate",
        },
        _launch=_sglang_args),
}
```
If Task 1 found that a name differs, the implementer edits the table **and** the verification doc's "decisions" section stays the source of truth. If Task 1 dropped an arm (e.g. SGLang NGRAM unavailable), the builder raises `ValueError("<engine> lacks <method> at <tag>; see p0b-engine-verification.md")` and the test asserting that arm is adjusted with the reason in the commit message.

- [ ] **Step 5: Run tests** — `pytest tests/bench/runner -q` → all pass.

- [ ] **Step 6: Commit** — `git add bench/runner/__init__.py bench/runner/engine.py bench/runner/config.py tests/bench/runner/ && git commit -m "bench: engine tables and run config for the sweep runner"`

---

### Task 3: Sweep expansion and state

**Files:**
- Create: `bench/runner/sweep.py`, `bench/runner/state.py`, `bench/configs/base.yaml`, `bench/configs/p0b_variance.yaml`
- Test: `tests/bench/runner/test_sweep.py`, `tests/bench/runner/test_state.py`

**Interfaces:**
- `load_sweep(path) -> dict` (YAML; `base:` file reference merged under the sweep's own keys); `expand(sweep: dict, sweep_id: str) -> list[RunConfig]` — crosses every list-valued key under `axes:` (order-preserving), applies `reps:` (each config repeated with `rep` 0..N-1 folded into `run_id` = `f"{sweep_id}-{index:04d}-r{rep}"`), resolves `trace_file`/`trace_sha256`/`trace_version` from `workload` via `WORKLOAD_TRACES = {"A": "chat", "B": "summarization", "C": "structured", "MT-shallow": "multiturn_shallow", ...}` and the trace's `.meta.json`; `schedule(configs, seed) -> list[RunConfig]` shuffled with `random.Random(seed)`.
- `SweepState(path)` with `load()`, `mark(run_id, status, reason=None)`, `pending() -> list[str]`, `requeue(run_id) -> bool` (False when the sweep's retry budget `max_retries_total` is exhausted), `to_dict()`; persisted as `state.json` after every mutation; statuses `pending|running|done|invalid|requeued`.

- [ ] **Step 1: Write the YAML files**

`bench/configs/base.yaml`:
```yaml
# Shared defaults for every sweep. Every value here is recorded per run in config.yaml.
phase: p0b
model: Qwen/Qwen2.5-3B-Instruct-AWQ
model_revision: "<T1 sha>"
quantization: awq
draft_model: Qwen/Qwen2.5-0.5B-Instruct
draft_revision: "<T1 sha>"
draft_quantization: null
spec_method: off
spec_k: null
ngram_lookup_max: 4
load_mode: concurrency
concurrency: 8
request_rate: null
burstiness: 1.0
num_prompts: 200
cache_state: cold
prefix_caching: true
max_model_len: 2048
max_num_seqs: 64
chunked_prefill_tokens: 1024
cudagraph_capture_sizes: [1, 2, 4, 8, 16, 32, 64]
sampling: {temperature: 1.0, top_p: 1.0, top_k: -1, repetition_penalty: 1.0}
ignore_eos: true
max_tokens_cap: null
warmup_requests: 8
cooldown_temp_c: 55
cooldown_min_s: 60
gpu_headroom_mb: 256
seed: 0
extra_body: {}
```

`bench/configs/p0b_variance.yaml`:
```yaml
base: bench/configs/base.yaml
sweep_id: p0b-variance
rq_tag: calibration
schedule_seed: 20260919
max_retries_total: 3
reps: 10
axes:
  engine: [vllm]
  workload: [A]
  concurrency: [8]
```

- [ ] **Step 2: Write the failing tests**

`tests/bench/runner/test_sweep.py`:
```python
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
```

`tests/bench/runner/test_state.py`:
```python
import json

from bench.runner.state import SweepState


def test_state_persists_and_resumes(tmp_path):
    p = tmp_path / "state.json"
    s = SweepState(p, run_ids=["a", "b", "c"], max_retries_total=1)
    assert s.pending() == ["a", "b", "c"]
    s.mark("a", "running"); s.mark("a", "done")
    s.mark("b", "invalid", reason="acceptance zero")
    assert s.requeue("b") is True and s.pending() == ["c", "b"]        # re-queued to the END
    s.mark("b", "invalid", reason="again")
    assert s.requeue("b") is False                                       # budget exhausted
    s2 = SweepState.load(p)
    assert s2.pending() == ["c"] and s2.to_dict()["runs"]["a"]["status"] == "done"
    assert s2.to_dict()["retries_used"] == 1 and s2.to_dict()["runs"]["b"]["reason"] == "again"
    assert json.loads(p.read_text())["max_retries_total"] == 1
```

- [ ] **Step 3: Run tests to verify they fail.**

- [ ] **Step 4: Implement `sweep.py` and `state.py`**

```python
# bench/runner/sweep.py
"""Sweep YAML → resolved RunConfigs → seeded schedule. Configs are data; nothing here launches anything."""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path

import yaml

from .config import RunConfig

WORKLOAD_TRACES = {"A": "chat", "B": "summarization", "C": "structured",
                   "MT-shallow": "multiturn_shallow", "MT-medium": "multiturn_medium", "MT-deep": "multiturn_deep"}


def load_sweep(path: str | Path) -> dict:
    sweep = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(sweep["base"]).read_text(encoding="utf-8")) if "base" in sweep else {}
    merged = {**base, **{k: v for k, v in sweep.items() if k != "base"}}
    merged["_source"] = str(path)
    return merged


def _trace_for(workload: str, traces_dir: Path, version: int = 1) -> tuple[str, str, int]:
    name = WORKLOAD_TRACES[workload]
    trace = traces_dir / f"{name}_v{version}.jsonl"
    meta = json.loads((traces_dir / f"{name}_v{version}.meta.json").read_text(encoding="utf-8"))
    return str(trace), meta["trace_sha256"], version


def expand(sweep: dict, sweep_id: str) -> list[RunConfig]:
    axes = sweep.get("axes", {})
    keys = list(axes)
    reps = int(sweep.get("reps", 1))
    traces_dir = Path(sweep.get("traces_dir", "bench/traces"))
    fixed = {k: v for k, v in sweep.items() if k not in ("axes", "reps", "base", "sweep_id", "schedule_seed",
                                                          "max_retries_total", "traces_dir", "_source", "gpu_headroom_mb")}
    out: list[RunConfig] = []
    for index, combo in enumerate(itertools.product(*[axes[k] for k in keys])):
        point = {**fixed, **dict(zip(keys, combo))}
        trace_file, sha, ver = _trace_for(point["workload"], traces_dir, int(point.get("trace_version", 1)))
        point.pop("trace_version", None)
        for rep in range(reps):
            out.append(RunConfig(run_id=f"{sweep_id}-{index:04d}-r{rep}", sweep_id=sweep_id,
                                 image="",  # filled by lifecycle from ENGINES[engine].image
                                 trace_file=trace_file, trace_sha256=sha, trace_version=ver,
                                 gpu_memory_utilization=0.0, free_vram_mb_at_start=0,   # resolved at run time
                                 concurrency=point.get("concurrency"), **{k: v for k, v in point.items() if k != "concurrency"}))
    return out


def schedule(configs: list[RunConfig], seed: int) -> list[RunConfig]:
    order = list(configs)
    random.Random(seed).shuffle(order)
    return order
```
(The implementer reconciles `RunConfig`'s required fields with what `base.yaml` supplies; `phase`, `rq_tag` come from the sweep/base; unknown keys raise `TypeError` from the dataclass — that is the intended strictness.)

```python
# bench/runner/state.py
"""Per-sweep progress, persisted after every mutation so `resume` never repeats a finished run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class SweepState:
    def __init__(self, path: Path, run_ids: list[str] | None = None, max_retries_total: int = 0):
        self.path = Path(path)
        self.max_retries_total = max_retries_total
        self.retries_used = 0
        self.order: list[str] = list(run_ids or [])
        self.runs: dict[str, dict] = {r: {"status": "pending", "reason": None, "attempts": 0} for r in self.order}
        if run_ids is not None:
            self._save()

    @classmethod
    def load(cls, path: Path) -> "SweepState":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        s = cls(path, None, d["max_retries_total"])
        s.retries_used, s.order, s.runs = d["retries_used"], d["order"], d["runs"]
        return s

    def mark(self, run_id: str, status: str, reason: str | None = None) -> None:
        r = self.runs[run_id]
        r["status"] = status
        r["reason"] = reason
        r["updated_at"] = datetime.now(timezone.utc).isoformat()
        if status == "running":
            r["attempts"] += 1
        self._save()

    def requeue(self, run_id: str) -> bool:
        if self.retries_used >= self.max_retries_total:
            return False
        self.retries_used += 1
        self.order.remove(run_id)
        self.order.append(run_id)
        self.runs[run_id]["status"] = "requeued"
        self._save()
        return True

    def pending(self) -> list[str]:
        return [r for r in self.order if self.runs[r]["status"] in ("pending", "requeued")]

    def to_dict(self) -> dict:
        return {"max_retries_total": self.max_retries_total, "retries_used": self.retries_used,
                "order": self.order, "runs": self.runs}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
```

- [ ] **Step 5: Run tests; commit** — `git add bench/runner/sweep.py bench/runner/state.py bench/configs/base.yaml bench/configs/p0b_variance.yaml tests/bench/runner/test_sweep.py tests/bench/runner/test_state.py && git commit -m "bench: sweep expansion, seeded scheduling and resumable sweep state"`

---

### Task 4: Docker wrapper and readiness

**Files:**
- Create: `bench/runner/docker.py`, `bench/runner/readiness.py`
- Test: `tests/bench/runner/test_readiness.py` (docker.py is a thin subprocess wrapper; tested only through `FakeDocker` in lifecycle tests, plus one smoke assertion that its command lines are well-formed)

**Interfaces:**
- `class Docker` with the same method signatures as `FakeDocker`: `run(image, name, args, gpus=True, network_host=True, mounts=(), env=None, entrypoint=None) -> container_id` (builds `docker run -d --name … [--gpus all] [--network host] -v … -e … [--entrypoint …] image args…`), `stop(name, timeout=30)` (`docker stop -t` then `docker rm -f`), `is_running(name)`, `image_digest(image)`, `pip_freeze(image)` (`docker run --rm --entrypoint pip image freeze`), `container_logs(name)`; `command_lines: list[list[str]]` recorded for tests.
- `wait_healthy(http, base_url, health_path, timeout_s, poll_s=2.0, docker=None, container=None) -> float` (seconds waited; raises `TimeoutError` including the last 40 lines of container logs when `docker`/`container` given); `warmup(http, base_url, model, prompts: list[str], max_tokens=8) -> int` (POSTs `/v1/completions` per prompt; returns count); `reset_cache(http, base_url, spec: EngineSpec) -> None` (POST/GET per `reset_cache_method`; raises on non-2xx).

- [ ] **Step 1: Failing tests**

`tests/bench/runner/test_readiness.py`:
```python
import pytest

from bench.runner.engine import ENGINES
from bench.runner.readiness import reset_cache, wait_healthy, warmup


def test_wait_healthy_polls_until_200(fake_http):
    fake_http.healthy_after = 3
    waited = wait_healthy(fake_http, "http://localhost:8000", "/health", timeout_s=10, poll_s=0.0)
    assert fake_http.polls == 4 and waited >= 0


def test_wait_healthy_times_out_with_logs(fake_http, fake_docker):
    fake_http.healthy_after = 10**6
    fake_docker.logs["c"] = "\n".join(f"line{i}" for i in range(100))
    with pytest.raises(TimeoutError) as e:
        wait_healthy(fake_http, "http://localhost:8000", "/health", timeout_s=0.01, poll_s=0.0, docker=fake_docker, container="c")
    assert "line99" in str(e.value) and "line10" not in str(e.value)   # last 40 lines only


def test_warmup_posts_each_prompt_to_completions(fake_http):
    n = warmup(fake_http, "http://localhost:8000", "m", ["p1", "p2", "p3"], max_tokens=4)
    assert n == 3
    assert [u for u, _ in fake_http.posts] == ["http://localhost:8000/v1/completions"] * 3
    assert fake_http.posts[0][1] == {"model": "m", "prompt": "p1", "max_tokens": 4, "temperature": 0}


def test_reset_cache_uses_engine_route(fake_http):
    reset_cache(fake_http, "http://localhost:8000", ENGINES["vllm"])
    assert fake_http.posts[-1][0] == "http://localhost:8000" + ENGINES["vllm"].reset_cache_path
```

- [ ] **Step 2: Implement**

```python
# bench/runner/docker.py
"""Thin docker CLI wrapper. Every method maps to one command line; nothing here knows about engines."""
from __future__ import annotations

import subprocess


class Docker:
    def __init__(self):
        self.command_lines: list[list[str]] = []

    def _run(self, *cmd: str, check: bool = True) -> str:
        self.command_lines.append(list(cmd))
        return subprocess.run(cmd, check=check, capture_output=True, text=True).stdout.strip()

    def run(self, image, name, args, gpus=True, network_host=True, mounts=(), env=None, entrypoint=None) -> str:
        cmd = ["docker", "run", "-d", "--name", name, "--ipc=host"]
        if gpus:
            cmd += ["--gpus", "all"]
        if network_host:
            cmd += ["--network", "host"]
        for host_path, container_path, mode in mounts:
            cmd += ["-v", f"{host_path}:{container_path}:{mode}"]
        for k, v in (env or {}).items():
            cmd += ["-e", f"{k}={v}"]
        if entrypoint:
            cmd += ["--entrypoint", entrypoint]
        return self._run(*cmd, image, *args)

    def stop(self, name: str, timeout: int = 30) -> None:
        self._run("docker", "stop", "-t", str(timeout), name, check=False)
        self._run("docker", "rm", "-f", name, check=False)

    def is_running(self, name: str) -> bool:
        return self._run("docker", "inspect", "-f", "{{.State.Running}}", name, check=False) == "true"

    def image_digest(self, image: str) -> str:
        return self._run("docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}")

    def pip_freeze(self, image: str) -> str:
        return self._run("docker", "run", "--rm", "--entrypoint", "pip", image, "freeze")

    def container_logs(self, name: str) -> str:
        return subprocess.run(["docker", "logs", name], capture_output=True, text=True).stdout + \
               subprocess.run(["docker", "logs", name], capture_output=True, text=True).stderr
```

```python
# bench/runner/readiness.py
"""Server readiness: 'port open' is not 'model loaded and warm' (spec §4 item 5)."""
from __future__ import annotations

import time

from .engine import EngineSpec


def wait_healthy(http, base_url, health_path, timeout_s, poll_s=2.0, docker=None, container=None) -> float:
    t0 = time.monotonic()
    while True:
        try:
            if http.get(base_url + health_path, timeout=5).status_code == 200:
                return time.monotonic() - t0
        except Exception:
            pass
        if time.monotonic() - t0 > timeout_s:
            tail = ""
            if docker is not None and container:
                tail = "\n".join(docker.container_logs(container).splitlines()[-40:])
            raise TimeoutError(f"{base_url}{health_path} not healthy after {timeout_s}s\n{tail}")
        time.sleep(poll_s)


def warmup(http, base_url, model, prompts, max_tokens=8) -> int:
    n = 0
    for p in prompts:
        r = http.post(f"{base_url}/v1/completions", json={"model": model, "prompt": p, "max_tokens": max_tokens, "temperature": 0}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"warmup request failed: {r.status_code}")
        n += 1
    return n


def reset_cache(http, base_url, spec: EngineSpec) -> None:
    url = base_url + spec.reset_cache_path
    r = http.post(url, json={}, timeout=30) if spec.reset_cache_method == "POST" else http.get(url, timeout=30)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"cache reset failed: {spec.reset_cache_method} {url} -> {r.status_code}")
```

- [ ] **Step 3: Run tests; commit** — `git commit -m "bench: docker wrapper and server readiness (health, warmup, cache reset)"`

---

### Task 5: GPU monitor, metrics scraper, environment capture

**Files:**
- Create: `bench/runner/gpu_monitor.py`, `bench/runner/metrics_scraper.py`, `bench/runner/env_capture.py`
- Test: `tests/bench/runner/test_gpu_monitor.py`, `test_metrics_scraper.py`, `test_env_capture.py`

**Interfaces:**
- `read_gpu(nvidia_smi_cmd: list[str]) -> dict` runs `<cmd> --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader,nounits` and returns keys `used_mb,total_mb,sm_util,mem_util,sm_clock,mem_clock,temp_c,power_w,throttle_reasons` (hex string; `power_w` may be `None`).
- `WSL_SMI = ["nvidia-smi"]`, `WIN_SMI = ["/mnt/c/Windows/System32/nvidia-smi.exe"]`.
- `GpuSampler(out_path, interval_s=1.0, wsl_cmd=WSL_SMI, win_cmd=WIN_SMI, reader=read_gpu)` thread with `start()/stop()`; each sample line: `{"t": monotonic, "wall": iso, "used_total_mb": win.used_mb, "used_ours_mb": wsl.used_mb, "used_host_mb": max(0, win.used_mb - wsl.used_mb), ...sm/mem clocks, temp, power (from win), throttle_reasons, "wsl_error"/"win_error" strings when a reader fails}` — spec §7 `gpu_samples.jsonl`. **Design note for the writeup:** WSL-side `nvidia-smi` sees only this VM's usage; the Windows-side one sees the whole card, so the host share is the difference. Both were verified callable on 2026-09-18.
- `parse_prometheus(text) -> dict[str, float]` (name{labels} → value; keep the label string in the key), `MetricsScraper(http, base_url, spec, out_path, interval_s=1.0)` thread writing `{"t", "wall", "raw": {name: value}, "kv_usage", "running", "waiting", "spec_accepted", "spec_draft", "prefix_hits", "prefix_queries"}` resolved through `spec.metric_names` (missing → `None`); `summarise(samples) -> dict` (mean acceptance rate = Δaccepted/Δdraft over the run, peak kv_usage, max waiting).
- `capture_env(cfg, docker, spec, launch_args, client_image, extra) -> dict` per spec §7 `env.json`: image + digest; `pip_freeze` (from the image, cached per image per sweep); host `nvidia-smi` driver/CUDA lines (WSL and Windows); `power_max_limit_w` (Windows); `clocks_pinned`; WSL kernel (`uname -r`); docker version; `client_image`, `client_output_schema_version`; `bench_git_sha` (reuse `bench.capture.cli.git_sha`); `windows_power_mode` (from `powercfg /getactivescheme` via `/mnt/c/Windows/System32/powercfg.exe`, best effort); `launch_cmd` verbatim.

- [ ] **Step 1: Failing tests** (fakes: a `reader` returning fixed dicts; Prometheus text fixture with `vllm:` names; `FakeDocker`)

`tests/bench/runner/test_gpu_monitor.py`:
```python
import json, time

from bench.runner.gpu_monitor import GpuSampler, read_gpu


def test_read_gpu_parses_csv(monkeypatch):
    import bench.runner.gpu_monitor as gm
    monkeypatch.setattr(gm.subprocess, "check_output", lambda cmd, text=True, timeout=5: "512, 6141, 87, 45, 2055, 7001, 61, 63.20, 0x0000000000000004\n")
    d = read_gpu(["nvidia-smi"])
    assert d == {"used_mb": 512, "total_mb": 6141, "sm_util": 87, "mem_util": 45, "sm_clock": 2055, "mem_clock": 7001,
                 "temp_c": 61, "power_w": 63.2, "throttle_reasons": "0x0000000000000004"}


def test_read_gpu_handles_na(monkeypatch):
    import bench.runner.gpu_monitor as gm
    monkeypatch.setattr(gm.subprocess, "check_output", lambda cmd, text=True, timeout=5: "0, 6141, 0, 0, 2055, 7001, 49, [N/A], 0x0\n")
    assert read_gpu(["x"])["power_w"] is None


def test_sampler_writes_host_and_ours(tmp_path):
    calls = {"n": 0}
    def reader(cmd):
        calls["n"] += 1
        return {"used_mb": 1000 if "exe" in cmd[0] else 700, "total_mb": 6141, "sm_util": 1, "mem_util": 1, "sm_clock": 1,
                "mem_clock": 1, "temp_c": 50, "power_w": 20.0, "throttle_reasons": "0x0"}
    s = GpuSampler(tmp_path / "g.jsonl", interval_s=0.01, wsl_cmd=["nvidia-smi"], win_cmd=["nvidia-smi.exe"], reader=reader)
    s.start(); time.sleep(0.1); s.stop()
    rows = [json.loads(l) for l in (tmp_path / "g.jsonl").read_text().splitlines()]
    assert len(rows) >= 3
    assert rows[0]["used_total_mb"] == 1000 and rows[0]["used_ours_mb"] == 700 and rows[0]["used_host_mb"] == 300
    assert "t" in rows[0] and "wall" in rows[0] and rows[0]["power_w"] == 20.0


def test_sampler_records_reader_errors_instead_of_dying(tmp_path):
    def reader(cmd):
        if "exe" in cmd[0]:
            raise RuntimeError("no windows smi")
        return {"used_mb": 1, "total_mb": 2, "sm_util": 0, "mem_util": 0, "sm_clock": 0, "mem_clock": 0, "temp_c": 0, "power_w": None, "throttle_reasons": "0x0"}
    s = GpuSampler(tmp_path / "g.jsonl", interval_s=0.01, wsl_cmd=["a"], win_cmd=["b.exe"], reader=reader)
    s.start(); time.sleep(0.05); s.stop()
    row = json.loads((tmp_path / "g.jsonl").read_text().splitlines()[0])
    assert row["used_ours_mb"] == 1 and row["used_total_mb"] is None and "no windows smi" in row["win_error"]
```

`tests/bench/runner/test_metrics_scraper.py`:
```python
import json, time

from bench.runner.engine import ENGINES
from bench.runner.metrics_scraper import MetricsScraper, parse_prometheus, summarise

PROM = """# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage.
vllm:gpu_cache_usage_perc{model_name="m"} 0.42
vllm:num_requests_running{model_name="m"} 8.0
vllm:num_requests_waiting{model_name="m"} 3.0
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 120.0
vllm:spec_decode_num_draft_tokens_total{model_name="m"} 200.0
vllm:prefix_cache_hits_total{model_name="m"} 50.0
vllm:prefix_cache_queries_total{model_name="m"} 100.0
"""


def test_parse_prometheus_keeps_labels_and_values():
    d = parse_prometheus(PROM)
    assert d['vllm:gpu_cache_usage_perc{model_name="m"}'] == 0.42 and len(d) == 7


def test_scraper_resolves_engine_names_and_summarises(fake_http, tmp_path):
    fake_http.metrics_text = PROM
    s = MetricsScraper(fake_http, "http://localhost:8000", ENGINES["vllm"], tmp_path / "m.jsonl", interval_s=0.01)
    s.start(); time.sleep(0.05); s.stop()
    rows = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    assert rows[0]["kv_usage"] == 0.42 and rows[0]["running"] == 8 and rows[0]["spec_accepted"] == 120
    rows[-1]["spec_accepted"], rows[-1]["spec_draft"] = 320.0, 400.0     # simulate progress over the run
    summ = summarise(rows)
    assert summ["acceptance_rate_mean"] == (320 - 120) / (400 - 200) and summ["kv_usage_peak"] == 0.42 and summ["waiting_max"] == 3


def test_summarise_without_spec_counters_gives_none():
    rows = [{"kv_usage": 0.1, "running": 1, "waiting": 0, "spec_accepted": None, "spec_draft": None, "prefix_hits": None, "prefix_queries": None}]
    assert summarise(rows)["acceptance_rate_mean"] is None
```

`tests/bench/runner/test_env_capture.py`:
```python
from bench.runner.engine import ENGINES
from bench.runner.env_capture import capture_env
from tests.bench.runner.test_engine import _cfg


def test_capture_env_has_the_spec_fields(fake_docker, monkeypatch):
    import bench.runner.env_capture as ec
    monkeypatch.setattr(ec, "_host_lines", lambda: {"wsl_nvidia_smi": "Driver 610", "win_nvidia_smi": "Driver 610.88", "power_max_limit_w": 100.0,
                                                    "uname": "6.6", "docker_version": "28.0.4", "windows_power_mode": "High performance"})
    env = capture_env(_cfg(), fake_docker, ENGINES["vllm"], ["--model", "m"], client_image="vllm/vllm-openai:v0.29.0",
                      extra={"clocks_pinned": False, "client_output_schema_version": "v0.29.0-save-detailed"})
    for k in ("image", "image_digest", "pip_freeze", "wsl_nvidia_smi", "win_nvidia_smi", "power_max_limit_w", "uname",
              "docker_version", "windows_power_mode", "clocks_pinned", "client_image", "client_output_schema_version",
              "bench_git_sha", "launch_cmd", "captured_at"):
        assert k in env, k
    assert env["image_digest"].startswith("sha256:") and env["launch_cmd"] == ["--model", "m"]
```

- [ ] **Step 2: Implement** the three modules per the interfaces (threads use `threading.Event` for stop; every write is one JSON line + flush; reader exceptions are caught per source and written into the sample as `wsl_error`/`win_error`; `capture_env` composes `_host_lines()` — which shells out to both `nvidia-smi`s, `uname -r`, `docker --version`, `powercfg.exe /getactivescheme` with try/except → `None` — plus docker digest/pip freeze and `bench.capture.cli.git_sha()`).

- [ ] **Step 3: Run tests; commit** — `git commit -m "bench: GPU sampler (WSL+Windows views), engine metrics scraper, environment capture"`

---

### Task 6: Client wrapper and output schema

**Files:**
- Create: `bench/runner/client.py`, `bench/runner/schema.py`
- Test: `tests/bench/runner/test_client.py`, `tests/bench/runner/test_schema.py`

**Interfaces:**
- `CLIENT_OUTPUT_SCHEMA_VERSION = "vllm-v0.29.0-save-detailed"`; `REQUIRED_CLIENT_KEYS = {"completed": int, "duration": float, "total_input_tokens": int, "total_output_tokens": int, "request_throughput": float, "output_throughput": float, "ttfts": list, "itls": list, "input_lens": list, "output_lens": list, "errors": list}` (exact set finalised from Task 1's doc); `validate_client_output(d: dict) -> None` raises `SchemaError` naming the missing/mistyped key; also checks `len(ttfts) == len(itls) == len(output_lens) == len(errors)`.
- `build_client_command(cfg, base_url, result_dir_in_container, result_filename) -> list[str]`: `vllm bench serve --backend openai --base-url <base_url> --endpoint /v1/completions --model <model> --tokenizer <model> --dataset-name custom --dataset-path /traces/<file> --skip-chat-template --num-prompts N --seed S --save-result --save-detailed --result-dir … --result-filename … --temperature/--top-p/--top-k …` plus `--max-concurrency C --request-rate inf` (concurrency mode) or `--request-rate λ --burstiness b` (poisson), `--ignore-eos` when set, `--custom-output-len -1` (per-row `output_tokens`; flag value per Task 1), `--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99`, `--extra-body '<json>'` when non-empty; every flag name copied from `bench/docs/p0b-engine-verification.md`.
- `run_client(docker, cfg, spec, run_dir: Path, traces_dir: Path, client_image: str) -> dict`: runs the command in a **GPU-less** container (`gpus=False`, `--network host`, mounts `traces_dir:/traces:ro` and `run_dir:/results:rw`, entrypoint `vllm`), waits for exit (polls `is_running`), copies nothing (results are on the bind mount), loads `run_dir/client_raw.json`, validates, returns the dict; on non-zero exit raises with the container logs.

- [ ] **Step 1: Failing tests** — `test_schema.py`: valid fixture passes; missing `ttfts` → `SchemaError` mentioning `ttfts`; wrong type → mentions the key; length mismatch → mentions "length". `test_client.py`: command for concurrency mode contains `--max-concurrency 8`, `--request-rate inf`, `--dataset-name custom`, `--skip-chat-template`, `--ignore-eos`, `--seed 0`, `--temperature 1.0`; poisson mode contains `--request-rate 4.0 --burstiness 1.0` and no `--max-concurrency`; `extra_body` renders as one JSON arg; `run_client` with `FakeDocker` (pre-writing `client_raw.json` into `run_dir`) returns the validated dict and used `gpus=False` (check `fake_docker.calls[0]`), and raises with logs when the fake reports a non-zero exit (add `exit_code` support to `FakeDocker.run` via a `fail_next` flag).

- [ ] **Step 2: Implement; run; commit** — `git commit -m "bench: vllm bench serve client wrapper with output schema validation"`

---

### Task 7: Summary, assertions

**Files:**
- Create: `bench/runner/summary.py`, `bench/runner/assertions.py`
- Test: `tests/bench/runner/test_summary.py`, `tests/bench/runner/test_assertions.py`

**Interfaces:**
- `write_requests(client: dict, trace_rows: list[dict], cfg, out_path) -> list[dict]`: one line per request with spec §7 fields — `request_id` (index), `trace_record_id` (from the trace row order **as the client consumed it**: with `--disable-shuffle` absent the tool shuffles by `--seed`; the runner passes `--disable-shuffle` so row i ↔ request i, and Task 1 confirms the flag), `call_role`, `workload`, `ttft_ms_client = ttfts[i]*1000`, `tpot_ms_client = mean(itls[i])*1000` (None if no ITLs), `itl_ms = [x*1000 …]`, `e2e_ms_derived = (ttfts[i] + sum(itls[i]))*1000`, `prompt_tokens = input_lens[i]`, `output_tokens = output_lens[i]`, `error = errors[i] or None`, `finish_reason = None` (not in client output), `schema_valid = None`.
- `goodput_grid(requests, ttft_grid_ms=(250,500,750,1000,1500,2000), tpot_grid_ms=(25,50,75,100,150)) -> dict[str, float]` fraction of completed requests meeting each (ttft, tpot) pair; keys `"ttft<=500&tpot<=50"`.
- `build_summary(client, requests, gpu_rows, metric_rows, cfg, timing) -> dict` with p50/p90/p99 of TTFT/TPOT/ITL/e2e computed from `requests` (stored as `*_derived`) **and** the client's own `median_*`/`p99_*` copied as `*_client`; `output_tok_s`, `req_s` from the client; acceptance summary from `metrics_scraper.summarise`; `peak_used_ours_mb`, `peak_used_host_mb`, `host_share_drift_mb = max(host) - min(host)`, `power_w` p50/p90/max, `clock_cv = std(sm_clock)/mean(sm_clock)`, `throttled = any(reasons & HW_SLOWDOWN|SW_THERMAL|HW_THERMAL|POWER_BRAKE)`, `error_rate`, `goodput_pre_registered` at `ttft<=500&tpot<=50` (spec §3.3), `goodput_by_threshold`, `valid`/`invalid_reason` filled by assertions.
- `check(cfg, summary, spec_on: bool, thresholds) -> tuple[bool, str | None]` in `assertions.py`: spec on ⇒ `acceptance_rate_mean` not None and > 0 and `spec_draft` advanced; `error_rate <= thresholds.max_error_rate` (0.01); `host_share_drift_mb <= thresholds.max_host_drift_mb` (256); `throttled` alone does **not** invalidate (it is recorded; runs are excluded at analysis time by policy) — spec §6 step 7 lists it as a flag; `completed == num_prompts`.

- [ ] **Step 1: Failing tests** (use `client_json` fixture; a tiny trace list; gpu/metric rows built inline): `write_requests` maps index → trace row and converts seconds→ms; `goodput_grid` fraction correct on a hand-built set; `build_summary` derived p50 equals the median of the inputs and `*_client` equals the fixture's `median_ttft_ms`; `clock_cv` = 0 for constant clocks; `throttled` true when reasons include `0x8`; assertions: acceptance zero with spec on → invalid "acceptance"; error rate 5% → invalid; clean → valid.

- [ ] **Step 2: Implement; run; commit** — `git commit -m "bench: per-request records, run summary with goodput grid, and validity assertions"`

---

### Task 8: Lifecycle and CLI

**Files:**
- Create: `bench/runner/lifecycle.py`, `bench/runner/cli.py`, `bench/runner/__main__.py`
- Test: `tests/bench/runner/test_lifecycle.py`

**Interfaces:**
- `run_one(cfg: RunConfig, paths: RunPaths, *, docker, http, spec, gpu_reader, clock, sleep) -> dict` executes spec §6 steps 1–9 with injectable collaborators (`clock`/`sleep` for cooldown in tests):
  1. pre-flight: `docker ps` shows no `bench-*` container; read host+ours VRAM (Windows + WSL readers); `cfg.free_vram_mb_at_start = total - host_used`; `cfg.gpu_memory_utilization = resolve_gpu_memory_fraction(...)`; attempt clock pin only if `paths.sweep_opts.try_clock_pin` (record result);
  2. `docker.run(spec.image, f"bench-{cfg.run_id}", spec.build_launch_args(cfg, frac), mounts=[(hf_cache, "/root/.cache/huggingface", "ro")])`;
  3. `wait_healthy` (timeout 600 s) → `warmup(N prompts from the trace)` → `reset_cache` unless warm;
  4. start `GpuSampler` + `MetricsScraper`;
  5. `run_client` → `client_raw.json`;
  6. stop samplers;
  7. `write_requests`, `build_summary`, `assertions.check` → `summary.valid`;
  8. `docker.stop`; wait until WSL-side `used_mb` returns within 200 MB of the pre-run value (timeout 60 s; failure recorded as `vram_leak_mb`);
  9. cooldown: sleep until Windows-side `temp_c <= cfg.cooldown_temp_c` **and** `cfg.cooldown_min_s` elapsed (poll 5 s, cap 15 min, record observed).
  Writes `config.yaml`, `env.json`, `requests.jsonl`, `gpu_samples.jsonl`, `engine_metrics.jsonl`, `client_raw.json`, `summary.json`, `meta.json`, `log.txt` (container logs + runner log) into `paths.run_dir`; returns the summary.
- `run_sweep(sweep_path, results_root, *, resume=False, collaborators…) -> SweepReport`: expands/schedules (or loads `state.json`), writes `sweep.yaml` copy + `schedule.json`, iterates `state.pending()`, calls `run_one`, on `valid: False` or exception → `state.requeue` (logged), stops when pending is empty or the retry budget is exhausted; prints a one-line status per run.
- CLI: `python -m bench.runner check-env` (docker + nvidia runtime + both nvidia-smi readers + images present + model revisions on disk match `base.yaml` + trace shas match metas + a `vllm bench serve --help` probe validates `REQUIRED_CLIENT_KEYS` names appear in `--help`/source → writes `check_env.json`), `run <sweep.yaml> [--results bench/results]`, `resume <sweep_id>`, `probe <name>` (Task 10).

- [ ] **Step 1: Failing tests** — `test_lifecycle.py` drives `run_one` fully with `FakeDocker`, `FakeHTTP` (metrics text from the scraper test), a `gpu_reader` returning fixed values, `clock`/`sleep` stubs, and a `run_client` monkeypatched to write `client_json` into the run dir. Assert: the docker `run` call carries the engine's launch args and the HF mount; the sequence of HTTP calls is health…, warmup posts, then the reset route; all nine artifact files exist; `summary.json["valid"] is True`; `meta.json` has `schedule_index`, `attempt`, cooldown observed; with `cache_state="warm"` no reset call happens; with a metrics text whose `spec_draft` never advances and `spec_method="draft"` the summary is `valid: False` with reason containing "acceptance". A second test drives `run_sweep` over a 3-run sweep where run 2 is invalid once: assert it is re-queued to the end, re-run, and `state.json` ends with 3 `done`.

- [ ] **Step 2: Implement; run the whole `tests/bench` suite; commit** — `git commit -m "bench: run lifecycle, sweep loop with resume, and the runner CLI"`

---

### Task 9: Echo server (harness-ceiling stand-in)

**Files:**
- Create: `bench/echo_server/__init__.py`, `bench/echo_server/server.py`
- Test: `tests/bench/runner/test_echo_server.py`

**Interfaces:** `aiohttp` app: `GET /health` → 200; `POST /v1/completions` accepts the OpenAI payload, streams `max_tokens` SSE chunks (`data: {"choices":[{"text":"x"}]}`) with `--per-token-ms` delay (default 5 ms) and a final `usage` chunk, honouring `stream: true`; `GET /metrics` → empty; `python -m bench.echo_server --port 8000 --per-token-ms 5`. Registered in `engine.py` as `ENGINES["echo"]` (no image; `lifecycle` skips docker for it and launches a subprocess) — add in this task with `verified_against="local"`.

- [ ] **Step 1: Failing test** — start the server in-process on a free port via `aiohttp.test_utils`, POST a streaming request with `max_tokens=5`, assert 5 content chunks + usage and that elapsed ≥ 5 × per-token delay.
- [ ] **Step 2: Implement; commit** — `git commit -m "bench: OpenAI-compatible echo server for measuring the client's own ceiling"`

---

### Task 10: Calibration configs and probe commands

**Files:**
- Create: `bench/configs/p0b_ceiling.yaml`, `bench/configs/p0b_parity.yaml`, `bench/configs/p0b_ignore_eos.yaml`, `bench/configs/p0b_probes.yaml`; `bench/runner/probes.py`
- Modify: `bench/runner/cli.py` (`probe <name>`)
- Test: `tests/bench/runner/test_probes.py` (pure functions only)

**Probes** (`bench/runner/probes.py`, each writes `bench/results/probes/<name>-<timestamp>.json`):
- `oom_signal`: launches vLLM with `--gpu-memory-utilization` stepping 0.90 → 0.95 → 0.98 → 1.00 while the Windows-side sampler runs; for each: did the container fail cleanly (exit + OOM text in logs) or come up and serve? If it serves at a fraction that exceeds physically free memory, run 20 requests and record `output_throughput` vs the 0.85 baseline — a collapse without an OOM is the WDDM-paging signature (spec §3.1). Writes `{"fractions": [...], "outcome": "clean_oom|spill|fits", "throughput_ratio": …}`.
- `cudagraph_cost`: launch with capture sizes `[]`-equivalent (smallest legal list `[1]`), `[1,2,4,8]`, `[1,2,4,8,16,32,64]`; record WSL-side `used_mb` after readiness for each → the graph line of the budget table.
- `host_reservation`: 60 s of Windows-side samples with no container running → `used_host_mb` idle (expected ≈ 0 on this hybrid-graphics laptop); then during a 60 s c=8 run → drift.
- `clock_pin`: `nvidia-smi.exe -lgc 2055,2055` → record exit code and whether `clocks.sm` holds under load; `-rgc` after. Outcome sets `try_clock_pin` in `base.yaml` and `clocks_pinned` in every later `env.json`.
- `parity` (uses `p0b_parity.yaml`): one trace row → both engines → `usage.prompt_tokens` from each vs the row's `prompt_tokens_qwen`; pass iff all three equal.
- `ignore_eos_acceptance` (uses `p0b_ignore_eos.yaml`): vLLM draft@k=3 and ngram@k=3, c=1, 100 prompts of `chat_v1`, with and without `--ignore-eos`; report acceptance rate for each; pre-registered rule: if they differ by > 10% relative, P2 reports acceptance over the natural-length prefix (spec §3.1 probe).

`p0b_ceiling.yaml`: `engine: [echo]`, `load_mode: poisson`, `request_rate: [8, 16, 32, 64, 128, 256]`, `num_prompts: 500`, `reps: 2`, `workload: [A]`. The ceiling is the highest rate at which achieved `request_throughput ≥ 0.95 × request_rate` and p99 TTFT stays within 2× of the rate-8 value. Spec §9 requires ceiling ≥ 3× the study's peak rate; the peak concurrency-mode rate is estimated from P1's expected knee (~64 concurrent × 1/(2.8 calls × ~1 s) ≈ 25 req/s), so the ladder must reach ≥ 75 req/s cleanly.

- [ ] **Step 1: Write the YAMLs and the probe functions with unit tests for the pure parts** (`ceiling_from_rows(rows) -> float`, `parity_verdict(a, b, expected) -> bool`, `acceptance_delta(with, without) -> float`).
- [ ] **Step 2: Commit** — `git commit -m "bench: P0b calibration sweeps and hardware probes"`

---

### Task 11: Run the calibration (runbook — GPU time ≈ 3–4 h)

- [ ] **Step 1:** `python -m bench.runner check-env` → all green; `git status --porcelain` empty.
- [ ] **Step 2:** probes in this order (each ≤ 20 min): `clock_pin`, `host_reservation`, `cudagraph_cost`, `oom_signal`, `parity`, `ignore_eos_acceptance`. Record outcomes; if `clock_pin` fails, set `try_clock_pin: false` in `base.yaml` and raise `cooldown_temp_c` to 50 / `cooldown_min_s` to 90 (the pre-registered contingency) **before** the variance sweep and commit that change.
- [ ] **Step 3:** `python -m bench.runner run bench/configs/p0b_ceiling.yaml` (echo server; ~15 min).
- [ ] **Step 4:** `python -m bench.runner run bench/configs/p0b_variance.yaml` (10 × ~6 min incl. cooldown ≈ 1 h). Re-run `resume` if interrupted.
- [ ] **Step 5:** Copy `bench/results/p0b-variance/*/summary.json`, `bench/results/p0b-ceiling/*/summary.json`, and `bench/results/probes/*.json` are read by Task 12's analysis; nothing else is committed from `bench/results/`.

---

### Task 12: Analysis and the P0b writeup

**Files:**
- Create: `bench/analysis/__init__.py`, `bench/analysis/load.py`, `bench/analysis/p0b_calibration.py`, `bench/docs/p0b-calibration.md` (generated), `bench/docs/p0b-writeup.md`
- Test: `tests/bench/runner/test_analysis.py` (on synthetic summaries)

**Interfaces:** `load_summaries(results_root, sweep_id, valid_only=True) -> list[dict]`; `variance_floor(summaries) -> dict` with, per metric in (`ttft_p50_derived`, `tpot_p50_derived`, `output_tok_s`), `mean`, `std`, `cv`, `min`, `max`, `n`; `ceiling(summaries) -> dict` (`ceiling_req_s`, table by rate); `render_markdown(...)`.

- [ ] **Step 1:** implement + tests on synthetic inputs; run `python -m bench.analysis.p0b_calibration` → `bench/docs/p0b-calibration.md` + `.json`.
- [ ] **Step 2:** write `bench/docs/p0b-writeup.md`: engine/client verification summary (from Task 1); host telemetry model (two `nvidia-smi` views); probe outcomes vs the pre-registered expectations (clock pin, OOM signal, graph cost, host reservation, parity, ignore_eos); **variance floor** (the number every later difference must beat) and **harness ceiling** (with the 3× margin arithmetic); the `check-env` snapshot; deviations from spec §3.1's budget table with the measured graph and reservation lines; carry-forward list for the P1 plan (concurrency ladder, `max_num_seqs`, cooldown settings, sampling pins).
- [ ] **Step 3:** update the spec: §3.1 budget table measured lines; §6 clock-pin outcome; §9 P0b exit row → met/not met with the two numbers. Commit: `git commit -m "bench: P0b calibration analysis, writeup and spec re-evaluation"`

---

## Self-review

- **Spec coverage:** §4 client + readiness (Tasks 4, 6); §6 lifecycle steps 1–9, monitoring of host share + power, `check-env`, engine tables as data, invariants (Tasks 2, 5, 7, 8); §7 every artifact file and field (Tasks 5, 7, 8); §3.1 probes — OOM signal, CUDA-graph cost, host reservation, `ignore_eos` × acceptance — plus the chat-template parity check of §4 and the clock-pin contingency of §6 (Task 10/11); §8 risk 6 verify-first (Task 1); §9 P0b exit — variance floor, harness ceiling ≥ 3× peak, flags verified at pins, probes done (Tasks 11–12); sampling pins (Task 6 command builder). Sweep-level `sweep.yaml`/`schedule.json`/`state.json`/`check_env.json` (Tasks 3, 8).
- **Placeholder scan:** flag names marked `# T1` are deliberately sourced from Task 1's document rather than this plan — that is the brief's own rule, not a placeholder; `base.yaml`'s `<T1 sha>` values are filled in Task 1. No "TBD".
- **Type consistency:** `RunConfig` field names are used identically in `engine.py`, `sweep.py`, `client.py`, `summary.py`; `EngineSpec.metric_names` keys match `MetricsScraper` output keys and `summarise`; `FakeDocker`/`Docker` share one method surface; `run_one` collaborator names match `lifecycle` tests.
