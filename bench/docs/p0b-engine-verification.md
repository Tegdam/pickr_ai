# P0b Task 1 — engine, client and model verification at the pins

Date: 2026-09-18. Spec §8 risk 6: verify first, build nothing on assumptions. Everything below was read from `--help` output, from source inside the pinned images, or observed in a smoke run on the target machine (WSL2, RTX 4050 Laptop 6 GB). Nothing is quoted from memory of documentation. The **Decisions** section at the end is the contract Task 2's launch tables copy from.

Host facts (2026-09-18): Windows 11, WSL2 kernel `5.15.167.4-microsoft-standard-WSL2`, Ubuntu 24.04.2, native Docker `28.0.4` in the distro with the `nvidia` runtime (`Runtimes: io.containerd.runc.v2 nvidia runc`, `nvidia-container-cli 1.17.8`), NVIDIA driver `610.88` (CUDA UMD 13.3). `docker run --gpus all` works. The app venv is `env/bin/python`.

---

## 1. Pins and digests

| Component | Tag | Digest (`docker image inspect --format '{{index .RepoDigests 0}}'`) |
|---|---|---|
| vLLM (server + client) | `vllm/vllm-openai:v0.29.0` | `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1` |
| SGLang | `lmsysorg/sglang:v0.5.20-runtime` | `sha256:00b02004501e402332827ffd5343a225a8960d99adc990b37d6f31085b8f6800` |

Versions inside the images (from `pip list` / `python -c`):

```
vllm/vllm-openai:v0.29.0   (Ubuntu 24.04.3, python 3.12.3, binary is `python3` — `python` is NOT on PATH)
  vllm 0.29.0, torch 2.13.0+cu130 (cuda 13.0), nvcc 13.0.88, flashinfer-python 0.6.18,
  transformers 5.16.1, tokenizers 0.23.2, triton 3.7.1, aiohttp 3.14.3, numpy 2.2.6
  NOT installed: pandas, datasets (the `vllm[bench]` extra = pandas, matplotlib, seaborn, datasets, scipy, plotly)

lmsysorg/sglang:v0.5.20-runtime   (Ubuntu 24.04.4, python 3.12.3 at /opt/sglang/bin/python)
  sglang 0.5.20 (/sgl-workspace/sglang/python), torch 2.13.0+cu130 (cuda 13.0), flashinfer-python 0.6.18,
  transformers 5.12.1, tokenizers 0.22.2, triton 3.7.1, xgrammar 0.2.1, llguidance 1.8.0, outlines 0.1.11
```

Two facts about the vLLM image that cost time and are worth knowing before Task 2:

- `vllm serve --help` **cannot be generated on a GPU-less container** — parser construction infers the device and raises `RuntimeError: Failed to infer device type`. Run it with `--gpus all`. `vllm bench serve --help` works on CPU.
- `vllm serve --help` prints only a short index; the full flag list needs `--help=all` (2011 lines captured).

## 2. Model revisions and config facts

Downloaded into the WSL ext4 HF cache (`/home/edamiba/.cache/huggingface/hub`) at pinned revisions, mounted read-only into every container at `/root/.cache/huggingface`. The cache has **no `refs/main`** for these repos (snapshots were fetched by sha), which matters for offline resolution — see §6.

| Model | Revision sha | Snapshot dir (inside containers) |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct-AWQ` | `3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd` | `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct-AWQ/snapshots/3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd` |
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | `/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775` |
| `BAAI/bge-small-en-v1.5` | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | `/root/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |

`config.json` checks (verbatim from the snapshot dirs):

```
Qwen2.5-3B-Instruct-AWQ: {'model_type': 'qwen2', 'architectures': ['Qwen2ForCausalLM'], 'num_hidden_layers': 36,
  'num_key_value_heads': 2, 'num_attention_heads': 16, 'hidden_size': 2048, 'vocab_size': 151936,
  'tie_word_embeddings': True, 'max_position_embeddings': 32768, 'torch_dtype': 'float16'}   head_dim = 128
  quantization_config = {'bits': 4, 'group_size': 128, 'modules_to_not_convert': None, 'quant_method': 'awq', 'version': 'gemm', 'zero_point': True}
Qwen2.5-0.5B-Instruct: {'model_type': 'qwen2', 'architectures': ['Qwen2ForCausalLM'], 'num_hidden_layers': 24,
  'num_key_value_heads': 2, 'num_attention_heads': 14, 'hidden_size': 896, 'vocab_size': 151936,
  'tie_word_embeddings': True, 'max_position_embeddings': 32768, 'torch_dtype': 'bfloat16'}   head_dim = 64
  quantization_config = None
bge-small-en-v1.5: {'model_type': 'bert', 'num_hidden_layers': 12, 'num_attention_heads': 12, 'hidden_size': 384,
  'vocab_size': 30522, 'max_position_embeddings': 512, 'torch_dtype': 'float32'}
```

**All facts match spec §3.1** (target 36 / 2 / 128, draft 24 / 2 / 64, AWQ 4-bit group 128). No budget-table correction needed. Chat template: both Qwen repos ship the identical `chat_template` (sha256 `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`, 2507 chars) — this is the hash for `env.json`.

## 3. vLLM 0.29.0 — flags, speculative methods, routes, metrics

### 3.1 Server flags (from `vllm serve --help=all`, verbatim)

```
  --model MODEL ...                                    (positional model_tag also accepted)
  --revision REVISION   The specific model version to use. ... (default: None)
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
  --quantization QUANTIZATION, -q QUANTIZATION        (Literal includes 'awq', 'awq_marlin', ...)
  --max-model-len MAX_MODEL_LEN
  --max-num-seqs MAX_NUM_SEQS                          (default: None)
  --max-num-batched-tokens MAX_NUM_BATCHED_TOKENS      (default: None)  Parse human-readable integers like '1k'
  --enable-chunked-prefill, --no-enable-chunked-prefill  (default: None)
  --long-prefill-token-threshold LONG_PREFILL_TOKEN_THRESHOLD  (default: 0)
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION     ... If unspecified, will use the default value of 0.92.
  --kv-cache-memory-bytes KV_CACHE_MEMORY_BYTES        Size of KV Cache per GPU in bytes. (alternative to the fraction)
  --enable-prefix-caching, --no-enable-prefix-caching  Whether to enable prefix caching. (default: None)
  --seed SEED           Random seed for reproducibility. ... (default: 0)
  --port PORT           Port number. (default: 8000)
  --enforce-eager, --no-enforce-eager                  (default: False)   -- prohibited by spec §3.1
  --cudagraph-capture-sizes CUDAGRAPH_CAPTURE_SIZES [CUDAGRAPH_CAPTURE_SIZES ...]   Sizes to capture cudagraph.
  --max-cudagraph-capture-size MAX_CUDAGRAPH_CAPTURE_SIZE
  --compilation-config COMPILATION_CONFIG, -cc COMPILATION_CONFIG   e.g. `{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`
                        (also -cc.cudagraph_capture_sizes+ ... ; default cudagraph_mode FULL_AND_PIECEWISE)
  --speculative-config SPECULATIVE_CONFIG, -sc SPECULATIVE_CONFIG   Should either be a valid JSON string or JSON keys passed individually.
  --generation-config GENERATION_CONFIG   Defaults to "auto" (loaded from model path); "vllm" = no generation config
  --scheduling-policy {fcfs,priority}
  --show-hidden-metrics-for-version SHOW_HIDDEN_METRICS_FOR_VERSION
  --cudagraph-metrics, --no-cudagraph-metrics          (default: False)
```

Observed default capture list at `--max-num-seqs 32`: `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]` (target only) and `[1, 2, 4, ... 256]` in the spec arms; `--cudagraph-capture-sizes 1 2 4 8 16 32` was honoured (`cudagraph_capture_sizes': [1, 2, 4, 8, 16, 32]` in the engine config log).

### 3.2 Speculative methods (from `vllm/config/speculative.py` source)

```
SpeculativeMethod Literal: "ngram", "medusa", "mlp_speculator", "draft_model", "suffix", "custom_class", EagleModelTypes, NgramGPUTypes, DSparkModelTypes,
field method: Optional[Literal['ngram', 'medusa', 'mlp_speculator', 'draft_model', 'suffix', 'custom_class', 'eagle', 'eagle3', ... 'mtp', ... 'dflash', 'ngram_gpu', 'dspark']] default=None
field model: str | None default=None
field num_speculative_tokens: int (Gt 0), default None (required)
field prompt_lookup_max / prompt_lookup_min: int | None (Ge 1); if both None for ngram -> both default to 5; if one given the other copies it
field revision: str | None default=None
field quantization: ... default=None
field draft_tensor_parallel_size, max_model_len, enforce_eager, disable_padded_drafter_batch (bool, False)
  712: if self.model in ("ngram", "[ngram]"): self.method = "ngram"   /   715: else: self.method = "draft_model"
 1485: def uses_draft_model(self): return self.method == "draft_model"
```

**Both `draft_model` and `ngram` are listed methods at v0.29.0**, and both launched and produced non-zero acceptance counters (§6). `k` maps to the JSON key `num_speculative_tokens`; ngram window to `prompt_lookup_max` / `prompt_lookup_min`.

### 3.3 Routes (from `vllm/entrypoints/**/api_router.py`)

```
entrypoints/serve/instrumentator/health.py:22:  @router.get("/health", response_class=Response)
entrypoints/serve/instrumentator/basic.py:53:   @router.get("/version")
entrypoints/serve/dev/cache/api_router.py:20:   @router.post("/reset_prefix_cache")
    async def reset_prefix_cache(raw_request, reset_running_requests: bool = Query(default=False), reset_external: bool = Query(default=False))
    Returns `{"success": bool}`. The reset fails (`success=false`) while blocks are still held, e.g. by running requests
entrypoints/serve/dev/cache/api_router.py:47:   @router.post("/reset_mm_cache")
entrypoints/openai/completion/api_router.py:36: "/v1/completions"
entrypoints/openai/chat_completion/api_router.py:42: "/v1/chat/completions"
/metrics: prometheus_fastapi_instrumentator .expose(app) in entrypoints/serve/instrumentator/metrics.py
```

**Gate:** `entrypoints/launchers/api_server/routers.py:35: if envs.VLLM_SERVER_DEV_MODE: register_vllm_dev_api_routers(app)` — the cache router is only mounted with `VLLM_SERVER_DEV_MODE=1`. Verified empirically: with the env var `POST /reset_prefix_cache -> 200 {"success":true}`, `GET -> 405`; without it `POST -> 404`.

### 3.4 Metric names (from `vllm/v1/metrics/loggers.py` and `vllm/v1/spec_decode/metrics.py`)

```
vllm:kv_cache_usage_perc           Gauge   "KV-cache usage. 1 means 100 percent usage."   (NOT vllm:gpu_cache_usage_perc — that name does not exist at 0.29.0; grep count 0 on the live /metrics)
vllm:num_requests_running          Gauge
vllm:num_requests_waiting          Gauge   (+ vllm:num_requests_waiting_by_reason{reason=capacity|deferred})
vllm:prefix_cache_queries          Counter "Prefix cache queries, in terms of number of queried tokens."   exposed as vllm:prefix_cache_queries_total
vllm:prefix_cache_hits             Counter "Prefix cache hits, in terms of number of cached tokens."       exposed as vllm:prefix_cache_hits_total
vllm:num_preemptions               Counter -> vllm:num_preemptions_total
vllm:spec_decode_num_drafts        Counter -> vllm:spec_decode_num_drafts_total
vllm:spec_decode_num_draft_tokens  Counter -> vllm:spec_decode_num_draft_tokens_total
vllm:spec_decode_num_accepted_tokens         -> vllm:spec_decode_num_accepted_tokens_total
vllm:spec_decode_num_accepted_tokens_per_pos -> vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0".."k-1"}
vllm:request_success_total{finished_reason=stop|length|abort|error|repetition}
also: vllm:prompt_tokens_total, vllm:generation_tokens_total, vllm:time_to_first_token_seconds, vllm:inter_token_latency_seconds,
      vllm:e2e_request_latency_seconds, vllm:request_queue_time_seconds, vllm:request_prefill_time_seconds, vllm:request_decode_time_seconds
```

Labels observed: `{engine="0",model_name="<served name>"}`. Under `HF_HUB_OFFLINE=1` vLLM rewrites the model id to the snapshot path (`arg_utils.py:866 HF_HUB_OFFLINE is True, replace model_id [...] to model_path [...]`), and without `--served-model-name` the `model_name` label becomes that path. There is **no "emitted" counter**; acceptance length = 1 + accepted/drafts, acceptance rate = accepted/draft_tokens (this is also how `vllm bench serve` computes its `spec_decode_*` result keys, see §5).

### 3.5 WSL2-specific: the V2 model runner needs pinned memory

First launch failed: `RuntimeError: UVA is not available` from `v1/worker/gpu/buffer_utils.py:47 UvaBuffer.__init__` — vLLM 0.29.0 defaults to the new `GPUModelRunnerV2`, which needs host-mapped pinned memory, and `platforms/cuda.py:290 is_pin_memory_available()` returns False under WSL by default. Two documented escape hatches in `vllm/envs.py`:

```
# On WSL2 with a compatible kernel (>= 4.19.121), pinned memory is supported but disabled by default due to a small
# performance regression. Set to 1 when pinned memory or UVA is required (e.g. CPU offloading or v2 model runner).
"VLLM_WSL2_ENABLE_PIN_MEMORY"
# Flag to control the v2 model runner. If unset, use config defaults.
"VLLM_USE_V2_MODEL_RUNNER"
```

With `VLLM_WSL2_ENABLE_PIN_MEMORY=1` the baseline came up on the V2 runner (`model_runner.py:404`). But both speculative arms silently fell back to the V1 runner:

```
Model Runner V2 does not yet support speculative method 'draft_model'; using the V1 model runner instead.
Model Runner V2 does not yet support ngram/ngram_gpu speculative decoding; using the V1 model runner instead.
```

so baseline-vs-spec would compare two model runners. Verified that `VLLM_USE_V2_MODEL_RUNNER=0` puts the baseline on the V1 runner (`gpu_model_runner.py:5471`) cleanly. See Decisions.

## 4. SGLang 0.5.20 — flags, speculative methods, routes, metrics

### 4.1 Server flags (from `python -m sglang.launch_server --help`, verbatim)

```
  --model-path MODEL_PATH, --model MODEL_PATH
  --revision REVISION
  --served-model-name SERVED_MODEL_NAME
  --quantization {awq,fp8,mxfp8,gptq,gptq_marlin,awq_marlin,bitsandbytes,gguf,...,unquant,humming}
  --context-length CONTEXT_LENGTH
  --mem-fraction-static MEM_FRACTION_STATIC   The fraction of the memory used for static allocation (model weights and KV cache memory pool).
  --max-total-tokens MAX_TOTAL_TOKENS         (alternative: absolute KV pool size)
  --max-running-requests MAX_RUNNING_REQUESTS
  --max-queued-requests MAX_QUEUED_REQUESTS
  --chunked-prefill-size CHUNKED_PREFILL_SIZE  The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.
  --max-prefill-tokens MAX_PREFILL_TOKENS
  --schedule-policy {lpm,random,fcfs,dfs-weight,lof,priority,routing-key,hrrn}
  --random-seed RANDOM_SEED
  --port PORT / --host HOST
  --disable-radix-cache   Disable RadixAttention for prefix caching.
  --enable-metrics        Enable log prometheus metrics.
  --enable-cache-report   Return number of cached tokens in usage.prompt_tokens_details for each openai request.
  --stream-interval STREAM_INTERVAL   (default 1)
  --attention-backend {triton,torch_native,...,flashinfer,...}   (auto-selected: "Attention backend not specified. Use flashinfer backend by default.")
  --cuda-graph-config CUDA_GRAPH_CONFIG            Per-phase CUDA graph settings as JSON ... JSON wins over the per-phase --cuda-graph-* convenience flags and over legacy flags.
  --cuda-graph-backend-decode {full,breakable,tc_piecewise,disabled}
  --cuda-graph-backend-prefill {full,breakable,tc_piecewise,disabled}
  --cuda-graph-max-bs-decode / --cuda-graph-max-bs-prefill
  --cuda-graph-bs-decode CUDA_GRAPH_BS_DECODE [CUDA_GRAPH_BS_DECODE ...]   Explicit list of batch sizes to capture for the decode cuda graph.
  --cuda-graph-bs-prefill CUDA_GRAPH_BS_PREFILL [...]
  --disable-prefill-cuda-graph / --disable-decode-cuda-graph
  --disable-cuda-graph  Deprecated. Use --cuda-graph-backend-{decode,prefill}=disabled instead.
```

**The brief's `--cuda-graph-bs` / `--cuda-graph-max-bs` do not exist at 0.5.20** (grep of the help and of `arg_groups/fields/exec_.py` finds only the `-decode`/`-prefill` variants). `--cuda-graph-bs-decode 1 2 4 8 16 32` was honoured (`Capture draft decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[1, 2, 4, 8, 16, 32]`). Default decode list at 0.80 was `bs=[1, 2, 4, 8]` (memory-derived). SGLang also captures a **prefill** graph (`backend=breakable`, num_tokens=[4, 8, ..., 640, ...]) which took 36–41 s and 0.36–0.40 GB on every launch.

### 4.2 Speculative flags and methods

```
  --speculative-algorithm SPECULATIVE_ALGORITHM   Speculative algorithm. Builtins: EAGLE, EAGLE3, NEXTN, STANDALONE, NGRAM, DFLASH, DSPARK, UNO.
  --speculative-draft-model-path SPECULATIVE_DRAFT_MODEL_PATH, --speculative-draft-model SPECULATIVE_DRAFT_MODEL_PATH
  --speculative-draft-model-revision SPECULATIVE_DRAFT_MODEL_REVISION
  --speculative-draft-model-quantization {awq,...,unquant,...}   The quantization method for speculative model.
  --speculative-num-steps SPECULATIVE_NUM_STEPS         The number of steps sampled from draft model in Speculative Decoding.
  --speculative-eagle-topk SPECULATIVE_EAGLE_TOPK       The number of tokens sampled from the draft model in eagle2 each step.
  --speculative-num-draft-tokens SPECULATIVE_NUM_DRAFT_TOKENS   The number of tokens sampled from the draft model in Speculative Decoding.
  --speculative-ngram-min-bfs-breadth (default 1) / --speculative-ngram-max-bfs-breadth (default 10)
  --speculative-ngram-match-type {BFS,PROB} (default BFS) / --speculative-ngram-max-trie-depth (default 18) / --speculative-ngram-capacity (default 10_000_000)
  --speculative-accept-threshold-single / --speculative-accept-threshold-acc / --speculative-use-rejection-sampling (EAGLE/EAGLE3 only)
```

`srt/speculative/spec_info.py:46-47`: `STANDALONE = auto()`, `NGRAM = auto()` — **both are accepted values**. Defaulting logic (`srt/arg_groups/speculative_hook.py`):

```
_auto_choose_speculative_params: if cfg.speculative_algorithm == "STANDALONE": return (3, 1, 4)   # (num_steps, topk, num_draft_tokens)
_handle_ngram: speculative_eagle_topk = cfg.speculative_ngram_max_bfs_breadth (topk is NOT user-settable for NGRAM)
               if num_draft_tokens is None: 12 ("set to 12 by default for ngram speculative decoding")
               if num_steps is None: num_draft_tokens // topk
```

`k` mapping: for STANDALONE with topk 1, `--speculative-num-steps k --speculative-num-draft-tokens k+1` (verified `3`/`4` -> `/get_server_info` reports `speculative_num_steps: 3, speculative_eagle_topk: 1, speculative_num_draft_tokens: 4`). For NGRAM, `--speculative-num-draft-tokens k+1` is the verify window; `--speculative-eagle-topk` is overridden to `ngram_max_bfs_breadth` (10) regardless of what is passed (observed: passed 1, server info says 10). `srt/managers/tokenizer_manager.py:2940`: per-request `num_proposed_drafts = spec_verify_ct * (speculative_num_draft_tokens - 1)`, so `k = num_draft_tokens - 1` is the accounting SGLang itself uses.

Two launch requirements found by failure (verbatim errors):

1. `--speculative-draft-model-path Qwen/Qwen2.5-0.5B-Instruct` with `HF_HUB_OFFLINE=1` fails before the server starts: `speculative_hook.py:54 _resolve_speculative_algorithm_alias -> get_config(speculative_draft_model_path)` is called **without the revision**, and the cache has no `refs/main` -> `huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled.` Fix: pass the draft as its snapshot directory.
2. The draft inherits the target's `--quantization` (`serving_hook.py: speculative_draft_model_quantization=cfg.quantization` when unset) -> `ValueError: Cannot find the config file for awq` from `standalone_worker_v2.py:83`. Fix: `--speculative-draft-model-quantization unquant` (resolves to `None`, `serving_hook.py:647`).

### 4.3 Routes (from `srt/entrypoints/http_server.py`)

```
662: @app.get("/ready")            200 if tokenizer_manager.is_ready() else 503
669: @app.get("/health")           (also /health_generate) -- returns 200 without generating unless SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION
990: @app.api_route("/flush_cache", methods=["GET", "POST"])   @auth_level(AuthLevel.ADMIN_OPTIONAL)
     async def flush_cache(timeout: float = Query(0.0, ge=0.0)):  """Flush the radix cache."""
     success -> "Cache flushed.\nPlease check backend logs for more details. (When there are running or waiting requests, the operation will not be performed.)\n"
808: @app.get("/get_server_info")  (deprecated alias; /server_info)  -- returns the resolved server args (used below to confirm resolved spec params)
     /metrics is added by add_prometheus_middleware(app) only when --enable-metrics
     /v1/completions, /v1/chat/completions, /v1/models, /generate, ...
```

Verified: `GET /health -> 200`, `GET /ready -> 200`, `GET /flush_cache -> 200`, `POST /flush_cache -> 200`; after the flush `sglang:cache_hit_rate` dropped to 0.0 and re-sending the same prompt added 0 to `sglang:cached_tokens_total` (the flush is effective). Note the HTTP status is 200 even when the flush is refused (message text differs), so the runner must check the body, not the code.

### 4.4 Metric names (from `srt/observability/metrics_collector.py`, verbatim definitions)

```
sglang:num_running_reqs        Gauge  "The number of running requests."
sglang:num_queue_reqs          Gauge  "The number of requests in the waiting queue."
sglang:num_used_tokens         Gauge  "The number of used tokens."         (KV occupancy in tokens)
sglang:token_usage             Gauge  "The token usage."                    (KV occupancy as a fraction of max_total_num_tokens)
sglang:max_total_num_tokens    Gauge   (KV pool size in tokens)
sglang:kv_cache_memory_usage_gb Gauge "Memory used by the KV cache pools in GB."
sglang:cache_hit_rate          Gauge  "The prefix cache hit rate."          (per-interval, not cumulative)
sglang:cached_tokens_total     Counter "Number of cached prompt tokens by source (device/host/storage)."  labels + cache_source
sglang:prompt_tokens_total     Counter "Number of prefill tokens processed."  labels + is_streaming
sglang:generation_tokens_total Counter
sglang:num_requests_total      Counter
sglang:spec_accept_length      Gauge  "Mean acceptance length of speculative decoding (accepted drafts + bonus token per forward)."
sglang:spec_accept_rate        Gauge  "Speculative acceptance rate (`accepted drafts / proposed drafts` in batch)."
sglang:spec_num_steps / sglang:spec_num_draft_tokens   Gauge  "Currently active speculative_num_steps / _num_draft_tokens"
sglang:spec_verify_calls_total Counter "Number of speculative decoding verification calls."
sglang:spec_cap_length, sglang:spec_block_accept_length   Gauge (0.0 here)
also: sglang:time_to_first_token_seconds, sglang:inter_token_latency_seconds, sglang:e2e_request_latency_seconds, sglang:queue_time_seconds,
      sglang:gen_throughput, sglang:num_retracted_reqs, sglang:evicted_tokens_total, sglang:uncached_prompt_tokens_histogram
```

Labels observed: `{engine_type="unified",model_name="Qwen/Qwen2.5-3B-Instruct-AWQ",moe_ep_rank="0",pp_rank="0",tp_rank="0"}` (plus `is_streaming` / `cache_source` where noted). **SGLang has no cumulative accepted-tokens / draft-tokens counters** — `spec_accept_*` are most-recent-interval gauges; the only cumulative spec counter is `spec_verify_calls_total`. The cumulative equivalent is per request: pass `"return_spec_tokens_details": true` in the request body (`openai/protocol.py:361`, `serving_completions.py:272`) and the non-streaming response carries

```
"sglext": {"spec_tokens_details": {"spec_accept_rate": 0.6825, "spec_accept_length": 3.0476, "spec_cap_length": 0.0, "spec_block_accept_length": 0.0,
           "spec_num_correct_drafts": 43, "spec_num_proposed_drafts": 63, "spec_verify_ct": 21, "spec_correct_drafts_histogram": [3, 5, 1, 12], "spec_cap_lens_histogram": []}}
```

(verbatim from a 64-token row-3 request on STANDALONE k=3). Summing `spec_num_correct_drafts` / `spec_num_proposed_drafts` across requests gives the cumulative counts that vLLM exposes on `/metrics`. The stock client ignores `sglext`, so the runner has to collect it itself (correctness/acceptance pass, not the load run).

Prefix-cache "queries": SGLang has no per-token queries counter; the pair to record is `sglang:cached_tokens_total{cache_source="device"}` (hits, tokens) against `sglang:prompt_tokens_total` (all prefill tokens), plus the per-request `usage.prompt_tokens_details.cached_tokens` when `--enable-cache-report` is set.

## 5. Client — `vllm bench serve` at v0.29.0 (from `--help=all` and `vllm/benchmarks/serve.py`, `datasets/datasets.py`, `lib/endpoint_request_func.py`)

```
  --backend {vllm,openai,openai-chat,openai-audio,openai-embeddings,...}   (default: openai)
      ASYNC_REQUEST_FUNCS: "vllm": async_request_openai_completions, "openai": async_request_openai_completions, "openai-chat": async_request_openai_chat_completions
  --endpoint ENDPOINT   API endpoint. (default: /v1/completions)
  --base-url BASE_URL / --host / --port
  --model MODEL         Name of the model. If not specified, will fetch the first model from the server's /v1/models endpoint.
  --tokenizer TOKENIZER / --served-model-name
  --dataset-name {sharegpt,burstgpt,sonnet,random,random-mm,random-rerank,hf,custom,custom_audio,custom_image,prefix_repetition,spec_bench,speed_bench,timed_trace}
  --dataset-path DATASET_PATH
  --custom-output-len CUSTOM_OUTPUT_LEN   Number of output tokens per request. Unless it is set to -1, the value overrides potential output length loaded from the dataset. It is used only for custom dataset. (default: 256)
  --skip-chat-template  Skip applying chat template to prompt for datasets that support it. (default: False)
  --disable-shuffle     Disable shuffling of dataset samples for deterministic ordering. (default: False)
  --no-oversample       Do not oversample if the dataset has fewer samples than num-prompts. (default: False)
  --num-prompts NUM_PROMPTS   (default: 1000)     (0 or negative = all rows, per CustomDataset.sample)
  --num-warmups NUM_WARMUPS   Number of warmup requests. (default: 0)   -- all warmups re-send request 0's prompt
  --ready-check-timeout-sec   Ready check will be skipped by default. (default: 0)   -- the "initial single prompt test run" request is NOT sent when 0 (verified via prefix_cache_queries_total == 265 + 6659)
  --max-concurrency MAX_CONCURRENCY  (default: None)
  --request-rate REQUEST_RATE  (default: inf) ; --burstiness BURSTINESS (default: 1.0 = Poisson)
  --seed SEED
  --ignore-eos          Set ignore_eos flag when sending the benchmark request.   -> payload["ignore_eos"] = True
  --temperature / --top-p / --top-k / --min-p / --repetition-penalty / --presence-penalty / --frequency-penalty   (default: None; only added when set)
  --extra-body EXTRA_BODY   A JSON string representing extra body parameters to include in each request.   -> payload.update(extra_body)
  --percentile-metrics PERCENTILE_METRICS   Allowed metric names are "ttft", "tpot", "itl", "e2el". defaults to "ttft,tpot,itl"
  --metric-percentiles METRIC_PERCENTILES   Comma-separated ... Default value is "99"
  --goodput GOODPUT [...]   "ttft:ms tpot:ms e2el:ms"
  --save-result / --save-detailed / --append-result / --result-dir / --result-filename / --label / --metadata KEY=VALUE
  --request-id-prefix REQUEST_ID_PREFIX   (default: bench-<random>-)
  --profile / --probe-request-rate / --ramp-up-* / --plot-timeline   (unused)
```

Custom dataset semantics (`CustomDataset`, verbatim):

- rows need `prompt`; `output_tokens` "is optional and has to be provided only if 'custom-output-len' argument is None or -1". Our trace rows carry both.
- `random.seed(self.random_seed); if not getattr(self, "disable_shuffle", False): random.shuffle(self.data)` — **`--disable-shuffle` exists and is honoured**; `request_id = request_id_prefix + str(i)` with `i` the row index. Verified: `input_lens` from a 20-prompt run equal the first 20 rows' `prompt_tokens_qwen` exactly (`[265, 267, 271, 116, 697, 260, 243, 284, 757, 267, 250, 275, 258, 116, 710, 275, 265, 272, 112, 699]`), on both engines.
- `if not skip_chat_template: prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], ...)` — with `--skip-chat-template` the trace's pre-templated `<|im_start|>...` text is sent verbatim as `prompt`.
- `--custom-output-len -1` -> `expected_output_len = int(item["output_tokens"])` -> `payload["max_tokens"]`. Verified: `output_lens == trace output_tokens` for all 20 rows with `--ignore-eos`.
- `maybe_oversample_requests` repeats rows when `num_prompts > len(dataset)` unless `--no-oversample`.
- The completions request func sends `{"model", "prompt", "repetition_penalty": 1.0, "max_tokens", "logprobs", "stream": true, "stream_options": {"include_usage": true}}` (+ `ignore_eos`, `extra_body`, sampling params).

Metric definitions (`endpoint_request_func.py` / `serve.py:595-625`): `ttft` = wall time to the first SSE chunk with `choices`; `itl` = wall time between successive chunks with `choices` (**inter-chunk**, not inter-token); `latency` (= e2el) = time to the last chunk; `tpot = (latency - ttft) / (output_len - 1)`; `output_len` = `usage.completion_tokens` from the final usage chunk (falls back to re-tokenizing the text); `prompt_len` is overwritten with `usage.prompt_tokens` when the server sends it (both engines do).

`--save-detailed` output keys (`serve.py:1273-1290`, kept only with `--save-detailed`): `input_lens`, `output_lens`, `ttfts`, `itls` (list per request), `start_times`, `generated_texts`, `errors`; always: `duration`, `completed`, `failed`, `total_input_tokens`, `total_output_tokens`, `request_throughput`, `request_goodput`, `output_throughput`, `total_token_throughput`, `max_output_tokens_per_s`, `max_concurrent_requests`, `rtfx`, `mean/median/std/p{N}_{ttft,tpot,itl,e2el}_ms`, plus `date, endpoint_type, backend, label, model_id, tokenizer_id, num_prompts, request_rate, burstiness, max_concurrency`. When the server exposes `vllm:spec_decode_*_total` the client also writes `spec_decode_acceptance_rate`, `spec_decode_acceptance_length`, `spec_decode_num_drafts`, `spec_decode_draft_tokens`, `spec_decode_accepted_tokens`, `spec_decode_per_position_acceptance_rates` as before/after deltas of `/metrics` (`fetch_spec_decode_metrics`, `serve.py:190-250`); SGLang's `sglang:spec_*` are not scraped (prefix mismatch).

Observed full key list of a saved file: `['backend', 'burstiness', 'completed', 'date', 'duration', 'endpoint_type', 'errors', 'failed', 'generated_texts', 'input_lens', 'itls', 'label', 'max_concurrency', 'max_concurrent_requests', 'max_output_tokens_per_s', 'mean_e2el_ms', 'mean_itl_ms', 'mean_tpot_ms', 'mean_ttft_ms', 'median_e2el_ms', 'median_itl_ms', 'median_tpot_ms', 'median_ttft_ms', 'model_id', 'num_prompts', 'output_lens', 'output_throughput', 'p50_e2el_ms', 'p50_itl_ms', 'p50_tpot_ms', 'p50_ttft_ms', 'p90_e2el_ms', 'p90_itl_ms', 'p90_tpot_ms', 'p90_ttft_ms', 'p99_e2el_ms', 'p99_itl_ms', 'p99_tpot_ms', 'p99_ttft_ms', 'request_goodput', 'request_rate', 'request_throughput', 'rtfx', 'start_times', 'std_e2el_ms', 'std_itl_ms', 'std_tpot_ms', 'std_ttft_ms', 'tokenizer_id', 'total_input_tokens', 'total_output_tokens', 'total_token_throughput', 'ttfts']`.

**The pinned image cannot run the custom dataset as shipped**: `ImportError: Please install vllm[bench] for bench support` (root cause `ModuleNotFoundError: No module named 'pandas'` at `datasets.py:2535 pd.read_json`). `pip install pandas` inside the client container fixes it (pandas is the only missing piece exercised; `datasets` is only needed for the `hf` dataset). The client also needs `--tokenizer <snapshot dir>` under `HF_HUB_OFFLINE=1` because `Qwen/Qwen2.5-3B-Instruct-AWQ` cannot be resolved without `refs/main`.

## 6. Smoke results

All smokes: one engine at a time, `--network host`, HF cache read-only, `HF_HUB_OFFLINE=1`, target `--max-model-len/--context-length 2048`, memory fraction 0.80, seed 0, max 32 running requests. Every container was stopped and removed before the next launch (`docker ps -a` at the end shows only the unrelated, long-exited `scenesweep-api`).

### 6.1 Chat-template parity (spec §4) — trace row 0 (`record_id=q000002-c0`) on `/v1/completions` with the pre-templated prompt

| Source | `prompt_tokens` |
|---|---|
| trace `prompt_tokens_qwen` | **265** |
| vLLM 0.29.0 `usage.prompt_tokens` | **265** |
| SGLang 0.5.20 `usage.prompt_tokens` | **265** |

All three agree (`MATCH` on every launch, including the spec arms; row 19: 699 on both engines; the full first-20 vector matches on both engines). Both engines also produced the identical greedy text `{"is_injection": false, "is_off_topic": false}` (15 completion tokens, `finish_reason: stop`). The completions-endpoint-with-pre-templated-text path the trace was built for is therefore the parity-safe path; the server-side chat template is never applied.

### 6.2 vLLM launches

| Arm | Env | Result | Model load | KV available | Graphs | Container VRAM (WSL `nvidia-smi`) |
|---|---|---|---|---|---|---|
| target only (default = V2 runner) | `VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_SERVER_DEV_MODE=1` | healthy in 88 s | 1.95 GiB | 1.46 GiB = 42,384 tokens ("Maximum concurrency for 2,048 tokens per request: 20.70x") | 0.09 + 0.05 GiB, default list to 64 | 4397 MiB |
| target only, **V1 runner**, capture list `1 2 4 8 16 32`, `--max-num-batched-tokens 2048` | + `VLLM_USE_V2_MODEL_RUNNER=0` | healthy in 79–83 s | 1.95 GiB | 1.51 GiB = 44,112 tokens | 0.06 GiB | 4407 MiB |
| `draft_model` k=3 (`{"method":"draft_model","model":"Qwen/Qwen2.5-0.5B-Instruct","revision":"7ae55...","num_speculative_tokens":3}`) | pin memory only (no dev mode) | healthy in 100 s; V1 runner forced | 2.88 GiB (target+draft) | **0.5 GiB = 10,976 tokens** ("5.36x") | 0.42 GiB (default list to 256) | 4415 MiB |
| `ngram` (`{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4}`) | pin memory only | healthy in 88 s; V1 runner forced | 1.95 GiB | 1.31 GiB = 38,144 tokens | 0.28 GiB | 4419 MiB |

First launch without `VLLM_WSL2_ENABLE_PIN_MEMORY=1` exited in 52 s with `RuntimeError: UVA is not available` (§3.5). The `draft_model` launch logged `ERROR [repo_utils.py:119] Error retrieving safetensors: Cannot reach https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/7ae55.../model.safetensors: offline mode is enabled` twice and then loaded the draft from the cache anyway — noise, but pass the snapshot dir instead. Kernel: `awq.py:448] Using MarlinLinearKernel for AutoAWQMarlinLinearMethod` (vLLM's `--quantization awq` resolves to `auto_awq` -> Marlin). Attention: `Using FLASH_ATTN attention backend ... FlashAttention version 2`. Engine init ≈ 30 s of which torch.compile ≈ 20 s (cached under `/root/.cache/vllm` inside the container, lost with the container).

Spec counters after 20 greedy requests (rows 0–19, max_tokens 64), verbatim:

```
draft_model k=3:
vllm:spec_decode_num_drafts_total{engine="0",model_name="Qwen/Qwen2.5-3B-Instruct-AWQ"} 133.0
vllm:spec_decode_num_draft_tokens_total{...} 399.0
vllm:spec_decode_num_accepted_tokens_total{...} 336.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{...,position="0"} 122.0 / position="1" 111.0 / position="2" 103.0
ngram k=3, lookup 4:
vllm:spec_decode_num_drafts_total 56.0 / num_draft_tokens_total 168.0 / num_accepted_tokens_total 93.0 / per_pos 42.0, 31.0, 20.0
```

Routes: `GET /health -> 200`, `GET /version -> 200 {"version":"0.29.0"}`, `POST /reset_prefix_cache -> 200 {"success":true}` (dev mode), `GET /reset_prefix_cache -> 405`, `POST /reset_prefix_cache -> 404` without dev mode. Reset effectiveness: `prefix_cache_queries_total 1068 / hits_total 736` -> reset -> re-send row 0 -> `queries 1333 / hits 736` (265 new queries, 0 new hits).

### 6.3 SGLang launches

| Arm | Result | Weights | KV pool (`max_total_num_tokens`) | Container VRAM | Notes |
|---|---|---|---|---|---|
| target only, `--quantization awq` | healthy 95–101 s | 1.98 GB | 57,014 tokens (0.98 + 0.98 GB) | 5477–5529 MiB | `available_gpu_mem=0.00 GB` after graphs; prefill graph 40.8 s / 0.40 GB; decode graphs `bs=[1, 2, 4, 8]` (default) |
| target only, `--mem-fraction-static 0.60` | healthy 91 s | 1.98 GB | 28,239 tokens | 4521 MiB | `available_gpu_mem=0.72 GB` |
| target only, **`--quantization awq_marlin`** | healthy **288 s** | 1.98 GB | 56,786 tokens | 5345 MiB | see kernel finding below |
| STANDALONE k=3 (draft = snapshot dir, `unquant`, steps 3 / topk 1 / draft tokens 4) | healthy 112–114 s | 1.98 + 0.98 GB | **17,142 tokens** (target 0.29+0.29 GB fp16, draft 0.10+0.10 GB bf16) | 5909–5913 MiB | `--cuda-graph-bs-decode 1 2 4 8 16 32` honoured for draft decode/extend graphs |
| NGRAM (steps 3 / draft tokens 4) | healthy 97 s | 1.98 GB | 57,014 tokens | 5565 MiB | "The mixed chunked prefill are disabled because of using ngram speculative decoding."; topk resolved to 10 |

Two STANDALONE launches failed before these (offline draft resolution; draft inheriting `awq`) — errors and fixes in §4.2.

Spec gauges/counters after 20 greedy requests, verbatim (labels elided):

```
STANDALONE: sglang:spec_accept_length 3.375   sglang:spec_accept_rate 0.7916666666666666   sglang:spec_verify_calls_total 139.0   spec_num_steps 3.0   spec_num_draft_tokens 4.0
NGRAM:      sglang:spec_accept_length 2.8     sglang:spec_accept_rate 0.6                  sglang:spec_verify_calls_total 261.0   spec_num_steps 3.0   spec_num_draft_tokens 4.0
```

Baseline scheduler gauges after the same load: `num_running_reqs 1.0`, `num_queue_reqs 0.0`, `cache_hit_rate 0.6766`, `cached_tokens_total{cache_source="device"} 4200.0`, `prompt_tokens_total 6930.0`, `kv_cache_memory_usage_gb 1.957`, `max_total_num_tokens 57014.0`.

**Kernel finding.** Every SGLang launch with `--quantization awq` logged: `Detected that the model can run with awq_marlin, however you specified quantization=awq explicitly, so forcing awq. Use quantization=awq_marlin for faster inference` and `awq quantization is not fully optimized yet. The speed can be slower than non-quantized models.` With the stock client at concurrency 4 over the first 20 chat rows (`--ignore-eos`, per-row output lengths):

| Server | Mean TPOT | Mean ITL | Mean TTFT | Mean E2EL | duration |
|---|---|---|---|---|---|
| vLLM V1 runner, awq (= Marlin) | 14.64 ms | 14.76 ms | 322 ms | 796 ms | 4.81 s |
| SGLang awq, 0.80 | 75.47 ms | 75.85 ms | 486 ms | 2921 ms | 19.16 s |
| SGLang awq, 0.60 | 76.55 ms | 78.10 ms | 482 ms | 2989 ms | 19.67 s |
| SGLang **awq_marlin**, 0.80 | **15.68 ms** | 15.57 ms | 338 ms | 838 ms | 5.00 s |

The 5x decode gap is the kernel, not memory (0.60 changed nothing) and not the engine. Both engines must run the Marlin AWQ kernel (Decisions). The Marlin launch spent ~4 min in prefill-graph capture before `/health` answered, so readiness timeouts must allow ≥ 6 min for SGLang.

### 6.4 Client end-to-end (both engines)

`vllm bench serve` from a GPU-less `vllm/vllm-openai:v0.29.0` container (`--entrypoint bash -c "pip install pandas && exec vllm bench serve ..."`), `--backend openai --endpoint /v1/completions --dataset-name custom --dataset-path /traces/chat_v1.jsonl --custom-output-len -1 --skip-chat-template --disable-shuffle --no-oversample --num-prompts 20 --max-concurrency 4 --request-rate inf --ignore-eos --temperature 0 --seed 0 --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result --save-detailed`:

- vLLM: 20/20 success, `input_lens == trace prompt_tokens_qwen` True, `output_lens == trace output_tokens` True, `len(itls[i]) == output_lens[i]-1` **True** for all i, `vllm:request_success_total{finished_reason="length"} 20.0`.
- SGLang baseline (awq and awq_marlin): 20/20, both equalities True, `len(itls[i]) == output_lens[i]-1` **True** (`--stream-interval` default 1 streams per token).
- SGLang STANDALONE: 20/20, both equalities True, but `len(itls[i]) == output_lens[i]-1` **False** — `[4, 4, 4, 38, 2, 4]` chunks for `[14, 14, 14, 135, 9, 14]` tokens: accepted tokens arrive bundled per verify step, so the client's `itl` is inter-chunk latency under speculation while `tpot` stays per-token. A streaming request on SGLang does emit the final usage chunk (`"choices":[],"usage":{"prompt_tokens":265,"total_tokens":273,"completion_tokens":8,"reasoning_tokens":0}`), so `output_lens` are server-reported on both engines.

vLLM under speculation was not run through the streaming client in this task; the same bundling is expected (vLLM also emits accepted tokens per step) and must be checked in Task 2's calibration before ITL is reported for any spec arm.

## 7. Clock-pin attempt and host telemetry (Step 7)

Windows-side `nvidia-smi.exe` (driver 610.88), called from WSL in a non-elevated shell, verbatim:

```
$ /mnt/c/Windows/System32/nvidia-smi.exe -lgc 2055,2055
The current user does not have permission to change clocks for GPU 00000000:01:00.0.
Terminating early due to previous errors.
$ ... -lgc 2055 / -lmc 8001 / -rgc / -rmc      -> same permission error
$ ... -ac 8001,2055
The requested functionality has been deprecated.
$ ... -pl 100
Changing power management limit is not supported in current scope for GPU: 00000000:01:00.0.
$ nvidia-smi -lgc 2055,2055        (WSL side)
The current user does not have permission to change clocks for GPU 00000000:01:00.0.   (exit 4)
$ ... -q -d CLOCK   -> Applications Clocks / Default Applications Clocks: "Requested functionality has been deprecated"; Max Clocks: SM 3105 MHz, Memory 8001 MHz
```

**Clock pinning is unavailable from this environment** (permission-denied from both sides, application clocks deprecated, power limit not settable). Spec §6 step 1's contingency is in force: raised cooldown between runs and `clock_cv` (SM-clock coefficient of variation from the per-second telemetry) as a covariate. Untested: an elevated (Administrator) Windows prompt might accept `-lgc`; that is a controller decision, and the result — either way — belongs in `env.json`.

Idle host telemetry (Windows side, `--query-gpu=...`):

```
memory.used 0 MiB, memory.total 6141 MiB, power.draw 10.65 W, power.max_limit 100.00 W (power.limit [N/A], min 5 W, default 55 W),
clocks.sm 2055 MHz, clocks.mem 7001 MHz, clocks.max.sm 3105 MHz, temperature 54 C, clocks_event_reasons.active 0x0000000000000024, pstate P3, utilization 0 %
GPU: NVIDIA GeForce RTX 4050 Laptop GPU, compute_cap 8.9, VBIOS 95.07.42.00.99
```

Host reservation at idle = **0 MiB** (hybrid graphics — the desktop is on the iGPU; spec §3.1's ~0.5 GB DWM line is 0 on this machine). The WSL-side `nvidia-smi` reports the same memory numbers (0 MiB idle; 4397–5913 MiB during the smokes above), `power.limit [N/A]`, and matching SM clock/temperature — it is usable for in-run telemetry, the Windows binary adds `power.max_limit`, `pstate` and throttle reasons. `power.draw` is reported on both sides (10.65 W vs 12.41 W idle, sampled seconds apart).

Measured budget lines for spec §3.1 (0.80 fraction, 2048 ctx): vLLM target-only leaves 1.46–1.51 GiB for KV with 0.06–0.14 GiB of graphs; adding the fp16 draft leaves 0.5 GiB (10,976 tokens) — the draft costs ~1 GiB of weights plus 0.42 GiB of graphs at the default capture list, i.e. spec prediction 2 ("roughly halves maximum concurrency") is if anything optimistic under vLLM at this fraction. SGLang's 0.80 means something different (static allocation only): its containers sit at 5.3–5.9 GiB of 6.1 with `available_gpu_mem=0.00 GB`, versus vLLM's 4.4 GiB at the "same" 0.80.

---

## 8. Decisions (contract for Task 2)

**Final pins:** `vllm/vllm-openai:v0.29.0` (`sha256:c291...1ae1`) for server and client; `lmsysorg/sglang:v0.5.20-runtime` (`sha256:00b0...6800`). Models at the shas in §2. **No arm is dropped and no pin is stepped back**: `draft_model` and `ngram` work on vLLM, `STANDALONE` and `NGRAM` work on SGLang, all with non-zero acceptance. RQ3's speculation comparison is symmetric.

**Common to every container:** `--gpus all --network host`, `-v /home/edamiba/.cache/huggingface:/root/.cache/huggingface:ro`, `-e HF_HUB_OFFLINE=1`; models by repo id + `--revision <sha>` for the target, and **draft models by snapshot directory** (both engines; the only form that resolves offline on both). Memory fractions are set from measured free memory per spec §3.1, and equalised by *measured footprint* (WSL `nvidia-smi memory.used` + engine-reported KV tokens), never by the nominal fraction, because 0.80 means different things to the two engines.

### vLLM launch (per arm)

| Purpose | Flag / env | Value used in smokes |
|---|---|---|
| model / revision / served name | `--model Qwen/Qwen2.5-3B-Instruct-AWQ --revision 3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd --served-model-name Qwen/Qwen2.5-3B-Instruct-AWQ` | (served name required under offline mode so the `model_name` label and the client's `--model` are the repo id) |
| quantization | `--quantization awq` | resolves to Marlin (`MarlinLinearKernel for AutoAWQMarlinLinearMethod`); `awq_marlin` is also an accepted literal if explicit symmetry with SGLang is preferred — record whichever in config |
| context | `--max-model-len 2048` | |
| max running requests | `--max-num-seqs N` | 32 |
| chunked prefill size | `--enable-chunked-prefill --max-num-batched-tokens 2048` | |
| memory fraction | `--gpu-memory-utilization 0.80` (or `--kv-cache-memory-bytes`) | |
| seed | `--seed 0` | |
| port | `--port 8000` | |
| CUDA-graph capture list | `--cudagraph-capture-sizes 1 2 4 8 16 32` | verified honoured; `--enforce-eager` never |
| prefix cache | `--enable-prefix-caching` / `--no-enable-prefix-caching` | default None = on for this model |
| spec: draft model | `--speculative-config '{"method":"draft_model","model":"<0.5B snapshot dir>","num_speculative_tokens":k}'` | k=3 verified with `"model":"Qwen/Qwen2.5-0.5B-Instruct","revision":"7ae55..."` |
| spec: ngram | `--speculative-config '{"method":"ngram","num_speculative_tokens":k,"prompt_lookup_max":4}'` | `prompt_lookup_min` copies max when omitted |
| **required env** | `-e VLLM_WSL2_ENABLE_PIN_MEMORY=1 -e VLLM_USE_V2_MODEL_RUNNER=0 -e VLLM_SERVER_DEV_MODE=1` | pin memory: WSL2 pinned-memory gate; V2=0: **all vLLM arms on the V1 model runner** (spec arms are forced there anyway, and the baseline must not silently run a different runner); dev mode: mounts `/reset_prefix_cache` |
| health | `GET /health` -> 200 | poll; `GET /version` gives `{"version":"0.29.0"}` for env.json |
| cache reset | `POST /reset_prefix_cache` -> `{"success":true}` (405 on GET, 404 without dev mode) | check `success`; retry while requests are in flight |
| metrics | `GET /metrics` | `vllm:kv_cache_usage_perc`, `vllm:num_requests_running`, `vllm:num_requests_waiting`, `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total`, `vllm:num_preemptions_total`, `vllm:spec_decode_num_drafts_total`, `vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_accepted_tokens_total`, `vllm:spec_decode_num_accepted_tokens_per_pos_total{position}` (no "emitted" counter exists; emitted = accepted + drafts) |

### SGLang launch (per arm)

| Purpose | Flag | Value used in smokes |
|---|---|---|
| model / revision / served name | `--model-path Qwen/Qwen2.5-3B-Instruct-AWQ --revision 3559b226e8ce77211e2c1bd7ddfb7686fec4d6dd --served-model-name Qwen/Qwen2.5-3B-Instruct-AWQ` | |
| quantization | **`--quantization awq_marlin`** | `awq` forces the unoptimised kernel (5x slower decode, §6.3) |
| context | `--context-length 2048` | |
| max running requests | `--max-running-requests N` | 32 |
| chunked prefill size | `--chunked-prefill-size 2048` | (default was already 2048) |
| memory fraction | `--mem-fraction-static 0.80` (or `--max-total-tokens`) | static allocation only; graphs/activations come on top |
| seed | `--random-seed 0` | |
| port | `--port 30000 --host 0.0.0.0` | |
| CUDA-graph capture list | `--cuda-graph-bs-decode 1 2 4 8 16 32` (+ optionally `--cuda-graph-bs-prefill`, `--cuda-graph-backend-prefill`) | `--cuda-graph-bs` / `--cuda-graph-max-bs` do not exist at 0.5.20; prefill graphs cost 36–41 s and ~0.4 GB per launch and are on by default |
| prefix cache | on by default; `--disable-radix-cache` to turn off | |
| metrics | `--enable-metrics` (required for `/metrics`) ; `--enable-cache-report` for per-request cached tokens | |
| spec: draft model | `--speculative-algorithm STANDALONE --speculative-draft-model-path <0.5B snapshot dir> --speculative-draft-model-quantization unquant --speculative-num-steps k --speculative-eagle-topk 1 --speculative-num-draft-tokens k+1` | k=3 verified (3 / 1 / 4) |
| spec: ngram | `--speculative-algorithm NGRAM --speculative-num-steps k --speculative-num-draft-tokens k+1` | topk is forced to `--speculative-ngram-max-bfs-breadth` (default 10); window-size flags do not exist at 0.5.20 (`min/max-bfs-breadth`, `max-trie-depth` are the tunables) |
| health | `GET /health` -> 200 (no generation) ; `GET /ready` -> 200/503 | poll `/ready` then `/health`; allow ≥ 6 min with awq_marlin |
| cache reset | `GET` or `POST /flush_cache` -> 200 with body starting `Cache flushed.` | status is 200 even when refused — check the body |
| metrics names | `sglang:num_used_tokens` + `sglang:token_usage` (KV usage), `sglang:max_total_num_tokens`, `sglang:num_running_reqs`, `sglang:num_queue_reqs` (waiting), `sglang:cached_tokens_total{cache_source="device"}` (prefix hits, tokens) vs `sglang:prompt_tokens_total` (queries proxy), `sglang:spec_accept_length` / `sglang:spec_accept_rate` (interval gauges), `sglang:spec_verify_calls_total` (cumulative) | cumulative accepted/proposed drafts only per request via `"return_spec_tokens_details": true` -> `sglext.spec_tokens_details.{spec_num_correct_drafts, spec_num_proposed_drafts, spec_verify_ct, spec_accept_rate, spec_accept_length}` |
| container | `--shm-size 2g` | used in all smokes |

### Client (`vllm bench serve`, GPU-less container of the vLLM image)

- Container: `docker run --rm --network host -v <traces>:/traces:ro -v <hf cache>:/root/.cache/huggingface:ro -v <out>:/out -e HF_HUB_OFFLINE=1 --entrypoint bash vllm/vllm-openai:v0.29.0 -c "pip install -q pandas && exec vllm bench serve ..."` — or a derived image `FROM vllm/vllm-openai:v0.29.0; RUN pip install pandas` recorded in `env.json`. The runtime `pip install` needs network (~15 s); the derived image is the reproducible option.
- Backend: `--backend openai --endpoint /v1/completions` (completions endpoint; `openai-chat` is not used — the trace is pre-templated and parity holds on completions).
- Model/tokenizer: `--model Qwen/Qwen2.5-3B-Instruct-AWQ --tokenizer <3B snapshot dir>`.
- Dataset: `--dataset-name custom --dataset-path /traces/<trace>_v1.jsonl --custom-output-len -1 --skip-chat-template --disable-shuffle --no-oversample --num-prompts N` (`-1` is mandatory: the default is 256; `--output-len` would override it — never pass `--output-len`).
- Ordering guarantee: `--disable-shuffle` -> request `i` is trace row `i`, `request_id = <prefix><i>`; use `--request-id-prefix` per run.
- Termination: `--ignore-eos` for P1/P2 (fixed `output_len` = trace `output_tokens`); omit it for P3 natural termination.
- Sampling: `--temperature 0` for greedy; `--top-p/--top-k` only when set. `--extra-body '<json>'` merges into every request body (e.g. `{"return_spec_tokens_details": true}` on SGLang, `response_format` for P3; note `repetition_penalty: 1.0` is always sent).
- Load: `--max-concurrency C --request-rate inf` (closed loop) or `--request-rate R --burstiness 1.0` (Poisson).
- Output: `--save-result --save-detailed --result-dir /out --result-filename <run>.json --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --metadata k=v ...`; per-request arrays are `input_lens`, `output_lens`, `ttfts`, `itls`, `start_times`, `generated_texts`, `errors`. Leave `--ready-check-timeout-sec` at 0 (no pre-run request) and `--num-warmups 0`; the runner does its own warmup + cache reset (spec §4 step 5).
- Metric semantics to state in the writeups: `ttft` = first chunk; `itl` = inter-*chunk*; `tpot = (e2el - ttft)/(output_len-1)`; under speculation chunks bundle accepted tokens (SGLang verified: 4 chunks for 14 tokens), so report `tpot`/`e2el` as the per-token latency metrics for spec arms and treat `itl` as per-step latency; record `len(itls[i])` vs `output_lens[i]` as the tripwire.

### Platform notes carried into `env.json`

`docker 28.0.4 (native WSL2, nvidia runtime, nvidia-container-cli 1.17.8)`, WSL kernel `5.15.167.4-microsoft-standard-WSL2`, driver `610.88`, GPU `RTX 4050 Laptop 6141 MiB`, `power.max_limit 100 W`, `power.default_limit 55 W`, host idle VRAM 0 MiB, clock pinning unavailable (contingency: cooldown + `clock_cv`), chat-template sha256 `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`, vLLM env `VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_V2_MODEL_RUNNER=0`.
