# Pickr Inference Benchmarking — Design Spec

**Status:** approved for full rigor, weeks-scale scope (see §9).
**Source brief:** `smartshop-inference-benchmark-brief.md` (repo root; written against a placeholder app name "SmartShop" — this spec uses the real app name, Pickr, throughout).

---

## 1. Problem framing

Pickr is an AI shopping assistant (`app/agents.py`: `CoordinatorAgent` routing to six specialized agents) that calls OpenAI's hosted API exclusively (`gpt-3.5-turbo`, `text-embedding-3-small`) — there is no local, open-weight model anywhere in the app, and no control over batching, speculative decoding, or quantization on a closed hosted endpoint.

The benchmark brief's entire premise (spec-decode crossover, vLLM vs. SGLang, quantization, KV-cache co-tenancy) requires a self-hosted, open-weight model under vLLM/SGLang. This spec bridges that gap without touching `app/`:

- **Capture** Pickr's real prompt-construction logic by driving `CoordinatorAgent` in-process with a generated query set (sampled from the real `data/*.csv` catalog, so routing and prompt templates are genuine app behavior, not hand-written approximations) and recording the exact messages sent to OpenAI via a pure-observation wrapper around `app/openai_client.py`'s client.
- **Replay** the frozen trace against a locally-served open-weight model stack on vLLM and SGLang, sweeping the engine/spec-decode/concurrency/quantization/structured-output axes the brief cares about.

This keeps the workload "imposed by the product, not synthesized to justify a benchmark" (brief §3) at the prompt-construction level, while accepting that the *serving* side is necessarily a stand-in model, since OpenAI's hosted endpoint cannot be instrumented this way.

**Hard rule preserved:** no refactor of `app/` code for benchmarking convenience. The capture wrapper observes; it does not change behavior. `bench/` is a fully separate deliverable — its dependencies (vLLM, SGLang) never enter `requirements.txt` or the Dockerfile, so the deployed app's footprint is untouched.

---

## 2. Traffic source (pluggable, per approved decision)

`bench/harness/capture.py` supports two trace sources behind one interface, selected by config:

- **`synthetic`** (default, available now): a query generator samples realistic queries from `data/products.csv` / `data/reviews.csv` / `data/store_policies.csv` against each of Pickr's six real routing paths, driven through `CoordinatorAgent.handle_query` in-process. The OpenAI client wrapper records the exact system+user messages, real input token counts, and real `gpt-3.5-turbo` output text/token counts to a JSONL trace file. This exercises real code paths and produces real prompt lengths; output lengths are a *reference* distribution for calibration, since actual benchmark output length is a property of the locally-served model/decoding config at run time, not literally reproduced from OpenAI.
- **`production`** (swap in once Pickr is deployed live on HF Spaces): the same trace file format, populated from real request logs instead of the generator. No other harness code changes — `replay.py` consumes trace files identically regardless of source.

**Disclosure requirement carried into every writeup:** while `synthetic` is in use, length distributions are labeled "derived from a representative synthetic query set exercised through the real app's real code paths," not production traffic. Switching to `production` removes this caveat entirely and is a pure config change, not a re-architecture.

---

## 3. Model stack

| Role | Model | Notes |
|---|---|---|
| Target | Qwen2.5-3B-Instruct, AWQ 4-bit (~2GB) | Chosen over 7B/8B alternatives specifically so the full concurrency sweep (1→32) and RQ4 co-tenancy analysis have headroom to run without OOM dominating the results. |
| Draft (standalone spec-decode) | Qwen2.5-0.5B-Instruct, same tokenizer/family | Cross-family draft models collapse acceptance rates (brief §9) — same-family is non-negotiable. |
| Spec-decode (no draft model) | ngram / prompt-lookup | Tried especially on Workload B (review summarization) — high expected acceptance at near-zero VRAM cost, per brief §9. |
| Spec-decode (best-effort) | EAGLE-3 | Only if a compatible checkpoint exists for Qwen2.5-3B-Instruct. Likely unavailable at this size — see §8 risk 3. |
| Embedding | small local embedding model (~0.3GB) | For RQ4's "target + draft + embedding co-resident" memory accounting. Specific model TBD at P0 build time. |

Flag names for spec-decode config (`--speculative-config` vs `--speculative-algorithm`/`--speculative-draft-model-path`, etc.) will be verified against each engine's installed `--help` output at build time, never assumed from documentation or training data (brief §9).

---

## 4. Workload mapping

| Workload | Source agents | Character |
|---|---|---|
| **A — Interactive chat** | Recommendation, Comparison, PriceComparison, StorePolicy, FAQ | batch≈1, TTFT/TPOT-sensitive |
| **B — Bulk summarization** | ReviewSummarization | high-batch, offline, throughput-bound; naturally large prompts (review text blocks) |
| **C — Structured JSON** | Recommendation/Comparison prompts, reused | **Does not exist in Pickr today** — agents return prose. The JSON-schema constraint is imposed at the harness's decoding config on reused prompts, not by changing app output format. This is a benchmark-only construct layered onto real prompts. |
| **A+B mixed** | A and B traces interleaved | for RQ3/RQ4 contention analysis |

---

## 5. Staged experiment matrix

Full cross-product (2 engines × 4 spec-methods × 5 draft-lengths × 6 concurrency × 4 workloads × 2 structured × 2 quant ≈ 7,680 configs × 3-5 reps) is intractable. Staged, coarse-then-dense:

- **P1 — Baselines.** Engine{vLLM,SGLang} × Workload{A,B} × spec=off × AWQ × concurrency{1,8,32}. 12 configs × 3 reps = 36 runs. Establishes variance floor; verifies the client isn't the bottleneck before the server (brief §8, most common way benchmark studies get silently invalidated).
- **P2 — RQ1 crossover, coarse pass.** Engine{2} × spec-method{ngram, standalone-draft} × draft-length{1,3,5,8} (dropping 2 — brackets the known diminishing-returns curve without a redundant point) × concurrency{1,4,16,32} × workload{A,B}. 2×2×4×4×2 = 128 configs × 3 reps = 384 runs.
- **P2 — dense resample.** Only near the crossover region the coarse pass finds: add concurrency{2,8}, bump to 5 reps, restricted to the spec-method(s)/draft-length(s) that mattered.
- **P3 — RQ2 constrained decoding.** Re-run the crossover-relevant subset from P2 with `structured=on` on Workload C, plus correctness verification (greedy token-for-token diff vs. non-speculative baseline; JSON-schema conformance rate). Not a fresh full sweep.
- **P4 — RQ4 co-tenancy / mixed load.** Engine × prefix-cache{on,off} × A+B mixed schedule. Small matrix; depth (per-model VRAM/KV-cache accounting) over breadth.
- **P5 — Jetson.** Timeboxed replication of exactly one P2 config plus the ngram/summarization finding. Ship without it if it overruns the box (brief §8).

**Quantization axis deviation:** FP16 for a 3B model (~6GB alone) leaves no room for KV cache, draft, or embedding. Full FP16 sweep is not feasible on this hardware; FP16 is run only as a single batch=1 calibration baseline, not across the matrix.

---

## 6. Harness architecture

```
bench/
  harness/
    loadgen.py        # open-loop, configurable arrival process (Poisson default), async HTTP
                        # against the engine's OpenAI-compatible endpoint, streams tokens -> TTFT/ITL
    replay.py         # feeds a frozen trace file through loadgen per a config's arrival/concurrency
    capture.py         # pickr instrumentation: wraps app/openai_client.py's client, drives
                        # CoordinatorAgent with generated (or production, once available) queries,
                        # records exact prompts -> trace files. Pluggable trace source (§2).
    metrics_client.py  # scrapes each engine's own Prometheus /metrics (acceptance rate, KV
                        # utilization, queue depth)
    gpu_monitor.py     # background NVML/nvidia-smi sampler: VRAM, SM util, clocks, temp, power
    env_capture.py     # snapshots driver/CUDA/torch/engine-version+SHA/model-revision/launch-cmd
    correctness.py     # spec-decode token-for-token diff vs. greedy baseline; JSON-schema
                        # conformance check
    orchestrator.py    # sequences configs in randomized order, enforces cooldown between runs,
                        # asserts acceptance-rate-nonzero before recording, retries on failure
  configs/             # one declarative YAML per experiment config
  traces/              # versioned fixed input traces (one per workload)
  results/             # raw JSONL, one dir per run, env metadata included
  analysis/            # scripts/notebooks -> figures; reads only from results/, never re-runs
  figures/
  docs/                # per-phase writeups
```

Both vLLM and SGLang expose an OpenAI-compatible `/v1/chat/completions`, so `loadgen.py` is engine-agnostic. One engine live at a time, fresh process, fresh CUDA context (brief §7.7).

**Platform:** vLLM and SGLang are Linux-only. This machine's RTX 4050 (6GB) is visible under WSL2 Ubuntu via GPU passthrough (verified via `nvidia-smi`) — all engine/harness work runs inside WSL2, not native Windows. `nvidia-smi -lgc` clock-pinning behavior under WSL2 passthrough is unverified and must be checked early in P0 (see §8 risk 5).

---

## 7. Result artifact schema

Each run writes `bench/results/<run_id>/`:

- `config.yaml` — resolved config that produced this run
- `env.json` — driver/CUDA/torch/engine version+SHA/flash-attn or flashinfer version/model revision hash/full launch command/GPU name
- `requests.jsonl` — per request: id, timestamps, TTFT, per-token ITLs, e2e latency, input/output token counts, workload tag, trace ref, accepted-tokens (spec decode), schema-valid (structured output), error if any
- `gpu_samples.jsonl` — periodic VRAM/SM-util/clock/temp/power samples
- `engine_metrics.jsonl` — periodic scrape of engine's own acceptance-rate/KV-util/queue-depth
- `summary.json` — computed p50/p90/p99 TTFT/TPOT/e2e, throughput (tok/s, req/s), goodput%, mean acceptance rate, peak VRAM, throttle-detected flag, acceptance-assertion pass/fail
- `meta.json` — run id, `bench/` git SHA, timestamp, phase, RQ tag

---

## 8. Assumptions/risks flagged against the brief

1. **§7.2 length distributions** assume production traffic exists; it doesn't yet (app not deployed live). Mitigated by the pluggable synthetic/production trace source (§2), with explicit disclosure while synthetic is in use.
2. **Workload C** doesn't exist in Pickr's current agents (prose only, no JSON schema output). The constraint is imposed at the harness's decoding config on reused prompts (§4).
3. **EAGLE-3** ("if a checkpoint fits") is genuinely conditional on Qwen2.5-3B checkpoint availability, which is unconfirmed and may not exist. The brief itself hedges this; likely resolves to "not tested."
4. **FP16 quantization arm** is infeasible as a full sweep at this model size on 6GB; narrowed to a single batch=1 calibration point (§5).
5. **WSL2 thermal control** — `nvidia-smi -lgc` under GPU passthrough is unverified against bare-metal Linux behavior. Must be checked in P0; if it doesn't work, requirement #6 needs an explicit caveat in every writeup rather than silent omission.
6. **Full cross-product infeasibility** — staged design cuts ~7,680 configs to a few hundred, risking a missed crossover between coarse-grid points. The dense-resample stage mitigates but doesn't eliminate this; stated as a limitation, not silently absorbed.

---

## 9. Scope decision and phasing

Per explicit approval: **full rigor as scoped**, not a fast directional pilot. The weeks-scale estimate reflects the brief's own non-negotiables (§7: minimum 3 reps with randomized order and cooldowns, thermal logging, correctness verification as part of the experiment not a side task, full environment capture) — almost all of that time is P0 engineering (a correct open-loop load generator with TTFT/ITL capture, Prometheus scraping, NVML monitoring, env snapshotting, cooldown-aware orchestration), not GPU wall-clock. Once P0 exists, later phases are comparatively fast since most rep/cooldown time is unattended.

| Phase | Scope | Estimate |
|---|---|---|
| P0 | Harness + capture pipeline + plotting skeleton | 1-2 weeks |
| P1 | Baselines, variance floor, client-not-bottleneck check | 2-3 days |
| P2 | RQ1 crossover (coarse + dense resample) + writeup | 1-2 weeks |
| P3 | RQ2 constrained decoding + correctness verification | 3-5 days |
| P4 | RQ4 co-tenancy/mixed load | 3-5 days |
| P5 | Jetson (timeboxed) | ≤1 week, cut if it overruns |

Total: roughly 5-7 weeks part-time. A writeup ships after each phase (brief §8's own instruction), so results are visible incrementally rather than only at the end.

---

## 10. Next step

This spec covers the overall staged design. Per the brainstorming process, only **P0** gets a detailed implementation plan next (via the writing-plans skill) — P2 onward depend on what P0/P1 actually measure, so planning them in detail now would be premature.
