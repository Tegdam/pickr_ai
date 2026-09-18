# Pickr Inference Benchmarking — Design Spec v2

**Status:** approved in brainstorming 2026-09-16/17; supersedes `2026-09-02-pickr-inference-benchmark-design.md` in full.
**Source brief:** `smartshop-inference-benchmark-brief.md` at commit `b7902da` (the version with §7 Phases folded in). The brief uses the placeholder name "SmartShop"; this spec uses the real app name, Pickr.
**Scope of this document:** the whole staged study. Only Phase 0 gets a detailed implementation plan next; Phase 1+ plans are written after Phase 0's calibration numbers exist, because those numbers set the later ladders.

---

## 1. Problem framing

Pickr (`app/agents.py`: `CoordinatorAgent` routing to specialised agents; `app/conversation.py` for multi-turn state; `app/guardrails.py`) calls OpenAI's hosted API exclusively — `gpt-3.5-turbo` for generation, `text-embedding-3-small` for the policy index. There is no local model in the app, and a hosted endpoint exposes none of the axes the brief studies (batching, speculation, quantization, KV cache).

The bridge, unchanged in principle from v1: **capture** Pickr's real prompt construction by driving `CoordinatorAgent` in-process and recording the exact messages it sends, then **replay** those frozen traces against a self-hosted open-weight model on vLLM and SGLang. The workload stays "imposed by the product" at the prompt-construction level; the serving side is necessarily a stand-in model.

Two facts changed since v1:

- Pickr now makes real JSON-mode calls (`response_format={"type":"json_object"}`) in `CoordinatorAgent._classify_intent`'s LLM fallback (`app/agents.py:264`) and in guardrails (`app/guardrails.py:72`). These are `json_object` mode, not schema-constrained, so they belong to the interactive workload; Workload C remains a decoding-config overlay on real prompts (§3.2).
- The app is live on Hugging Face Spaces with LangSmith tracing wired through `wrap_openai` in `app/openai_client.py`, so real emitted prompts are exportable. Traffic to date is the author's own testing.

**Hard rule preserved:** no change to `app/` for benchmarking convenience. The capture wrapper observes only. `bench/` is a separate deliverable on the `benchmarking` branch; its dependencies never enter `requirements.txt` or the `Dockerfile`.

---

## 2. Decisions taken in brainstorming

| # | Decision | Chosen | Rejected |
|---|---|---|---|
| D1 | SLO | Brief's example, pre-registered with a mechanical fallback ladder and a sensitivity curve (§3.3) | Looser perception-derived SLO; deriving SLO from P1 data |
| D2 | Trace source | In-process generator as corpus; LangSmith export as validation set (§3.2) | Generator only; LangSmith only |
| D3 | Engine isolation | One pinned Docker container per run; client/runner/monitor in a host `uv` venv under WSL2 (§4) | Two host venvs |
| D4 | Load client | `vllm bench serve` for *both* engines, run from a GPU-less container on `--network host` (§4) | SGLang `bench_serving.py` (fallback if A's pinned version lacks custom-dataset or per-request output) |
| D5 | Model family | Qwen2.5-3B-Instruct-AWQ target, Qwen2.5-0.5B-Instruct draft (§3.1) | Llama-3.2-3B/1B (4× KV cost); Qwen3-4B/0.6B (2.6 GB weights, 147 KB/token KV, thinking mode) |
| D6 | EAGLE arm | **Dropped.** No EAGLE/EAGLE-3 head exists on the HF hub for Qwen2.5-3B-Instruct (verified 2026-09-16; all "Qwen2.5-3B eagle" hits are unrelated LoRA/SFT repos; Qwen2.5 EAGLE-3 heads start at 7B). Reported as a finding: on 6 GB the available drafting methods are prompt-lookup and a standalone draft model. | Switching to Qwen3-4B for `AngelSlim/Qwen3-4B_eagle3` |
| D7 | Runner architecture | Single Python package, declarative sweep files expanded to resolved run-configs, plain `subprocess` + `docker` CLI (§6) | Shell/Makefile; workflow frameworks |
| D8 | Load mode per phase | P1–P3 concurrency-controlled (`--max-concurrency N`, unlimited rate); P4 open-loop Poisson (`--request-rate λ`) (§5) | One mode for all phases |

---

## 3. Pre-work deliverables (brief §5)

### 3.1 Model selection and VRAM budget

Model facts (from the Qwen2.5 `config.json`s; re-verified against the downloaded files in P0b):

| | Qwen2.5-3B-Instruct-AWQ (target) | Qwen2.5-0.5B-Instruct (draft) |
|---|---|---|
| Layers / KV heads / head dim | 36 / 2 / 128 | 24 / 2 / 64 |
| Params | 3.09B (0.31B tied embeddings) | 0.49B (0.14B tied embeddings) |
| Weights resident | 2.77B × ~0.52 B (4-bit + group scales) + 0.31B × 2 B fp16 embeddings ≈ **2.1 GB** | fp16 **1.0 GB**; AWQ ≈ **0.46 GB** |
| KV per token (fp16) | 2·36·2·128·2 B = **36 KB** | 2·24·2·64·2 B = **12 KB** |
| Tokenizer | Qwen2.5, vocab 151,936 — identical | same |

Embedder: `BAAI/bge-small-en-v1.5` (33M params, ~67 MB fp16), served by its own process for co-tenancy, which costs a second CUDA context.

Budget (GB), three residency conditions:

| Item | Target only | + draft fp16 | + draft + embedder |
|---|---|---|---|
| Windows host / DWM reservation (WSL2 shares the GPU with the desktop) | ~0.5 **measured in P0b** | 0.5 | 0.5 |
| CUDA context + engine workspace + CUDA graphs | ~0.6 **measured in P0b** with a pinned capture-size list | 0.6 | 0.6 |
| Target weights | 2.1 | 2.1 | 2.1 |
| Draft weights | — | 1.0 | 1.0 |
| Embedder + second CUDA context | — | — | ~0.35 |
| Activations / sampler peak, **bounded by a pinned chunked-prefill size** | ~0.3 | 0.3 | 0.3 |
| **Remaining for KV** | **~2.5** | **~1.5** | **~1.15** |
| KV per token (target + draft) | 36 KB | 48 KB | 48 KB |
| **Predicted max resident context** | **~70k tok** | **~31k tok** | **~24k tok** |
| At ~1.5k tok/request (placeholder until §3.2 measures the chat median) | **~45 concurrent** | **~20** | **~16** |

**Pre-registered predictions this budget makes:**

1. Target-only, the P1 concurrency ladder to 32 is compute-bound rather than KV-bound, **provided the §3.2 measured chat median stays below ~2k tokens.** At 1.5k, 32 requests use ~48k of ~70k tokens — a 1.5× margin, not headroom; at 2k it is KV-bound. §3.2 can invalidate this prediction and the P0a writeup says which way it went.
2. The fp16 draft costs ~1 GB of weights plus 33% more KV per token: roughly halves maximum concurrency. That is RQ4's "price of speculation" quantified in advance.
3. Under full co-tenancy, concurrency 32 queues or fails; P4 is expected to find that edge.

**Constraints on how the budget is applied:**

- `--gpu-memory-utilization` (and SGLang's equivalent) is a fraction of *total* memory. It is set from the *measured free* amount at run start, recorded alongside that measurement in `config.yaml`, never 0.9 by reflex.
- `--enforce-eager` is prohibited as a memory fix: it changes decode latency and would confound every latency measurement. CUDA-graph cost is controlled with an explicit reduced capture-size list, recorded in config.
- Chunked prefill is pinned to a fixed chunk size, recorded in config, so peak activation is bounded by construction rather than scaling with the concurrency axis P1 sweeps.
- The draft-in-AWQ residency arm (P4c) is a **cost-and-quality** comparison: 4-bit hurts a 0.5B model proportionally more than a 3B one, and draft quality *is* acceptance rate. Acceptance rate is recorded for both draft variants and reported next to VRAM, so a quality effect is not misread as a memory effect.

**P0b hardware probes this section requires:**

- **OOM-signal probe.** WSL2 runs the WDDM driver model, which can page GPU allocations to host RAM instead of failing. Deliberately over-allocate past the budget and observe. If it fails cleanly, record that. If it spills, build a detector — bandwidth collapse without a memory error, allocated-vs-reserved divergence as the tripwire — and apply it to every later run. Either outcome is a reportable 6 GB / WSL2 finding.
- **Host reservation** measured at idle and tracked during runs (§6).
- **CUDA-graph and workspace cost** measured for the pinned capture list.
- **`ignore_eos` × acceptance-rate probe.** `ignore_eos` (§3.2) forces generation past the natural stopping point, so the draft predicts in a region the target would never have produced. Measure acceptance rate on a small sample with and without `ignore_eos`. If they diverge, the pre-registered handling is to report acceptance over the natural-length prefix only (the runner records per-request natural stop position from the no-`ignore_eos` reference so the prefix is identifiable); the P2 writeup states which rule applied.

### 3.2 Trace capture

**Pipeline:** `bench/capture/` samples queries from `data/products.csv`, `data/reviews.csv`, `data/store_policies.csv` using templates per real routing path (recommendation, comparison, price comparison, store policy, FAQ, stock, review summary), seeded and deterministic, and drives `CoordinatorAgent.handle_query` in-process. A recording wrapper around `app.openai_client.client` captures every OpenAI call — messages, `response_format`, `temperature`, `max_tokens`, response text, `usage` — without altering behaviour. Calls are real (gpt-3.5-turbo) so that output lengths and multi-turn state are genuine. **Sample size: 1,500–2,000 queries** (~$2–3), because the records split across seven routing paths and three call roles, and the per-cell quantiles feed both §3.1 prediction 1 and the P0a validation check; ~500 would leave cells too thin. The capture is run once at this size — re-running later means a new seed and a provenance break. Multi-turn conversations run through `app/conversation.py` so history accumulates exactly as the app does it; they are exported as **three P4b traces with distinct turn-depth profiles** (shallow 2–3 turns, medium 4–5, deep 6+), each versioned separately.

**One user turn is several LLM calls.** Guardrails fire a JSON-mode call on every query; the intent classifier fires on keyword-miss; then the agent call. Each LLM call is one trace record with `call_role: guardrail|classifier|agent`. The per-turn call structure is itself reported.

**Workload tagging:**

- **A — interactive:** every call on the interactive path — all agents except review summarisation, plus guardrail and classifier calls.
- **B — bulk summarisation:** `ReviewSummarizationAgent` calls.
- **C — structured:** A's recommendation and comparison prompts re-issued with a product-card JSON schema kept in `bench/traces/schemas/`. The constraint is a decoding-config overlay on real prompts; the app's own JSON calls are `json_object` mode and stay in A.
- **A+B mixed** is a runner-level composition of A and B traces (P4a), not a separate file.

**Output-length policy (pre-registered):** P1/P2 set per-record `output_len` from the reference distribution with `ignore_eos`, so every config generates the same token count and throughput is comparable. P3 uses natural termination with a `max_tokens` cap, because `ignore_eos` under a closed JSON grammar is meaningless; output length is then a per-config observation and is reported.

**Token counting:** with the Qwen2.5 tokenizer (the one the benchmark runs), tiktoken counts alongside for reference.

**Export:** `bench/traces/<workload>_v<N>.jsonl` in the client's custom-dataset format, plus `<workload>_v<N>.meta.json` holding length quantiles per workload and call role, generator seed, app git SHA, record counts by `provenance: generated|real`. Trace files are immutable once versioned; a change is a new version.

**Validation set:** the LangSmith export of real emitted prompts lands in `bench/traces/validation/` in the same format. A check script compares generator-vs-real prompt-length quantiles per workload and asserts agreement within a stated tolerance; the result goes in the P0a writeup. Because real traffic is developer testing, writeups describe it as "developer testing traffic through the deployed app," never "production traffic."

### 3.3 SLO (pre-registered)

**Condition:** cold prefix cache, first turn, chat workload (A).
**Bounds:** TTFT p90 < 500 ms **and** TPOT p90 < 50 ms.

**Grounding:** a 2.1 GB target at ~190 GB/s cannot beat ~11 ms/token at batch=1; realistic achieved bandwidth on a laptop 4050 puts the unloaded floor at ~20–30 ms, so 50 ms is ~2× the floor — reachable, and tight enough to bite as concurrency rises. 500 ms TTFT is the conventional responsiveness threshold; the risk is prompt length, since Pickr prompts carry system text plus retrieved context.

**Mechanical fallback:** if P1's unloaded batch=1 chat run violates a bound, that bound moves to the lowest rung of its pre-declared ladder that the unloaded run clears — TTFT 500 ms → 1 s → 2 s; TPOT 50 → 75 → 100 ms. The ladder is fixed here; the rung is determined by P1 data, not chosen. Any move is logged with the P1 measurement that triggered it.

**Reporting:** headline goodput at the pre-registered SLO, plus a supplementary goodput-gap vs threshold sensitivity curve computed from stored per-request TTFT/TPOT, with the 2×-measured-batch=1-p50 point labelled on it. If the co-tenancy penalty appears at every threshold, the conclusion is threshold-independent. Consequence for §7: per-request TTFT and TPOT are stored raw, never only as percentiles.

---

## 4. Runtime and tooling

**Engines:** one pinned Docker image per engine (`vllm/vllm-openai:<tag>`, `lmsysorg/sglang:<tag>`), one fresh container per run. A fresh container is the fresh CUDA context §9.7 of the brief requires. The HF cache is a bind-mount on the WSL2 ext4 filesystem (never `/mnt/c`), shared by both engines. Whether Docker Desktop's WSL integration or a native `docker` inside the distro is used is decided in P0b (§8 risk 4); whichever it is, its version is in `env.json`.

**Client:** `vllm bench serve` for both engines (D4), from a GPU-less container of the same pinned vLLM image on `--network host`, so the client version is captured with the server's. Both engines expose OpenAI-compatible chat, so the `openai-chat` backend serves both. Using one client removes the cross-tool TTFT/ITL definition mismatch the brief warns about; the client's definitions are still documented in the P0b writeup. Fallback if the pinned version lacks custom-dataset input or per-request output: SGLang's `bench_serving.py`.

**Chat-template parity (P0b, before any comparative run).** The `openai-chat` backend sends messages and each *server* applies its own chat template. If vLLM and SGLang resolve Qwen2.5-3B-Instruct's template differently, identical input yields different prompt token counts — silently shifting prefill cost and prefix-cache behaviour, a phantom engine difference that would invalidate RQ3 without ever failing. P0b sends one fixed request to each engine and compares the reported prompt token counts. If they differ, either the template is pinned explicitly on both or the client switches to the completions endpoint with pre-templated text. The resolved template's hash is recorded in `env.json` either way.

**What the stock client does not cover — built by the runner:**

1. **Multi-turn replay** (P4b Arm 2 only): ordered turns with a conventional history-resending client — `replay_multiturn.py`. Arm 1 needs no replayer: it is Pickr's own per-request stream (chat + condense traces) through the stock client.
2. **Concurrent mixed load** (P4a): two client instances (chat Poisson + batch) against one server, coordinated by the runner.
3. **Correctness** (P2d, P3): greedy token-identity diff and schema validation — `correctness.py`, not a load test.
4. **Extra metrics:** acceptance rate, KV occupancy, clocks, temperature, power — scraped from `/metrics` and NVML.
5. **Server readiness and cache state.** "Port open" is not "model loaded." Readiness = poll the engine's health route until 200 → N warmup requests → **explicit prefix-cache reset** (vLLM and SGLang each expose a reset/flush route; exact routes verified in P0b) → clock starts. Warmup prompts share Pickr's system-prompt prefix, so without the reset a "cold" run isn't. `cache_state: warm` skips the reset and says so in the artifact.

---

## 5. Experiment matrix

**Universal rules** (brief §7): ≥ 3 reps per config, 5 where noted; distributions with variance, never bare means; warmup discarded; cache state, engine version, clocks, temperature recorded per run; execution order randomised with a logged seed.

**Load mode (D8):** the brief puts concurrency on P1/P2's x-axis and also says closed-loop understates tails. Both hold for different questions: P1–P3 are concurrency-controlled because the knee and the crossover are functions of batch size; P4 is open-loop Poisson because goodput and tails are functions of offered load. `load_mode` is recorded per run.

| Phase | Axes (× 3 reps unless noted) | Runs | Cuts and justification |
|---|---|---|---|
| **P0b** | Engine method-availability check first; chat-template parity check; 10 identical runs of one mid config (vLLM, A, c=8, spec off) → variance floor; echo-server request-rate ladder → harness ceiling; OOM-signal probe; CUDA-graph / host-reservation measurement; `ignore_eos` × acceptance probe | ~25 | — |
| **P1** | engine{vLLM, SGLang} × workload{A, B} × c{1, 2, 4, 8, 16, 32} | 72 | **FP16 dropped:** 3.09B × 2 B = 6.2 GB of weights alone. Optional single GPTQ-Int8 batch=1 point quantifies AWQ's own speed effect. Ladder extends to 64 by rule: while top-rung throughput > 1.1× the rung below. |
| **P2a** k-sweep | engine{2} × workload{A, B} × method{ngram, draft} × k{1, 2, 3, 5, 8}, at c=1 | 120 | Full k × c cross (600 runs) cut: k is a per-step verification-cost trade-off most visible at batch=1, and the brief's k plot is at batch=1. Selects k* per (engine, workload, method). **Tie rule:** if the top two k values' confidence intervals overlap, both are carried into P2c's dense points rather than picking by point estimate. |
| **P2b** crossover | engine{2} × workload{A, B} × method{off, ngram@k*, draft@k*} × c{P1 ladder + 2 dense points around the P1 knee} | 288 | `off` is re-run inside this sweep so the speedup ratio's denominator shares thermal state and randomisation with its numerator. |
| **P2c** k-shift check | at the 2 dense points: k = the rung below k* (and the tied k, where the P2a tie rule fired) | 48 (+24 per tie) | Guards the P2a cut — optimal k likely shrinks as batch grows. If the lower rung wins there, it is reported and the crossover recomputed at it. |
| **P2d** correctness | engine{2} × method{2} × k{5} at c=1, 1 run each on a 50-prompt subset (25 from A, 25 from B), fixed seed, **greedy**: token-identity vs `off` = 20 runs. Control: `off` at c=1 vs c=8 per engine = 4 runs, to separate batch-numerics divergence from speculation divergence | 24 | Cheap; not a load test. Identity is verified under greedy decoding and **assumed** to carry to the sampled regime the rest of the study runs in (per-call temperature from the trace records, matching the app); rejection sampling guarantees distributional, not token, equivalence there, so P2/P3 acceptance rates are reported as measured under sampling. |
| **P3** | workload C: engine{2} × method{off, ngram@k*, draft@k*} × constrained{off, on} × c{ladder}; schema validity parsed on every output; hang/crash log per cell | 216 | A/B excluded — constraints only mean something on structured output. Methods restricted to P2 winners. Natural termination with `max_tokens` cap (§3.2). **Pre-registered cut under time pressure:** constrained{off} runs at the 2 dense points only, not the full ladder (−48 runs); the cut is announced before the sweep starts, never mid-sweep. |
| **P4a** goodput | engine{2} × condition{chat alone, + batch naive, + batch with engine priority control} × offered λ{5 points}, open-loop | 90 | **λ ladder is fixed in absolute req/s**, derived once from the chat-alone measurement (spanning below → above the rate at which chat-alone reaches the P1 knee) and held constant across all three conditions, so the three curves share an x-axis. Converting per-condition via Little's law would use latency that changes with the condition. An engine with no priority control yields a cell marked "none available" — a finding, not a gap. |
| **P4b** prefix caching, **two arms** | cache{off, vLLM APC, SGLang RadixAttention} × **3 distinct multi-turn traces** (shallow 2–3 turns, medium 4–5, deep 6+). **Arm 1 — real, primary:** Pickr's own request stream (every request shares the identical system prompt; the condense call carries a transcript bounded at `HISTORY_WINDOW` = 6 rows) — hit rate and TTFT under repeated shared-prefix traffic, a per-request effect, with the condense call as the bounded turn-indexed case. **Arm 2 — synthetic, labelled:** the same conversations replayed with a conventional history-resending client, the pattern most chat apps use — TTFT vs turn index, explicitly marked NOT Pickr's behaviour. | 54 | **Amended 2026-09-19 (P0a finding):** Pickr never resends history to the agent — `handle_conversational_query` condenses the follow-up into a standalone query — so there is no large growing prefix on the interactive path and the original turn-indexed design would have measured a workload the app doesn't produce. The gap between the arms is the finding: how much prefix-cache benefit an application forfeits by not resending history. |
| **P4c** memory table | residency{target, + draft fp16, + draft AWQ, + draft + embedder} × max-concurrency probe | 12 | Draft AWQ reported as cost-and-quality (§3.1). |
| **P5** | Jetson Orin Nano Super: P1 ladder (one engine) + P2b batch=1 points | ~42 | Hard timebox; ships as "where it stopped" if it expires. |

**Total ≈ 990 runs** at ~5–7 min each (container start, readiness, ~200 requests, cooldown) ≈ 90 hours of unattended GPU time, dominated by P2/P3. Resume-from-state (§6) makes it tolerable.

**Pre-registered graph predictions** (an inversion is a signal, not a surprise):

- P1: knee at c ≈ 16–32 for A. **B — amended 2026-09-19 (P0a finding):** the original prediction ("lower for B, whose long prompts fill KV first") is withdrawn before any run: the catalog holds a median of 2 reviews per product (~51 chars each), so B prompts are *short*, and the longest call on a summarisation turn is the output-guardrail faithfulness check, not the summary. Revised prediction: B's knee is at or above A's, and B is compute-bound, not KV-bound. The withdrawn prediction and its reason stay in the record. Large engine gaps at baseline mean a config problem.
- P2: crossover K below the P1 knee; ngram beats draft on B (summaries copy their sources), draft beats ngram on A.
- P3: direction genuinely open — grammar overhead pushes K left, predictable JSON boilerplate pushes acceptance up. No prediction; both outcomes are reported as found.
- P4a: naive mixing collapses chat goodput above ~50% of knee load; priority control recovers a large fraction. P4b **(amended 2026-09-19):** Arm 1 — TTFT flat across the request stream with a high hit rate on the shared system prefix under APC/Radix, and only a small, bounded rise on the condense call up to turn 3 (the window then slides and the prefix changes); Arm 2 — prefix-off TTFT grows ~linearly with turn, APC/Radix hold it nearly flat unless evicted under memory pressure, which on 6 GB is expected at higher turn counts. The writeup states plainly that Pickr's architecture limits its own prefix-cache opportunity — a finding about real applications versus benchmark assumptions.

---

## 6. Harness architecture

```
bench/
  runner/
    cli.py               # bench run <sweep.yaml> | resume | check-env | capture
    sweep.py             # expands a sweep file into resolved run-configs; shuffles with a logged seed
    engine.py            # Engine ABC + VllmEngine / SglangEngine: image tag, launch args,
                         #   health route, cache-reset route, /metrics field names
    lifecycle.py         # one run, start to cooldown (below)
    client.py            # builds/runs the `vllm bench serve` command; validates and parses its JSON
    gpu_monitor.py       # NVML sampler thread → gpu_samples.jsonl (ours vs host split)
    metrics_scraper.py   # polls engine /metrics → engine_metrics.jsonl
    env_capture.py       # image digest, container pip freeze, driver/CUDA, model revisions,
                         #   launch cmd, power limit + power mode, docker version, host reservation
    state.py             # sweep progress: pending/done/invalid/requeued; retry budget
    correctness.py       # P2d/P3: greedy diff vs baseline; schema validation
    replay_multiturn.py  # P4b Arm 2: ordered turns with a history-resending client (NOT Pickr's behaviour)
  capture/               # §3.2 generator + recorder + LangSmith export + validation check
  configs/               # sweep files, one per phase or sub-phase
  traces/                # versioned JSONL + meta; schemas/; validation/
  results/  analysis/  figures/  docs/
```

**Run lifecycle:**

1. **Pre-flight.** Assert no other GPU process is visible from WSL; read host reservation; attempt `nvidia-smi -lgc`. If pinning is unavailable (likely under WSL2 — the driver passthrough usually lacks the privilege), the pre-registered contingency applies: cooldown threshold is raised, `clocks_pinned: false` is recorded, and per-run clock-trace variance (`clock_cv`) is stored as a covariate so analysis can test whether drift correlates with any result.
2. `docker run` with the pinned image, HF cache bind-mount, GPU access, resolved launch args.
3. **Readiness** as in §4 item 5: health route → warmup → cache reset unless warm.
4. Start the NVML sampler and the `/metrics` scraper.
5. Run the client; keep its raw output file.
6. Stop samplers.
7. **Assertions before recording:** if speculation is on, acceptance counters advanced and are non-zero; throttle flag computed from the clock trace; host-share drift below its threshold; client error rate below threshold; client output validated against the expected schema. A failed assertion marks the run `valid: false` with a reason and **re-queues it to the end of the sweep** — never an immediate retry, which would run at a thermal state the randomised schedule did not intend. Total retries per sweep are capped so a systematic failure cannot burn a night.
8. `docker stop`; verify VRAM returned to the pre-run baseline (catches leaked contexts).
9. **Cooldown** until GPU temperature ≤ threshold *and* a minimum wall-clock gap has elapsed.

**Monitoring the shared GPU.** `gpu_monitor.py` samples total used VRAM and our container's usage separately; `used_host = total − ours`. The Windows compositor allocates from the same pool mid-run and is invisible to WSL as a process, so the runner flags any run where the host share moves by more than a configured threshold. It also samples **actual power draw** per run, not only the configured limit: laptop dynamic boost shifts budget between CPU and GPU at runtime, so sustained GPU power varies with client CPU load — which the co-located client directly affects. The per-run power distribution is stored so a correlation with concurrency is visible rather than absorbed.

**`check-env` (before any sweep):** model revisions on disk match the sweep config; client output schema validates on a probe run; clock-pin result; host reservation at sweep start; docker and driver versions. A mid-sweep re-download pulling a different revision would otherwise be nearly invisible afterwards.

**Engine abstraction is data, not code:** flag names, routes, and metric field names are per-engine tables verified in P0b against `--help` output. Adding a method or a priority flag is a table change.

**Invariants:**

- The runner never *recomputes* a metric it takes from the engine. Any metric it derives (e.g. TTFT/ITL from raw timestamps for a parity check) is stored alongside the engine's or client's own value, never replacing it.
- `analysis/` never imports `runner/` and never launches anything.
- A run is reproducible from its results directory plus the trace file it names by version and SHA-256 (traces are not copied per run).

---

## 7. Result artifact schema

One directory per run: `bench/results/<sweep_id>/<run_id>/`. Timestamps are monotonic-clock seconds with a wall-clock anchor in `meta.json`.

| File | Contents |
|---|---|
| `config.yaml` | Fully resolved run-config: engine, image tag, model + draft + embedder revision hashes, launch args verbatim, spec method/k, workload, `load_mode`, `cache_state`, CUDA-graph capture list, chunked-prefill size, gpu-memory-utilization *and* the measured free VRAM it was derived from, warmup count, seed, `trace: {file, version, sha256}` |
| `env.json` | Image digest; `pip freeze` from inside the container; driver/CUDA runtime; torch/flashinfer/xformers versions; engine git SHA if exposed; GPU name/VBIOS; configured power limit and Windows power mode; WSL kernel; docker flavour + version; `clocks_pinned` + pinned values; client tool version + `client_output_schema_version`; `bench/` git SHA |
| `requests.jsonl` | Per request: `request_id`, `trace_record_id`, `call_role`, `workload`, `t_send`, `t_first_token`, `t_last_token`, `ttft_ms`, `tpot_ms`, `itl_ms[]`, `e2e_ms`, `prompt_tokens`, `output_tokens`, `finish_reason`, `error`, `schema_valid` (P3), `turn_index` + `conversation_id` (P4b). Timing fields carry a `_client` suffix for the tool's value and a `_derived` suffix where the runner recomputes from raw timestamps — both kept |
| `gpu_samples.jsonl` | NVML at fixed interval: `used_total_mb`, `used_ours_mb`, `used_host_mb`, `sm_util`, `mem_util`, `sm_clock`, `mem_clock`, `temp_c`, `power_w`, `throttle_reasons` |
| `engine_metrics.jsonl` | `/metrics` scrape at fixed interval, raw field names preserved per engine: acceptance counters, accepted tokens per step, KV usage, running/waiting queues, prefix-cache hit/query counters |
| `client_raw.json` | The tool's own output, untouched |
| `summary.json` | Computed once by the runner from the files above: p50/p90/p99 TTFT/TPOT/ITL/e2e; tok/s; req/s; mean acceptance rate and tokens/step; peak `used_ours_mb`; peak `used_host_mb` and `host_share_drift_mb`; `power_w` distribution; `clock_cv`; `throttled`; `host_drift_flag`; `valid` + `invalid_reason`; goodput at the pre-registered SLO and `goodput_by_threshold` grid |
| `meta.json` | `run_id`, `sweep_id`, phase, RQ tag, `schedule_index`, `attempt`, `requeued_from`, wall-clock start/end, cooldown observed (seconds, start/end temperature), monotonic-to-wall anchor |
| `log.txt` | Container stdout/stderr + runner log, for hang/crash forensics |

Sweep level, `bench/results/<sweep_id>/`: `sweep.yaml` (source, copied), `schedule.json` (shuffled order + seed), `state.json` (per-run status, retry budget consumed), `check_env.json` (pre-flight results).

**Analysis contract:** `summary.json` for plots; `requests.jsonl` for distributions and the SLO sensitivity curve; `gpu_samples.jsonl` and `engine_metrics.jsonl` for the P4 memory table and drift tests. Runs with `valid: false` are excluded by default and counted in every figure caption.

---

## 8. Assumptions and risks against the brief

Ordered by expected impact.

1. **Results are "WSL2 on a laptop," not "Linux on a 4050."** Driver passthrough, WDDM paging, the DWM reservation, and OEM power limits (4050 TGP spans 35–115 W) all sit between us and bare-metal numbers. Framed as part of the differentiator; every writeup states the platform; `env.json` records power limit and power mode; the laptop is on mains, "Best performance," lid open — a checklist item.
2. **This is a snapshot comparison, not an engine verdict.** Pinned vLLM vX vs SGLang vY on one date measures two fast-moving projects at a point in time. Every writeup states the versions *in the claim itself*, not only in an appendix, and disclaims generalisation to "vLLM vs SGLang" as products.
3. **Clock pinning is probably unavailable under WSL2.** Contingency pre-registered in §6 step 1.
4. **Docker Desktop is a layer we may not need.** Native `docker` inside the WSL2 distro puts fewer moving parts between container and GPU and removes one versioned component from the results. Evaluated in P0b; whichever stays is recorded in `env.json`.
5. **The §3.2 measured chat median can invalidate the P1 compute-bound prediction** (§3.1 prediction 1).
6. **Engine method availability** — verify first, do not pre-build for asymmetry. Current vLLM lists `draft_model` among its speculative methods alongside `ngram`, and current SGLang lists `STANDALONE` and `NGRAM` alongside the EAGLE family. If both hold at our pins, RQ3's speculation comparison is symmetric and neither arm needs a version workaround. P0b checks this *first*, against `--help` and a smoke run, before anything else is built on it. The brief's `bench_speculative.py` tip is an EAGLE tuner and does not apply with EAGLE dropped.
7. **"Traffic imposed by the product"** is true of prompt construction, not of who sent the queries — handled by the validation-set framing in §3.2.
8. **Client co-location** is low-risk at ≤ 64 concurrency; P0b's 3× ceiling test is the proof. If the client container competes with the engine for CPU, the ceiling test catches it before it corrupts timing.
9. **HF cache must live on WSL2 ext4**, not `/mnt/c`: Docker Desktop bind-mounts from the Windows filesystem are slow enough to distort load and warmup time. Not a measurement risk after readiness, but a startup-time one.
10. **Workload C's prompts were not written for schema output.** Re-issuing A's recommendation/comparison prompts under a schema overlay isolates the constraint's effect, but the model's natural output may fight the grammar in a way a purpose-written prompt would not — plausibly lowering acceptance and raising constraint overhead. **Known bias direction: toward finding a *larger* RQ2 effect.** The P3 writeup states this rather than leaving it to a reviewer.
11. **Reference output lengths come from gpt-3.5-turbo**, a different model. Neutralised in P1/P2 by fixed `output_len`; in P3 natural termination makes output length a per-config observation.

---

## 9. Estimates

| Phase | Work | Estimate |
|---|---|---|
| **P0a** | Capture generator + recorder, trace export, LangSmith export, validation check; writeup. **Exit:** traces v1 exist with validated quantiles; §3.1 prediction 1 re-evaluated against the measured median | 4–5 days |
| **P0b** | Runner, collection, env capture, engine-flag verification (first), calibration runs, OOM/pin/graph/host-reservation probes, docker-flavour decision; writeup. **Exit:** variance floor and harness ceiling (≥ 3× the study's peak rate) quantified; engine flags verified at pins; probes done | 1.5 weeks |
| **P1** | 72 runs (~7 h GPU) + optional Int8 point; knee analysis; SLO fallback evaluated; writeup | 3 days |
| **P2** | ~480 runs (~45 h GPU, 4–5 nights) + correctness pass; crossover analysis; writeup | 2 weeks |
| **P3** | 216 runs (~20 h) + failure forensics; writeup | 1 week |
| **P4** | Multi-turn replayer, priority-control discovery, ~130 runs, memory table; writeup | 1 week |
| **P5** | Jetson, hard timebox | ≤ 1 week |
| **Synthesis** | Assemble the blog series from per-phase writeups; cross-phase figures; version-stamped claims | 3–4 days |

**~8–9 weeks part-time.** GPU nights are unattended.

---

## 10. Next step

Write the implementation plan for **P0a and P0b only** (writing-plans skill). P1+ plans follow their predecessors' calibration numbers. The first P0b task is the engine method-availability check (§8 risk 6), because its outcome shapes the P2 configs and the engine tables in `engine.py`.
