# SmartShop Inference Benchmarking — Project Brief (v2)

**Purpose:** context and constraints for planning and implementing an inference-serving benchmark study on top of an existing SmartShop application. Read this fully before proposing a plan.

**Companion document:** `methodology.md` defines the phases (P0–P5), what each measures, and the expected result shapes. This brief covers context, constraints, and traps. Read both.

**Changed in v2:** the harness is no longer built from scratch (§6); model selection and trace capture are now explicit pre-work (§5).

---

## 1. Who this is for and what already exists

- The author is an HPC performance engineer with a benchmarking background, moving toward inference-infrastructure work. Assume fluency with roofline reasoning, controlling for variance, and statistical rigor. **Do not** explain what TTFT or batching is.
- **SmartShop is already implemented.** It is an AI shopping assistant: product recommendations, review summarization, price comparison, conversational interface.
- This project is the *performance analysis layer* on top of it — a separate deliverable, in its own directory.

**Hard rule: do not refactor SmartShop's application code to make benchmarking easier.** Instrument it, wrap it, or replay traffic captured from it. The scientific value depends on the workload being the real application's workload, not a benchmark-friendly rewrite of it. If a change to app code is genuinely unavoidable, flag it and explain why.

---

## 2. Hardware

| Platform | Role |
|---|---|
| RTX 4050 laptop, **6 GB VRAM**, ~190 GB/s | Primary. All headline results come from here. |
| Jetson Orin Nano Super | Secondary. Edge-deployment comparison only, final phase, timeboxed. |

6 GB is the binding constraint and **it is the differentiator, not a limitation to apologize for.** Nearly all published vLLM/SGLang numbers come from A100/H100-class hardware. Memory-constrained results are the contribution. Say so in the writeup.

Laptop GPUs thermally throttle. Thermal control is a first-class methodology concern.

---

## 3. What makes this project worth doing

SmartShop naturally generates **two contending workload regimes on one GPU**:

- **Workload A — Interactive chat.** Batch≈1, latency-bound. TTFT and TPOT matter.
- **Workload B — Bulk review summarization.** High batch, offline, throughput-bound. No latency SLO.
- **Workload C — Structured output.** Product cards / comparison tables as schema-constrained JSON. Runs interactive or batch.

The traffic mix is imposed by the product, not synthesized to justify a benchmark. Preserve that framing throughout.

---

## 4. Research questions

Ordered by expected value of the result. Full experimental design in `methodology.md`.

**RQ1 — Where is the speculative-decoding crossover?** Speculation trades compute for latency. It should win decisively at batch=1 and degrade or invert as the GPU saturates. Locate the crossover on this hardware and explain it in roofline terms.

**RQ2 — How does constrained decoding move that crossover?** Hypothesis: JSON schema constraints shift it to a *lower* batch size. Under-benchmarked publicly. Spec-decode × structured-output has a history of correctness bugs in both engines — **correctness verification here is part of the experiment.**

**RQ3 — vLLM vs SGLang under mixed load.** Not a generic bake-off. Test where the architectures actually differ: prefix-cache behavior on multi-turn replay (RadixAttention vs APC), grammar backend throughput, scheduling fairness when batch and interactive traffic contend.

**RQ4 — Co-tenancy under memory pressure.** Target + draft + embedding model resident in 6 GB. How much KV cache does the draft model cost, and when does that cost exceed the speedup it buys?

**RQ5 — Edge portability (Jetson).** Timeboxed hard.

---

## 5. Pre-work: three decisions before any code

**5.1 — Model selection (blocking).** Everything downstream depends on this. Required:
- A **target model** that fits in ~2 GB quantized (4-bit AWQ or GPTQ), leaving ~3 GB for KV cache after overhead. A 3B-class instruct model is the expected landing spot.
- A **draft model** from the *same family with the same tokenizer* (a 0.5B–1B sibling). Cross-family drafts collapse acceptance rates — this is non-negotiable.
- Confirm an **EAGLE-class checkpoint** exists for the chosen target, or drop that arm of the experiment.
- A small **embedding model** for the recommendation path (co-tenancy, RQ4).

Deliverable: a written VRAM budget — weights, draft, embedder, activations, remaining KV — with the predicted max concurrent context in tokens. This is the first number the study tests.

**5.2 — Trace capture (blocking).** The benchmark replays SmartShop's *real* traffic shape, not uniform synthetic lengths.
- Measure SmartShop's actual input and output length distributions per workload (A, B, C).
- Export fixed, versioned trace files in the benchmark tools' dataset format (JSONL).
- Report the distributions in the writeup — a benchmark whose input shape is undocumented is unreproducible.

**5.3 — SLO definition.** Fix the interactive service-level objective in advance (e.g. TTFT < 500 ms and p90 TPOT < 50 ms) and defend it. Goodput in P4 is meaningless without it, and choosing it after seeing results is fitting the target to the data.

---

## 6. Harness strategy: reuse, don't rebuild

**Do not write a load generator from scratch.** Both engines ship serving benchmarks that already do most of Phase 0's work:

- `vllm bench serve`
- SGLang's `bench_serving.py`

Both drive an OpenAI-compatible endpoint — so **one client can test both engines** — support open-loop Poisson arrivals via a request-rate flag, accept custom datasets, and emit TTFT/TPOT/ITL distributions. They are maintained by people who benchmark for a living.

**What to build on top (this is the actual scope):**

1. **Trace files** for workloads A/B/C in the tools' dataset format (from §5.2).
2. **A sweep runner** — loops the config matrix, launches and tears down servers cleanly, pins clocks (`nvidia-smi -lgc`), enforces cooldowns, randomizes execution order, retries on transient failure.
3. **Environment capture** — driver, CUDA, engine version + git SHA, model revision, flash-attn/flashinfer version, full launch command, written into every result artifact.
4. **Extra metric collection** the stock tools don't cover — acceptance rate, KV-cache occupancy, GPU temperature and clock traces, scraped from engine metrics endpoints and `nvidia-smi`.
5. **Analysis and plotting** — reads only from stored results, never re-runs the benchmark.

**Critical compatibility check, before any cross-engine comparison:** verify that both tools compute TTFT and ITL *the same way*. Definitions differ subtly between benchmark scripts (what counts as "first token" with streaming, whether queueing time is included). If they differ, normalize by computing metrics yourself from raw per-request timestamps. Undetected, this manufactures phantom engine differences and invalidates RQ3.

**Tooling note:** this is a single well-specified deliverable, not an evolving multi-phase product. Plain Claude Code with this brief as context is sufficient — a heavier spec-driven framework is not warranted here.

---

## 7. Metrics

Per-request and per-run: **TTFT**, **TPOT/ITL** (p50/p90/p99), **end-to-end latency**, **output token throughput**, **request throughput**, **goodput** (against the §5.3 SLO), **acceptance rate / accepted tokens per forward pass** (from engine metrics, never inferred), **peak VRAM and KV-cache utilization**, **GPU utilization and achieved bandwidth** where obtainable, and **clock/temperature traces** for throttle detection.

Always report distributions and variance, never bare means. Report run count and confidence intervals.

---

## 8. Methodology requirements

Non-negotiable; the project's credibility rests on them. (Phase-by-phase detail in `methodology.md`.)

1. **Fixed input traces**, versioned in-repo. Every engine and config sees byte-identical input.
2. **Realistic length distributions** sampled from measured SmartShop traffic — never uniform 512-in/256-out.
3. **Warmup then discard**, fixed count.
4. **Cache state explicitly controlled** and stated per run. Cold-cache and warm-cache runs are different experiments.
5. **Minimum 3 runs per config** (5 preferred), variance reported.
6. **Thermal control** — pin clocks, log temperature, enforce cooldowns, mark throttled runs, randomize config order so drift can't correlate with a treatment.
7. **Isolation** — one engine at a time, clean process, fresh CUDA context, no background GPU work of any kind during a run.
8. **Environment captured in every artifact.** Engine versions matter enormously; results without them are worthless.
9. **Output correctness** — speculative decoding must be token-identical to non-speculative greedy decoding under a fixed seed. **Verify empirically.** Any divergence is a finding. Also validate JSON schema conformance rate under constrained decoding.

---

## 9. Known traps

- **Engine flags change frequently between releases.** Never trust flag names from documentation, blog posts, or training data. Verify against the installed version (`vllm serve --help`, `python -m sglang.launch_server --help`) and pin exact versions. Names differ across engines (vLLM's `--speculative-config` JSON vs SGLang's `--speculative-algorithm` / `--speculative-draft-model-path`).
- **SGLang ships `bench_speculative.py`** for auto-tuning the steps/topk/draft-token triple — use it to narrow the P2 sweep instead of brute-forcing.
- **Cross-family draft models collapse acceptance rates.** Same family, same tokenizer.
- **Silent spec-decode disablement.** Some configs cause an engine to ignore speculation without erroring. **Assert acceptance-rate metrics are non-zero before recording any run.**
- **Spec-decode + structured output has a bug history** in both engines (hangs, truncation at the first constrained-choice token, crashes with `json_schema`). Assume nothing works until verified; a reproducible failure is a publishable finding.
- **Draft length has sharply diminishing returns** past ~5–8 tokens, and each token costs VRAM. On 6 GB the optimum will likely be lower than published numbers suggest. Measure it.
- **Try `ngram` / prompt-lookup on the summarization path specifically.** Summaries copy heavily from source reviews, so acceptance should be high at near-zero VRAM cost. Likely the best cost/benefit finding available on this hardware.
- **Closed-loop load generators understate tail latency.** Use open-loop with a defined arrival distribution.
- **Quantization interacts with speculation.** A 4-bit target and FP16 draft have different numerics; verify output equivalence per quantization setting.
- **Client/server co-location.** Load generator and server share one laptop. If the client shows any sign of being the limiter, move it to a second machine over LAN rather than shaving the safety margin.

---

## 10. Repository conventions

```
bench/
  runner/         # sweep orchestration, server lifecycle, env capture
  traces/         # versioned fixed input traces (JSONL)
  configs/        # one declarative file per experiment config
  results/        # raw JSONL, one dir per run, env metadata included
  analysis/       # scripts -> figures; reads only from results/
  figures/
  docs/           # per-phase writeups
```

- Every run writes a self-describing artifact: config + environment + raw per-request records, reproducible from that artifact alone.
- Configs are declarative and diffable, never embedded in scripts.
- Analysis never re-runs the benchmark.

---

## 11. Deliverable

A blog series with a specific, non-obvious, hardware-grounded headline — something in the shape of:

> Speculative decoding delivered N× TPOT improvement at batch=1 but net-negative throughput above batch=K; enabling JSON schema constraints moved that crossover down to K′.

Numbers are unknown until measured — **do not assume the direction of any result.** If speculation is a loss across the entire tested range on 6 GB, that is a legitimate and useful finding, and the writeup should say so plainly.

Ship a short writeup at the end of each phase rather than saving everything for the end.

---

## 12. What I want from you first

Before writing code:

1. The §5.1 model selection and VRAM budget, with reasoning.
2. A proposed staged experiment design — the full matrix pruned, with explicit justification for what was cut.
3. The result-artifact schema.
4. A sketch of the sweep runner's architecture, and confirmation of what the stock benchmark tools do and don't cover.
5. A list of assumptions in this brief you think are wrong or risky.
6. A rough time estimate per phase.

Push back on anything here that doesn't hold up. Do not start implementing until the design is agreed.
