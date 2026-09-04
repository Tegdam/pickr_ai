# SmartShop Inference Benchmarking — Project Brief

**Purpose of this document:** context and constraints for planning and implementing an inference-serving benchmark study on top of an existing SmartShop application. Read this fully before proposing a plan.

---

## 1. Who this is for and what already exists

- The author is an HPC performance engineer with a benchmarking background, moving toward inference-infrastructure work. Assume fluency with roofline reasoning, hardware counters, controlling for variance, and statistical rigor. **Do not** explain what TTFT or batching is.
- **SmartShop is already implemented.** It is an AI shopping assistant: product recommendations, review summarization, price comparison, conversational interface.
- This project is the *performance analysis layer* on top of it. It is a separate deliverable.

**Hard rule: do not refactor SmartShop's application code to make benchmarking easier.** Instrument it, wrap it, or replay traffic captured from it. The scientific value depends on the workload being the real application's workload, not a benchmark-friendly rewrite of it. If a change to app code is genuinely unavoidable, flag it explicitly and explain why.

---

## 2. Hardware

| Platform | Role |
|---|---|
| RTX 4050 laptop, **6 GB VRAM** | Primary benchmark platform. All headline results come from here. |
| Jetson Orin Nano Super | Secondary. Edge-deployment comparison only, final phase. Not a co-equal platform. |

6 GB is the binding constraint and **it is a feature of this study, not a limitation to apologize for.** Nearly all published vLLM/SGLang numbers come from A100/H100 class hardware. Memory-constrained results are the differentiated contribution. Say so in the writeup.

Laptop GPUs thermally throttle. Thermal control is a first-class methodology concern (see §7).

---

## 3. What makes this project worth doing

SmartShop naturally generates **two contending workload regimes on one GPU**:

- **Workload A — Interactive chat.** Batch≈1, latency-bound. TTFT and TPOT are what matter.
- **Workload B — Bulk review summarization.** High batch, offline, throughput-bound. No latency SLO.
- **Workload C — Structured output.** Product cards / comparison tables / recommendation payloads returned as schema-constrained JSON. Can run interactive or batch.

The traffic mix is imposed by the product, not synthesized to justify a benchmark. Preserve that framing throughout.

---

## 4. Research questions

Ordered by expected value of the result.

**RQ1 — Where is the speculative-decoding crossover?**
Speculative decoding trades compute for latency. It should win decisively at batch=1 and degrade or invert as the GPU saturates. Locate the crossover point on this hardware and explain it in roofline terms (memory-bandwidth-bound → compute-bound transition).

**RQ2 — How does constrained decoding move that crossover?**
Grammar-constrained generation changes the cost of each verification step. Hypothesis: enabling JSON schema constraints shifts the crossover to a *lower* batch size. This is under-benchmarked publicly. Note that spec-decode × structured-output interaction has a history of correctness bugs in both engines (see §9) — **correctness verification here is part of the experiment, not a side task.**

**RQ3 — vLLM vs SGLang under mixed load.**
Not a generic throughput bake-off — those exist. Test where the engines' architectures actually differ: prefix-cache behavior on multi-turn conversation replay (SGLang RadixAttention vs vLLM APC), grammar backend throughput, and scheduling fairness when batch and interactive traffic contend.

**RQ4 — Co-tenancy under memory pressure.**
Target model + draft model + embedding model must be resident simultaneously in 6 GB. Quantify: how much KV cache does the draft model cost, and at what point does that cost exceed the speedup it buys? This is the most practically useful result for anyone running small-GPU inference.

**RQ5 — Edge portability (Jetson).** Does any of this survive on ARM/Orin? Timeboxed.

---

## 5. Experiment matrix

Axes to sweep:

| Axis | Values |
|---|---|
| Engine | vLLM, SGLang |
| Spec-decode method | off, ngram/prompt-lookup, draft-model (standalone), EAGLE-3 if a checkpoint fits |
| Draft length (`num_speculative_tokens` / `num_draft_tokens`) | 1, 2, 3, 5, 8 |
| Concurrency | 1, 2, 4, 8, 16, 32 (extend until saturation or OOM) |
| Workload | A (chat), B (summarization), C (JSON), A+B mixed |
| Structured output | off, JSON schema on |
| Quantization | 4-bit (AWQ or GPTQ) baseline; FP16 only if it fits |

Full cross-product is too large. **Propose a staged design:** coarse sweep first to find interesting regions, then dense sampling only near crossover points. Justify what gets cut.

---

## 6. Metrics

Record per-request and per-run:

- **TTFT** — time to first token (distribution, not just mean)
- **TPOT / ITL** — inter-token latency, p50/p90/p99
- **End-to-end latency**
- **Output token throughput** (tok/s, system-wide)
- **Request throughput** (req/s)
- **Goodput** — requests meeting a stated SLO (define one, e.g. TTFT < 500 ms and p90 TPOT < 50 ms). This is what makes the mixed-load result legible.
- **Acceptance rate / accepted tokens per forward pass** — the causal variable behind every spec-decode result. Extract from engine metrics endpoints, not inferred.
- **Peak VRAM and KV-cache utilization** — mandatory given the 6 GB constraint
- **GPU utilization, SM occupancy, achieved memory bandwidth** where obtainable
- **Clock frequency and temperature over time** — for throttle detection

Always report distributions and variance, never bare means. Report the number of runs and confidence intervals.

---

## 7. Methodology requirements

These are non-negotiable; the whole project's credibility rests on them.

1. **Fixed input traces.** Capture or generate a fixed set of prompts per workload, versioned in the repo. Every engine and config sees byte-identical input.
2. **Realistic length distributions.** Do not use uniform 512-in/256-out. Measure SmartShop's actual input/output length distributions and sample from them. Report them.
3. **Warmup then discard.** Fixed warmup request count before measurement; discard warmup entirely.
4. **Cache state control.** Prefix caching must be explicitly on or off and stated per run. A cold-cache run and a warm-cache run are different experiments. Never let cache state drift.
5. **Repetition.** Minimum 3 independent runs per config, ideally 5. Report variance.
6. **Thermal control.** Pin clocks where possible (`nvidia-smi -lgc`), log temperature throughout, enforce a cooldown between runs, and mark runs where throttling occurred. Randomize config execution order so thermal drift does not correlate with a treatment.
7. **Isolation.** One engine at a time, clean process, fresh CUDA context. No background GPU work.
8. **Environment capture.** Log driver, CUDA, PyTorch, engine version + git SHA, flash-attn/flashinfer version, model revision hash, and the full launch command into every result artifact. Engine versions matter enormously here and results are worthless without them.
9. **Output correctness.** Speculative decoding must be output-equivalent to non-speculative greedy decoding. **Verify this empirically** — greedy, fixed seed, compare token-for-token. Any divergence is a finding worth reporting on its own. Also validate JSON schema conformance rate under constrained decoding.

---

## 8. Suggested phasing

**P0 — Harness.** Load generator (open-loop with configurable arrival process, not closed-loop — closed-loop hides queueing effects), metrics collection, result schema, environment capture, plotting. Engine-agnostic through the OpenAI-compatible API so the same client hits both.

**P1 — Baselines.** Both engines, no spec decode, no constraints. Workloads A and B separately. Establish variance floor and confirm the harness is not the bottleneck. **Verify the client is not saturating before the server is** — this is the most common way benchmark studies get silently invalidated.

**P2 — RQ1 (crossover).** The headline result. Sweep spec-decode method × draft length × concurrency.

**P3 — RQ2 (constrained decoding).** Add JSON schema. Includes correctness validation.

**P4 — RQ4 (co-tenancy / mixed load).** A+B contending. Prefix-cache comparison. Memory-budget analysis.

**P5 — Jetson.** Timeboxed hard — ARM builds of these engines can consume a week. If it exceeds the box, ship without it and note it as future work.

Ship a writeup at the end of each phase rather than saving everything for the end.

---

## 9. Known traps

- **Engine flags change frequently between releases.** Do not trust flag names from documentation, blog posts, or training data. Verify against the installed version (`vllm serve --help`, `python -m sglang.launch_server --help`) and pin exact versions in the repo. Names differ across engines (`--speculative-config` vs `--speculative-algorithm` / `--speculative-draft-model-path`).
- **Cross-family draft models collapse acceptance rates.** Draft and target must share a tokenizer and family.
- **Silent spec-decode disablement.** Some configurations cause an engine to ignore speculative decoding without erroring. Always assert that acceptance-rate metrics are non-zero before recording a run.
- **Spec-decode + structured output has a bug history** in both engines (hangs, truncation at first constrained-choice token, intermittent crashes with json_schema). Assume nothing works until verified. If it breaks on the installed version, that reproducible failure is itself a publishable finding.
- **Draft length has sharply diminishing returns** past roughly 5–8 tokens, and each extra token costs VRAM. On 6 GB the optimum will likely be lower than published numbers suggest — that is a result, so measure it rather than assuming it.
- **`ngram` / prompt-lookup should be tried on the summarization path specifically.** Summaries copy heavily from source reviews, so acceptance should be high at near-zero VRAM cost. Likely one of the best cost/benefit findings available on this hardware.
- **Closed-loop load generators** understate tail latency. Use open-loop with a defined arrival distribution.
- **Quantization interacts with spec decode.** A 4-bit target and FP16 draft have different numerics; verify output equivalence separately per quantization setting.

---

## 10. Repository conventions

Propose a layout along these lines and confirm before building:

```
bench/
  harness/        # load generator, metrics client, trace replay
  configs/        # one declarative file per experiment config
  traces/         # versioned fixed input traces
  results/        # raw JSONL, one dir per run, env metadata included
  analysis/       # notebooks / scripts -> figures
  figures/
  docs/           # per-phase writeups
```

Requirements:
- Every run writes a self-describing artifact: config + env + raw per-request records. Reproducible from that artifact alone.
- Analysis reads only from `results/`; never recompute by re-running the benchmark.
- Configs are declarative and diffable, not embedded in scripts.

---

## 11. Deliverable

The target output is a blog series with a specific, non-obvious, hardware-grounded headline — something in the shape of:

> Speculative decoding delivered N× TPOT improvement at batch=1 but net-negative throughput above batch=K; enabling JSON schema constraints moved that crossover down to K′.

Numbers are unknown until measured — **do not assume the direction of any result.** If speculative decoding turns out to be a loss across the entire tested range on 6 GB, that is a legitimate and useful finding, and the writeup should say so plainly.

---

## 12. What I want from you first

Before writing code:

1. A proposed staged experiment design with the full matrix pruned down, and an explicit justification for what was cut.
2. A harness architecture sketch.
3. The result-artifact schema.
4. A list of assumptions in this brief you think are wrong or risky.
5. A rough time estimate per phase.

Push back on anything here that does not hold up. Do not start implementing until the design is agreed.
