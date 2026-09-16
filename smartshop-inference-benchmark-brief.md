# SmartShop Inference Benchmarking — Project Brief

**Purpose:** context, constraints, and experimental plan for an inference-serving benchmark study built on top of an existing SmartShop application. This is the single authoritative document for the project — read it fully before proposing a plan.

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

Ordered by expected value of the result. Each maps to a phase in §7.

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

**5.3 — SLO definition.** Fix the interactive service-level objective in advance (e.g. TTFT < 500 ms and p90 TPOT < 50 ms) and defend it. Goodput in Phase 4 is meaningless without it, and choosing it after seeing results is fitting the target to the data.

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

## 7. Phases

Each phase states what effect it isolates, what gets collected, what gets plotted, and when it's done. **Predict the shape of each graph before running it** — that way an unexpected shape is a signal rather than a surprise, and the "deviations that mean something" notes tell you which.

**Universal rules, stated once and applying to every phase:** minimum 3 runs per configuration (5 preferred); results reported as distributions with variance, never bare means; warmup requests discarded; cache state, engine version, clock speeds and temperature recorded per run; configuration execution order randomized so thermal drift cannot masquerade as a treatment effect.

### Phase 0 — Harness calibration

**Studies the effect of nothing, deliberately.** Before measuring the system, prove the instrument is honest.

**Build:** trace files (§5.2), sweep runner, metric collection, environment capture — the §6 scope.

**Collect:** replay the same trace against the same config repeatedly, and against a trivial echo server.

**Produce:** two numbers, not plots — the **variance floor** (run-to-run noise with nothing changed; every later "difference" must beat this to count) and the **harness ceiling** (the request rate at which the client itself maxes out).

**The idea is to show** the instrument is neither the bottleneck nor the noise source. The most common way benchmark studies silently die is a client that saturates before the server does; from then on you are measuring your own stopwatch.

**Done when:** variance floor is quantified, and the harness sustains ≥ 3× the highest request rate any later phase will use. *Why "any later phase":* calibration happens once, so its ceiling must cover the study's worst case (taken from the config matrix) or the instrument needs re-validating mid-study. *Why 3× rather than "fast enough":* open-loop Poisson traffic is bursty, with short windows well above the average rate; the client grows heavier as per-request instrumentation is added; and a client near its own limit corrupts timing — delaying sends and timestamp captures — long before it visibly fails. The factor 3 is a conventional safety margin; the non-negotiable part is that the headroom is *measured*.

*Expected graph: none. The desired result here is a flat, boring line. If Phase 0 produces an interesting plot, something is wrong.*

### Phase 1 — Baselines

**Studies the effect of concurrency alone** — no speculation, no constraints, no mixing.

**Collect:** throughput, TTFT, TPOT distributions **for various values of** concurrency (1, 2, 4, 8, 16, 32 — extend until saturation or OOM), per engine × per workload.

**Plot:** throughput vs concurrency (one curve per engine, one panel per workload), plus p50/p99 latency vs concurrency.

**The idea is to show** the saturation curve every later phase is compared against: steep rise while decode is bandwidth-bound and extra requests are nearly free, then a bend and flattening at the **knee** where the GPU runs out of spare compute or KV space. The knee's location per engine per workload is the finding, and it tells Phase 2 where to look densely.

**Deviations that mean something:** a curve flattening suspiciously early on the chat workload usually means KV cache filled before compute did — check memory counters; that's a real 6 GB finding. A curve that never flattens means concurrency wasn't pushed high enough. Large engine gaps *at baseline* suggest a config problem, not an architectural difference.

**Done when:** the knee is located for all four engine × workload combinations with variance bars, and nothing about the harness looks like the limiter.

### Phase 2 — The crossover (headline)

**Studies the combined effect of speculative decoding and concurrency.**

**Collect:** TPOT, throughput, and — critically — **acceptance rate** (from engine counters, never inferred) **for various values of** speculation method (off / prompt-lookup / draft-model / EAGLE-class if available), draft length k (1, 2, 3, 5, 8), and concurrency (the Phase 1 ladder, sampled densely near the knee), per engine, on both workloads.

**Plot:** *speedup ratio vs concurrency* — speculation-ON ÷ speculation-OFF, for both TPOT and throughput, with the 1.0 line drawn boldly. Second plot: speedup vs draft length k at batch=1, one curve per workload.

**The idea is to show** the crossover concurrency K where speculation flips from win to loss — and to *explain* it. At batch=1 the compute units are idle, so guess-and-verify is nearly free and the latency win is large; as concurrency rises, batching has already put them to work, the guessing steals time from real requests, and the ratio slides below 1.0. Tie K's location to the Phase 1 knee — that's the roofline story told with your own data. The k-sweep tells the companion story: guessing further ahead pays less and less while renting more KV space, so the optimum k on 6 GB is likely smaller than values quoted from 80 GB cards.

**Deviations that mean something:** if speculation *never* wins even at batch=1, check acceptance rate first — near-zero means a tokenizer/family mismatch or a silently disabled speculator. If it never *loses* at high concurrency, the GPU likely never reached saturation — cross-check the Phase 1 knee. If summarization with prompt-lookup shows dramatically higher acceptance than chat, that's not an anomaly — it's the expected "summaries copy their sources" effect and a publishable observation.

**Done when:** K is located per engine per workload with error bars, acceptance rate is recorded at every point, and speculative output is verified token-identical to baseline greedy decoding (any divergence reported as a finding).

### Phase 3 — Constrained decoding

**Studies the combined effect of grammar-constrained output and speculation** — Phase 2 repeated with schema-valid JSON enforced.

**Collect:** Phase 2's metrics plus **schema-validity rate** (did every output parse?) and a **correctness log** (hangs, truncations, crashes — this combination has a documented bug history in both engines, so failures are data) **for** the Phase 2 axes with constraints ON vs OFF as the new dimension, using each engine's default grammar backend.

**Plot:** the Phase 2 crossover chart drawn twice on the same axes — free-form vs JSON-enforced — so the crossover's movement is visible as a horizontal shift. Plus a table of validity rate and failure count per configuration.

**The idea is to show** whether, and in which direction, forcing structure changes the economics of speculation. Hypothesis: K moves **left**, because grammar-checking adds per-step work to a system that saturates earlier. Counter-force: JSON boilerplate is highly predictable, pushing acceptance *up*. Which wins is genuinely open — that's what makes this the most novel chart in the project.

**Deviations that mean something:** if the JSON curve crosses *later* instead, acceptance gains beat grammar overhead — that inverts the hypothesis and is arguably the more interesting result; lead with it. If validity rate dips below 100% anywhere, stop and investigate before trusting any performance number from that cell — a fast engine emitting broken JSON is not fast.

**Done when:** both curves exist with error bars, every cell has a validity rate, and every failure is reproducible with pinned version and config attached.

### Phase 4 — Co-tenancy and mixed load

**Studies the combined effect of workload mixing and memory pressure** — bulk summarization and interactive chat on the same 6 GB card simultaneously, with the embedding model also resident. The most production-shaped phase.

**Collect:** goodput (chat requests/sec meeting the §5.3 SLO), TTFT/TPOT tails, KV-cache occupancy **for various values of** offered chat load, in three conditions — chat **alone**, chat **+ background batch job, naive sharing**, and chat + batch **with whatever priority/scheduling controls each engine offers**. Separately: a multi-turn replay measuring TTFT **for various values of** turn number, with prefix caching off, with vLLM APC, and with SGLang RadixAttention.

**Plot:** goodput vs offered chat load (three curves, per engine); TTFT vs conversation turn (three curves); plus a memory-budget table showing where all 6 GB went in each condition and the measured cost of the draft model's residency in lost concurrency.

**The idea is to show** two things. First, the price of co-tenancy — how badly the batch job starves interactive traffic when nobody schedules, and how much scheduling buys back, told through goodput because raw throughput can hide it entirely. Second, the prefix-caching story: chat resends its whole history every turn, so without reuse TTFT should grow roughly linearly with turn number while APC and RadixAttention hold it nearly flat. That's the sharpest architectural difference between the engines, measured on the workload that stresses it.

**Deviations that mean something:** if naive mixing barely hurts goodput, the batch job probably wasn't saturating the GPU during the measurement window — check overlap and occupancy traces. If prefix-cached TTFT climbs anyway, the cache is being evicted under memory pressure — cross-reference KV occupancy; on 6 GB, *cache eviction under pressure* is a likely and very reportable finding.

**Done when:** the three-curve goodput chart exists per engine against a pre-committed SLO, the prefix chart exists with cache hit-rates recorded, and the memory table accounts for every gigabyte.

### Phase 5 — Edge (Jetson), timeboxed

**Studies the effect of platform alone:** the best configurations from Phases 1–4, transplanted unchanged onto the Jetson Orin Nano.

**Collect:** the Phase 1 saturation curve and the Phase 2 batch=1 speculation measurement only. **Produce** a comparison table — 4050 vs Jetson: decode ceiling, knee location, speculation benefit — plus honest notes on what refused to build or run.

**The idea is to show** portability, not new science: whether laptop-GPU conclusions transfer to an edge module with different bandwidth and thermals, and to bound the pain of getting these engines onto ARM at all.

**The timebox is real.** ARM builds have historically consumed weeks. If it expires, this phase ships as "attempted, here is exactly where it stopped" — itself useful — and the project concludes without it. No headline result depends on Phase 5.

### How the phases add up

| Phase | Isolates | Key output | Feeds |
|---|---|---|---|
| P0 | nothing (calibration) | variance floor, harness ceiling | everything |
| P1 | concurrency | saturation knee per engine × workload | P2's search region |
| P2 | speculation × concurrency | **the crossover K** + best draft length | P3's baseline |
| P3 | + JSON constraints | how K moves under grammar | the novel chart |
| P4 | mixing × memory pressure | goodput gap + prefix-caching TTFT | the production story |
| P5 | platform | portability table | the edge epilogue |

Ship a short writeup at the end of each phase rather than saving everything for the end — the blog series then assembles itself from those instead of from memory.

---

## 8. Metrics

Per-request and per-run: **TTFT**, **TPOT/ITL** (p50/p90/p99), **end-to-end latency**, **output token throughput**, **request throughput**, **goodput** (against the §5.3 SLO), **acceptance rate / accepted tokens per forward pass** (from engine metrics, never inferred), **peak VRAM and KV-cache utilization**, **GPU utilization and achieved bandwidth** where obtainable, and **clock/temperature traces** for throttle detection.

Always report distributions and variance, never bare means. Report run count and confidence intervals.

---

## 9. Methodology requirements

Non-negotiable; the project's credibility rests on them.

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

## 10. Known traps

- **Engine flags change frequently between releases.** Never trust flag names from documentation, blog posts, or training data. Verify against the installed version (`vllm serve --help`, `python -m sglang.launch_server --help`) and pin exact versions. Names differ across engines (vLLM's `--speculative-config` JSON vs SGLang's `--speculative-algorithm` / `--speculative-draft-model-path`).
- **SGLang ships `bench_speculative.py`** for auto-tuning the steps/topk/draft-token triple — use it to narrow the Phase 2 sweep instead of brute-forcing.
- **Cross-family draft models collapse acceptance rates.** Same family, same tokenizer.
- **Silent spec-decode disablement.** Some configs cause an engine to ignore speculation without erroring. **Assert acceptance-rate metrics are non-zero before recording any run.**
- **Spec-decode + structured output has a bug history** in both engines (hangs, truncation at the first constrained-choice token, crashes with `json_schema`). Assume nothing works until verified; a reproducible failure is a publishable finding.
- **Draft length has sharply diminishing returns** past ~5–8 tokens, and each token costs VRAM. On 6 GB the optimum will likely be lower than published numbers suggest. Measure it.
- **Try `ngram` / prompt-lookup on the summarization path specifically.** Summaries copy heavily from source reviews, so acceptance should be high at near-zero VRAM cost. Likely the best cost/benefit finding available on this hardware.
- **Closed-loop load generators understate tail latency.** Use open-loop with a defined arrival distribution.
- **Quantization interacts with speculation.** A 4-bit target and FP16 draft have different numerics; verify output equivalence per quantization setting.
- **Client/server co-location.** Load generator and server share one laptop. If the client shows any sign of being the limiter, move it to a second machine over LAN rather than shaving the safety margin.

---

## 11. Repository conventions

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

## 12. Deliverable

A blog series with a specific, non-obvious, hardware-grounded headline — something in the shape of:

> Speculative decoding delivered N× TPOT improvement at batch=1 but net-negative throughput above batch=K; enabling JSON schema constraints moved that crossover down to K′.

Numbers are unknown until measured — **do not assume the direction of any result.** If speculation is a loss across the entire tested range on 6 GB, that is a legitimate and useful finding, and the writeup should say so plainly.

---

## 13. What I want from you first

Before writing code:

1. The §5.1 model selection and VRAM budget, with reasoning.
2. A proposed staged experiment design — the §7 matrix pruned, with explicit justification for what was cut.
3. The result-artifact schema.
4. A sketch of the sweep runner's architecture, and confirmation of what the stock benchmark tools do and don't cover.
5. A list of assumptions in this brief you think are wrong or risky.
6. A rough time estimate per phase.

Push back on anything here that doesn't hold up. Do not start implementing until the design is agreed.
