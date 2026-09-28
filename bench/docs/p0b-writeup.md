# P0b — Calibration: what this laptop actually gives you, and what the harness can measure

*Phase 0b of the Pickr inference-serving benchmark. Spec: `docs/superpowers/specs/2026-09-17-pickr-inference-benchmark-design-v2.md` §3.1, §3.3, §6, §9. Engine-flag contract: `bench/docs/p0b-engine-verification.md`. Probes and runs executed 2026-09-25/26 on WSL2 (Ubuntu) + native Docker, RTX 4050 Laptop 6 GB, driver-reported power limit 100 W, Windows power scheme **Balanced** (see §11). Pins: vLLM `v0.29.0` @ `sha256:c2914767…`, SGLang `v0.5.20-runtime` @ `sha256:00b02004…`, client image `bench-client:v0.29.0` @ `sha256:6b6e3050…`, Qwen2.5-3B-Instruct-AWQ @ `3559b226`, Marlin kernel resolved on both engines.*

P0b exists to answer two questions before any comparison is attempted: **what does the platform actually give us**, and **what differences is this harness capable of detecting**. Both answers are numbers below, and several of them contradict the spec's estimates.

## 1. What was run

| | |
|---|---|
| Engine-flag verification | Every flag, route, metric name and env value pinned in `bench/docs/p0b-engine-verification.md` (the binding contract, controller rulings P1–P15) |
| Hardware probes | 4 — `host_reservation`, `oom_signal`, `clock_pin`, `cudagraph_cost` (2 runs; the first was confounded, §5) |
| Parity smoke | 2 runs (1 per engine) — the gate before any sweep |
| Power-mode A/B | 20 runs — `power_overlay` as an axis under a randomised block design (§10a) |
| Harness ceiling | 16 runs — 8 request-rate rungs × 2 reps against the echo server |
| `ignore_eos` × acceptance | 4 runs — vLLM, draft + ngram, `ignore_eos` true/false |
| Variance floor | 10 identical runs — vLLM, workload A, c=8, spec off, 200 prompts, **twice** (once per power mode) |
| Total | 62 recorded runs + 6 probe result files, all under `bench/results/` |

Nothing failed silently: every recorded run is `valid=True`, every run returned its VRAM (`vram_leak_mb: 0`), and no run left a container behind.

## 2. The memory budget, measured against §3.1's prediction

vLLM's own reported breakdown, warm compile cache, resolved fraction 0.79 (`env.json.engine_memory`, identical across the parity and variance runs):

| Item | §3.1 predicted (GB) | **Measured (GiB)** | |
|---|---|---|---|
| Host / WSL2 reservation invisible to CUDA | ~0.5 | **1.05** | 2.1× the estimate |
| Target weights (`model_load_gib`) | 2.1 | 1.95 | |
| Non-torch / CUDA context (folded into `weights_plus_non_torch` 2.10) | — | ~0.15 | |
| Peak activation, chunked prefill pinned at 2048 | ~0.3 | 0.22 | |
| CUDA graphs, pinned list `[1,2,4,8,16,32,64]` | (part of a ~0.6 line) | 0.07 | ~3× less than estimated |
| **KV cache** | **~2.5** | **2.41** | |
| **Max resident context** | **~70k tok** | **70,240 tok** | |

Two independent confirmations fall out of this. KV per token measured **2.41 GiB ÷ 70,240 = 35.98 KB**, matching the 36 KB/token computed from `config.json` in §3.1 — the GQA arithmetic was right. And vLLM's own `max_concurrency_x` at `max_model_len` 2048 is **34.3×**, against §3.1's "~45 concurrent at 1.5k tokens" placeholder.

**The headline prediction (~70k KV tokens) was right, but partly by luck.** Two errors of opposite sign cancelled: the host reservation was underestimated by ~0.55 GiB, and the context/workspace/graph line was overestimated by ~0.38 GiB. The spec's method was sound; its two hardest-to-estimate rows were both wrong, and only their sum survived. Worth stating plainly rather than claiming a clean hit.

**Consequence for P1's prediction 1**, which needed the measured chat median below ~2k tokens: at the P0a-measured 265-token median, concurrency 32 occupies ≈ 8.5k of 70,240 KV tokens — an **8× margin**. Prediction 1 now stands on measured numbers at both ends, so any early flattening of the P1 chat curve points at compute or scheduling, not KV exhaustion.

## 3. The 6 GB is really ~4.95 GB, and nothing in `nvidia-smi` says so

With no process running, `nvidia-smi` reports **6141 MB total and 6141 MB free**, and both of its views (Windows-side and WSL-side) report **0 MiB used**. CUDA, in the same state, sees **4.95 of 6.0 GiB free**. The gap — **≈ 1072 MB ≈ 1.05 GiB** — is reserved by the WDDM/desktop stack and is invisible to every `nvidia-smi` query we make.

This is not an inference from one engine. SGLang, launching independently, logs `avail mem=4.93 GB` at load begin — a second engine reaching the same number by its own path.

The practical consequence is a startup failure that looks like a bug and is not: a `--gpu-memory-utilization` derived from `nvidia-smi` free memory asks for memory that does not exist. That is exactly how vLLM died during the first `host_reservation` probe attempt, with

> `ValueError: Free memory on device cuda:0 (4.95/6.0 GiB) on startup is less than desired GPU memory utilization (0.87, 5.22 GiB)`

**Ruling P36** therefore sets the memory fraction from *measured CUDA-visible free* minus a per-engine headroom (`mem_headroom_mb_total` 1280 for vLLM, 1536 for SGLang), resolving to **0.79 for vLLM and 0.74 for SGLang**, recorded per run in `config.yaml`. No run in this study uses a fraction chosen by reflex.

## 4. WDDM fails cleanly — it does not page

The spec's largest platform worry (§3.1, §8 risk 1) was that WDDM would page GPU allocations to host RAM instead of failing, silently turning a memory experiment into a bandwidth experiment. The `oom_signal` probe over-allocates past the budget and watches for bandwidth collapse without a memory error:

| fraction | requested | outcome | ready_s | tok/s | tok/s vs baseline |
|---|---|---|---|---|---|
| 0.78 (baseline) | 4790 MB | served | 46.4 | 181.8 | — |
| 0.80 | 4913 MB | served | 48.4 | 178.0 | 0.979 |
| 0.82 | 5036 MB | served | 48.4 | 179.8 | 0.989 |
| **0.84** | **5158 MB** | **`clean_oom`** | — | — | — |

Throughput is **flat to within 2 %** right up to the boundary, then the engine fails at startup with an explicit memory error. There is no degradation ramp, so there is no paging to detect, and **the spill detector the spec asked us to build is not needed** — a negative result that removes a tripwire from every later run.

The boundary is not arbitrary. 0.82 asks for 5036 MB and fits; 0.84 asks for 5158 MB and does not; CUDA-visible free is **5069 MB**. The clean-failure boundary *is* the invisible-reservation limit, to within the ladder's 0.02 resolution. Two independent measurements, one number.

Recorded outcome: `clean_oom_boundary`. Our operating fraction (0.79) sits below the lowest tested-and-served fraction, with the 0.21 GiB slack visible in §2.

## 5. CUDA-graph cost — and the confound that nearly hid it

The pinned capture list costs **0.07 GiB = 2,512 KV tokens** (3.6 % of KV capacity) against capturing a single batch size. That is the number §3.1 asked for, and it is cheap enough that `--enforce-eager` stays prohibited as a memory fix.

Getting there required catching a confound in our own probe. The first `cudagraph_cost` run re-resolved the memory fraction per launch, so `kv_tokens` tracked the fraction rather than the graphs:

| capture list | graph GiB | KV tokens — **first (confounded)** run | KV tokens — **pinned-fraction** re-run |
|---|---|---|---|
| `[1]` | 0.00 | 44,416 | 72,752 |
| `[1,2,4,8]` | 0.02 | 43,568 | 71,888 |
| `[1,2,4,8,16,32,64]` | 0.07 | **58,016** | 70,240 |

The third column is not noise, it is non-monotone: capturing *more* graphs appeared to *buy* 13,600 KV tokens. A graph-memory cost cannot be negative, which is what exposed the bug. **Ruling P37** pins one fraction across all capture lists and waits for an idle GPU before each launch; the re-run is monotone and the cost is 2,512 tokens.

The lesson generalises past this probe: a probe that varies two things at once produces a number with the right units and the wrong meaning, and only a sign or monotonicity check catches it.

## 6. Compile-cache warmth moves 26,000 KV tokens

Two launches with identical flags reported KV capacities of **43,664** and **70,240** tokens. The cause is `torch.compile` cache warmth: a cold cache holds compilation workspace a warm one does not, and vLLM sizes KV from what remains.

| | weights + non-torch | peak activation | KV |
|---|---|---|---|
| Cold compile cache | 2.63 GiB | 0.67 GiB | 1.5 GiB |
| **Warm** | **2.10 GiB** | **0.22 GiB** | **2.41 GiB** |

A 26,576-token swing in KV capacity is larger than most effects this study intends to measure, and it is invisible in the launch command. **Ruling P38** records `compile_cache_warm` per run and parses the engine-reported memory breakdown into `env.json` (`bench/runner/enginelog.py`), so no cross-run comparison can straddle a cache-warmth boundary without it being visible in the artifacts. Every run in §9–§12 is warm (`compilation_s` 0.8–0.87 s).

## 7. The two engines do not size memory the same way

The parity smoke exposed a structural asymmetry that will shape RQ1/RQ3/RQ4. SGLang captures **prefill** CUDA graphs across 42 batch sizes by default; vLLM captures none. Pinning `--cuda-graph-bs-prefill` to the decode list drops prefill graph memory **0.47 → 0.08 GB** and leaves SGLang 0.49 GB spare instead of 0.12 GB.

Two corrections to what that appeared to buy, both found by re-measuring rather than reasoning:

- **KV capacity is unchanged at 48,154 tokens either way.** SGLang allocates its static pool and KV *before* graph capture, while vLLM sizes KV from what remains *after*. So matching the two engines' fraction flags does **not** match their KV capacity.
- **Startup is not improved.** 7 capture sizes took 225 s where 42 took 212 s; the ~220 s is a fixed per-launch cost.

Pinning therefore buys symmetry and headroom, not KV or startup. Prefill graphs stay **enabled**: disabling a default optimisation on one engine would bias the engine comparison, which is the thing the study is for.

The asymmetry that remains is the important one: **vLLM 70,240 vs SGLang 48,154 KV tokens** — a 1.46× difference in the resource most likely to drive a concurrency knee. `kv_cache_tokens` is a *shared pool* measured in token-slots, not a per-request limit: 70,240 slots × 36 KiB/token = 2.41 GiB, and against `max_model_len` 2048 it is what vLLM reports as `max_concurrency_x: 34.3`. It is an allocated quantity, not an architectural one — whatever remained after weights, activations and graphs at the resolved fraction, which is why compile-cache warmth alone moved it by 26,576 slots (§6).

**Only about 60 % of that gap is the engines' doing.** The two are running at different fractions, and that difference is ours:

| | resolved fraction | requested | KV tokens | KV bytes |
|---|---|---|---|---|
| vLLM | 0.79 (headroom 1280 MB) | 4.74 GiB | 70,240 | 2.41 GiB |
| SGLang | 0.74 (headroom 1536 MB) | 4.44 GiB | 48,154 | 1.65 GiB |

The KV gap is 0.76 GiB, of which **0.30 GiB is simply the 0.05 difference in fraction** — a consequence of the per-engine headroom *we* chose in ruling P36, not of engine behaviour. Only the residual **~0.46 GiB** is SGLang's genuinely larger non-KV footprint (static pool, prefill graphs, workspace).

Both engines page KV at the same **36 KiB/token**, and this is measured on each side rather than assumed from the shared model: vLLM reports 2.41 GiB for 70,240 slots (**35.98 KiB**), SGLang reports `kv_cache_memory_usage_gb 1.957` against `max_total_num_tokens 57,014` (**35.97 KiB**). Both match the 36,864 B computed from `config.json` (2 × 36 layers × 2 KV heads × 128 head dim × 2 bytes), so token counts are directly comparable between the engines and the token↔byte conversion below is exact.

That decomposition decides it: a comparison in which ~40 % of the KV asymmetry traces to our own headroom setting is not measuring the engines, so the headroom difference is a harness artifact to remove rather than a finding to report.

**Decision (2026-09-28): equalise, at 44,000 token slots.** From P1 onward every engine-comparison sweep pins the pool to the same size on both engines — `--kv-cache-memory-bytes` on vLLM (bytes), `--max-total-tokens` on SGLang (tokens) — through one `kv_cache_tokens` config field expressed in tokens, because tokens are the unit that drives concurrency and the unit both engines report back.

Why 44,000: it is 8.6 % below SGLang's measured 48,154 ceiling, leaving margin against a heavier allocation on another day, and 1.6× above the realistic worst case — c=64 × ~420 tokens per request ≈ 27k slots, given P0a's 265-token median prompt and ~70-token output. The cost is explicit: 64 simultaneous p99-length requests (~61k slots) would queue on **both** engines. That is the intended consequence. The equalised budget is the resource under test, and vLLM no longer earns a later knee for having been handed more KV than SGLang. Every knee claim states the pinned value.

**The pin is verified, not trusted.** Assertions rule 3b fails a run unless the engine's own reported pool size lands within 2 % of what was asked (each engine rounds to its own block or page size), and a pool size that cannot be read back fails too — an unverifiable equal-KV claim is not worth making, and a silently ignored flag would leave the engines unequal while this document claimed otherwise. `bench/configs/p0b_kv_parity.yaml` is the gate: one run per engine, before P1 depends on any of it.

One boundary is deliberately left closed. `kv_cache_tokens` is **not** set in `base.yaml`, and `validate()` refuses it while speculation is on: the draft model adds 12,288 B/token of its own KV, and whether vLLM's byte flag covers the draft pool as well as the target's is unverified at our pin. Converting tokens to bytes there would be guessing, and an equal-KV claim built on a guess is worse than an unequal comparison stated honestly — so P2 verifies the draft accounting before its equal-KV spec arms run.

Startup cost also differs sharply, which sets the wall-clock budget for every later sweep: vLLM `ready_s` **50.4 s**, SGLang **272.9 s** — 5.4×.

*Provenance caveat:* the 48,154-token and prefill-graph figures come from a manual verification launch (`docker logs`), not from a recorded run artifact — the parity smoke ran before the SGLang log parser landed in `8762d68`, so `p0b-parity-0001-r0/env.json` has `engine_memory` all-null. One SGLang smoke run will capture the breakdown into the schema; until it does, treat these two numbers as verified but not archived.

## 8. Chat-template parity — the gate before any sweep

Both engines tokenised the same pre-rendered trace row to **265 prompt tokens**, with `prompt_token_mismatches: 0` across both runs and the Marlin kernel resolved on both. This is the check that makes any cross-engine latency claim meaningful: had the engines applied different chat templates, every downstream comparison would have been between different prompts.

Both parity runs were `valid=True`, with no leaked container and no VRAM leak.

## 9. Harness ceiling: ≥ 1024 req/s

Eight rungs against the echo server (5 ms/token, no GPU), 500 prompts, 2 reps each — so any limit found is the client's and the runner's own:

| requested | offered | offered/requested | achieved req/s | send window | drain tail | TTFT p99 |
|---|---|---|---|---|---|---|
| 8 | 8.0 | 1.000 | 8.0 | 62.4 s | 0.24 s | 8 ms |
| 16 | 16.0 | 1.000 | 15.7 | 31.2 s | 0.61 s | 8 ms |
| 32 | 32.0 | 1.000 | 30.5 | 15.6 s | 0.81 s | 8 ms |
| 64 | 64.0 | 1.000 | 57.4 | 7.8 s | 0.92 s | 8–9 ms |
| 128 | 127.9 | 1.000 | 100.2 | 3.9 s | 1.08 s | 9 ms |
| 256 | 255.9 | 0.999 | 155.7 | 1.95 s | 1.26 s | 9–12 ms |
| 512 | 511.6 | 0.999 | 214.8 | 0.98 s | 1.36 s | 47–66 ms |
| 1024 | 1016.2 | 0.993 | 263.6 | 0.49 s | 1.41 s | 79–121 ms |

Zero request errors in 16 runs; host CPU peaked at 10.9 %.

**Achieved throughput is the wrong criterion, and this ladder shows why.** Read the achieved column alone and the harness appears to top out near 64 req/s. It does not. `req_s = n / (send_window + drain_tail)`, and the drain tail converges on **1.41 s — the duration of the single longest row in the trace** (257 output tokens × 5 ms = 1.29 s; longest request measured 1.449 s). The send window halves each rung; once it is shorter than that fixed tail, the tail owns the denominator. It is an artifact of finite `num_prompts` against a heavy-tailed output-length distribution: the median row is 14 output tokens (~77 ms), and peak in-flight requests stay ≤ 32 even at 1024 req/s.

**Ruling P30**'s offered-rate criterion — arrival timestamps from `requests.jsonl` — tracks the requested rate to within 0.1 % through 512 and 1.0 % at 1024. Genuine strain does appear, but in TTFT p99 (flat at 8 ms through 256, then 47–66 ms at 512 and 79–121 ms at 1024) and in e2e p50 at the top rung only; the arrival process stays faithful throughout.

So **1024 req/s is a lower bound**, not a located limit — the ladder never found the client's ceiling. Against P4a's λ ladder, which derives from P1's measured knee and should top out between 10 and 60 req/s, this is a **≥ 17× margin** where §9 asks for 3×. Pre-registered re-check: extend the ladder before P4a if P1's knee puts the top λ above ~340 req/s.

## 10. Variance floor — and the materiality ladder it pre-registers

Ten identical runs (vLLM, workload A, c=8, spec off, 200 prompts). All 10 `valid`, `completed=200` and `error_rate=0.0` on every run, zero VRAM leak, `throttled=False` and `sw_throttle_fraction=0.00` everywhere, `ready_s` 44.3–48.4 s, `peak_used_ours_mb` **identical at 4897** on all ten. The machine was quiet by measurement, not assertion: host `cpu_busy` max 12.4–14.4 %, `loadavg_1m` 0.75–2.14.

| metric | cv % | observed range % |
|---|---|---|
| e2e p90 | 0.62 | 2.20 |
| e2e p50 | 0.72 | 2.31 |
| req/s, output tok/s | 0.84 | 2.82 |
| ITL p50 | 0.86 | 3.17 |
| TPOT p90 | 1.02 | 3.64 |
| TTFT p90 | 1.17 | 3.67 |
| TPOT p50 | 1.22 | 3.82 |
| TTFT p50 | 1.73 | 6.32 |
| e2e p99 | 2.21 | 6.42 |
| ITL p99 | 2.88 | 10.94 |
| TPOT p99 | 3.44 | 13.54 |
| **TTFT p99** | **7.33** | **22.20** |

Central tendency is tight — every p50/p90 metric within 1.73 %. The tails are an order of magnitude noisier, and TTFT p99 is the noisiest real metric by a wide margin.

**Pre-registered materiality ladder (new; carried into the spec as §3.4, beside the SLO).** These thresholds hold **within a session only** — §10a records an observed ~3 % offset between sessions against a 0.49 % block-to-block cv inside one, which is why §3.4 also requires comparison axes to be blocked. A difference is reported as material only if it exceeds:

| metric family | measured cv | **material if >** |
|---|---|---|
| any p50 / p90 metric, req/s, tok/s | ≤ 1.73 % | **5 %** |
| TPOT / ITL / e2e p99 | ≤ 3.44 % | **10 %** |
| TTFT p99 | 7.33 % | **25 %** |

Anything smaller is reported as *within the noise floor*. Required rep count follows from cv/√n, so a cell whose claim rests on a smaller difference needs more reps rather than a softer adjective. **This ladder is fixed now, before P1 produces a single number** — choosing it after seeing results would forfeit the pre-registration the rest of the design rests on.

Two readings to keep honest:

- **`goodput_pre_registered` was exactly 0.960 on all ten runs** (sd 0) — the same 192/200 requests cleared the SLO every time. With 200 prompts the metric is quantised at 0.005, so cv = 0 means "stable to within one request", not infinite precision.
- **No run-order effect, so the first run of a sweep is not discarded.** r0 has the only elevated `clock_cv_busy` (0.245 vs 0.000–0.060) *and* the worst TPOT p99 (33.83 ms), which looks like a first-launch clock ramp — but TTFT p99's worst run is r4 (801.7 ms) at `clock_cv_busy` 0.000, and r0's TTFT p99 is mid-pack. The tail noise is inherent jitter. n=1 on the clock observation; revisit only if it recurs.

## 10a. Session drift, and how a sequential A/B overstated an effect by 4×

The floor above was measured twice: under the Windows **Balanced** power mode (2026-09-26) and under **Best performance** (2026-09-27). Compared sequentially, Best performance looked clearly better — every latency metric 4–5 % faster, throughput **+4.28 %**, all 13 informative metrics shifting coherently in the same direction. The reading offered at the time was that metric-wide coherence is what a genuine system-wide change looks like rather than what noise looks like.

That reading was wrong, and the way it was wrong is the most useful thing in this document.

Because the two arms ran on different days, "the power mode caused it" was not separable from "the day was different". So the comparison was re-run as a **randomised block design** (`p0b_power_ab.yaml`, 20 runs): `power_overlay` became a sweep axis, the runner set the mode itself per run and verified it took effect, and the schedule paired the arms so each consecutive block held one of each. All 20 runs valid, and every run's configured arm was checked against its recorded effective overlay.

**Paired within-block differences, which is what the design bought:**

| metric | paired diff | 95 % CI | blocks favouring Performance |
|---|---|---|---|
| TTFT p50 | −1.20 % | [−2.59, +0.19] | 8/10 |
| TTFT p90 | −0.38 % | [−0.85, +0.09] | 7/10 |
| TPOT p50 | −0.89 % | [−1.75, −0.02] | 7/10 |
| TPOT p90 | −0.78 % | [−1.43, −0.13] | **9/10** |
| TPOT p99 | −1.12 % | [−2.24, −0.00] | 7/10 |
| ITL p99 | −1.89 % | [−3.80, +0.02] | 7/10 |
| e2e p50 | −0.88 % | [−1.76, −0.00] | 7/10 |
| e2e p90 | −0.93 % | [−2.01, +0.15] | **9/10** |
| req/s, tok/s | +0.58 % | [−0.09, +1.25] | 7/10 |
| TTFT p99 | +1.76 % | [−9.03, +12.55] | 5/10 |

**Best performance is worth about 1 %, not 4–5 %.** A small real effect survives — the direction favours it in 7–9 of 10 blocks and several intervals exclude zero — but every difference is far below its §3.4 threshold, so it is reported as *within the noise floor*.

**Where the other 3 % was.** Comparing like arm to like arm across dates isolates it:

| same config, same power mode, different day | req/s | e2e p50 |
|---|---|---|
| Balanced: 2026-09-26 → 2026-09-28 | **+0.32 %** | −0.19 % |
| Performance: 2026-09-27 → 2026-09-28 | **−3.24 %** | +3.19 % |

The Balanced arm reproduced to within 0.3 % two days apart. The **2026-09-27 session simply ran ~3 % fast**, and the sequential comparison charged that session effect to the power mode. It decomposes almost exactly: **4.28 % observed ≈ 0.6 % real + 3.2 % session offset.**

**How much this does and does not establish.** These are the only two like-for-like inter-session deltas we hold, and they disagree by a factor of ten (+0.32 % against −3.24 %). So a ~3 % excursion is an **observed possibility, not a measured typical magnitude**: n=2 supports an existence proof and no distribution, and a week of dedicated re-measurement would yield only ~6 deltas — a standard deviation with roughly a ±30 % confidence interval, still not a quotable number. It is therefore measured going forward instead, by an **anchor run**: one unchanged reference config at the start of every session, which accumulates deltas contemporaneously with the data they qualify and flags an odd session *before* its results are trusted.

**Within a session, by contrast, the harness is very stable** — and this is what the blocking rule actually rests on. The A/B ran ~1 h as 10 time-ordered blocks: the first five blocks and the last five differed by **−0.11 %** in req/s, with a **0.49 % block-to-block cv**. So the Sep 27 anomaly was a *step* fixed for the whole session, not a drift accumulating with runtime, which is exactly why pairing arms inside a session removes it.

The offset's cause is unidentified — busy SM clock (2670 MHz), GPU power p90 (≈68 W), peak temperature (62–63 °C) and host CPU were indistinguishable between arms and between sessions. Behaving as session-fixed rather than runtime-accumulating points at state constant across a session: driver or kernel state since boot, a background process, or thermal starting conditions.

**Why the coherence argument failed, precisely.** A session-level drift moves every metric together *because they all derive from the same requests*. So coherence across metrics separates "something systematic" from "random noise", but it cannot separate "the treatment" from "the day". The correlation among metrics was noted at the time and then the opposite conclusion was drawn from it.

**The consequence is much larger than the power mode.** An excursion of the size observed here dwarfs the 0.49 % within-session block-to-block cv, so §10's materiality ladder is only valid for comparisons whose arms sit in one session. P2 is ~480 runs across 4–5 nights, comparing engines: had vLLM and SGLang been split across nights, a 3 % session effect could have been read as an engine difference, and 3 % is comparable to real engine differences. Spec §3.4 therefore now requires any axis carrying a headline comparison — the engine axis above all — to be scheduled as a randomised block design, with results reported as within-block differences.

A free shuffle is not sufficient for this, which is worth stating because it is the obvious thing to reach for: over these 10+10 runs it produced a 5-run single-arm streak and left one arm's mean position 1.6 of 20 slots ahead of the other's. Blocking caps the streak at 2 and equalises mean position by construction.

## 11. What WSL2 refuses, and what we substitute

**Clock pinning is unavailable.** `nvidia-smi --lock-gpu-clocks` returns code 4:

> `The current user does not have permission to change clocks for GPU 00000000:01:00.0.`

Both the pin and the reset fail identically, so §6 step 1's pre-registered contingency applies: clocks are *reported*, not controlled. `clock_cv` substitutes — and **ruling P41** had to fix how it is computed. Over all client-phase samples it measures idle/boost transitions rather than thermal drift whenever the load does not fill the sampling window: the parity smoke read 210 MHz for 11 of 17 samples with 2685 MHz spikes, giving `clock_cv` 1.37. `summary.json` now also carries **`clock_cv_busy`** (`sm_util > 0`) and `busy_sample_fraction`. Across the variance runs `clock_cv` sat at 0.68–0.80 with ~0.5 busy fraction, while `clock_cv_busy` was **0.000–0.060 in 9 of 10 runs** — clocks are stable under load, and the raw statistic would have said the opposite.

**Host GPU share is unobservable.** The `host_reservation` probe found both `nvidia-smi` views reporting identically — 0 MiB at idle, 4835 MB under load — with `views_track_each_other: true` and `drift_mb: 0.0`. The two-view split the spec designed (§6) cannot separate our usage from the desktop's on this platform. Across the 10 variance runs `host_share_drift_mb` was 0 on nine and 60 MB on one, with `host_drift_flag` never set. It is retained as a diagnostic, and the study states that host contention is bounded by the quiet-machine checklist rather than measured.

**The machine ran in the wrong power mode.** `env.json` records the Windows power scheme as **Balanced** (`381b4222…`) with a 100 W driver power limit, while §8 risk 1's checklist specifies "Best performance". Nothing throttled (`throttled=False`, `sw_throttle_fraction=0.00` in every run), so Balanced did not cost measurable performance — but the §10 variance floor was measured under Balanced, and **switching to Best performance before P1 would invalidate it**. The recommendation is to keep **Balanced for the whole study** and amend the checklist, rather than switch and re-measure the floor. This is a decision, not a fait accompli; it is listed in §14.

## 12. `ignore_eos` × acceptance rate

vLLM, k=3, c=1, 100 workload-A prompts, one run per cell:

| spec method | `ignore_eos` | acceptance | TPOT p50 | TTFT p50 |
|---|---|---|---|---|
| draft | true | 0.7494 | 9.38 ms | 76.2 ms |
| draft | false | 0.7689 | 9.12 ms | 78.7 ms |
| ngram | true | 0.5561 | 12.07 ms | 28.1 ms |
| ngram | false | 0.5726 | 11.94 ms | 28.0 ms |

Relative divergence is **−2.5 % (draft)** and **−2.9 % (ngram)**, both well inside the pre-registered 10 % band, so **P2 reports acceptance over the full generation** and the natural-length-prefix rule does not apply. One pre-registered contingency resolved.

These are also the study's first spec-decode numbers, and they preview RQ2's trade-off: the standalone draft accepts far better than ngram (0.75 vs 0.56) and wins on decode (9.4 vs 12.1 ms TPOT), but pays at prefill (76–79 vs 28 ms TTFT — the draft model's own forward pass).

**Limitation, recorded not fixed.** Output lengths come from the trace, whose median row asks for only 14 output tokens, so the model rarely reaches a natural EOS before `max_tokens` in either arm — total output tokens were 2964 (`ignore_eos` true) vs 2961 (false), near-identical *by construction*. The result therefore means "the rule does not fire at trace lengths", not "the rule can never fire". If P2 reports acceptance on long generations, this probe needs a re-run with a long-output workload first.

## 13. Exit criteria

> **Exit:** variance floor and harness ceiling (≥ 3× the study's peak rate) quantified; engine flags verified at pins; probes done.

**Met.**

| Criterion | Result |
|---|---|
| Variance floor quantified | §10 — cv 0.62–7.33 % by metric, plus a pre-registered materiality ladder |
| Harness ceiling ≥ 3× peak | §9 — ≥ 1024 req/s, a ≥ 17× margin (lower bound) |
| Engine flags verified at pins | `bench/docs/p0b-engine-verification.md`; parity smoke §8, both engines 265 tokens, Marlin on both |
| Probes done | §3–§7, §11 — host reservation, OOM signal, clock pin, CUDA-graph cost, plus the unplanned compile-cache finding |
| Docker flavour decided | Native `docker` inside WSL2 (Docker Desktop not used); recorded in `env.json` |

Artifacts: `bench/results/{probes,p0b-parity,p0b-ceiling,p0b-ceiling-hi,p0b-ignore-eos,p0b-variance,p0b-variance-perf,p0b-power-ab}/` — 62 runs, each reproducible from its own `config.yaml` plus the trace SHA-256, with `schedule.json` recording the seed and the blocked axis where one applies.

## 14. Carried forward

**Decisions needed before P1 runs:**

1. ~~**KV equalisation across engines**~~ **Decided (§7): equalise at 44,000 token slots**, pinned per engine and verified from each engine's own reported pool size. Remaining work is the gate run, `bench/configs/p0b_kv_parity.yaml` (1 run per engine), which must pass before P1 relies on the pin.
2. ~~**Power mode**~~ **Settled (§10a).** P1 and everything after run under the Windows **Best performance** overlay, and `base.yaml` now pins `power_overlay: performance` so each run asserts its mode instead of inheriting the machine's state. The justification is that it is measurably never worse and matches the field's convention — **not** that it is worth the 4–5 % a sequential comparison appeared to show. The interleaved block design put the real effect at about **1 %**, below every §3.4 threshold, and resolved the rest into ~3 % session drift. §3.4's ladder is keyed to the Performance floor, whose cv values are equal or better, so the 5/10/25 % thresholds stand unchanged.

**Fixes before P2:**

3. **tokens/step is missing from `summary.json`**, which spec §7 requires. `metrics_scraper.py` gives vLLM only the accepted/draft ratio while its SGLang branch returns `spec_accept_length_last` — and SGLang's accept length *is* tokens/step. Left as-is, RQ2's cross-engine comparison would compare different quantities. Fix: derive vLLM's as `1 + k × acceptance_rate` and record both engines under one key.
4. ~~**`max_tokens_cap` is declared and never plumbed** — wire it or delete it.~~ **Withdrawn:** `bench/runner/config.py` documents it as deliberately reserved for P3's natural-termination arm, where the client stops passing per-row `output_tokens` (`--custom-output-len -1`). It is unused rather than dead, and `p0b_ignore_eos.yaml` setting it to 512 was inert only because no trace row reaches that length. Nothing to fix; P3 reads it.

**Deferred measurements:**

5. **One SGLang smoke run** to capture its memory breakdown into `env.json` (§7 provenance caveat).
6. **A long-output `ignore_eos` re-run** if P2 reports acceptance on long generations (§12).
7. **P0a addendum B2** (LangSmith validation) after the trace-leak fix and real traffic on the live Space.
