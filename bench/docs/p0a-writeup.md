# P0a — Trace capture: what Pickr's traffic actually looks like

*Phase 0a of the Pickr inference-serving benchmark. Spec: `docs/superpowers/specs/2026-09-17-pickr-inference-benchmark-design-v2.md` §3.2. Capture date 2026-09-18, seed `20260919`, app at `2ee5298` (tracked tree clean; the raw records carry a spurious `-dirty` suffix on `52fe499` caused by the capture's own untracked output files — fixed in `3f4ef71` before export, and the export's SHA-256 is byte-identical either way).*

## 1. What was captured

A seeded generator produced **1,400 single-turn queries** and **120 conversations** (40 each of shallow 2–3 turns, medium 4–5, deep 6–8; 559 turns) from Pickr's real catalog, and drove them through the app's own `CoordinatorAgent` and condensation path in-process. An observe-only wrapper around the shared OpenAI client recorded every chat-completion call: **5,530 calls over 1,959 turns**, 0 failed units, 0 fail-open warnings on stderr, 100% of records with OpenAI `usage` present.

| | |
|---|---|
| Queries by intent (single-turn) | recommendation 30% (category+price 20, brand 7, browse 3), review 22%, comparison 14%, store policy 14%, price comparison 10%, stock 7%, capabilities 3% |
| Phrasing | 80% "keyword" (written to hit a routing rule), 20% "natural" (written to miss all rules) |
| Routed via | keyword rule 45%, LLM classifier fallback 55% |
| Guardrail-blocked turns | 18 of 1,959 (kept: they are real behaviour, recorded as `guardrail_input`-only turns) |
| Calls by role | guardrail_input 1,959 · agent 1,163 · guardrail_output 1,163 · classifier 806 · condense 439 |
| Unrecorded calls per turn | 2 moderation-endpoint calls when an LLM agent answers (1 otherwise); 1 embeddings call on FAQ turns |
| OpenAI cost | 1,789,840 prompt + 128,241 completion tokens ≈ **$1.09** at gpt-3.5-turbo rates |

Everything is in git: `bench/traces/raw/` (queries, raw capture, empty stderr log) and the six `bench/traces/*_v1.jsonl` trace files with `.meta.json` sidecars carrying counts, quantiles, the generator seed, the app sha, the tokenizer revision (`Qwen/Qwen2.5-3B-Instruct` @ `aa8e7253…`, chat-template sha `cd8e9439…`), and each file's SHA-256.

## 2. One user turn is not one LLM call

The brief's "Workload A — interactive chat" is, in Pickr, a **serial chain of 2.82 LLM calls per turn on average**. The observed sequences (turn 0, status ok):

| n | routed agent | via | call sequence |
|---|---|---|---|
| 373 | ProductRecommendationAgent | classifier | guardrail_input → classifier → agent → guardrail_output |
| 264 | ReviewSummarizationAgent | keyword | guardrail_input → agent → guardrail_output |
| 176 | StorePolicyAgent | keyword | guardrail_input *(deterministic agent, no LLM)* |
| 166 | ProductComparisonAgent | keyword | guardrail_input → agent → guardrail_output |
| 124 | PriceComparisonAgent | keyword | guardrail_input *(deterministic)* |
| 89 | StockAvailabilityAgent | keyword | guardrail_input *(deterministic)* |
| 67 | ProductRecommendationAgent | classifier | guardrail_input → classifier *(classifier chose a deterministic agent path)* |
| 60 | ReviewSummarizationAgent | classifier | guardrail_input → classifier → agent → guardrail_output |
| 42 | FAQAgent | classifier | guardrail_input → classifier → agent → guardrail_output |
| 42 | CapabilitiesAgent | keyword | guardrail_input *(deterministic)* |
| 41 | ProductComparisonAgent | classifier | guardrail_input → classifier → agent → guardrail_output |
| 30 / 22 / 19 | PriceComparison / Stock / Capabilities | classifier | guardrail_input → classifier |

From turn 1 of a conversation, `condense` precedes all of the above.

Two things follow. **Every recommendation query pays a classifier call** — there is no keyword rule for "recommend", so 100% of them route via the LLM fallback (the 5 "keyword" recommendation turns hit another rule by accident). And **user-perceived turn latency is the sum of the chain**: the SLO's TTFT is the `guardrail_input` call's TTFT; the turn's end-to-end latency adds two to four more sequential calls. Analysis therefore reports A **per call role** and additionally **per turn**, never pooled.

## 3. Length distributions (Qwen2.5 tokens; prompts pre-rendered through the chat template)

| trace | n | role | prompt p10 / p50 / p90 / p99 | output p50 / p90 |
|---|---|---|---|---|
| **chat_v1** (A) | 4,375 | all | 242 / **265** / 755 / 877 | 14 / 67 |
| | 1,959 | guardrail_input | – / 265 / 274 / – | 14 / 14 |
| | 806 | classifier | – / 246 / 251 / – | 7 / 7 |
| | 805 | agent | – / 288 / 310 / – | 70 / 165 |
| | 805 | guardrail_output | – / **759** / 824 / – | 9 / 9 |
| **summarization_v1** (B) | 716 | all | 71 / 346 / 597 / 636 | 12 / 54 |
| | 358 | agent | – / **90** / 130 / – | 44 / 60 |
| | 358 | guardrail_output | – / 562 / 618 / – | 9 / 9 |
| **structured_v1** (C, schema overlay on A's rec/comparison agent prompts) | 760 | agent | 316 / 490 / 510 / 517 | 72 / 168 |
| **multiturn_shallow_v1** | 61 | condense | 135 / 202 / 315 / 422 | 10 / 15 |
| **multiturn_medium_v1** | 138 | condense | 168 / 282 / 472 / 608 | 11 / 18 |
| **multiturn_deep_v1** | 240 | condense | 168 / 302 / 467 / 616 | 10 / 18 |

Three findings the spec did not anticipate:

- **`guardrail_output` is the long call on every turn, not the agent call.** Its prompt carries the full retrieved context *and* the generated answer for the faithfulness check (~760 tokens in A, ~560 in B). Workload A's length distribution is dominated by a call role the spec assumed was short.
- **Workload B prompts are short in this catalog.** The catalog holds a median of 2 reviews per product (~51 characters each), so the summarisation *agent* prompt has p50 = 90 tokens. The brief's picture of B as "naturally large prompts" does not hold for Pickr; B is a short-prompt, short-output batch workload. Nothing was synthesised to change this.
- **Outputs are short everywhere.** Guardrail and classifier calls return ~7–14 tokens of JSON; agent answers p50 ≈ 70 tokens (A) / 44 (B). Output length for C is the prose answer's length reused as `output_tokens` — a proxy, since the JSON-cards completion will differ; P3 uses natural termination with a cap and reports what it observes.

## 4. Multi-turn shape: growth is bounded by design

Pickr never resends conversation history to the agent. `handle_conversational_query` condenses the follow-up against the last `HISTORY_WINDOW = 6` rows into a standalone query; the agent sees only that. The only prompt that grows with the conversation is the condense call — and it stops growing at turn 3:

| profile | turn 1 | turn 2 | turn 3 | turn 4 | turn 5 | turn 6 | turn 7 |
|---|---|---|---|---|---|---|---|
| shallow (p50 / p90) | 174 / 275 | 299 / 381 | | | | | |
| medium | 200 / 286 | 283 / 391 | 394 / 518 | 364 / 488 | | | |
| deep | 169 / 243 | 229 / 363 | 342 / 466 | 340 / 484 | 348 / 493 | 370 / 487 | 340 / 431 |

The window slides after three exchanges, so from turn 4 the prompt is a *different* ~350-token transcript each turn rather than a longer one. This is why P4b was restructured into two arms (spec §5, amended 2026-09-19): **Arm 1** measures prefix-cache behaviour on Pickr's real stream — identical system prompt on every request, bounded condense growth — and **Arm 2** replays the same conversations through a conventional history-resending client, clearly labelled as not Pickr's behaviour. The gap between the arms is how much prefix-cache benefit an application forfeits by not resending history. The multi-turn trace files contain only the condense calls, in turn order; the conversations' other calls live in `chat_v1` tagged by `conversation_id`.

## 5. Validation against real traffic: deferred, not failed

Prompt construction is verified by construction — every record is the exact message list Pickr's own code emitted. What is unverified is query *distribution* realism: whether the generator's query mix resembles what real users would send. The LangSmith project holds no deployed-Space traffic (launch-week traces aged out of the 14-day retention window and the Space has been sleeping since), so this check is deferred rather than failed. With only developer-testing traffic available it would have been a weak check at best. The check script (`bench/capture/validate.py`) is retained and runs unchanged if traffic accumulates; a small-n addendum is planned once the deployed app has been exercised again.

Two things were verified while diagnosing: the LangSmith integration itself works (one real query with tracing on produced the expected three `llm` child runs with `usage_metadata`), and the app's test suite leaks `@traceable` fixture roots into the project when run with `.env` loaded — logged as an open issue in `docs/decision-log.md`; it must be fixed before the addendum so the validation set is not contaminated.

## 6. Spec §3.1 prediction 1, re-evaluated

The VRAM budget predicted that, target-only, the P1 concurrency ladder to 32 would be compute-bound rather than KV-bound **provided the measured chat median stayed below ~2k tokens**. Measured chat prompt median = **265 tokens** (p90 755, p99 877). At concurrency 32 that is ≈ 8.5k of ~70k available KV tokens (≈ 24k even at p90) — a 3–8× margin. **Prediction 1 stands**, and the expectation for P1 is now stronger than when it was written: any early flattening of the chat curve would point at compute or scheduling, not KV exhaustion. This is the first pre-registered prediction the study has tested; the second (B's knee lower than A's because of long prompts) was withdrawn before any run once the catalog's review density was measured (§3, and spec §5).

## 7. Exit criteria (amended 2026-09-18)

> Traces v1 exist with per-workload, per-call-role length distributions measured in Qwen tokens, and §3.1 prediction 1 re-evaluated against the measured chat median. External validation deferred, with the check script retained and the reason logged.

Met. Files: `bench/traces/{chat,summarization,structured,multiturn_shallow,multiturn_medium,multiturn_deep}_v1.jsonl` + `.meta.json`; raw source `bench/traces/raw/capture_20260919.jsonl`; this document.

## 8. Carried forward

- **P0b** verifies that the pinned `vllm bench serve` honours per-row `output_tokens` from these files and that both engines report identical `prompt_tokens` for one pre-rendered row (chat-template parity).
- **Sampling parameters** must be pinned explicitly per run (the app runs guardrail/classifier/condense at temperature 0 and agent calls at the SDK default; the client sets one temperature per run — trace rows carry `app_temperature` so the deviation can be stated).
- **P1** predictions as amended: A compute-bound to 32; B's knee at or above A's.
- **Addendum B2** (validation, small n) after the trace-leak fix and a batch of real queries on the live Space.
