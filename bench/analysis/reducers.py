"""P0b hardware/harness calibration reducers (spec §3.1, §9 P0b exit; doc
bench/docs/p0b-engine-verification.md §7-8).

Moved out of `bench.runner.probes` (final-review fix wave, Minors) so the
P0b calibration analysis can be imported and run standalone -- from a report
notebook, say -- without pulling in `bench.runner`'s docker/subprocess/GPU
machinery at all. **This module must never import `bench.runner`**
(`bench.runner.probes` imports these functions back, not the other way
around); `tests/bench/test_analysis_reducers.py` asserts this statically.

Five pure functions, unit-tested directly against hand-built inputs:

- `ceiling_from_rows`: reduces the echo-server request-rate ladder
  (`p0b_ceiling.yaml`'s `summary.json`s) to the harness ceiling (spec §9:
  ceiling >= 3x the study's peak rate). Final-review fix I1/P30: a rung
  passes on the *offered* rate actually achieved (`(N-1) / span` over
  `requests.jsonl`'s own `start_time`s, via `ceiling_rows_from_sweep`), not
  the client's own `req_s` (which can under-report on a ramp) -- `req_s` is
  still carried in the table for reference. The ceiling is the highest rung
  strictly below the first failing rung (monotone): a later rung "passing"
  past a failure never resurrects the ceiling.
- `ceiling_rows_from_sweep`: reads a real (or fixture) `p0b-ceiling` sweep
  directory's per-run `config.yaml`/`summary.json`/`requests.jsonl` files and
  averages reps per `request_rate`, producing `ceiling_from_rows`'s input
  directly. Runs whose `summary.json["valid"]` is `False` are skipped
  outright (I1: an invalid run's throughput/TTFT numbers say nothing about
  the harness's own ceiling).
- `parity_verdict`: reduces the chat-template parity sweep
  (`p0b_parity.yaml`'s `summary.json`s, one `usage.prompt_tokens` per engine)
  to a pass/fail against the trace row's own token count (spec §4).
- `acceptance_delta`: the pre-registered `ignore_eos` x acceptance-rate rule
  (spec §3.1): a >10% relative divergence between the with/without
  `--ignore-eos` acceptance rates means P2 reports acceptance over the
  natural-length prefix only.
- `throttle_baseline`: per-bit `clocks_throttle_reasons.active` fractions
  over a set of GPU samples (the `host_reservation` probe's idle phase), via
  `bench.gpu_throttle` -- the exact bit-fraction computation
  `bench.runner.summary.build_summary` uses for a real run, so the bit
  semantics never drift between the two.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import yaml

from bench.gpu_throttle import throttle_bits_fraction, throttle_reason_values


def ceiling_from_rows(rows: list[dict], baseline_floor_ms: float = 25.0) -> dict:
    """Spec §9 P0b exit: the harness ceiling is the highest `request_rate` R
    (from `p0b_ceiling.yaml`'s per-rate rows: `{"request_rate", "req_s",
    "offered_req_s", "ttft_p99_ms"}`) at which the ACHIEVED offered rate
    (`offered_req_s`, computed from `requests.jsonl`'s own request-issue
    timestamps -- see `ceiling_rows_from_sweep`) stays within 5% of the
    requested rate AND `ttft_p99_ms` stays within the baseline's own noise
    band (`max(2x baseline, baseline + baseline_floor_ms)` -- the `+floor`
    term keeps a near-zero baseline TTFT from making the 2x test trivially
    tight). `req_s` (the client's own reported throughput) is carried in the
    table for reference but is no longer the pass/fail signal (I1/P30: it can
    under-report during a ramp even when the harness is keeping up).

    Final-review fix I1/P30 (monotone ceiling): rungs are walked in
    ascending-rate order and the ceiling is the highest rate seen *before*
    the first failing rung -- a later rung that happens to "pass" past that
    first failure (a non-monotone table) never resurrects or raises the
    ceiling. Returns `{"ceiling_req_s": R or None, "table": rows sorted by
    rate, each with a "passes" bool}`; `None` if the first rung already
    fails (or the input is empty)."""
    sorted_rows = sorted(rows, key=lambda r: r["request_rate"])
    if not sorted_rows:
        return {"ceiling_req_s": None, "table": []}

    baseline_ttft = sorted_rows[0]["ttft_p99_ms"]
    ttft_threshold = max(2 * baseline_ttft, baseline_ttft + baseline_floor_ms)

    table: list[dict] = []
    ceiling: float | None = None
    first_failure_seen = False
    for row in sorted_rows:
        passes = (
            row["offered_req_s"] >= 0.95 * row["request_rate"]
            and row["ttft_p99_ms"] <= ttft_threshold
        )
        table.append({**row, "passes": passes})
        if first_failure_seen:
            continue
        if passes:
            ceiling = row["request_rate"]
        else:
            first_failure_seen = True
    return {"ceiling_req_s": ceiling, "table": table}


def _offered_req_s(requests_path: Path, attempted) -> float | None:
    """I1/P30: the rate actually offered to the engine, from `requests.jsonl`'s
    own per-request `start_time`s -- `(N-1) / (start_times[-1] - start_times[0])`,
    N = `attempted` (spec's own attempted count, not just `len(requests.jsonl)`,
    which can be shorter than `attempted` on the client's own error paths).
    `None` when there aren't at least two start times to span, or attempted is
    falsy/missing."""
    if not attempted or not requests_path.exists():
        return None
    rows = []
    for line in requests_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    start_times = [r["start_time"] for r in rows if r.get("start_time") is not None]
    if len(start_times) < 2:
        return None
    span = start_times[-1] - start_times[0]
    if span <= 0:
        return None
    return (attempted - 1) / span


def ceiling_rows_from_sweep(sweep_dir) -> list[dict]:
    """Reads a `p0b-ceiling` sweep directory's per-run `config.yaml`
    (`request_rate`), `summary.json` (`req_s`, `ttft_p99_client`, `valid`,
    `attempted`) and `requests.jsonl` (`start_time`s, for `_offered_req_s`),
    and averages reps per rate, producing `ceiling_from_rows`'s input
    directly from a real (or fixture) sweep directory. Runs missing either
    `config.yaml`/`summary.json`, or whose `config.yaml` carries no
    `request_rate` (not a poisson-mode run), are skipped -- as is any run
    whose `summary.json["valid"]` is explicitly `False` (I1/P30: an invalid
    run's throughput/TTFT numbers say nothing about the harness ceiling)."""
    sweep_dir = Path(sweep_dir)
    by_rate: dict[float, list[dict]] = {}
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        config_path, summary_path = run_dir / "config.yaml", run_dir / "summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        rate = cfg.get("request_rate")
        if rate is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("valid") is False:
            continue
        by_rate.setdefault(rate, []).append({
            "req_s": summary.get("req_s"),
            "ttft_p99_ms": summary.get("ttft_p99_client"),
            "offered_req_s": _offered_req_s(run_dir / "requests.jsonl", summary.get("attempted")),
        })

    rows = []
    for rate, entries in by_rate.items():
        req_s_vals = [e["req_s"] for e in entries if e["req_s"] is not None]
        ttft_vals = [e["ttft_p99_ms"] for e in entries if e["ttft_p99_ms"] is not None]
        offered_vals = [e["offered_req_s"] for e in entries if e["offered_req_s"] is not None]
        rows.append({
            "request_rate": rate,
            "req_s": statistics.fmean(req_s_vals) if req_s_vals else None,
            "ttft_p99_ms": statistics.fmean(ttft_vals) if ttft_vals else None,
            "offered_req_s": statistics.fmean(offered_vals) if offered_vals else None,
        })
    return rows


def parity_verdict(prompt_tokens_by_engine: dict[str, int], expected: int) -> dict:
    """Spec §4 chat-template parity: pass iff `prompt_tokens_by_engine` is
    non-empty and every engine's reported `usage.prompt_tokens` equals
    `expected` (the trace row's own `prompt_tokens_qwen`) -- an empty dict
    never passes vacuously."""
    return {
        "pass": bool(prompt_tokens_by_engine) and all(v == expected for v in prompt_tokens_by_engine.values()),
        "values": dict(prompt_tokens_by_engine),
        "expected": expected,
    }


def acceptance_delta(with_ignore_eos: float | None, without: float | None) -> dict:
    """Spec §3.1 pre-registered rule: if the with/without `--ignore-eos`
    acceptance rates differ by more than 10% relative, P2 reports acceptance
    over the natural-length prefix only. `rel_diff` is `None` (and the rule
    does not apply) when either rate is missing or `without` is zero."""
    if with_ignore_eos is None or without is None or without == 0:
        return {"rel_diff": None, "prefix_rule_applies": False}
    rel_diff = (with_ignore_eos - without) / without
    return {"rel_diff": rel_diff, "prefix_rule_applies": abs(rel_diff) > 0.10}


def throttle_baseline(gpu_rows: list[dict]) -> dict:
    """Per-bit fraction of samples (with a non-None `throttle_reasons`)
    having each `clocks_throttle_reasons.active` bit set, over `gpu_rows`
    (the `host_reservation` probe's idle-phase samples). Delegates to
    `bench.gpu_throttle` -- the exact computation
    `bench.runner.summary.build_summary` uses for a real run -- so the bit
    semantics never drift from the runner's."""
    return throttle_bits_fraction(throttle_reason_values(gpu_rows))
