"""The session-anchor series: every anchor run to date, and the deltas between.

P0b found a session can sit ~3% off another for an unknown reason while runs
inside a session agree to ~0.5% (writeup §10a). The anchor is one unchanged
reference run per session; this reports the series so an anomalous session is
visible at the time, and so the inter-session delta sample grows as a byproduct
of the real work rather than needing a dedicated measurement campaign.

Deliberately reads only `bench/results/anchor-*/`, never the runner -- analysis
must not import the harness (ruling P34).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

# The anchor's own config is fixed, so these are directly comparable run to run.
METRICS = ("req_s", "ttft_p50_client", "tpot_p50_client", "e2e_p50_client")
# An anchor this far from the running median marks the session as suspect. Set
# from the within-session block-to-block cv of 0.49% (§10a): 3x it, so ordinary
# run-to-run noise does not raise the flag but the observed ~3% excursion does.
SUSPECT_PCT = 1.5


def anchor_runs(results_root="bench/results") -> list[dict]:
    """One dict per anchor run, oldest first, keyed by its sweep's date stamp."""
    out = []
    for sweep in sorted(Path(results_root).glob("anchor-*")):
        if not sweep.is_dir():
            continue
        for run in sorted(p for p in sweep.iterdir() if p.is_dir()):
            summary = run / "summary.json"
            if not summary.exists():
                continue
            s = json.loads(summary.read_text())
            row = {"session": sweep.name.removeprefix("anchor-"), "run_id": run.name,
                   "valid": s.get("valid")}
            for m in METRICS:
                row[m] = s.get(m)
            env = run / "env.json"
            if env.exists():
                e = json.loads(env.read_text())
                row["power_overlay"] = e.get("windows_power_overlay_effective")
                row["compile_cache_warm"] = e.get("compile_cache_warm")
            out.append(row)
    return out


def series_report(rows: list[dict], metric: str = "req_s", suspect_pct: float = SUSPECT_PCT) -> dict:
    """Median, each session's deviation from it, and which sessions look off.

    The median is used rather than the mean so one bad session does not drag the
    reference it is being judged against.
    """
    vals = [r[metric] for r in rows if isinstance(r.get(metric), (int, float)) and r.get("valid")]
    if not vals:
        return {"n": 0, "median": None, "sessions": [], "suspect": []}
    med = st.median(vals)
    sessions, suspect = [], []
    for r in rows:
        v = r.get(metric)
        if not isinstance(v, (int, float)):
            continue
        dev = 100 * (v - med) / med
        sessions.append({"session": r["session"], metric: v, "dev_pct": dev,
                         "valid": r.get("valid")})
        if abs(dev) > suspect_pct:
            suspect.append(r["session"])
    return {"n": len(vals), "median": med, "sessions": sessions, "suspect": suspect,
            "spread_pct": (100 * (max(vals) - min(vals)) / med) if len(vals) > 1 else 0.0}


def main() -> None:
    rows = anchor_runs()
    if not rows:
        print("no anchor runs yet -- run bench/scripts/anchor.sh at the start of a session")
        return
    print(f"=== session anchor series ({len(rows)} run(s)) ===")
    print(f"{'session':18} {'req_s':>9} {'ttft_p50':>10} {'tpot_p50':>10} {'e2e_p50':>10} "
          f"{'valid':>6} {'dev vs median':>14}")
    rep = series_report(rows)
    dev_by_session = {s["session"]: s["dev_pct"] for s in rep["sessions"]}
    for r in rows:
        dev = dev_by_session.get(r["session"])
        flag = "  <-- SUSPECT" if dev is not None and abs(dev) > SUSPECT_PCT else ""
        print(f"{r['session']:18} {r.get('req_s') or 0:>9.3f} {r.get('ttft_p50_client') or 0:>10.2f} "
              f"{r.get('tpot_p50_client') or 0:>10.2f} {r.get('e2e_p50_client') or 0:>10.2f} "
              f"{str(r.get('valid')):>6} {(f'{dev:+.2f}%' if dev is not None else '-'):>14}{flag}")

    if rep["n"] < 2:
        print("\nOne anchor so far: no inter-session delta yet. The series becomes useful "
              "from the second session.")
        return
    print(f"\nmedian req_s = {rep['median']:.3f} over {rep['n']} valid anchor(s); "
          f"full spread {rep['spread_pct']:.2f}%")
    print(f"suspect threshold ±{SUSPECT_PCT}% (3x the 0.49% within-session block cv)")
    if rep["suspect"]:
        print(f"SUSPECT sessions: {', '.join(rep['suspect'])} -- treat their data with care; "
              f"prefer comparisons blocked inside a session")
    else:
        print("no session flagged")
    vals = [s["req_s"] for s in rep["sessions"] if isinstance(s.get("req_s"), (int, float))]
    if len(vals) >= 3:
        cv = 100 * st.stdev(vals) / st.mean(vals)
        print(f"inter-session cv so far = {cv:.2f}% (n={len(vals)}; needs ~25 to pin to ±20%)")


if __name__ == "__main__":
    main()
