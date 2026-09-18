"""Compare the generator's prompt-length distributions against real emitted
prompts, per (workload, call_role). Both sides are rendered and counted with
the same tokenizer, so the only difference is who wrote the query.

Real traffic carries no routing information (LangSmith runs don't include the
coordinator's log line), so real records are assigned to workloads by
call_role alone: they all land in A. B's real count is therefore 0 and its
cell reads "insufficient_real" — the report says so rather than hiding it.

Per-cell status is one of "pass", "fail", "insufficient_real", or
"insufficient_generated".
"""
from __future__ import annotations

from .export import assign_workload, quantiles
from .tokens import QwenTokenizer

NOTE_ROUTING = ("routed_agent is unknown for real traffic, so real records are assigned by call_role only "
                "(all to A); B and C cells have no real counterpart and are reported as insufficient_real.")


def _lengths(records: list[dict], tokenizer: QwenTokenizer) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in records:
        w = assign_workload(r)
        if w is None:
            continue
        key = f"{w}/{r['call_role']}"
        out.setdefault(key, []).append(tokenizer.count(tokenizer.render(r["messages"])))
    return out


def _rel(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a - b) / b


def compare(generated: list[dict], real: list[dict], tokenizer: QwenTokenizer,
            tolerance: float = 0.25, min_n: int = 10) -> dict:
    g = _lengths(generated, tokenizer)
    r = _lengths(real, tokenizer)
    cells = {}
    for key in sorted(set(g) | set(r)):
        qg, qr = quantiles(g.get(key, [])), quantiles(r.get(key, []))
        d50, d90 = _rel(qg["p50"], qr["p50"]), _rel(qg["p90"], qr["p90"])
        if qr["n"] < min_n:
            status = "insufficient_real"
        elif qg["n"] == 0:
            status = "insufficient_generated"
        elif abs(d50) <= tolerance and abs(d90) <= tolerance:
            status = "pass"
        else:
            status = "fail"
        cells[key] = {"n_generated": qg["n"], "n_real": qr["n"],
                      "p50_generated": qg["p50"], "p50_real": qr["p50"],
                      "p90_generated": qg["p90"], "p90_real": qr["p90"],
                      "rel_diff_p50": d50, "rel_diff_p90": d90, "status": status}
    if any(c["status"] == "fail" for c in cells.values()):
        overall = "fail"
    elif any(c["status"] == "pass" for c in cells.values()):
        overall = "pass"
    else:
        overall = "insufficient"
    return {"tolerance": tolerance, "min_n": min_n, "cells": cells, "overall_status": overall,
            "notes": [NOTE_ROUTING]}


def _num(x):
    return "—" if x is None else f"{x:.0f}"


def _pct(x):
    return "—" if x is None else f"{x:+.0%}"


def to_markdown(report: dict) -> str:
    lines = ["# Generated vs real prompt-length validation", "",
             f"Tolerance ±{report['tolerance']:.0%} on p50 and p90; cells with < {report['min_n']} real records "
             f"are marked insufficient_real and do not affect the overall result.", "",
             "| cell | n gen | n real | p50 gen | p50 real | Δp50 | p90 gen | p90 real | Δp90 | status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in report["cells"].items():
        lines.append(f"| {key} | {c['n_generated']} | {c['n_real']} | {_num(c['p50_generated'])} | {_num(c['p50_real'])} | "
                     f"{_pct(c['rel_diff_p50'])} | {_num(c['p90_generated'])} | {_num(c['p90_real'])} | "
                     f"{_pct(c['rel_diff_p90'])} | {c['status']} |")
    lines += ["", f"**Overall: {report['overall_status']}**", ""]
    lines += [f"- {n}" for n in report["notes"]]
    return "\n".join(lines) + "\n"
