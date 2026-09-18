"""python -m bench.capture <generate|capture|export|langsmith|validate>"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .driver import capture_all, read_raw
from .export import export_all
from .generator import Catalog, from_json, generate_conversations, generate_queries, to_json
from .langsmith_export import export_langsmith
from .tokens import QwenTokenizer
from .validate import compare, to_markdown


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _generate(a) -> int:
    catalog = Catalog.from_app()
    qs = generate_queries(catalog, n=a.n_single, seed=a.seed)
    convs = generate_conversations(catalog, n_per_profile=a.n_per_profile, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"seed": a.seed, **to_json(qs, convs)}, indent=1), encoding="utf-8")
    print(f"wrote {len(qs)} queries and {len(convs)} conversations ({sum(len(c.turns) for c in convs)} turns) to {a.out}")
    return 0


def _capture(a) -> int:
    qs, convs = from_json(json.loads(Path(a.queries).read_text(encoding="utf-8")))
    n = capture_all(qs, convs, Path(a.out), workers=a.workers, app_git_sha=git_sha())
    print(f"wrote {n} new records to {a.out} ({len(read_raw(Path(a.out)))} total)")
    return 0


def _export(a) -> int:
    tok = QwenTokenizer.load(model_id=a.tokenizer, revision=a.tokenizer_revision)
    try:
        paths = export_all(Path(a.raw), Path(a.out_dir), version=a.version, tokenizer=tok, seed=a.seed,
                           app_git_sha=git_sha())
    except FileExistsError as e:
        print(f"refusing to overwrite: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"refusing to export: {e}", file=sys.stderr)
        return 1
    for p in paths:
        print(p)
    return 0


def _langsmith(a) -> int:
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc) if a.since else None
    n = export_langsmith(a.project, Path(a.out), since=since)
    print(f"wrote {n} real records to {a.out}")
    return 0


def _validate(a) -> int:
    for label, p in (("--generated", a.generated), ("--real", a.real)):
        if not Path(p).exists():
            print(f"{label} file not found: {p}", file=sys.stderr)
            return 1
    tok = QwenTokenizer.load(model_id=a.tokenizer, revision=a.tokenizer_revision)
    report = compare(read_raw(Path(a.generated)), read_raw(Path(a.real)), tok, tolerance=a.tolerance, min_n=a.min_n)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(to_markdown(report), encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"overall: {report['overall_status']} -> {a.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench.capture")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--n-single", type=int, default=1400)
    g.add_argument("--n-per-profile", type=int, default=40)
    g.add_argument("--out", required=True)
    g.set_defaults(fn=_generate)

    c = sub.add_parser("capture")
    c.add_argument("--queries", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=4)
    c.set_defaults(fn=_capture)

    e = sub.add_parser("export")
    e.add_argument("--raw", required=True)
    e.add_argument("--out-dir", default="bench/traces")
    e.add_argument("--version", type=int, required=True)
    e.add_argument("--seed", type=int, required=True)
    e.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    e.add_argument("--tokenizer-revision", default=None)
    e.set_defaults(fn=_export)

    l = sub.add_parser("langsmith")
    l.add_argument("--project", required=True)
    l.add_argument("--out", required=True)
    l.add_argument("--since", default=None, help="ISO date, UTC")
    l.set_defaults(fn=_langsmith)

    v = sub.add_parser("validate")
    v.add_argument("--generated", required=True)
    v.add_argument("--real", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--tolerance", type=float, default=0.25)
    v.add_argument("--min-n", type=int, default=10)
    v.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B-Instruct")
    v.add_argument("--tokenizer-revision", default=None)
    v.set_defaults(fn=_validate)
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)
