"""`python -m bench.echo_server --port 8000 --per-token-ms 5 --host 0.0.0.0`

Launched by `bench/runner/lifecycle.py`'s `start_engine` as a plain subprocess
(`ENGINES["echo"]`, `spec.image is None`) -- see that module for how its
stdout/stderr are captured for `log.txt`/crash diagnostics.
"""
from __future__ import annotations

import argparse

from aiohttp import web

from .server import create_app


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench.echo_server")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--per-token-ms", type=int, default=5)
    p.add_argument("--host", default="0.0.0.0")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = create_app(per_token_ms=args.per_token_ms)
    # access_log=None: a real sweep sends thousands of requests through this
    # process, and nothing in lifecycle.py drains its stderr pipe while it is
    # running (only after it exits) -- per-request access logging would fill
    # that pipe's OS buffer and deadlock the subprocess.
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
