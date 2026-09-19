"""Single source of truth for the repo root (ruling C1/P28).

Docker rejects relative `-v` bind-mount sources ("<path> includes invalid
characters for a local volume name"), and the runner can be invoked from any
cwd (`python -m bench.runner ...` from wherever the user happens to be) --
every default config/results/traces path must therefore be anchored to the
repo root rather than left relative. `lifecycle.py`, `cli.py`, `probes.py`
and `sweep.py` all import `REPO_ROOT` from here instead of each re-deriving
it (previously `lifecycle.py` alone defined its own `_REPO_ROOT`).
"""
from __future__ import annotations

from pathlib import Path

# bench/runner/paths.py -> parents[0]=bench/runner, [1]=bench, [2]=repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
