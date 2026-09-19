"""bench.analysis.reducers must never import bench.runner (ruling: the runner
may import the analysis package, but analysis must run standalone -- e.g.
from a report notebook that never touches docker/subprocess/GPU code at
all). Statically checks the module's own import statements via `ast` rather
than inspecting `sys.modules` (which could already have bench.runner loaded
by an unrelated test in the same process, masking a real violation)."""
from __future__ import annotations

import ast
from pathlib import Path

import bench.analysis.reducers as reducers


def _imported_module_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_reducers_module_never_imports_bench_runner():
    source = Path(reducers.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = _imported_module_names(tree)
    offenders = [name for name in imported if name == "bench.runner" or name.startswith("bench.runner.")]
    assert offenders == []


def test_reducers_exposes_the_five_pure_functions():
    for name in ("ceiling_from_rows", "ceiling_rows_from_sweep", "parity_verdict",
                 "acceptance_delta", "throttle_baseline"):
        assert hasattr(reducers, name), name
