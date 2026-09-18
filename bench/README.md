# bench/ — Pickr inference-serving benchmark

Separate deliverable from the app (see docs/superpowers/specs/2026-09-17-pickr-inference-benchmark-design-v2.md).

## P0a: trace capture

All commands from the repo root, inside WSL2, with `OPENAI_API_KEY` set:

    python -m bench.capture generate  --seed 20260919 --out bench/traces/raw/queries_20260919.json
    python -m bench.capture capture   --queries bench/traces/raw/queries_20260919.json --out bench/traces/raw/capture_20260919.jsonl --workers 4
    python -m bench.capture export    --raw bench/traces/raw/capture_20260919.jsonl --version 1
    python -m bench.capture langsmith --project <LANGSMITH_PROJECT> --out bench/traces/validation/langsmith_20260919.jsonl
    python -m bench.capture validate  --generated bench/traces/raw/capture_20260919.jsonl --real bench/traces/validation/langsmith_20260919.jsonl --out bench/docs/p0a-validation.md

Tests: `pytest tests/bench -q`
