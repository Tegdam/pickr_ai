"""Hand-rolled schema for `vllm bench serve --save-result --save-detailed`
output at the pinned vLLM version (doc §5, §8 "Client" table). No jsonschema
dependency (bench/requirements.txt stays at pyyaml / requests / aiohttp,
doc §8 Requirements).

The key set and per-key types below are the minimum the doc's §5 "output keys"
list guarantees (`--save-detailed` per-request arrays plus the always-present
summary scalars); the real output carries many more keys (percentile/mean/etc,
doc §5 "Observed full key list") but this runner only depends on these.
"""
from __future__ import annotations

CLIENT_OUTPUT_SCHEMA_VERSION = "vllm-v0.29.0-save-detailed"

# doc §5 ("--save-detailed output keys") / §8 Client "Output" bullet: always-present
# summary scalars plus the per-request arrays that --save-detailed adds.
REQUIRED_CLIENT_KEYS: dict[str, type] = {
    "completed": int,
    "duration": float,
    "total_input_tokens": int,
    "total_output_tokens": int,
    "request_throughput": float,
    "output_throughput": float,
    "ttfts": list,
    "itls": list,
    "input_lens": list,
    "output_lens": list,
    "generated_texts": list,
    "errors": list,
    "start_times": list,
}

# Per-request arrays that must all describe the same N requests (doc §5: one
# entry per completed-or-failed request, index i <-> trace row i under
# --disable-shuffle).
_PER_REQUEST_LIST_KEYS = (
    "ttfts", "itls", "input_lens", "output_lens", "generated_texts", "errors", "start_times",
)


class SchemaError(Exception):
    pass


def _type_ok(value, expected: type) -> bool:
    if expected is float:
        # json ints (e.g. "duration": 2) are valid floats; bool is never a number here.
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate_client_output(d: dict) -> None:
    for key, expected in REQUIRED_CLIENT_KEYS.items():
        if key not in d:
            raise SchemaError(f"client output missing required key {key!r}")
        if not _type_ok(d[key], expected):
            raise SchemaError(
                f"client output key {key!r} has wrong type: expected {expected.__name__}, "
                f"got {type(d[key]).__name__}"
            )
    lengths = {k: len(d[k]) for k in _PER_REQUEST_LIST_KEYS}
    if len(set(lengths.values())) > 1:
        raise SchemaError(f"client output per-request lists have mismatched lengths: {lengths}")
