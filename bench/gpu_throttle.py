"""Pure nvidia-smi `clocks_throttle_reasons.active` bit-fraction helpers.

Shared by `bench.runner.summary.build_summary` (a real run's own
gpu_samples.jsonl) and `bench.analysis.reducers.throttle_baseline` (the
`host_reservation` probe's idle-phase samples) so the bit semantics can never
drift between the two call sites. Lives outside both `bench.runner` and
`bench.analysis` on purpose: the analysis package must never import the
runner package (it needs to run standalone, e.g. from a report notebook that
never touches docker/subprocess), so this tiny shared piece sits above both.
"""
from __future__ import annotations

THROTTLE_BITS = (0x4, 0x8, 0x20, 0x40, 0x80)


def throttle_reason_values(gpu_rows: list[dict]) -> list[int]:
    values: list[int] = []
    for row in gpu_rows:
        reasons = row.get("throttle_reasons")
        if reasons is None:
            continue
        try:
            values.append(int(reasons, 16))
        except (TypeError, ValueError):
            continue
    return values


def throttle_bits_fraction(values: list[int]) -> dict:
    """Fraction of samples (with non-None reasons) having each bit set."""
    if not values:
        return {f"0x{b:x}": None for b in THROTTLE_BITS}
    n = len(values)
    return {f"0x{b:x}": sum(1 for v in values if v & b) / n for b in THROTTLE_BITS}
