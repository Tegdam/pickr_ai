"""Engine /metrics scraper: parses Prometheus text exposition format and
resolves engine-specific metric names through `EngineSpec.metric_names`
(spec %7 metrics.jsonl).

See bench/docs/p0b-engine-verification.md %3.4 (vLLM) and %4.4 (SGLang) for
the metric names this resolves and their label caveats -- in particular
SGLang's `sglang:cached_tokens_total` is labelled by `cache_source`
(device/host/storage) and only the device label is the prefix-hit proxy
(doc %6.3/%4.4); `_resolve`'s `prefer_label` handles that.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .engine import EngineSpec

METRIC_KEYS = ("kv_usage", "running", "waiting", "spec_accepted", "spec_draft",
               "spec_drafts", "prefix_hits", "prefix_queries")

# Metric names with more than one label combination in the raw text where
# only one is the value we want -- prefer the sample whose key contains this
# label fragment when present.
_PREFERRED_LABEL = {"sglang:cached_tokens_total": 'cache_source="device"'}


def parse_prometheus(text: str) -> dict[str, float]:
    """name{labels} (or bare name) -> value; the label string stays in the key."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not name:
            continue
        try:
            out[name] = float(value)
        except ValueError:
            continue
    return out


def _resolve(raw: dict[str, float], name: str, prefer_label: str | None = None) -> float | None:
    """Match `name` in `raw` ignoring labels: an exact bare-name key, or the
    first key that starts with `name + "{"`. When `prefer_label` is given and
    more than one labelled sample matches, prefer the one whose label string
    contains it (e.g. SGLang's `cache_source="device"`)."""
    candidates = [k for k in raw if k == name or k.startswith(name + "{")]
    if not candidates:
        return None
    if prefer_label:
        for k in candidates:
            if prefer_label in k:
                return raw[k]
    return raw[candidates[0]]


class MetricsScraper:
    """Background thread writing one JSON line per poll to `out_path`."""

    def __init__(self, http, base_url: str, spec: EngineSpec, out_path, interval_s: float = 1.0):
        self.http = http
        self.base_url = base_url
        self.spec = spec
        self.out_path = Path(out_path)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _sample(self) -> dict:
        t = time.monotonic()
        wall = datetime.now(timezone.utc).isoformat()
        raw: dict[str, float] = {}
        error = None
        try:
            resp = self.http.get(self.base_url + self.spec.metrics_path, timeout=5)
            raw = parse_prometheus(resp.text)
        except Exception as exc:
            error = str(exc)

        row: dict = {"t": t, "wall": wall, "raw": raw}
        for key in METRIC_KEYS:
            name = self.spec.metric_names.get(key)
            row[key] = _resolve(raw, name, _PREFERRED_LABEL.get(name)) if name else None
        if error is not None:
            row["error"] = error
        return row

    def _run(self) -> None:
        with self.out_path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                row = self._sample()
                f.write(json.dumps(row) + "\n")
                f.flush()
                self._stop.wait(self.interval_s)


def _values(rows: list[dict], key: str) -> list[float]:
    return [r[key] for r in rows if r.get(key) is not None]


def _delta(rows: list[dict], key: str):
    vals = _values(rows, key)
    if len(vals) < 2:
        return None
    return vals[-1] - vals[0]


def _peak(rows: list[dict], key: str):
    vals = _values(rows, key)
    return max(vals) if vals else None


def summarise(rows: list[dict]) -> dict:
    """Aggregate a run's metric samples: acceptance rate and prefix hit rate
    as end-to-end deltas over the run (not an average of per-sample rates --
    the counters are cumulative), plus peaks for the gauges."""
    d_accepted, d_draft = _delta(rows, "spec_accepted"), _delta(rows, "spec_draft")
    acceptance_rate_mean = d_accepted / d_draft if d_accepted is not None and d_draft else None

    d_hits, d_queries = _delta(rows, "prefix_hits"), _delta(rows, "prefix_queries")
    prefix_hit_rate = d_hits / d_queries if d_hits is not None and d_queries else None

    return {
        "acceptance_rate_mean": acceptance_rate_mean,
        "drafts_delta": _delta(rows, "spec_drafts"),
        "kv_usage_peak": _peak(rows, "kv_usage"),
        "waiting_max": _peak(rows, "waiting"),
        "running_max": _peak(rows, "running"),
        "prefix_hit_rate": prefix_hit_rate,
    }
