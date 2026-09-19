"""Engine /metrics scraper: parses Prometheus text exposition format and
resolves engine-specific metric names through `EngineSpec.metric_names`
(spec %7 metrics.jsonl).

See bench/docs/p0b-engine-verification.md %3.4 (vLLM) and %4.4 (SGLang) for
the metric names this resolves and their caveats:

- SGLang's `sglang:cached_tokens_total` is labelled by `cache_source`
  (device/host/storage) and only the device label is the prefix-hit proxy
  (doc %6.3/%4.4); `_resolve`'s `prefer_label` handles that.
- SGLang's `sglang:prompt_tokens_total` is labelled by `is_streaming` and
  every series must be summed to get the total prefill-token count (doc
  %4.4's "queries" proxy); `_SUM_LABELS` handles that.
- SGLang has no cumulative accepted/draft-token counters at all -- its
  `spec_accept_length`/`spec_accept_rate` are per-interval gauges, not
  counters (doc %4.4: "spec_accept_* are most-recent-interval gauges"), so a
  vLLM-style Δaccepted/Δdraft over the run would be a fabricated number for
  SGLang. `summarise` branches on the `engine` stamped into every row instead.
"""
from __future__ import annotations

import json
import math
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

# Metric names whose label series must be summed to get the total (doc %4.4:
# sglang:prompt_tokens_total is split by is_streaming, with no unlabelled
# aggregate series).
_SUM_LABELS = {"sglang:prompt_tokens_total"}


def parse_prometheus(text: str) -> dict[str, float]:
    """name{labels} (or bare name) -> value; the label string stays in the
    key. Tolerates an optional trailing sample timestamp (3rd whitespace-
    separated field) and drops non-finite values (`inf`/`nan`)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        name, value_str = fields[0], fields[1]
        try:
            value = float(value_str)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        out[name] = value
    return out


def _resolve(raw: dict[str, float], name: str, prefer_label: str | None = None) -> float | None:
    """Match `name` in `raw` ignoring labels: an exact bare-name key, or any
    key that starts with `name + "{"` (never a same-prefixed different metric
    like `name_total`/`name_bucket`, since those don't have `{` right after
    `name`). When `name` is in `_SUM_LABELS`, every matching label series is
    summed. Otherwise, when `prefer_label` is given and more than one
    labelled sample matches, prefer the one whose label string contains it
    (e.g. SGLang's `cache_source="device"`)."""
    candidates = [k for k in raw if k == name or k.startswith(name + "{")]
    if not candidates:
        return None
    if name in _SUM_LABELS:
        return sum(raw[k] for k in candidates)
    if prefer_label:
        for k in candidates:
            if prefer_label in k:
                return raw[k]
    return raw[candidates[0]]


class MetricsScraper:
    """Background thread writing one JSON line per poll to `out_path`."""

    JOIN_TIMEOUT_S = 5.0

    def __init__(self, http, base_url: str, spec: EngineSpec, out_path, interval_s: float = 1.0):
        self.http = http
        self.base_url = base_url
        self.spec = spec
        self.out_path = Path(out_path)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("MetricsScraper already started")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                # Still writing a sample -- block for real so the caller never
                # reads a torn last line off the tail of the file.
                self._thread.join()

    def _sample(self) -> dict:
        t = time.monotonic()
        wall = datetime.now(timezone.utc).isoformat()
        raw: dict[str, float] = {}
        error = None
        try:
            resp = self.http.get(self.base_url + self.spec.metrics_path, timeout=5)
            if resp.status_code != 200:
                raise RuntimeError(f"metrics endpoint returned {resp.status_code}")
            raw = parse_prometheus(resp.text)
        except Exception as exc:
            error = str(exc)

        row: dict = {"t": t, "wall": wall, "engine": self.spec.name, "raw": raw}
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


def _ratio_delta(rows: list[dict], numerator: str, denominator: str):
    num, den = _delta(rows, numerator), _delta(rows, denominator)
    return num / den if num is not None and den else None


def summarise(rows: list[dict]) -> dict:
    """Aggregate a run's metric samples. Acceptance accounting is
    engine-specific (doc %3.4 vs %4.4, see module docstring): vLLM has true
    cumulative accepted/draft-token counters, so acceptance_rate_mean is a
    real Δaccepted/Δdraft over the run; SGLang has no such counters, so that
    key is always None there and the SGLang-only gauge summaries are reported
    instead. `rows[0]["engine"]` (stamped by `MetricsScraper`) selects the
    branch; rows lacking it (e.g. hand-built in tests) default to the vLLM
    shape."""
    common = {
        "drafts_delta": _delta(rows, "spec_drafts"),
        "kv_usage_peak": _peak(rows, "kv_usage"),
        "waiting_max": _peak(rows, "waiting"),
        "running_max": _peak(rows, "running"),
        "prefix_hit_rate": _ratio_delta(rows, "prefix_hits", "prefix_queries"),
    }

    engine = rows[0].get("engine", "vllm") if rows else "vllm"
    if engine == "sglang":
        accepted_vals = _values(rows, "spec_accepted")
        rate_vals = [
            v for v in (
                _resolve(r.get("raw", {}), "sglang:spec_accept_rate")
                for r in rows if (r.get("running") or 0) > 0
            ) if v is not None
        ]
        return {
            "acceptance_rate_mean": None,
            "acceptance_source": "sglang_gauge",
            "spec_accept_length_last": accepted_vals[-1] if accepted_vals else None,
            "spec_accept_rate_mean": (sum(rate_vals) / len(rate_vals)) if rate_vals else None,
            **common,
        }

    return {
        "acceptance_rate_mean": _ratio_delta(rows, "spec_accepted", "spec_draft"),
        "acceptance_source": "vllm_counters",
        **common,
    }
