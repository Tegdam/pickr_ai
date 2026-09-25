"""GPU telemetry sampler: two nvidia-smi views on a background thread.

The WSL-side `nvidia-smi` sees only this VM's own usage; the Windows-side one
(reached through the Windows filesystem mount, since there is no `nvidia-smi`
binary inside WSL2's own kernel view of the GPU) sees the whole card. The
difference between the two is the rest of the host's reservation (spec %7
gpu_samples.jsonl). Both views were verified callable on 2026-09-18 (doc %7).

Measured 2026-09-25 (host_reservation probe): on this WSL2 build the two views
report the SAME device-wide figure, so `used_host_mb` is always 0 -- see
`env_capture.host_view_note`. Each sample therefore also carries host CPU
(P39), which is the one contention signal neither GPU view can provide and the
load generator is co-located with the engine.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

WSL_SMI = ["nvidia-smi"]
WIN_SMI = ["/mnt/c/Windows/System32/nvidia-smi.exe"]

_QUERY = ("memory.used,memory.total,utilization.gpu,utilization.memory,"
          "clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active")

_KEYS = ("used_mb", "total_mb", "sm_util", "mem_util", "sm_clock", "mem_clock",
         "temp_c", "power_w", "throttle_reasons")
_NUMERIC = {"used_mb": int, "total_mb": int, "sm_util": int, "mem_util": int,
            "sm_clock": int, "mem_clock": int, "temp_c": int, "power_w": float}


def _num(kind, raw: str):
    raw = raw.strip()
    if raw in ("[N/A]", "N/A"):
        return None
    return kind(raw)


def read_host_cpu(proc_stat: str = "/proc/stat", loadavg: str = "/proc/loadavg") -> dict:
    """P39: the load generator is co-located with the engine, so CPU contention
    on this laptop can starve the client and inflate TTFT with no trace in the
    GPU samples -- the one blind spot the two nvidia-smi views cannot cover.
    Sampled from /proc so it costs nothing and needs no extra dependency.

    `cpu_busy_pct` is computed against the previous call's counters (None on
    the first sample, which has no interval to difference against).
    """
    fields = Path(proc_stat).read_text().splitlines()[0].split()[1:]
    vals = [int(x) for x in fields[:8]]
    idle = vals[3] + vals[4]                                   # idle + iowait
    total = sum(vals)
    la1, la5, la15 = Path(loadavg).read_text().split()[:3]
    row = {"cpu_total_jiffies": total, "cpu_idle_jiffies": idle,
           "loadavg_1m": float(la1), "loadavg_5m": float(la5), "loadavg_15m": float(la15)}
    prev = read_host_cpu._prev
    read_host_cpu._prev = (total, idle)
    if prev is not None:
        d_total, d_idle = total - prev[0], idle - prev[1]
        row["cpu_busy_pct"] = round(100.0 * (d_total - d_idle) / d_total, 1) if d_total > 0 else None
    else:
        row["cpu_busy_pct"] = None
    return row


read_host_cpu._prev = None


def reset_host_cpu_counter() -> None:
    """Drop the cached counters so a fresh sampler's first interval is its own."""
    read_host_cpu._prev = None


def read_gpu(nvidia_smi_cmd: list[str]) -> dict:
    cmd = [*nvidia_smi_cmd, f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"]
    out = subprocess.check_output(cmd, text=True, timeout=5)
    parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    row = dict(zip(_KEYS, parts))
    for key, kind in _NUMERIC.items():
        row[key] = _num(kind, row[key])
    return row


class GpuSampler:
    """Background thread writing one JSON line per poll to `out_path`.

    Reader exceptions are caught per source (WSL, Windows) and recorded as
    `wsl_error`/`win_error` instead of killing the sampler.
    """

    JOIN_TIMEOUT_S = 5.0

    def __init__(self, out_path, interval_s: float = 1.0, wsl_cmd: list[str] = WSL_SMI,
                 win_cmd: list[str] = WIN_SMI, reader=read_gpu, host_reader=read_host_cpu):
        self.out_path = Path(out_path)
        self.interval_s = interval_s
        self.wsl_cmd = wsl_cmd
        self.win_cmd = win_cmd
        self.reader = reader
        self.host_reader = host_reader
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("GpuSampler already started")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        reset_host_cpu_counter()      # P39: this run's first interval is its own
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
        row: dict = {"t": t, "wall": wall}

        wsl = None
        try:
            wsl = self.reader(self.wsl_cmd)
        except Exception as exc:
            row["wsl_error"] = str(exc)

        win = None
        try:
            win = self.reader(self.win_cmd)
        except Exception as exc:
            row["win_error"] = str(exc)

        row["used_ours_mb"] = wsl["used_mb"] if wsl else None
        row["used_total_mb"] = win["used_mb"] if win else None
        if wsl and win and wsl["used_mb"] is not None and win["used_mb"] is not None:
            row["used_host_mb"] = max(0, win["used_mb"] - wsl["used_mb"])
            # I6: the same difference, but NOT clamped to 0 -- lets analysis
            # see whether the two nvidia-smi views actually move together
            # (see env_capture.host_view_note / probes._probe_host_reservation's
            # views_track_each_other): a clamped used_host_mb alone can't
            # distinguish "no host share" from "the WSL view briefly read
            # higher than the Windows view."
            row["view_diff_mb"] = win["used_mb"] - wsl["used_mb"]
        else:
            row["used_host_mb"] = None
            row["view_diff_mb"] = None

        # Windows view is preferred for the shared fields (it also carries
        # power.max_limit / throttle reasons that the WSL view lacks, doc %7);
        # fall back to the WSL view if the Windows one failed.
        primary = win or wsl or {}
        for key in ("sm_util", "mem_util", "sm_clock", "mem_clock", "temp_c", "power_w", "throttle_reasons"):
            row[key] = primary.get(key)

        # P39: host CPU alongside the GPU views -- the client is co-located, so
        # contention here is a client-starvation signal the GPU cannot show.
        if self.host_reader is not None:
            try:
                row.update(self.host_reader())
            except Exception as exc:
                row["host_cpu_error"] = str(exc)

        return row

    def _run(self) -> None:
        with self.out_path.open("a", encoding="utf-8") as f:
            while not self._stop.is_set():
                row = self._sample()
                f.write(json.dumps(row) + "\n")
                f.flush()
                self._stop.wait(self.interval_s)
