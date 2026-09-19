"""Per-sweep progress, persisted after every mutation so `resume` never repeats a finished run."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("pending", "running", "done", "invalid", "requeued")


class SweepState:
    """Tracks each run_id's status across a sweep and persists to `path` as JSON
    after every mutation. `SweepState.load(path)` resumes from that file."""

    def __init__(self, path: str | Path, run_ids: list[str] | None = None, max_retries_total: int = 0):
        self.path = Path(path)
        self.max_retries_total = max_retries_total
        self.retries_used = 0
        self.order: list[str] = list(run_ids or [])
        self.runs: dict[str, dict] = {r: {"status": "pending", "reason": None, "attempts": 0} for r in self.order}
        if run_ids is not None:
            self._save()

    @classmethod
    def load(cls, path: str | Path) -> "SweepState":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        s = cls(path, None, d["max_retries_total"])
        s.retries_used = d["retries_used"]
        s.order = d["order"]
        s.runs = d["runs"]
        return s

    def mark(self, run_id: str, status: str, reason: str | None = None) -> None:
        r = self.runs[run_id]
        r["status"] = status
        r["reason"] = reason
        r["updated_at"] = datetime.now(timezone.utc).isoformat()
        if status == "running":
            r["attempts"] += 1
        self._save()

    def requeue(self, run_id: str) -> bool:
        """Move run_id to the end of the queue as `requeued`. Returns False (no-op)
        once the sweep's total retry budget is exhausted."""
        if self.retries_used >= self.max_retries_total:
            return False
        self.retries_used += 1
        self.order.remove(run_id)
        self.order.append(run_id)
        self.runs[run_id]["status"] = "requeued"
        self._save()
        return True

    def pending(self) -> list[str]:
        return [r for r in self.order if self.runs[r]["status"] in ("pending", "requeued")]

    def to_dict(self) -> dict:
        return {
            "max_retries_total": self.max_retries_total,
            "retries_used": self.retries_used,
            "order": self.order,
            "runs": self.runs,
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
