import json

import pytest

from bench.runner.state import SweepState


def test_state_persists_and_resumes(tmp_path):
    p = tmp_path / "state.json"
    s = SweepState(p, run_ids=["a", "b", "c"], max_retries_total=1)
    assert s.pending() == ["a", "b", "c"]
    s.mark("a", "running"); s.mark("a", "done")
    s.mark("b", "invalid", reason="acceptance zero")
    assert s.requeue("b") is True and s.pending() == ["c", "b"]        # re-queued to the END
    s.mark("b", "invalid", reason="again")
    assert s.requeue("b") is False                                       # budget exhausted
    s2 = SweepState.load(p)
    assert s2.pending() == ["c"] and s2.to_dict()["runs"]["a"]["status"] == "done"
    assert s2.to_dict()["retries_used"] == 1 and s2.to_dict()["runs"]["b"]["reason"] == "again"
    assert json.loads(p.read_text())["max_retries_total"] == 1


def test_reset_stale_requeues_crashed_running_runs(tmp_path):
    """A run left `running` when the process died must not be dropped forever:
    `pending()` alone never surfaces it, but `reset_stale()` (called explicitly
    by the lifecycle on resume, never automatically by `load()`) flips it back
    to pending, preserving its attempts and its original queue position."""
    p = tmp_path / "state.json"
    s = SweepState(p, run_ids=["a", "b", "c"], max_retries_total=1)
    s.mark("a", "running")
    s2 = SweepState.load(p)
    assert "a" not in s2.pending()  # not auto-reset by load()
    assert s2.to_dict()["runs"]["a"]["attempts"] == 1
    assert s2.reset_stale() == ["a"]
    assert s2.pending() == ["a", "b", "c"]  # "a" restored to its original position
    assert s2.to_dict()["runs"]["a"]["attempts"] == 1  # attempts preserved
    s3 = SweepState.load(p)  # reset_stale() persisted
    assert s3.pending() == ["a", "b", "c"]


def test_mark_rejects_unknown_status(tmp_path):
    s = SweepState(tmp_path / "state.json", run_ids=["a"], max_retries_total=0)
    with pytest.raises(ValueError):
        s.mark("a", "bogus")


def test_requeue_unknown_run_id_raises_before_touching_budget(tmp_path):
    s = SweepState(tmp_path / "state.json", run_ids=["a"], max_retries_total=1)
    with pytest.raises(KeyError):
        s.requeue("zzz")
    assert s.retries_used == 0
