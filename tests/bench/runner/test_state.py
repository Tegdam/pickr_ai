import json

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
