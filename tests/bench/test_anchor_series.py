"""Session-anchor series: does it flag an anomalous session, and read real dirs?

The anchor exists because P0b saw one session sit ~3% off another while runs
inside a session agreed to ~0.5% (writeup §10a). Its whole value is raising a flag
at the time, so the threshold behaviour is worth pinning.
"""
import json

from bench.analysis.anchor_series import SUSPECT_PCT, anchor_runs, series_report


def _rows(*values):
    return [{"session": f"s{i}", "run_id": f"anchor-{i}", "valid": True, "req_s": v}
            for i, v in enumerate(values)]


def test_series_report_is_empty_before_any_anchor():
    assert series_report([]) == {"n": 0, "median": None, "sessions": [], "suspect": []}


def test_a_session_within_the_threshold_is_not_flagged():
    # ±0.5%-ish spread, i.e. ordinary within-session noise, must not raise a flag
    # or the check would cry wolf on every session.
    rep = series_report(_rows(13.80, 13.85, 13.83, 13.84))
    assert rep["suspect"] == []
    assert all(abs(s["dev_pct"]) < SUSPECT_PCT for s in rep["sessions"])


def test_a_three_percent_excursion_is_flagged():
    """The size actually observed between 2026-09-27 and 2026-09-28."""
    rep = series_report(_rows(13.80, 13.85, 13.83, 14.28))
    assert rep["suspect"] == ["s3"], rep["sessions"]


def test_the_reference_is_the_median_so_one_bad_session_does_not_move_it():
    """With a mean, a single large excursion would drag the reference toward
    itself and mask the next one."""
    good = series_report(_rows(13.80, 13.85, 13.83))["median"]
    with_outlier = series_report(_rows(13.80, 13.85, 13.83, 20.0))["median"]
    assert abs(with_outlier - good) < 0.05, (good, with_outlier)


def test_invalid_runs_are_excluded_from_the_reference():
    rows = _rows(13.80, 13.85)
    rows.append({"session": "bad", "run_id": "x", "valid": False, "req_s": 99.0})
    rep = series_report(rows)
    assert rep["n"] == 2
    assert rep["median"] < 14  # the invalid 99.0 did not enter the median
    assert "bad" in rep["suspect"]  # ...but it is still reported as off


def test_anchor_runs_reads_only_anchor_dirs_oldest_first(tmp_path):
    for name, req in (("anchor-20260101T0000Z", 13.8), ("anchor-20260102T0000Z", 13.9)):
        run = tmp_path / name / f"{name}-0000-r0"
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({
            "valid": True, "req_s": req, "ttft_p50_client": 55.0,
            "tpot_p50_client": 17.0, "e2e_p50_client": 266.0,
        }))
        (run / "env.json").write_text(json.dumps({"compile_cache_warm": True}))
    # A non-anchor sweep in the same results root must be ignored entirely.
    other = tmp_path / "p0b-variance" / "p0b-variance-0000-r0"
    other.mkdir(parents=True)
    (other / "summary.json").write_text(json.dumps({"valid": True, "req_s": 99.0}))

    rows = anchor_runs(tmp_path)
    assert [r["session"] for r in rows] == ["20260101T0000Z", "20260102T0000Z"]
    assert [r["req_s"] for r in rows] == [13.8, 13.9]
    assert rows[0]["compile_cache_warm"] is True


def test_anchor_runs_skips_a_run_with_no_summary(tmp_path):
    """A session interrupted before its anchor finished leaves a dir with no
    summary.json -- it must be skipped, not crash the series."""
    run = tmp_path / "anchor-20260103T0000Z" / "anchor-20260103T0000Z-0000-r0"
    run.mkdir(parents=True)
    assert anchor_runs(tmp_path) == []
