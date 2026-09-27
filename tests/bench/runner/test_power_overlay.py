"""Power-mode overlay: reading it, setting it, and refusing a set that lied.

Every test patches `_ps`, the single point where this module shells out to
powershell.exe -- the conftest autouse fixture makes an unpatched call a hard
failure, so no test can silently depend on the host's real power mode.
"""
import pytest

import bench.runner.power_overlay as po

PERF = "ded574b5-45a0-4f42-8737-46345c09c238"
BAL = "00000000-0000-0000-0000-000000000000"


def test_read_returns_labelled_effective_and_actual(monkeypatch):
    # Real shape: "<rc_eff> <eff_guid> <rc_act> <act_guid>"
    monkeypatch.setattr(po, "_ps", lambda s, timeout=60: f"0 {PERF} 0 {PERF}")
    got = po.read()
    assert got == {
        "effective": f"{PERF} (performance)",
        "actual": f"{PERF} (performance)",
    }


def test_read_degrades_to_none_when_the_shell_out_fails(monkeypatch):
    """env capture must never abort over a host probe -- same contract as every
    other tool reached through the Windows filesystem."""
    def boom(script, timeout=60):
        raise FileNotFoundError("powershell.exe")

    monkeypatch.setattr(po, "_ps", boom)
    assert po.read() == {"effective": None, "actual": None}


def test_read_reports_none_for_a_nonzero_api_return_code(monkeypatch):
    monkeypatch.setattr(po, "_ps", lambda s, timeout=60: f"1 {PERF} 0 {PERF}")
    got = po.read()
    assert got["effective"] is None
    assert got["actual"] == f"{PERF} (performance)"


def test_apply_returns_the_effective_label_on_success(monkeypatch):
    seen = {}

    def fake(script, timeout=60):
        seen["script"] = script
        return f"0 {BAL}"

    monkeypatch.setattr(po, "_ps", fake)
    assert po.apply("balanced") == f"{BAL} (balanced)"
    # The GUID is substituted with str.replace, never str.format: the embedded
    # C# is full of braces that .format() would read as fields.
    assert BAL in seen["script"]
    assert "__GUID__" not in seen["script"]


def test_apply_rejects_an_unknown_name_without_shelling_out(monkeypatch):
    def refuse(script, timeout=60):
        raise AssertionError("must not shell out for an unknown name")

    monkeypatch.setattr(po, "_ps", refuse)
    with pytest.raises(po.PowerOverlayError, match="unknown power overlay"):
        po.apply("turbo")


def test_apply_raises_when_the_api_returns_nonzero(monkeypatch):
    monkeypatch.setattr(po, "_ps", lambda s, timeout=60: f"5 {PERF}")
    with pytest.raises(po.PowerOverlayError, match="returned 5"):
        po.apply("performance")


def test_apply_raises_when_the_set_did_not_take_effect(monkeypatch):
    """The read-back is the whole point: a set that silently does nothing would
    mislabel a whole arm of an A/B as the other arm."""
    monkeypatch.setattr(po, "_ps", lambda s, timeout=60: f"0 {PERF}")
    with pytest.raises(po.PowerOverlayError, match="did not take effect"):
        po.apply("balanced")


def test_apply_raises_when_the_shell_out_fails(monkeypatch):
    def boom(script, timeout=60):
        raise OSError("no powershell")

    monkeypatch.setattr(po, "_ps", boom)
    with pytest.raises(po.PowerOverlayError, match="failed to set power overlay"):
        po.apply("performance")


def test_label_is_case_and_brace_insensitive():
    assert po.label("{DED574B5-45A0-4F42-8737-46345C09C238}") == f"{PERF} (performance)"
    assert po.label("11111111-1111-1111-1111-111111111111").endswith("(unrecognised)")
    assert po.label(None) is None


def test_overlay_names_cover_the_three_slider_positions():
    assert set(po.OVERLAYS) == {"balanced", "efficiency", "performance"}
    assert po.OVERLAYS["performance"] == PERF
    assert po.OVERLAYS["balanced"] == BAL
