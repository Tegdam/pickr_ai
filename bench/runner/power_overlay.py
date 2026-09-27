"""Windows 11 power-mode overlay: read what is in force, and set it per run.

Why this exists. The Windows power-mode slider ("Best power efficiency" /
"Balanced" / "Best performance") is an **overlay** on top of the active power
scheme, not a scheme of its own. Two consequences bit us in P0b:

1. `powercfg /getactivescheme` reports the same scheme (Balanced) whichever way
   the slider is set, so env.json could not distinguish two runs under
   different power modes (see env_capture).
2. `powercfg` on this Windows build has **no overlay verb at all** --
   `/overlaylist` is unsupported and `/overlaysetactiveschemeoverlay` returns
   "Invalid Parameters", with `/?` documenting nothing -- so the mode cannot be
   changed through powercfg either.

The slider itself calls `powrprof.dll`, so we call the same three entry points
through PowerShell P/Invoke. All three return ERROR_SUCCESS unelevated, which
is what makes an unattended interleaved A/B on power mode possible: the runner
flips the mode between runs itself, instead of a human flipping a slider.

`PowerGetEffectiveOverlayScheme` is the field analysis should trust. "Actual"
is what was last requested; "effective" is what the system is applying right
now, which can differ -- notably because AC and DC are separate settings inside
one scheme, so a mains blip changes the effective overlay without changing
anything we requested.
"""
from __future__ import annotations

import subprocess

# The three GUIDs the slider writes. All-zero means "no overlay", i.e. the
# scheme's own settings apply, which is what Windows reports for Balanced.
BALANCED = "00000000-0000-0000-0000-000000000000"
EFFICIENCY = "961cc777-2547-4f9d-8174-7d86181b8a7a"
PERFORMANCE = "ded574b5-45a0-4f42-8737-46345c09c238"

# Sweep-facing names -> GUID. `balanced` is the all-zero no-overlay state.
OVERLAYS = {
    "balanced": BALANCED,
    "efficiency": EFFICIENCY,
    "performance": PERFORMANCE,
}
NAMES_BY_GUID = {v: k for k, v in OVERLAYS.items()}

_POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"

# Compiled fresh per invocation (~1 s). That is once per run, against a run
# measured in minutes, so it is not worth caching into a background process.
_PS_PREAMBLE = """
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class PowerOverlay {
  [DllImport("powrprof.dll", SetLastError = true)]
  public static extern uint PowerSetActiveOverlayScheme(Guid OverlaySchemeGuid);
  [DllImport("powrprof.dll", SetLastError = true)]
  public static extern uint PowerGetEffectiveOverlayScheme(out Guid EffectiveOverlayGuid);
  [DllImport("powrprof.dll", SetLastError = true)]
  public static extern uint PowerGetActualOverlayScheme(out Guid ActualOverlayGuid);
}
'@
"""

_PS_READ = _PS_PREAMBLE + """
$eff = [Guid]::Empty; $act = [Guid]::Empty
$re = [PowerOverlay]::PowerGetEffectiveOverlayScheme([ref]$eff)
$ra = [PowerOverlay]::PowerGetActualOverlayScheme([ref]$act)
Write-Output "$re $eff $ra $act"
"""

# NB: substituted with str.replace, not str.format -- the embedded C# is full
# of braces, which .format() reads as fields.
_PS_SET = _PS_PREAMBLE + """
$rc = [PowerOverlay]::PowerSetActiveOverlayScheme([Guid]'__GUID__')
$eff = [Guid]::Empty
[void][PowerOverlay]::PowerGetEffectiveOverlayScheme([ref]$eff)
Write-Output "$rc $eff"
"""


class PowerOverlayError(RuntimeError):
    """Raised when the overlay cannot be read or did not take effect. Callers
    in the run path turn this into a PreflightError: a run under the wrong
    power mode is a silently wrong measurement, not a slow one."""


def _ps(script: str, timeout: int = 60) -> str:
    return subprocess.check_output(
        [_POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", script],
        text=True, timeout=timeout, stderr=subprocess.STDOUT,
    ).strip()


def label(guid: str | None) -> str | None:
    """`<guid> (<name>)`, so a reader never has to look a GUID up."""
    if guid is None:
        return None
    g = guid.strip("{}").lower()
    return f"{g} ({NAMES_BY_GUID.get(g, 'unrecognised')})"


def read() -> dict:
    """`{effective, actual}` as labelled GUIDs, or both None if unreadable.

    Never raises: env capture must degrade to None like every other host
    probe. Use `require_effective()` when a caller needs the value to exist.
    """
    try:
        out = _ps(_PS_READ).split()
        rc_eff, eff, rc_act, act = out[0], out[1], out[2], out[3]
    except Exception:
        return {"effective": None, "actual": None}
    return {
        "effective": label(eff) if rc_eff == "0" else None,
        "actual": label(act) if rc_act == "0" else None,
    }


def apply(name: str) -> str:
    """Set the overlay to `name` and return the labelled effective GUID.

    Raises PowerOverlayError if the name is unknown, the call fails, or the
    effective overlay afterwards is not what we asked for -- the read-back is
    the point, since a set that silently does nothing would produce a whole
    arm of runs mislabelled as the other arm.
    """
    if name not in OVERLAYS:
        raise PowerOverlayError(f"unknown power overlay {name!r}; expected one of {sorted(OVERLAYS)}")
    want = OVERLAYS[name]
    try:
        out = _ps(_PS_SET.replace("__GUID__", want)).split()
        rc, eff = out[0], out[1]
    except Exception as e:
        raise PowerOverlayError(f"failed to set power overlay to {name!r}: {e}") from e
    if rc != "0":
        raise PowerOverlayError(f"PowerSetActiveOverlayScheme({name}) returned {rc}, expected 0")
    got = eff.strip("{}").lower()
    if got != want:
        raise PowerOverlayError(
            f"power overlay did not take effect: asked for {name} ({want}), "
            f"effective is {label(got)}"
        )
    return label(got)
