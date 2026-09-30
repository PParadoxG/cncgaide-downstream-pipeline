"""Where NX lives, and how to talk to it.

Where the install *is* comes from :mod:`nx_location`, which answers the same
question for ``store.stamps`` - environment variable, then
``config/nx_local.json``, then the machine this line was built on.  The
license server is only reachable over VPN and has no grace period: both CLIs
fail in ~7 seconds when it is not, which doubles as the liveness probe.

``NX_ROOT`` and the three paths under it are resolved **on every read**, not
frozen when this module is first imported (node 26E finding E-09).  A refresh
that runs after ``config/nx_local.json`` was written, or a process that sets
``GATHERSEED_NX_ROOT`` before asking, has to get the answer that is true now;
an import-time snapshot made the answer depend on which module happened to be
imported first.  They stay spelled as module attributes because that is how
every reader already asks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import nx_location

# name -> where it sits under the install root
_UNDER_ROOT = {
    "NX_ROOT": (),
    "STEP242": ("NXBIN", "step242.exe"),
    "RUN_JOURNAL": ("NXBIN", "run_journal.exe"),
    "CUSTOMIZATION_XML": ("UGII", "customization_base.xml"),
}

# The fast-fail messages both CLIs print when the license server is down.
_LICENSE_FAIL_MARKERS = (
    "failed to initialize UFUN",
    "Cannot connect to license server",
    "License Error",
)


def _path(name: str) -> Path:
    return nx_location.nx_root().joinpath(*_UNDER_ROOT[name])


def __getattr__(name: str) -> Path:
    """``NX_ROOT`` / ``STEP242`` / ``RUN_JOURNAL`` / ``CUSTOMIZATION_XML``.

    PEP 562: reading one of these re-asks :func:`nx_location.nx_root`, so the
    answer follows the environment instead of the import order.
    """
    if name not in _UNDER_ROOT:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return _path(name)


def nx_available() -> bool:
    """Whether the NX install is present at all (no license check)."""
    return all(_path(name).is_file() for name, parts in _UNDER_ROOT.items() if parts)


def looks_like_license_failure(output: str) -> bool:
    return any(marker.lower() in output.lower() for marker in _LICENSE_FAIL_MARKERS)


def probe_license(timeout: float = 60.0) -> tuple[bool, str]:
    """Run a minimal journal to check the license server is reachable.

    Returns ``(ok, detail)``.  ``run_journal`` fails in ~7 s without VPN, so
    a generous timeout still gives a quick verdict.  Any failure that does not
    look like a license failure is reported verbatim - it is a different
    problem and must not be filed as "blocked".
    """
    if not nx_available():
        return False, f"NX install not found under {_path('NX_ROOT')}"
    probe: Path | None = None
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("import NXOpen\nNXOpen.Session.GetSession()\nprint('LICENSE_OK')\n")
            probe = Path(handle.name)
        completed = subprocess.run(
            [str(_path("RUN_JOURNAL")), str(probe)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if "LICENSE_OK" in output:
            return True, "license server reachable"
        return False, output.strip()[-400:] or f"run_journal exited {completed.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"license probe timed out after {timeout}s"
    finally:
        if probe is not None:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass
