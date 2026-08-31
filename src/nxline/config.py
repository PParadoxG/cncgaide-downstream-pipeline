"""Where NX lives, and how to talk to it.

Where the install *is* comes from :mod:`nx_location`, which answers the same
question for ``store.stamps`` - environment variable, then
``config/nx_local.json``, then the machine this line was built on.  The
license server is only reachable over VPN and has no grace period: both CLIs
fail in ~7 seconds when it is not, which doubles as the liveness probe.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import nx_location

NX_ROOT = nx_location.nx_root()
STEP242 = NX_ROOT / "NXBIN" / "step242.exe"
RUN_JOURNAL = NX_ROOT / "NXBIN" / "run_journal.exe"
CUSTOMIZATION_XML = NX_ROOT / "UGII" / "customization_base.xml"

# The fast-fail messages both CLIs print when the license server is down.
_LICENSE_FAIL_MARKERS = (
    "failed to initialize UFUN",
    "Cannot connect to license server",
    "License Error",
)


def nx_available() -> bool:
    """Whether the NX install is present at all (no license check)."""
    return STEP242.is_file() and RUN_JOURNAL.is_file() and CUSTOMIZATION_XML.is_file()


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
        return False, f"NX install not found under {NX_ROOT}"
    probe = NX_ROOT.parent / "_gatherseed_license_probe.py"
    try:
        import tempfile

        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("import NXOpen\nNXOpen.Session.GetSession()\nprint('LICENSE_OK')\n")
            probe = Path(handle.name)
        completed = subprocess.run(
            [str(RUN_JOURNAL), str(probe)],
            capture_output=True, text=True, timeout=timeout,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if "LICENSE_OK" in output:
            return True, "license server reachable"
        return False, output.strip()[-400:] or f"run_journal exited {completed.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"license probe timed out after {timeout}s"
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
