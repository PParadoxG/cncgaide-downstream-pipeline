"""Where the NX install is on *this* machine, resolved in one place.

Two modules need the answer and they sit on opposite sides of the seam:
``nxline.config`` drives the two CLIs, and ``store.stamps`` hashes the feature
schema that lives under the same root.  ``store`` may not import a kernel line
(``tests/test_node10_seams.py``), so the shared answer cannot live in
``nxline``; the value used to be spelled out as a literal in both files, which
meant a machine with NX somewhere else had to be told twice.

**Why this module is not in a stamped layer.**  Every other shared module sits
inside one - ``contract`` above all, which by design "invalidates everything".
This one deliberately does not, and the reason is that it cannot change any
artifact.  Pointing the pipeline at a different NX install already invalidates
the NX line through the mechanism that is supposed to catch it: the schema file
itself is hashed into the ``nx`` stamp (``EXTERNAL_LAYER_FILES``), so a
different install means a different hash means a rebuild.  Stamping the
resolver on top of that would invalidate both lines' every part for an edit
that provably moves nothing - a stamp that lies about what it covers is worse
than one layer fewer.

**Resolution order**, highest first:

1. ``GATHERSEED_NX_ROOT`` - unchanged, still wins, still what CI and one-off
   runs use.
2. ``config/nx_local.json`` - the machine-local file, ignored by git.  This is
   the surface a deploying agent fills in: it finds NX, writes one path, and
   the answer survives the shell it was found in.  ``config/nx_local.example.json``
   is the tracked template.
3. The built-in default - the machine this line was built on.

A malformed or unreadable ``nx_local.json`` falls through to the default rather
than raising: this module is imported at start-up by code that has nothing to
do with NX, and a typo in an optional file must not take the open-source line
down with it.  The malformed file is reported by :func:`local_config_problem`
so the caller that *does* care can say so.

Standard library only, like ``contract`` - it is imported from inside the NX
journal process too, where nothing else of ours is on the path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Environment variable that overrides everything else.
ENV_VAR = "GATHERSEED_NX_ROOT"

#: Repository-relative path of the machine-local override file.
LOCAL_CONFIG = "config/nx_local.json"

#: The install this line was built against.  A fact about one machine, kept as
#: the last fallback so the repository still runs here with no configuration.
DEFAULT_NX_ROOT = r"E:\NX2506"

#: Key read from ``nx_local.json``.
_KEY = "nxRoot"

# ``src/nx_location.py`` -> ``src`` -> repository root.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_problem: str | None = None


def _from_local_config() -> str | None:
    """The path in ``config/nx_local.json``, or None if it cannot be used."""

    global _problem
    # Re-read means re-judge: a file that has since been repaired has no
    # problem, and reporting the old one sends a reader after a fault that is
    # no longer there.
    _problem = None
    path = _PROJECT_ROOT / LOCAL_CONFIG
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _problem = f"{LOCAL_CONFIG} could not be read: {error}"
        return None
    if not isinstance(payload, dict):
        _problem = f"{LOCAL_CONFIG} must contain a JSON object"
        return None
    value = payload.get(_KEY)
    if value is None:
        _problem = f"{LOCAL_CONFIG} has no {_KEY!r} key"
        return None
    if not isinstance(value, str) or not value.strip():
        _problem = f"{LOCAL_CONFIG}: {_KEY!r} must be a non-empty string"
        return None
    return value.strip()


def nx_root() -> Path:
    """The NX install root: environment, then local config, then default."""

    from_env = os.environ.get(ENV_VAR)
    if from_env and from_env.strip():
        return Path(from_env.strip())
    from_file = _from_local_config()
    if from_file is not None:
        return Path(from_file)
    return Path(DEFAULT_NX_ROOT)


def source() -> str:
    """Which of the three answered, for diagnostics and for the setup docs."""

    if (os.environ.get(ENV_VAR) or "").strip():
        return f"environment variable {ENV_VAR}"
    if _from_local_config() is not None:
        return LOCAL_CONFIG
    return "built-in default"


def local_config_problem() -> str | None:
    """Why ``nx_local.json`` was ignored, if it was there and unusable."""

    _from_local_config()
    return _problem
