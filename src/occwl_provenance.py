"""Which occwl produced a piece of evidence, read from installer metadata.

occwl is the one dependency of the open-source line that is neither pinned by
a file in this repository nor reported by anything the pipeline writes: node
27X took it out of the tree and made it an install (``install/environment.yml``
pins a commit), which is right for licensing and left a gap - a machine with a
different occwl produces different AAGNet graphs, and nothing said so.  This
module closes it from both sides at once:

* ``occline.aagnet.per_face`` writes the answer into ``aagnet_evidence.json``,
  so a stored artifact names the occwl it came from;
* ``store.stamps`` hashes it into the ``aagnet`` layer stamp, so swapping occwl
  rebuilds every part of that line - the same thing the external NX schema file
  does for the ``nx`` layer.

Like ``nx_location``, this sits at the top of ``src`` rather than inside either
side: ``store`` may not import a kernel line, and the two callers are on
opposite sides of that rule.  **Unlike** ``nx_location`` it is listed in a layer
(``LAYERS["aagnet"]``), because editing it *does* change an artifact - it
decides three keys of the evidence file - and a stamp that does not cover what
it decides is a stamp that lies.

The package itself is never imported.  ``importlib.metadata`` reads the
installer's own records off disk, so asking which occwl is installed costs no
OCC import and works in a process that will never use it.
"""

from __future__ import annotations

import json
from importlib import metadata
from typing import Any

DISTRIBUTION = "occwl"

#: The vocabulary ``occwl_source`` may use.  A fixed set of categories, never
#: the URL or the directory: ``aagnet_evidence.json`` is hashed into the
#: artifact digest, and a machine-local path in there reports "the result
#: changed" when only the checkout moved (node 15's lesson, node 27 B-14's).
SOURCES = (
    "not_installed",                       # no such distribution
    "vcs_checkout",                        # PEP 610 vcs_info: a commit is known
    "editable_directory",                  # editable install, no vcs_info
    "local_directory",                     # installed from a directory
    "direct_url_without_vcs_info",         # a direct URL that named no commit
    "no_direct_url_metadata",              # an ordinary wheel / sdist install
    "unreadable_direct_url_metadata",      # the file is there and is not JSON
)


def occwl_provenance() -> dict[str, Any]:
    """``{occwl_version, occwl_commit, occwl_source}``, guesses excluded.

    ``occwl_commit`` is the commit *the installer recorded* (PEP 610
    ``direct_url.json``), and is ``None`` whenever the installer recorded
    none - an editable install from a working tree, most of all, which is what
    this repository's own environment has.  ``occwl_source`` says which of
    those cases it was, so a ``None`` commit is a named gap rather than a
    silence.  Upstream's version string ("3.0.0") has covered several different
    trees, so it is provenance, not identity; that is precisely why the commit
    is asked for separately instead of being inferred from it.
    """
    try:
        version = metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return {"occwl_version": None, "occwl_commit": None,
                "occwl_source": "not_installed"}
    except Exception:
        return {"occwl_version": None, "occwl_commit": None,
                "occwl_source": "not_installed"}

    raw: str | None = None
    try:
        raw = metadata.distribution(DISTRIBUTION).read_text("direct_url.json")
    except Exception:
        raw = None
    if not raw:
        return {"occwl_version": version, "occwl_commit": None,
                "occwl_source": "no_direct_url_metadata"}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {"occwl_version": version, "occwl_commit": None,
                "occwl_source": "unreadable_direct_url_metadata"}
    if not isinstance(payload, dict):
        return {"occwl_version": version, "occwl_commit": None,
                "occwl_source": "unreadable_direct_url_metadata"}

    commit = (payload.get("vcs_info") or {}).get("commit_id")
    if isinstance(commit, str) and commit.strip():
        return {"occwl_version": version, "occwl_commit": commit.strip(),
                "occwl_source": "vcs_checkout"}
    directory = payload.get("dir_info")
    if isinstance(directory, dict):
        return {"occwl_version": version, "occwl_commit": None,
                "occwl_source": "editable_directory" if directory.get("editable")
                else "local_directory"}
    return {"occwl_version": version, "occwl_commit": None,
            "occwl_source": "direct_url_without_vcs_info"}


def stamp_value() -> str:
    """The same answer as one canonical string, for a layer stamp."""
    return json.dumps(occwl_provenance(), sort_keys=True, separators=(",", ":"))
