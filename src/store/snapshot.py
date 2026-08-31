"""Freeze a run: copy ``runs/<from>`` to ``runs/<to>`` (records included).

A snapshot is an ordinary run directory afterwards - the GED, the viewer and
``store status`` read it like any other tag.  Its indexes are rebuilt once so
``outputTag`` names the new tag; the per-part files are copied verbatim.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from core.paths import run_paths
from store.index import rebuild_index
from store.sync import STAGING_DIR


def snapshot(project_root: str | Path, from_tag: str, to_tag: str) -> Path:
    root = Path(project_root).resolve()
    source = run_paths(root, from_tag)["run_root"]
    target = run_paths(root, to_tag)["run_root"]
    if not source.is_dir():
        raise FileNotFoundError(f"no run at {source}")
    if target.exists():
        raise FileExistsError(f"{target} already exists; pick another tag or remove it first")
    if from_tag == to_tag:
        raise ValueError("source and target tag are the same")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(STAGING_DIR, "*.tmp"))
    rebuild_index(root, to_tag)
    return target
