"""One ``record.json`` per part: the store's only addition to the run layout.

The record sits at the part level, above the per-line folders, because a part
has exactly one source STEP however many lines were built from it.  It holds
the source identity once (path + fingerprint) and then one *line block* per
line built, each carrying what that line produced (manifest entry, viewer
entry, verdict counts) and what it was built under (its own layer stamps and
the STEP sha it saw).

Keeping the stamps inside the line block rather than at the top is what lets
the two lines be out of date independently: a change to ``aagnet`` moves the
open-source block and leaves the NX block alone, and neither says anything
about whether the source STEP changed.

A line block's ``status`` is ``PASS``, ``FAIL`` or ``NOT_BUILT``.  The last one
records a reason that was never about this part - an unreachable licence, a
missing vendor install - and is always retried; ``FAIL`` is the part's own
problem and is not retried until something changes.

Every file the store writes goes through ``atomic_write_*``: write a sibling
``.tmp`` then ``os.replace`` it, so a reader never sees a half-written file
and a crash leaves either the old or the new version, never a torn one.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from core.paths import part_dir, run_paths

RECORD_SCHEMA = "v8.part_record.v2"
NOT_BUILT = "NOT_BUILT"
RECORD_NAME = "record.json"


# ------------------------------------------------------------- atomic write

def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: str | Path, payload, *, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(payload, indent=indent, ensure_ascii=False))


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ records

def record_path(project_root: str | Path, output_tag: str, part_id: str) -> Path:
    """The record sits at the *part* level, above the per-line folders: one
    part has one record, whatever each line did."""
    return part_dir(project_root, output_tag, part_id) / RECORD_NAME


def load_record(path: str | Path) -> dict | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (isinstance(payload, dict) and payload.get("partId")):
        return None
    # A record written before the two-line layout has no lines and no source
    # identity we can trust; treat it as absent rather than half-reading it.
    return payload if isinstance(payload.get("lines"), dict) else None


def save_record(path: str | Path, record: dict) -> None:
    record.setdefault("schemaVersion", RECORD_SCHEMA)
    atomic_write_json(path, record, indent=1)


def iter_records(run_root: str | Path) -> list[dict]:
    """Every readable record under ``runs/<tag>/parts``, sorted by part id."""
    parts = Path(run_root) / "parts"
    if not parts.is_dir():
        return []
    records = []
    for child in parts.iterdir():
        if not child.is_dir():
            continue
        record = load_record(child / RECORD_NAME)
        if record is not None and record.get("partId") == child.name:
            records.append(record)
    # Same order the input scan uses (case-insensitive), so the manifest reads
    # like the one a whole-run refresh used to write.
    return sorted(records, key=lambda item: item["partId"].lower())


def untracked_part_dirs(run_root: str | Path) -> list[str]:
    """Part directories that carry no record (runs made before the store)."""
    parts = Path(run_root) / "parts"
    if not parts.is_dir():
        return []
    return sorted(
        child.name for child in parts.iterdir()
        if child.is_dir() and load_record(child / RECORD_NAME) is None
    )


def records_by_id(project_root: str | Path, output_tag: str) -> dict[str, dict]:
    run_root = run_paths(project_root, output_tag)["run_root"]
    return {record["partId"]: record for record in iter_records(run_root)}


def new_record(*, part_id: str, source_step: str, source_step_abs: str,
               fingerprint: dict) -> dict:
    """An empty record: the part's source identity, and no lines built yet."""
    return {
        "schemaVersion": RECORD_SCHEMA,
        "partId": part_id,
        "sourceStep": source_step,
        "sourceStepAbs": source_step_abs,
        "fingerprint": dict(fingerprint),
        "lines": {},
    }


def line_block(
    *,
    status: str,
    stamps: dict | None = None,
    fingerprint_sha: str | None = None,
    device: str | None = None,
    job_id: str | None = None,
    entry: dict | None = None,
    viewer_entry: dict | None = None,
    heuristic_summary: dict | None = None,
    heuristic_counts: dict | None = None,
    artifact_digest: str | None = None,
    error: str | None = None,
    traceback: str | None = None,
    reason: str | None = None,
    produced_at: str | None = None,
    built_at: str | None = None,
) -> dict:
    """What one line produced for one part, and what it was built under.

    ``NOT_BUILT`` blocks carry a *reason* and no stamps: there is nothing to be
    out of date with respect to, and the next sync retries unconditionally.
    """
    now = utc_now()
    block = {
        "status": status,
        "stamps": dict(stamps or {}),
        "fingerprint": fingerprint_sha,
        "device": device,
        "producedAt": produced_at or now,
        "builtAt": built_at or now,
        "jobId": job_id,
        "error": error,
        "traceback": traceback,
        "entry": entry,
        "viewerEntry": viewer_entry,
        "heuristicSummary": heuristic_summary or {},
        "heuristicCounts": heuristic_counts or {},
        "artifactDigest": artifact_digest,
    }
    if reason is not None:
        block["reason"] = reason
    return block


def not_built_block(reason: str, *, job_id: str | None = None) -> dict:
    """A line the environment would not let us build.  Never a failure."""
    return {
        "status": NOT_BUILT,
        "reason": reason,
        "checkedAt": utc_now(),
        "jobId": job_id,
    }


def set_line(record: dict, line: str, block: dict) -> dict:
    record.setdefault("lines", {})[line] = block
    return record


def get_line(record: dict | None, line: str) -> dict | None:
    block = ((record or {}).get("lines") or {}).get(line)
    return block if isinstance(block, dict) else None


def line_passed(record: dict | None, line: str) -> bool:
    block = get_line(record, line)
    return bool(block and block.get("status") == "PASS" and block.get("entry"))
