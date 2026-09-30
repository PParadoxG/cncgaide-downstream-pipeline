"""STEP -> .prt via NX's standalone translator.

``step242.exe`` is the only working batch route: the NXOpen
``Step242Importer.Commit()`` crashes with a memory access violation under
batch, so translation happens out of process, one CLI call per part.
Translation is cached by content: a ``.prt`` is reused when the recorded
source sha256 matches, so a re-run without source changes costs nothing.

The translator writes a log of its own.  Left to itself it drops
``step242ug.log`` into whatever directory the pipeline happened to be started
from (node 26E finding E-12), so it is given both a working directory and a
name: the log lands beside the ``.prt`` it belongs to and a failure can point
at it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nxline import config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def translate(step_path: str | Path, work_dir: str | Path, part_id: str,
              *, source_label: str | None = None) -> dict:
    """Translate one STEP into ``<work_dir>/<part_id>.prt``.

    *source_label* is how the project names the source (a project-relative
    path).  It is recorded beside the content hash purely so a human can tell
    which file a cached ``.prt`` came from; nothing reads it back, which is
    exactly why it must not be this machine's absolute path (node 26E E-11,
    node 26P P-04).

    Returns ``{"status": "ok"|"cached"|"blocked"|"error", "prt": ..., "detail": ...}``.
    """
    step_path = Path(step_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    prt = work_dir / f"{part_id}.prt"
    marker = work_dir / f"{part_id}.source.json"
    log = work_dir / f"{part_id}.step242.log"

    source_hash = _sha256(step_path)
    if prt.is_file() and marker.is_file():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
            if recorded.get("sha256") == source_hash:
                return {"status": "cached", "prt": str(prt), "detail": "source unchanged"}
        except (OSError, ValueError):
            pass

    completed = subprocess.run(
        [str(config.STEP242), str(step_path.absolute()),
         f"o={prt.absolute()}", f"l={log.absolute()}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600, cwd=str(work_dir),
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if not prt.is_file():
        status = "blocked" if config.looks_like_license_failure(output) else "error"
        detail = output.strip()[-400:]
        if log.is_file():
            detail = f"{detail} (translator log: {log.name})"
        # Known pit: a stray default-named translation (`<stem>_step.prt`)
        # sitting next to the source makes step242 fail to create the PLM
        # part ("Failed to find/create NX OCC part").  Name the culprit so
        # the fix is obvious.
        stray = step_path.parent / f"{step_path.stem}_step.prt"
        if stray.is_file():
            detail = (
                f"stray translator artifact {stray.name} sits next to the source STEP and "
                f"blocks step242; move or delete it. Translator said: {detail}"
            )
        return {"status": status, "prt": str(prt), "detail": detail}
    marker.write_text(
        json.dumps({"sha256": source_hash,
                    "sourceStep": source_label or step_path.name}),
        encoding="utf-8",
    )
    return {"status": "ok", "prt": str(prt), "detail": output.strip()[-200:]}


#: What one part leaves in the translation cache.  The ``.prt`` is the cache
#: itself; the other two are scratch from the run that produced it.
_SCRATCH_SUFFIXES = (".summary.json", ".step242.log")


def sweep_cache(work_dir: str | Path, live_part_ids) -> dict:
    """Drop the run scratch of parts the store no longer holds, count the rest.

    ``runs/<tag>/nx/`` outlives every build by design - that is what makes a
    translation reusable - but nothing ever pruned it, so a corpus that has
    turned over N times keeps N generations (node 26E finding E-11).  Two
    different things live there and they are swept differently:

    * a ``.summary.json`` or a translator log is **scratch of one run**.  For a
      part the store no longer has, it answers nothing and is removed.
    * a ``.prt`` is a **content cache** worth about nine seconds.  Parts move in
      and out of ``inputs/`` all the time, so an orphan is counted and
      reported rather than deleted.

    *live_part_ids* is the set of parts the store currently holds, or ``None``
    when that is not known - in which case nothing is removed, because "I could
    not read the store" is not evidence that a file is an orphan.
    """
    work_dir = Path(work_dir)
    census = {"scratchRemoved": 0, "orphanTranslations": 0, "orphanBytes": 0}
    if live_part_ids is None or not work_dir.is_dir():
        return census
    live = set(live_part_ids)
    for child in sorted(work_dir.iterdir()):
        if not child.is_file():
            continue
        for suffix in _SCRATCH_SUFFIXES:
            if child.name.endswith(suffix):
                if child.name[: -len(suffix)] not in live:
                    try:
                        child.unlink()
                    except OSError:
                        continue
                    census["scratchRemoved"] += 1
                break
        else:
            if child.suffix == ".prt" and child.stem not in live:
                census["orphanTranslations"] += 1
                census["orphanBytes"] += child.stat().st_size
    return census
