"""STEP -> .prt via NX's standalone translator.

``step242.exe`` is the only working batch route: the NXOpen
``Step242Importer.Commit()`` crashes with a memory access violation under
batch, so translation happens out of process, one CLI call per part.
Translation is cached by content: a ``.prt`` is reused when the recorded
source sha256 matches, so a re-run without source changes costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from nxline.config import STEP242, looks_like_license_failure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def translate(step_path: str | Path, work_dir: str | Path, part_id: str) -> dict:
    """Translate one STEP into ``<work_dir>/<part_id>.prt``.

    Returns ``{"status": "ok"|"cached"|"blocked"|"error", "prt": ..., "detail": ...}``.
    """
    step_path = Path(step_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    prt = work_dir / f"{part_id}.prt"
    marker = work_dir / f"{part_id}.source.json"

    source_hash = _sha256(step_path)
    if prt.is_file() and marker.is_file():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
            if recorded.get("sha256") == source_hash:
                return {"status": "cached", "prt": str(prt), "detail": "source unchanged"}
        except (OSError, ValueError):
            pass

    completed = subprocess.run(
        [str(STEP242), str(step_path), f"o={prt}"],
        capture_output=True, text=True, timeout=600,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if not prt.is_file():
        status = "blocked" if looks_like_license_failure(output) else "error"
        detail = output.strip()[-400:]
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
        json.dumps({"sha256": source_hash, "sourceStep": str(step_path)}), encoding="utf-8"
    )
    return {"status": "ok", "prt": str(prt), "detail": output.strip()[-200:]}
