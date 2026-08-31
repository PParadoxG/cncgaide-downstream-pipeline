"""CPython driver for the NX line: one NX session, many parts.

Translates the requested STEPs (cached by content), then runs
``journal_fullnx.py`` once under ``run_journal.exe`` with every part in a
single job file - NX startup costs about a minute and is paid once per sync,
which is the whole reason this is batch shaped.

It writes artifacts and hands back per-part summaries; it does not touch
records or indexes.  Deciding what a part owes, publishing a line folder and
writing the record are the store's job (``store.sync``), and keeping
that split is what lets the NX line be planned, skipped and retried by exactly
the same machinery as the open-source one.

An unreachable licence comes back as ``blocked``: a named degradation about
the environment, never a verdict on a part.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from nxline.config import RUN_JOURNAL, looks_like_license_failure, nx_available
from nxline.translate import translate

_JOURNAL = Path(__file__).resolve().parent / "journal_fullnx.py"


def work_dir(project_root: str | Path, output_tag: str) -> Path:
    """Where translated ``.prt`` files and journal summaries live.

    Beside the parts rather than inside one, because a translation belongs to
    a source file rather than to a build of it: it survives a rebuild and is
    reused whenever the STEP's content has not moved.
    """
    return Path(project_root) / "runs" / output_tag / "nx"


def summary_path(project_root: str | Path, output_tag: str, part_id: str) -> Path:
    return work_dir(project_root, output_tag) / f"{part_id}.summary.json"


def run_nx_line(project_root: str | Path, output_tag: str, jobs: list[dict],
                *, timeout: float = 3600.0) -> dict:
    """Build the NX line for *jobs* in one session.

    *jobs* is ``[{"partId", "stepPath", "outDir"}]``; ``outDir`` is where the
    artifacts are written, which the store points at a staging directory so a
    half-finished part is never visible.

    Returns ``{"status": "ok" | "blocked" | "error" | "nothing_to_run",
    "detail", "parts": {partId: summary}}``, where each summary is either the
    journal's own (``status: success``) or a named failure.
    """
    root = Path(project_root).resolve()
    outcome: dict = {"status": "ok", "detail": "", "parts": {}}
    if not nx_available():
        outcome["status"] = "blocked"
        outcome["detail"] = "NX install not found"
        return outcome

    work = work_dir(root, output_tag)
    journal_jobs = []
    for job in jobs:
        part_id = job["partId"]
        step_path = Path(job["stepPath"])
        translated = translate(step_path, work, part_id)
        if translated["status"] in ("ok", "cached"):
            journal_jobs.append({
                "partId": part_id,
                "prt": translated["prt"],
                "outDir": str(job["outDir"]),
                "summary": str(summary_path(root, output_tag, part_id)),
                "sourceStep": job.get("sourceStep") or step_path.name,
            })
        else:
            outcome["parts"][part_id] = {
                "status": translated["status"], "error": translated["detail"]}

    if not journal_jobs:
        blocked = [state for state in outcome["parts"].values() if state["status"] == "blocked"]
        outcome["status"] = "blocked" if blocked else "nothing_to_run"
        if blocked:
            outcome["detail"] = blocked[0]["error"]
        return outcome

    jobfile = {"projectRoot": str(root), "outputTag": output_tag, "parts": journal_jobs}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(jobfile, handle)
        jobfile_path = handle.name
    try:
        # run_journal ignores PYTHONPATH; the journal wires sys.path itself
        # from its own location.  Only the job file travels by environment.
        environment = dict(os.environ, GATHERSEED_NX_JOB=jobfile_path)
        completed = subprocess.run(
            [str(RUN_JOURNAL), str(_JOURNAL)],
            capture_output=True, text=True, timeout=timeout, env=environment,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
    finally:
        try:
            os.unlink(jobfile_path)
        except OSError:
            pass

    outcome["journalTail"] = output.strip()[-600:]
    if looks_like_license_failure(output):
        outcome["status"] = "blocked"
        outcome["detail"] = output.strip()[-400:]
        return outcome

    for job in journal_jobs:
        part_id = job["partId"]
        path = Path(job["summary"])
        if not path.is_file():
            outcome["parts"][part_id] = {
                "status": "error",
                "error": "the journal wrote no summary; tail: " + output.strip()[-300:],
            }
            continue
        try:
            outcome["parts"][part_id] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            outcome["parts"][part_id] = {
                "status": "error", "error": f"unreadable summary: {error}"}
    return outcome
