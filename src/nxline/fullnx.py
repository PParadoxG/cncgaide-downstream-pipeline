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

**Every summary this driver reads back has to belong to this run.**  Summaries
live in ``runs/<tag>/nx/`` beside the translation cache, which outlives any
one build on purpose, so "the file is there" never meant "the journal answered
me" (node 26E finding E-01).  Each run mints a token, writes it into the job
file, and refuses any summary that does not echo it; the batch's own summaries
are cleared before the session starts, so a crash mid-batch cannot leave a
previous answer standing in for a part that was never built.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

from nxline import config
from nxline.config import looks_like_license_failure, nx_available
from nxline.translate import sweep_cache, translate

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


def _stored_part_ids(project_root: str | Path, output_tag: str) -> set[str] | None:
    """The parts this store currently holds, or ``None`` if it cannot be read.

    Used only to tell the translation cache which of its files are orphans.
    Not knowing is answered with ``None`` rather than an empty set: "I could
    not look" must never be read as "nothing is live".
    """
    parts = Path(project_root) / "runs" / output_tag / "parts"
    if not parts.is_dir():
        return None
    try:
        return {child.name for child in parts.iterdir() if child.is_dir()}
    except OSError:
        return None


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
    run_token = uuid.uuid4().hex
    journal_jobs = []
    for job in jobs:
        part_id = job["partId"]
        step_path = Path(job["stepPath"])
        source_label = job.get("sourceStep") or step_path.name
        translated = translate(step_path, work, part_id, source_label=source_label)
        if translated["status"] == "blocked":
            # The licence, not this part.  Every remaining translation would
            # start its own translator, wait out the same connection failure
            # and report the same sentence: nine parts of that is a minute and
            # a half spent learning one fact.  Since node 28W this is also the
            # *only* place the store hears about an unreachable licence - the
            # probe it used to run before every build is gone (node 26
            # PERF-02) - so the answer has to be the whole batch's.
            outcome["status"] = "blocked"
            outcome["detail"] = translated["detail"]
            outcome["parts"][part_id] = {"status": "blocked", "error": translated["detail"]}
            return outcome
        if translated["status"] in ("ok", "cached"):
            journal_jobs.append({
                "partId": part_id,
                "prt": translated["prt"],
                "outDir": str(job["outDir"]),
                "summary": str(summary_path(root, output_tag, part_id)),
                "sourceStep": source_label,
                "runToken": run_token,
            })
        else:
            outcome["parts"][part_id] = {
                "status": translated["status"], "error": translated["detail"]}

    census = sweep_cache(work, _stored_part_ids(root, output_tag))
    if census["orphanTranslations"]:
        print(f"nx translation cache: {census['orphanTranslations']} translated part(s) "
              f"no longer in the store ({census['orphanBytes'] / 1024:.0f} KiB)")
    outcome["translationCache"] = census

    if not journal_jobs:
        blocked = [state for state in outcome["parts"].values() if state["status"] == "blocked"]
        outcome["status"] = "blocked" if blocked else "nothing_to_run"
        if blocked:
            outcome["detail"] = blocked[0]["error"]
        return outcome

    # Last run's answers are not this run's, and an answer that is merely
    # *present* is the failure mode E-01 describes.  Clearing them first means
    # a stale file cannot be read back even if the token check were removed.
    for job in journal_jobs:
        try:
            Path(job["summary"]).unlink(missing_ok=True)
        except OSError:
            pass

    jobfile = {"projectRoot": str(root), "outputTag": output_tag,
               "runToken": run_token, "parts": journal_jobs}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(jobfile, handle)
        jobfile_path = handle.name
    try:
        # run_journal ignores PYTHONPATH; the journal wires sys.path itself
        # from its own location.  Only the job file travels by environment.
        environment = dict(os.environ, GATHERSEED_NX_JOB=jobfile_path)
        completed = subprocess.run(
            [str(config.RUN_JOURNAL), str(_JOURNAL)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, env=environment,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
    except subprocess.TimeoutExpired:
        # A hung session is about this machine, not about any part, but it is
        # not the licence either: name it, give every part in the batch the
        # same named failure, and let the sync carry on and record them.
        detail = f"the NX session timed out after {timeout}s and was killed"
        outcome["status"] = "error"
        outcome["detail"] = detail
        for job in journal_jobs:
            outcome["parts"][job["partId"]] = {"status": "error", "error": detail}
        return outcome
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
        summary = None
        if path.is_file():
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                outcome["parts"][part_id] = {
                    "status": "error", "error": f"unreadable summary: {error}"}
                continue
        if not isinstance(summary, dict) or summary.get("runToken") != run_token:
            outcome["parts"][part_id] = {
                "status": "error",
                "error": "the journal wrote no summary for this run; tail: "
                         + output.strip()[-300:],
            }
            continue
        outcome["parts"][part_id] = summary
    return outcome
