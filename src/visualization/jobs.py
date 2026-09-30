"""Run store commands on behalf of the viewer page, one subprocess at a time.

Generalises the old single-refresh runner into a small FIFO queue.  The shape
of what a job may do is fixed here, not by the request: a job *kind* maps to
one ``run_pipeline.py`` command line, and the request can only contribute
values that pass a strict pattern (a part id, a refresh mode).  A job therefore
cannot be talked into running anything but ``refresh`` and ``store remove``
against this server's own tag and input directory.

Every job keeps the tail of its log and its exit code; the last twenty
finished jobs stay readable through ``/api/jobs``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

LOG_TAIL = 80
HISTORY = 20
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
KINDS = ("refresh", "remove")
MODES = ("stale", "all")

PART_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
STEP_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.(step|stp)$", re.IGNORECASE)

JOB_ID_ENV = "GATHERSEED_JOB_ID"


def valid_part_id(value) -> bool:
    """Same alphabet ``make_part_id`` emits; never a path step."""
    return isinstance(value, str) and bool(PART_ID_RE.match(value)) and value not in (".", "..")


def valid_step_name(value) -> bool:
    """A bare file name ending in .step/.stp, with a real stem."""
    if not isinstance(value, str) or not STEP_NAME_RE.match(value) or value.startswith("."):
        return False
    stem = value.rsplit(".", 1)[0]
    return any(char.isalnum() for char in stem)


@dataclass
class Job:
    jobId: str
    kind: str
    args: dict
    state: str = "queued"            # queued | running | done | failed
    exitCode: int | None = None
    createdAt: str = ""
    startedAt: str | None = None
    finishedAt: str | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=LOG_TAIL))

    def to_dict(self) -> dict:
        return {
            "jobId": self.jobId,
            "kind": self.kind,
            "args": dict(self.args),
            "state": self.state,
            "exitCode": self.exitCode,
            "createdAt": self.createdAt,
            "startedAt": self.startedAt,
            "finishedAt": self.finishedAt,
            "log": list(self.log),
        }

    def touches(self, part_id: str) -> bool:
        """Does this job (re)build or remove *part_id*?"""
        if self.kind == "remove":
            return self.args.get("partId") == part_id
        only = self.args.get("only")
        return not only or part_id in only


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class JobRunner:
    """FIFO queue of white-listed subprocesses; one runs at a time."""

    def __init__(
        self,
        project_root: Path,
        output_tag: str,
        input_path: str = "inputs",
        *,
        popen=subprocess.Popen,
        python: str | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve()
        self._output_tag = output_tag
        self._input_path = input_path
        self._popen = popen
        self._python = python or sys.executable
        self._lock = threading.Lock()
        self._queue: deque[Job] = deque()
        self._current: Job | None = None
        self._history: deque[Job] = deque(maxlen=HISTORY)
        self._last_finished: Job | None = None
        self._counter = 0
        self._thread: threading.Thread | None = None
        self._process = None
        self._stopping = False

    # -- public properties -------------------------------------------------

    @property
    def input_path(self) -> str:
        return self._input_path

    @property
    def output_tag(self) -> str:
        return self._output_tag

    @property
    def running(self) -> bool:
        with self._lock:
            return self._current is not None

    # -- validation and argv ----------------------------------------------

    def validate(self, kind: str, args: dict) -> dict:
        """Return the cleaned argument dict or raise ``ValueError``."""
        if kind not in KINDS:
            raise ValueError(f"unknown job kind {kind!r}")
        args = dict(args or {})
        if kind == "refresh":
            mode = args.get("mode") or "stale"
            if mode not in MODES:
                raise ValueError(f"mode must be one of {MODES}")
            only = args.get("only") or []
            if not isinstance(only, list) or not all(valid_part_id(item) for item in only):
                raise ValueError("only must be a list of part ids")
            return {"mode": mode, "only": list(only)}
        part_id = args.get("partId")
        if not valid_part_id(part_id):
            raise ValueError("partId is not a valid part id")
        return {"partId": part_id, "deleteSource": bool(args.get("deleteSource"))}

    def argv(self, job: Job) -> list[str]:
        base = [self._python, str(self._project_root / "run_pipeline.py")]
        if job.kind == "refresh":
            command = base + [
                "refresh",
                "--input", self._input_path,
                "--output-tag", self._output_tag,
                "--device", "auto",
            ]
            if job.args["mode"] == "all":
                command.append("--all")
            if job.args["only"]:
                command += ["--only", *job.args["only"]]
            return command
        command = base + ["store", "remove", job.args["partId"], "--output-tag", self._output_tag]
        if job.args["deleteSource"]:
            command.append("--delete-source")
        return command

    # -- queue --------------------------------------------------------------

    def submit(self, kind: str, args: dict | None = None) -> Job:
        clean = self.validate(kind, args or {})
        with self._lock:
            self._counter += 1
            job = Job(jobId=f"job_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{self._counter}",
                      kind=kind, args=clean, createdAt=_now())
            self._queue.append(job)
        self._start_next()
        return job

    def _start_next(self) -> None:
        with self._lock:
            if self._stopping or self._current is not None or not self._queue:
                return
            job = self._queue.popleft()
            self._current = job
            job.state = "running"
            job.startedAt = _now()
            env = dict(os.environ)
            env[JOB_ID_ENV] = job.jobId
            # Both ends of the pipe are pinned to UTF-8.  Left to the locale
            # (cp936 on this machine) a child that prints one character the
            # code page cannot carry raises inside ``_drain``, and a build
            # that succeeded is reported as failed (node 26 F-17).
            env["PYTHONIOENCODING"] = "utf-8"
            try:
                process = self._popen(
                    self.argv(job),
                    cwd=str(self._project_root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )
            except Exception as error:  # noqa: BLE001 - a missing interpreter must not hang the queue
                job.log.append(f"could not start: {type(error).__name__}: {error}")
                job.state = "failed"
                job.exitCode = -1
                job.finishedAt = _now()
                self._current = None
                self._history.append(job)
                self._last_finished = job
                process = None
            self._process = process
        if process is None:
            self._start_next()
            return
        self._thread = threading.Thread(target=self._drain, args=(job, process), daemon=True)
        self._thread.start()

    def _drain(self, job: Job, process) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        with self._lock:
                            job.log.append(line)
            code = process.wait()
        except Exception as error:  # noqa: BLE001
            with self._lock:
                job.log.append(f"runner error: {type(error).__name__}: {error}")
            code = -1
        with self._lock:
            job.exitCode = code
            job.state = "done" if code == 0 else "failed"
            job.finishedAt = _now()
            self._history.append(job)
            self._last_finished = job
            self._current = None
            self._process = None
        self._start_next()

    def stop(self, timeout: float = 10.0) -> bool:
        """Take no more jobs and end the one that is running.

        A job outlives its runner otherwise: ``_drain`` is a daemon thread and
        dies with the process, but the ``run_pipeline.py refresh`` subprocess
        does not - it keeps writing into ``runs/<tag>/`` with nobody left to
        report it, which is exactly what ``view --replace`` does to a server
        that was mid-refresh (node 26 F-07).

        Returns True when nothing of this runner's is still running.  Queued
        jobs are dropped and marked, because the queue does not survive the
        server that owns it.
        """
        with self._lock:
            self._stopping = True
            process = self._process
            dropped = list(self._queue)
            self._queue.clear()
        for job in dropped:
            job.state = "failed"
            job.exitCode = -1
            job.finishedAt = _now()
            job.log.append("dropped: the server was stopped before this job started")
            with self._lock:
                self._history.append(job)
                self._last_finished = job
        if process is None:
            return True
        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=timeout)
        except Exception:  # noqa: BLE001 - already gone, or never really started
            return process.poll() is not None
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        return process.poll() is not None

    def wait(self, timeout: float = 30.0) -> bool:
        """Block until the queue drains (tests and shutdown); False on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._current is None and not self._queue:
                    return True
            time.sleep(0.02)
        return False

    # -- reporting ----------------------------------------------------------

    def jobs(self) -> dict:
        with self._lock:
            return {
                "current": self._current.to_dict() if self._current else None,
                "queue": [job.to_dict() for job in self._queue],
                "recent": [job.to_dict() for job in reversed(self._history)],
            }

    def jobs_for(self, part_id: str) -> dict | None:
        """The queued/running job that will touch *part_id*, if any."""
        with self._lock:
            candidates = ([self._current] if self._current is not None else []) + list(self._queue)
            for job in candidates:
                if job.touches(part_id):
                    return {"jobId": job.jobId, "state": job.state, "kind": job.kind, "args": dict(job.args)}
        return None

    def status(self) -> dict:
        """Compatible with the old single-runner shape, plus the queue."""
        with self._lock:
            current = self._current
            last = self._last_finished
            running = current is not None
            if running:
                log = list(current.log)
            elif last is not None:
                log = list(last.log)
            else:
                log = []
            exit_code = None if running else (last.exitCode if last is not None else None)
            payload = {
                "state": "running" if running else "idle",
                "exitCode": exit_code,
                "log": log,
                "current": current.to_dict() if current else None,
                "queue": [job.to_dict() for job in self._queue],
                "recent": [job.to_dict() for job in reversed(self._history)],
            }
        payload["dataRunId"] = self.data_run_id()
        return payload

    def data_run_id(self) -> str | None:
        """Run id currently on disk — the page compares it with its own.

        Read from disk rather than remembered, so a refresh run from a terminal
        is noticed just the same.
        """
        index = self._project_root / "runs" / self._output_tag / "viewer_index.json"
        try:
            with index.open(encoding="utf-8") as handle:
                return json.load(handle).get("runId")
        except (OSError, ValueError):
            return None  # missing or half-written: mid-refresh, not a change
