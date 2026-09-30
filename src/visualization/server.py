"""Local server for the web viewer: static page, run data and a small API.

Two mounts, because the two halves live in different places for good reason:
the page itself is source (tracked, one copy) while its data is a run artifact
(ignored, one copy per run tag).  Nothing is duplicated into ``runs/``.

    /                 -> src/visualization/  (index.html, web/, vendor/)
    /data/index.json  -> runs/<tag>/viewer_index.json
    /data/<part>/...  -> runs/<tag>/parts/<part>/...

Run data answers with a *weak* ETag naming the build it belongs to, and a
request may demand that build with ``If-Match``; see ``_if_match_satisfied``.

With ``GATHERSEED_VIEWER_DEBUG_DELAY`` set in the environment that starts the
server, any request may add ``?delayMs=N`` to arrive late - the fixture the
page's "an older answer must not land" guards are tested with.  Unset (the
default, and every production start) the parameter is ignored.

An HTTP origin is required: the page uses ES modules, ``fetch`` and a binary
mesh buffer, none of which work from ``file://``.

The API is a route table (method + path pattern -> handler).  Read endpoints
answer always; write endpoints run *store commands in a subprocess* through
the job queue and are switched off by ``--no-refresh-api`` (403):

    GET    /api/status            what is running, the run id on disk, who
                                  this server is, and a store summary
    GET    /api/parts             every part with its live freshness
    GET    /api/jobs              queue and history
    POST   /api/refresh           queue a sync (body: {"mode","only"})
    POST   /api/parts?name=x.step upload one STEP (raw body), queue its build
    DELETE /api/parts/<partId>    queue a removal (?source=1 also deletes the STEP)
    POST   /api/shutdown          stop this server

    GET    /api/similarity/configs   cost presets, their hashes, the weight specs
    GET    /api/similarity/rank      ?query=<id>[&config=..&weights=..&candidates=a,b&top=N&prefilter=N]
    GET    /api/similarity/compare   ?a=<id>&b=<id>[&config=..&weights=..]

The similarity endpoints are read-only and synchronous (a pair takes
milliseconds, a corpus ranking well under a second); they call the GED
package's ``SimilarityService`` and nothing else, and keep answering under
``--no-refresh-api``.

The write endpoints bind to localhost only, accept only values that match a
strict pattern, and require a custom header, which a cross-origin page cannot
set without a CORS preflight this server never answers.  ``/api/shutdown``
keeps working with ``--no-refresh-api``, because a server nobody can identify
or stop is exactly the one that ends up blocking the port.
"""

from __future__ import annotations

import http.server
import json
import os
import re
import socketserver
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from core.paths import make_part_id, resolve_input_path
from visualization import instances
from visualization.jobs import MAX_UPLOAD_BYTES, JobRunner, valid_part_id, valid_step_name

# A cross-origin page can send a simple POST, but not one carrying a custom
# header — that needs a preflight, and this server answers none.
REFRESH_HEADER = "X-Viewer-Refresh"

ASSET_ROOT = Path(__file__).resolve().parent

# What the ``/`` mount publishes: the pages themselves and the two directories
# they load from, and nothing else.  The mount used to be the whole package, so
# ``GET /server.py`` and ``GET /__pycache__/jobs.cpython-310.pyc`` answered 200
# (node 26 F-15).  Harmless while everything in here is the page's own source;
# a rule is what keeps it that way when it is not.
#
# A *rule* rather than a list of names, deliberately: the pages are whatever
# ``.html`` files sit in this directory (``index.html``, ``viewer.html``,
# ``compare.html`` today, and the next one somebody adds), and a list would
# have to be edited in step with them - the kind of edit that is forgotten
# until a page 404s.  Everything a page cannot be, a name cannot reach: the
# server's own modules, ``__pycache__``, the retired ``legacy/`` viewer.
ASSET_PAGE_SUFFIXES = (".html",)
ASSET_DIRS = ("web", "vendor")

# ``?delayMs=N`` makes one response arrive late, so a page's "an older answer
# must not land after a newer one" guards can be provoked on purpose instead
# of by luck (node 26 G-09, which was fixed with no way to test the timing).
#
# Off unless this environment variable is set when the server starts: a
# server that can be told to be slow by a query string is a server anyone can
# stall, and the production path must not gain even a branch it can be talked
# into.  With the switch off the parameter is ignored exactly like any other
# unknown one, which is the behaviour every existing test pins.
DEBUG_DELAY_ENV = "GATHERSEED_VIEWER_DEBUG_DELAY"
MAX_DEBUG_DELAY_MS = 5000

# The most a JSON request body may declare.  The API's own bodies are a
# handful of keys; the STEP upload endpoint reads raw bytes and has its own,
# much larger, limit (``jobs.MAX_UPLOAD_BYTES``).
MAX_JSON_BODY_BYTES = 64 * 1024

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".bin": "application/octet-stream",
    ".svg": "image/svg+xml",
}


# ------------------------------------------------------------------ request

class BodyTooLarge(ValueError):
    """The request declared more body than the endpoint will read."""


class Request:
    """What a handler may look at.  Built from the HTTP request, or by a test."""

    def __init__(self, method: str, path: str, *, query: dict | None = None, headers: dict | None = None,
                 body: bytes = b"", reader=None, content_length: int | None = None) -> None:
        self.method = method.upper()
        self.path = path
        self.query = dict(query or {})
        self.headers = {str(key).lower(): value for key, value in (headers or {}).items()}
        self._body = body
        self._reader = reader
        self.content_length = len(body) if content_length is None else int(content_length)

    def header(self, name: str, default=None):
        return self.headers.get(name.lower(), default)

    def read(self, size: int) -> bytes:
        if self._reader is not None:
            return self._reader(size)
        chunk, self._body = self._body[:size], self._body[size:]
        return chunk

    def json(self, max_bytes: int = MAX_JSON_BODY_BYTES) -> dict:
        """The JSON body, up to *max_bytes*.

        The declared length is checked *before* anything is read: a JSON
        endpoint here takes a few dozen bytes, and reading a declared 64 MiB
        into memory because a fetch passed the wrong object is not a thing
        this server should be able to be talked into (node 26 F-06).
        """
        if self.content_length <= 0:
            return {}
        if self.content_length > max_bytes:
            raise BodyTooLarge(f"body larger than {max_bytes} bytes")
        raw = self.read(self.content_length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError):
            raise ValueError("body is not valid JSON")
        if not isinstance(payload, dict):
            raise ValueError("body must be a JSON object")
        return payload


# ---------------------------------------------------------------------- api

class Api:
    """The handlers.  Each returns ``(status_code, json_payload)``."""

    def __init__(self, project_root: Path, output_tag: str, run_root: Path, runner: JobRunner | None,
                 presence: "_Presence", input_path: str = "inputs") -> None:
        self.project_root = Path(project_root).resolve()
        self.output_tag = output_tag
        self.run_root = Path(run_root)
        self.runner = runner
        self.presence = presence
        self.input_path = input_path
        self._similarity: dict[str, object] = {}
        self._similarity_lock = threading.Lock()
        self._rows: tuple[float, list[dict]] | None = None
        self._rows_lock = threading.Lock()
        self._run_id: tuple[float, str | None] | None = None
        self._run_id_lock = threading.Lock()

    # -- reads --------------------------------------------------------------

    def status(self, request: Request) -> tuple[int, dict]:
        state = (
            {"state": "disabled", "dataRunId": self._data_run_id(), "exitCode": None, "log": [],
             "current": None, "queue": [], "recent": []}
            if self.runner is None else self.runner.status()
        )
        # The identity travels with the status so a second terminal can name
        # the holder of the port instead of guessing.
        return 200, {**self.presence.identity, **state, "store": self._store_summary()}

    def parts(self, request: Request) -> tuple[int, dict]:
        return 200, {"parts": self._part_rows(), "inputPath": self.input_path, "outputTag": self.output_tag}

    def jobs(self, request: Request) -> tuple[int, dict]:
        if self.runner is None:
            return 200, {"current": None, "queue": [], "recent": [], "disabled": True}
        return 200, self.runner.jobs()

    # -- writes -------------------------------------------------------------

    def refresh(self, request: Request) -> tuple[int, dict]:
        try:
            body = request.json() if request.content_length else {}
            job = self.runner.submit("refresh", {"mode": body.get("mode") or "stale", "only": body.get("only") or []})
            self.forget_part_rows()
        except BodyTooLarge as error:
            return 413, {"error": str(error)}
        except ValueError as error:
            return 400, {"error": str(error)}
        return 202, {"started": True, "detail": "queued", "jobId": job.jobId, **self.runner.status()}

    def upload(self, request: Request) -> tuple[int, dict]:
        name = request.query.get("name")
        if not valid_step_name(name):
            return 400, {"error": "name must be a bare file name ending in .step/.stp (letters, digits, _ . -)"}
        length = request.content_length
        if length <= 0:
            return 400, {"error": "empty body"}
        if length > MAX_UPLOAD_BYTES:
            return 413, {"error": f"file larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"}
        input_dir = resolve_input_path(self.input_path, self.project_root)
        input_dir.mkdir(parents=True, exist_ok=True)
        target = input_dir / name
        overwrite = request.query.get("overwrite") in ("1", "true", "yes")
        if target.exists() and not overwrite:
            return 409, {"error": f"{name} already exists; pass overwrite=1 to replace it",
                         "partId": make_part_id(target, input_dir)}
        tmp = target.with_name(target.name + ".tmp")
        remaining = length
        try:
            with tmp.open("wb") as handle:
                while remaining > 0:
                    chunk = request.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    handle.write(chunk)
                    remaining -= len(chunk)
            if remaining > 0:
                tmp.unlink(missing_ok=True)
                return 400, {"error": "body shorter than Content-Length"}
            os.replace(tmp, target)
        except OSError as error:
            tmp.unlink(missing_ok=True)
            return 500, {"error": f"could not store the file: {error}"}
        part_id = make_part_id(target, input_dir)
        self.forget_part_rows()
        job = self.runner.submit("refresh", {"mode": "stale", "only": [part_id]})
        return 202, {"jobId": job.jobId, "partId": part_id, "file": str(target)}

    def remove(self, request: Request, part_id: str) -> tuple[int, dict]:
        part_id = unquote(part_id)
        if not valid_part_id(part_id):
            return 400, {"error": "invalid part id"}
        delete_source = request.query.get("source") in ("1", "true", "yes")
        self.forget_part_rows()
        job = self.runner.submit("remove", {"partId": part_id, "deleteSource": delete_source})
        return 202, {"jobId": job.jobId, "partId": part_id, "deleteSource": delete_source}

    def shutdown(self, request: Request) -> tuple[int, dict]:
        if request.header(instances.SHUTDOWN_HEADER) is None:
            return 403, {"error": f"missing {instances.SHUTDOWN_HEADER} header"}
        self.presence.request_stop("asked to stop by another process")
        return 202, {"stopping": True}

    # -- similarity (read-only, synchronous) --------------------------------

    def similarity(self, line: str | None = None):
        """The GED service over this run *for one line*, created on first use.

        One instance per line, because a line is a whole scale: its own
        corpus folders, its own representation and its own pair cache.  The
        page asks with ``?line=``; without one the answer is the open-source
        line, which is what every caller meant before node 21'.
        """
        from contract.registry import LINE_OCC

        key = line or LINE_OCC
        with self._similarity_lock:
            service = self._similarity.get(key)
            if service is None:
                from ged.retrieval.service import SimilarityService

                service = SimilarityService(self.run_root, line=key)
                self._similarity[key] = service
            return service

    def similarity_configs(self, request: Request) -> tuple[int, dict]:
        """Presets, weight specs — and, given ``config`` / ``weights`` /
        ``w_*`` parameters, the description of that configuration
        (``resolved: {name, hash, weights, preset}``) without computing a
        single pair: that is how the page shows the hash while a slider moves."""
        try:
            line = _line_param(request)
        except ValueError as error:
            return 400, {"error": str(error)}
        service = self.similarity(line)
        payload = service.configs()
        if any(key in request.query for key in ("config", "weights")) or any(
            key.upper() in presets_fields() for key in request.query
        ):
            try:
                config_name, weights = _cost_params(request)
                from ged.contract.representation import for_line
                from ged.cost.presets import resolve_config

                _, payload["resolved"] = resolve_config(
                    config_name, weights, representation=for_line(service.line))
            except (KeyError, ValueError) as error:
                return 400, {"error": str(error).strip("'\"")}
        return 200, payload

    def similarity_rank(self, request: Request) -> tuple[int, dict]:
        try:
            line = _line_param(request)
            query = _part_id_param(request, "query")
            config_name, weights = _cost_params(request)
            candidates = _id_list_param(request, "candidates")
            top = _count_param(request, "top")
            prefilter = _count_param(request, "prefilter")
        except ValueError as error:
            return 400, {"error": str(error)}
        return self._similarity_call(
            lambda service: service.rank(
                query, config_name=config_name, weights=weights, candidate_ids=candidates,
                top=top, prefilter_keep=prefilter,
            ),
            line,
        )

    def similarity_compare(self, request: Request) -> tuple[int, dict]:
        try:
            line = _line_param(request)
            part_a = _part_id_param(request, "a")
            part_b = _part_id_param(request, "b")
            config_name, weights = _cost_params(request)
        except ValueError as error:
            return 400, {"error": str(error)}
        return self._similarity_call(
            lambda service: service.compare(part_a, part_b, config_name=config_name, weights=weights),
            line,
        )

    def _similarity_call(self, call, line: str | None = None) -> tuple[int, dict]:
        from ged.retrieval.service import PartNotFound, ProjectionFailed

        try:
            return 200, call(self.similarity(line))
        except KeyError as error:  # unknown preset / weight key
            return 400, {"error": str(error).strip("'\"")}
        except ValueError as error:
            return 400, {"error": str(error)}
        except PartNotFound as error:
            return 404, {"error": str(error)}
        except ProjectionFailed as error:
            part_id = str(error).split(":", 1)[0]
            return 404, {"error": str(error), "corpus": {"failed": {part_id: str(error)}}}
        except Exception as error:  # noqa: BLE001 - the text, never the traceback
            return 500, {"error": f"{type(error).__name__}: {error}"}

    # -- helpers ------------------------------------------------------------

    def _data_run_id(self) -> str | None:
        try:
            with (self.run_root / "viewer_index.json").open(encoding="utf-8") as handle:
                return json.load(handle).get("runId")
        except (OSError, ValueError):
            return None

    #: How long the run id may be reused when stamping run data.  Every file
    #: of a part carries the same weak ETag, so a page opening one part asks
    #: for four of them in a burst; reading ``viewer_index.json`` once per
    #: request would put an open() in front of every artifact byte.  Same
    #: reasoning and same window as ``PART_ROWS_TTL`` (node 26 FP-5).
    #: ``/api/status`` deliberately does *not* use this: ``dataRunId`` is the
    #: field the page watches for drift, and it stays a live read.
    DATA_RUN_ID_TTL = 0.5

    def data_run_id(self) -> str | None:
        """The run id on disk, re-read at most every ``DATA_RUN_ID_TTL``."""
        now = time.monotonic()
        with self._run_id_lock:
            remembered = self._run_id
        if remembered is not None and now - remembered[0] < self.DATA_RUN_ID_TTL:
            return remembered[1]
        value = self._data_run_id()
        with self._run_id_lock:
            self._run_id = (now, value)
        return value

    def forget_data_run_id(self) -> None:
        """Drop the remembered run id: a build just replaced the run."""
        with self._run_id_lock:
            self._run_id = None

    #: How long a store scan may be reused.  The page polls ``/api/status``
    #: every two seconds and every poll used to walk the whole store twice
    #: (once for the summary, once if ``/api/parts`` followed) - and a scan is
    #: the most expensive read this server does (node 26 FP-5).  Short enough
    #: that a build finishing is seen within the same poll the page would have
    #: seen it in; the job state and ``dataRunId`` are read live either way.
    PART_ROWS_TTL = 0.5

    def _part_rows(self) -> list[dict]:
        from store.sync import store_status

        now = time.monotonic()
        with self._rows_lock:
            remembered = self._rows
        rows = None
        if remembered is not None and now - remembered[0] < self.PART_ROWS_TTL:
            rows = remembered[1]
        if rows is None:
            rows = store_status(self.project_root, self.output_tag, self.input_path)
            with self._rows_lock:
                self._rows = (now, rows)
        # The queue moves faster than the store does, so it is never reused.
        for row in rows:
            row["job"] = self.runner.jobs_for(row["partId"]) if self.runner is not None else None
        return rows

    def forget_part_rows(self) -> None:
        """Drop the remembered scan: something just changed the store."""
        with self._rows_lock:
            self._rows = None
        self.forget_data_run_id()

    def _store_summary(self) -> dict:
        try:
            rows = self._part_rows()
        except Exception as error:  # noqa: BLE001 - status must keep answering
            return {"error": f"{type(error).__name__}: {error}"}
        summary = {"parts": len(rows), "fresh": 0, "changed": 0, "failed": 0,
                   "sourceMissing": 0, "new": 0, "queued": 0, "rulesChanged": 0}
        for row in rows:
            state = row["light"]["state"]
            if state == "fresh":
                summary["fresh"] += 1
            elif state == "changed":
                summary["changed"] += 1
            elif state in ("source_missing", "untracked"):
                summary["sourceMissing"] += 1
            elif state == "new":
                summary["new"] += 1
            # A line that failed is the part's own problem; a line the
            # environment would not let us build is not, and is not counted
            # here - the page names it on the part instead.
            if any((line or {}).get("status") == "FAIL" for line in (row.get("lines") or {}).values()):
                summary["failed"] += 1
            if row.get("rulesChanged"):
                summary["rulesChanged"] += 1
            if row.get("job"):
                summary["queued"] += 1
        return summary


# ROUTES: (method, pattern, handler name, is_write).  A write needs the job
# runner (else 403) and the custom header (else 403); everything else is
# answered regardless of ``--no-refresh-api``.
ROUTES: list[tuple[str, re.Pattern, str, bool]] = [
    ("GET", re.compile(r"^/api/status/?$"), "status", False),
    ("GET", re.compile(r"^/api/parts/?$"), "parts", False),
    ("GET", re.compile(r"^/api/jobs/?$"), "jobs", False),
    ("POST", re.compile(r"^/api/refresh/?$"), "refresh", True),
    ("POST", re.compile(r"^/api/parts/?$"), "upload", True),
    ("DELETE", re.compile(r"^/api/parts/([^/]+)/?$"), "remove", True),
    ("POST", re.compile(r"^/api/shutdown/?$"), "shutdown", False),
    ("GET", re.compile(r"^/api/similarity/configs/?$"), "similarity_configs", False),
    ("GET", re.compile(r"^/api/similarity/rank/?$"), "similarity_rank", False),
    ("GET", re.compile(r"^/api/similarity/compare/?$"), "similarity_compare", False),
]


# ------------------------------------------------- similarity query parsing


def presets_fields() -> dict:
    from ged.cost.presets import WEIGHT_FIELDS

    return WEIGHT_FIELDS

#
# Every parameter is a part id (same pattern as the store endpoints), a
# preset name, a weight override or a small integer; anything else is a 400
# before the GED package is even asked.

def _line_param(request: Request) -> str | None:
    """``?line=occ|nx``.  Validated against the line registry, never a path."""
    from contract.registry import LINES

    value = request.query.get("line")
    if value is None or value.strip() == "":
        return None
    value = value.strip()
    if value not in LINES:
        raise ValueError(f"line: unknown kernel line {value!r}")
    return value


def _part_id_param(request: Request, name: str) -> str:
    value = request.query.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    if not valid_part_id(value):
        raise ValueError(f"{name}: invalid part id")
    return value


def _id_list_param(request: Request, name: str) -> list[str] | None:
    raw = request.query.get(name)
    if raw is None or raw.strip() == "":
        return None
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    for item in ids:
        if not valid_part_id(item):
            raise ValueError(f"{name}: invalid part id {item!r}")
    return ids


def _count_param(request: Request, name: str) -> int | None:
    """Non-negative integer; absent or 0 means "no limit"."""
    raw = request.query.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name}: expected an integer, got {raw!r}") from None
    if value < 0:
        raise ValueError(f"{name}: must be >= 0")
    return value or None


def _cost_params(request: Request) -> tuple[str | None, dict]:
    """``config=<preset>`` plus ``weights=<JSON | K=V,...>`` plus single
    ``w_sem=`` style overrides (which win over ``weights``)."""
    from ged.cost.presets import WEIGHT_FIELDS, parse_weights

    config_name = request.query.get("config") or None
    if config_name is not None and not re.match(r"^[A-Za-z0-9_.-]{1,64}$", config_name):
        raise ValueError("config: invalid preset name")
    try:
        weights = parse_weights(request.query.get("weights"))
        singles = {key: value for key, value in request.query.items() if key.upper() in WEIGHT_FIELDS}
        weights.update(parse_weights(",".join(f"{key}={value}" for key, value in singles.items())))
    except KeyError as error:
        raise ValueError(str(error).strip("'\"")) from None
    return config_name, weights


def dispatch(api: Api, request: Request) -> tuple[int, dict] | None:
    """Route one request; ``None`` means "not an API path"."""
    path = request.path
    for method, pattern, name, is_write in ROUTES:
        match = pattern.match(path)
        if not match:
            continue
        if method != request.method:
            continue
        if is_write:
            if api.runner is None:
                return 403, {"error": "refresh API disabled (server started with --no-refresh-api)"}
            if request.header(REFRESH_HEADER) is None:
                return 403, {"error": f"missing {REFRESH_HEADER} header"}
        handler = getattr(api, name)
        return handler(request, *match.groups())
    if path.startswith("/api/"):
        known = any(pattern.match(path) for _, pattern, _, _ in ROUTES)
        return (405, {"error": "method not allowed"}) if known else (404, {"error": "not found"})
    return None


# ------------------------------------------------------------------ handler

def debug_delay_seconds(query: dict, allowed: bool) -> float:
    """How long this request is asked to take, or 0.

    Anything that is not a non-negative number is 0, and the ceiling is
    ``MAX_DEBUG_DELAY_MS``: the switch exists to reorder two answers, not to
    hold a connection open.
    """
    if not allowed:
        return 0.0
    raw = query.get("delayMs")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value or value <= 0:  # NaN or nothing to wait for
        return 0.0
    return min(value, MAX_DEBUG_DELAY_MS) / 1000.0


def _make_handler(run_root: Path, api: Api, presence: "_Presence", allow_delay: bool = False):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802 - stdlib naming
            self._handle("GET")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

        def do_DELETE(self):  # noqa: N802
            self._handle("DELETE")

        def _handle(self, method: str) -> None:
            """Answer, whatever happens.

            The one thing a handler may never do is fall through with no
            response: ``socketserver`` then prints a traceback and closes the
            connection, and the client sees nothing at all (node 26 F-05).
            Only the exception's *type* goes down the wire - its message can
            carry an absolute path (F-11).
            """
            try:
                self._route(method)
            except Exception as error:  # noqa: BLE001 - a status code, never a traceback
                try:
                    self._json(500, {"error": type(error).__name__})
                except Exception:  # noqa: BLE001 - the socket is already gone
                    self.close_connection = True

        def _route(self, method: str) -> None:
            presence.touch()
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            query = {key: values[0] for key, values in parse_qs(parsed.query).items() if values}
            delay = debug_delay_seconds(query, allow_delay)
            if delay:
                time.sleep(delay)
            length = int(self.headers.get("Content-Length") or 0)
            consumed = {"n": 0}

            def reader(size: int) -> bytes:
                size = min(size, length - consumed["n"])
                if size <= 0:
                    return b""
                chunk = self.rfile.read(size)
                consumed["n"] += len(chunk)
                return chunk

            request = Request(method, path, query=query, headers=dict(self.headers.items()),
                              reader=reader, content_length=length)
            routed = dispatch(api, request)
            if routed is not None:
                code, payload = routed
                # A write that was refused still has to drain its body, or the
                # keep-alive connection is left mid-stream.  Above a sane size,
                # just close the connection instead.
                if consumed["n"] < length:
                    if length - consumed["n"] <= 1 << 20:
                        while consumed["n"] < length and reader(1 << 16):
                            pass
                    else:
                        self.close_connection = True
                self._json(code, payload)
                return

            if method != "GET":
                self.send_error(404, "not found")
                return
            path_is_asset = not path.startswith("/data/") and path != "/data/index.json"
            # Run data belongs to a build, and a page may insist on the build
            # it started reading.  The precondition is judged before the file
            # is looked for, the way HTTP judges preconditions: "not that
            # build" is a better answer than "no such file", and it is the
            # answer even when the file it asked for is gone because the new
            # build no longer has it.
            run_id = None if path_is_asset else api.data_run_id()
            if not path_is_asset:
                demanded = self.headers.get("If-Match")
                if demanded is not None and not _if_match_satisfied(demanded, run_id):
                    self._json(409, {
                        "error": "the run on disk is not the one this request asked for",
                        "runId": run_id,
                    })
                    return
            target = self._resolve(path)
            if target is None or not target.is_file():
                self.send_error(404, "not found")
                return
            payload = target.read_bytes()
            content_type = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
            # Assets (the page's own modules and vendor bundles) revalidate;
            # run data never caches at all.  See ``_asset_etag``.
            etag = _asset_etag(payload) if path_is_asset else None
            if etag is not None and self.headers.get("If-None-Match") == etag:
                self.send_response(304)
                self.send_header("ETag", etag)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            if etag is None:
                self.send_header("Cache-Control", "no-store")
                # Still no caching: the validator says which *build* answered,
                # not which bytes, so it is something to compare and to demand
                # back with ``If-Match`` - never something to store and reuse.
                if run_id is not None:
                    self.send_header("ETag", _data_etag(run_id))
            else:
                # ``no-cache`` is *revalidate every time*, not "do not cache".
                # Never a positive max-age: a changed module has to be seen on
                # a plain reload, and the viewer audit has been misled by a
                # stale module before (engineering-pitfalls, "Viewer audit").
                self.send_header("Cache-Control", "no-cache")
                self.send_header("ETag", etag)
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _resolve(self, path: str) -> Path | None:
            if path in ("/", ""):
                path = "/index.html"
            if path == "/data/index.json":
                return run_root / "viewer_index.json"
            if path.startswith("/data/"):
                return _safe_join(run_root / "parts", path[len("/data/"):])
            return _asset_path(path.lstrip("/"))

        def log_message(self, fmt, *args):  # only complain about misses
            if "404" in (fmt % args):
                super().log_message(fmt, *args)

    return Handler


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


class _Presence:
    """This server's identity, its last-request clock and its stop switch."""

    def __init__(self, identity: dict) -> None:
        self.identity = identity
        self._last = time.time()
        self._lock = threading.Lock()
        self._stop_reason: str | None = None
        self._server: _Server | None = None

    def bind(self, server: "_Server") -> None:
        self._server = server

    def touch(self) -> None:
        with self._lock:
            self._last = time.time()

    @property
    def idle_seconds(self) -> float:
        with self._lock:
            return time.time() - self._last

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def request_stop(self, reason: str) -> None:
        self._stop_reason = reason
        server = self._server
        if server is not None:
            threading.Thread(target=server.shutdown, daemon=True).start()


def _idle_watchdog(presence: _Presence, timeout: float) -> None:
    """Stop a server nobody is talking to.

    With the refresh API on, an open page polls ``/api/status``, so "no
    request for a while" really does mean "no page open".  This is why the
    timeout defaults on only for ``--no-browser``: if nobody asked us to open
    a browser, nobody is coming back to this server, and it should not be
    holding a port tomorrow morning.
    """
    while presence.stop_reason is None:
        remaining = timeout - presence.idle_seconds
        if remaining <= 0:
            presence.request_stop(f"idle for {timeout:.0f}s")
            return
        time.sleep(min(30.0, max(1.0, remaining)))


def _safe_join(root: Path, relative: str) -> Path | None:
    """Join under *root*, refusing anything that escapes it or cannot be a path.

    ``resolve()`` does not only raise ``ValueError`` for a path outside the
    root: on Windows it raises for a NUL byte in the name (node 26 F-05, which
    dropped the connection and printed a traceback) and ``OSError`` for a name
    the filesystem will not even look at.  All of them mean the same thing
    here - this is not a file under *root* - and all of them are answered the
    same way: no.
    """
    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return candidate


def _asset_etag(payload: bytes) -> str:
    """A strong ETag over the bytes actually being served.

    Content, not mtime: a checkout, a branch switch or an export restores
    files with new timestamps and identical contents, and the page should not
    have to re-download 2.4 MiB of vendor bundle for that (node 26 P-05).
    Paired with ``Cache-Control: no-cache``, which asks the browser to
    revalidate on *every* request - so a changed module is always fetched, and
    an unchanged one costs one 304 instead of 1.27 MB.
    """
    import hashlib

    return f'"{hashlib.sha256(payload).hexdigest()[:32]}"'


def _data_etag(run_id: str) -> str:
    """The validator run data answers with: the build, not the bytes.

    Weak on purpose.  Every file of one build shares this value, so it cannot
    mean "these bytes"; what it does mean is "this came out of run *x*", which
    is the only question a page assembling four files of one part has to ask
    (node 26 G-03, node 27F 6.2).
    """
    return f'W/"{run_id}"'


def _if_match_satisfied(header: str, run_id: str | None) -> bool:
    """Does an ``If-Match`` on run data name the build now on disk?

    Compared weakly - ``W/"r-3"`` and ``"r-3"`` are the same request here -
    because the validator is a build id and there is nothing strong about it.
    ``*`` means "whatever build you have", which is satisfied as soon as there
    is one.  No build on disk satisfies nothing: a page that demanded a run
    cannot be handed files from a run the server cannot name.

    This is a different conversation from the one static assets have.  There:
    strong ETag over the bytes, ``no-cache``, ``If-None-Match``, 304 - a
    cache revalidation.  Here: weak ETag over the build, ``no-store``,
    ``If-Match``, 409 - a torn-read guard.  Neither header appears on the
    other's path.
    """
    if run_id is None:
        return False
    wanted = {_data_etag(run_id), f'"{run_id}"'}
    for candidate in header.split(","):
        candidate = candidate.strip()
        if candidate == "*" or candidate in wanted:
            return True
    return False


def _asset_path(relative: str) -> Path | None:
    """The file behind a ``/`` request, or None if the page does not publish it.

    The mount is this directory's pages plus ``web/`` and ``vendor/`` - what
    the module docstring has always said it was.  It is not the package: this
    directory also holds the server's own source, its ``__pycache__`` and the
    retired ``legacy/`` viewer, and none of those is part of the page
    (node 26 F-15).
    """
    if not relative:
        return None
    head, separator, _rest = relative.replace("\\", "/").partition("/")
    if separator:
        return _safe_join(ASSET_ROOT, relative) if head in ASSET_DIRS else None
    return _safe_join(ASSET_ROOT, head) if head.endswith(ASSET_PAGE_SUFFIXES) else None


def serve(
    project_root: str | Path,
    output_tag: str = "latest",
    port: int = 8765,
    open_browser: bool = True,
    allow_refresh: bool = True,
    replace: bool = False,
    idle_timeout: float | None = None,
    input_path: str = "inputs",
) -> int:
    project_root = Path(project_root).resolve()
    run_root = (project_root / "runs" / output_tag).resolve()
    # A missing index is a cold start, not an error.  The page already copes:
    # it lists the STEP files found in the input directory as parts nobody has
    # built yet, and Add parts / Refresh are exactly what to do next.  Refusing
    # to serve here would kill the one launch that has to come from an icon
    # rather than a terminal — the first one, before any run exists.
    if not (run_root / "viewer_index.json").exists():
        todo = "add parts on the page, then Refresh" if allow_refresh else "run `run_pipeline.py refresh` first"
        print(f"[viewer] no finished run in {run_root} — starting empty ({todo})")

    if port == 0:
        port = instances.free_port()

    if replace:
        stopped, detail = instances.stop(port)
        if stopped:
            print(f"[viewer] stopped the viewer already on {port}")
        elif instances.probe(port) is not None:
            print(f"[viewer] could not take over port {port}: {detail}")
            return 1

    # No browser means nobody asked to look at this server, which is exactly
    # the case that leaves a port held by a process the user never sees.
    if idle_timeout is None:
        idle_timeout = 1800.0 if not open_browser else 0.0

    runner = JobRunner(project_root, output_tag, input_path) if allow_refresh else None
    presence = _Presence(instances.identity(port=port, output_tag=output_tag, run_root=run_root))
    api = Api(project_root, output_tag, run_root, runner, presence, input_path)

    # Threaded, because the handler speaks HTTP/1.1 and therefore keeps
    # connections alive: a single-threaded server would stall on the first
    # persistent connection while the page is still asking for the rest of its
    # modules.  Not SO_REUSEADDR, because on Windows that lets a second server
    # bind a port another one already holds while the older process keeps
    # answering — failing loudly beats serving somebody else's data.
    # Read once, at startup, from the environment - never from a request.
    allow_delay = bool(os.environ.get(DEBUG_DELAY_ENV))
    try:
        httpd = _Server(("127.0.0.1", port), _make_handler(run_root, api, presence, allow_delay))
    except OSError:
        # The bind still fails loudly, but now it says who is in the way.
        for line in instances.describe_conflict(port):
            print(line)
        return 1

    presence.bind(httpd)
    instances.write_lock(project_root, port=port, output_tag=output_tag, run_root=run_root)
    if idle_timeout and idle_timeout > 0:
        threading.Thread(target=_idle_watchdog, args=(presence, float(idle_timeout)), daemon=True).start()

    with httpd:
        url = f"http://127.0.0.1:{port}/index.html"
        print(f"[viewer] assets {ASSET_ROOT}")
        print(f"[viewer] data   {run_root}")
        print(f"[viewer] pid    {presence.identity['pid']}   port {port}   tag {output_tag}")
        print(f"[viewer] store actions from the page: {'on' if runner else 'off'}   input {input_path}")
        if idle_timeout and idle_timeout > 0:
            print(f"[viewer] idle timeout {idle_timeout:.0f}s (no browser opened; --idle-timeout 0 disables)")
        if allow_delay:
            print(f"[viewer] debug: ?delayMs=N honoured up to {MAX_DEBUG_DELAY_MS} ms ({DEBUG_DELAY_ENV} is set)")
        print(f"[viewer] open   {url}   (Ctrl+C to stop)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
            if presence.stop_reason:
                print(f"[viewer] stopped: {presence.stop_reason}")
        except KeyboardInterrupt:
            print("\n[viewer] stopped")
        finally:
            # Every way out of ``serve_forever`` lands here - Ctrl+C, the
            # idle watchdog, ``view --stop`` and the ``--replace`` takeover
            # all go through ``presence.request_stop``.  A job still running
            # is still writing into the run, so it does not get to outlive
            # the server that started it (node 26 F-07).
            if runner is not None and runner.running:
                print("[viewer] stopping the running job...")
                if not runner.stop():
                    print("[viewer] the job did not stop; it may still be writing to the run")
            instances.clear_lock(project_root, port)
    return 0
