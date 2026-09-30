"""Pair result cache keyed by the two parts' versions, the cost configuration
and the semantics of the code that priced them.

One entry = one ``GedResult`` (with its edit path) for an unordered pair of
parts under one configuration hash.  The key carries both parts'
``producedAt`` stamps, so a part rebuilt by the store invalidates every pair
it is in without anybody having to clean up: a stale entry simply never
matches again (lazy invalidation).  It carries the semantics stamp
(``ged.semantics``) for the same reason: the configuration hash is only the
eight weights, so a representation or cost-table change would otherwise be
served from entries computed before it.  Both live in the payload rather
than in the path, so a superseded entry is overwritten in place instead of
leaving an orphaned directory behind.  Preset configurations are persisted
under ``runs/<tag>/ged/cache/<hash>/<idA>__<idB>.json`` (generated artifact,
not hand-edited); custom weight sets live in memory only.

Entries are stored in *sorted* orientation (``graph_id_a`` is the smaller
id); ``orient`` flips a payload to the orientation a caller asked for.  The
flip is exact: the solver computes both directions and averages them, so
``ged(b, a)`` is ``ged(a, b)`` with the two direction blocks swapped.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from contract.registry import LINE_OCC
from ged.result import GedBreakdown, GedResult
from ged.semantics import SEMANTICS_STAMP, semantics_stamp


CACHE_DIR = ("ged", "cache")

# How many pair results one process keeps in memory.  The cache key contains
# the configuration hash and the hash is a function of the eight weights, so
# every position of a page slider is a new key and nothing ever collides with
# what came before: without a bound the server's memory is linear in "how
# many weight vectors has anyone tried today" (node 26 F-10).
MEMORY_ENTRIES = 512
#: The edit path is ~99.9% of a payload (77-135 KiB against well under one),
#: and only ``compare`` reads it - one pair at a time.  Ranking keeps the
#: light entries; the last few full ones are kept beside them so that looking
#: at the same pair twice is still free (FP-3).
FULL_ENTRIES = 8


def cache_dir(line: str) -> tuple[str, ...]:
    """Where one line's pair files live.

    The open-source line keeps ``ged/cache`` so nothing already on disk moves;
    any other line gets its own directory, because the file name is the two
    *part* ids and those are the same on every line.  Two lines writing into
    one directory would take turns overwriting each other's answers - the
    semantics stamp inside would stop a wrong hit, but only after the file
    had already been thrown away.
    """
    return CACHE_DIR if line == LINE_OCC else ("ged", line, "cache")


def features_dir(line: str) -> tuple[str, ...]:
    """Where one line's per-part pre-filter features live, beside its pairs."""
    return ("ged", "features") if line == LINE_OCC else ("ged", line, "features")


class PartFeatureCache:
    """The two numbers the pre-filter needs, without projecting the part.

    A ``rank`` whose every pair is already in the pair cache still had to
    project all nine parts, because the pre-filter asks each candidate for its
    leaf histogram and its ``L_ref`` - 127 of a warm ranking's 144 ms, and all
    of it thrown away again (node 26 FP-2).  Those two are cheap to write down
    and are a function of the same three things the pair cache is keyed on:
    the part's ``producedAt``, the semantics stamp of the tables that produced
    them, and the artifacts themselves.  The artifact fingerprint is the extra
    one: a pair result is a claim about two parts, but these are a claim about
    two *files*, and a file that changed under a stale ``producedAt`` must not
    be answered for.

    Generated artifact, in the run, never hand-edited; a miss costs exactly
    the projection it was avoiding.
    """

    def __init__(self, run_root: str | Path, line: str, semantics: str) -> None:
        self.root = Path(run_root).joinpath(*features_dir(line))
        self.semantics = semantics

    def path(self, part_id: str) -> Path:
        return self.root / f"{part_id}.json"

    @staticmethod
    def fingerprint(line_dir: Path) -> list[list] | None:
        """``[[name, size, mtime_ns], ...]`` for the files a projection reads."""
        stamps = []
        for name in ("result.json", "partmodel.json"):
            try:
                stat = (line_dir / name).stat()
            except OSError:
                return None
            stamps.append([name, stat.st_size, stat.st_mtime_ns])
        return stamps

    def get(self, part_id: str, produced_at: str, line_dir: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.path(part_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("semantics") != self.semantics or payload.get("producedAt") != produced_at:
            return None
        if payload.get("files") != self.fingerprint(line_dir):
            return None
        histogram = payload.get("typeHistogram")
        l_ref = payload.get("lRef")
        if not isinstance(histogram, dict) or not isinstance(l_ref, (int, float)):
            return None
        return {"typeHistogram": {str(k): int(v) for k, v in histogram.items()},
                "lRef": float(l_ref)}

    def put(self, part_id: str, produced_at: str, line_dir: Path,
            histogram: dict[str, int], l_ref: float) -> None:
        payload = {
            "schemaVersion": "v8.ged_features.v1",
            "partId": part_id,
            "producedAt": produced_at,
            "semantics": self.semantics,
            "files": self.fingerprint(line_dir),
            "typeHistogram": dict(sorted(histogram.items())),
            "lRef": float(l_ref),
        }
        if payload["files"] is None:
            return
        _write_json(self.path(part_id), payload)


def pair_order(id_a: str, id_b: str) -> tuple[str, str]:
    return (id_a, id_b) if id_a <= id_b else (id_b, id_a)


def orient(payload: dict[str, Any], id_a: str, id_b: str) -> dict[str, Any]:
    """A shallow copy of ``payload`` with ``a`` = ``id_a``, ``b`` = ``id_b``."""
    out = dict(payload)
    if out.get("graph_id_a") == id_a and out.get("graph_id_b") == id_b:
        return out
    if not (out.get("graph_id_a") == id_b and out.get("graph_id_b") == id_a):
        raise ValueError(f"payload is for {out.get('graph_id_a')}/{out.get('graph_id_b')}, not {id_a}/{id_b}")
    out["graph_id_a"], out["graph_id_b"] = id_a, id_b
    path = out.get("edit_path")
    if isinstance(path, dict):
        out["edit_path"] = {"a_to_b": path.get("b_to_a"), "b_to_a": path.get("a_to_b")}
    produced = out.get("producedAt")
    if isinstance(produced, dict):
        out["producedAt"] = {"a": produced.get("b"), "b": produced.get("a")}
    return out


def _breakdown_is_readable(breakdown: Any) -> bool:
    """Can ``result_from_payload`` still build a ``GedBreakdown`` from this?

    "Is it a dict" was not the same question: ``GedBreakdown(**payload)``
    raises on a key it does not know, so one entry written by a version with
    an extra cost channel turned every ``rank`` touching that pair into a 500
    whose text never mentioned the cache (node 26 F-08).  A file this reader
    cannot use is a miss, which is what the same module already does with a
    truncated file - and a miss is recomputed and overwritten in place.
    """
    if not isinstance(breakdown, dict):
        return False
    return set(breakdown) == set(GedBreakdown.__dataclass_fields__)


def result_from_payload(payload: dict[str, Any]) -> GedResult:
    """Rebuild the typed result from a cached payload (in its own orientation)."""
    config = payload.get("config")
    name = config.get("name") if isinstance(config, dict) else str(config)
    return GedResult(
        graph_id_a=str(payload["graph_id_a"]),
        graph_id_b=str(payload["graph_id_b"]),
        raw_cost=float(payload["raw_cost"]),
        upper_bound=float(payload["upper_bound"]),
        normalized_distance=float(payload["normalized_distance"]),
        similarity_percent=float(payload["similarity_percent"]),
        breakdown=GedBreakdown(**{k: float(v) for k, v in payload["breakdown"].items()}),
        edit_path=dict(payload.get("edit_path") or {}),
        config_name=str(name),
    )


class PairCache:
    def __init__(self, run_root: str | Path, *, semantics: str | None = None,
                 line: str = LINE_OCC, representation: str | None = None) -> None:
        self.run_root = Path(run_root)
        self.line = line
        self.root = self.run_root.joinpath(*cache_dir(line))
        if semantics is None:
            semantics = SEMANTICS_STAMP if representation is None else semantics_stamp(representation)
        self.semantics = semantics
        # Two bounded LRUs: everything, without its edit path, and the last
        # few complete payloads.  See MEMORY_ENTRIES / FULL_ENTRIES above.
        self._memory: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()
        self._full: "OrderedDict[tuple, dict[str, Any]]" = OrderedDict()
        self._unusable: set[str] = set()
        self._lock = threading.Lock()

    # -- keys -----------------------------------------------------------------

    @staticmethod
    def key(
        config_hash: str,
        id_a: str,
        id_b: str,
        produced_a: str | None,
        produced_b: str | None,
        semantics: str = "",
    ) -> tuple:
        first, second = pair_order(id_a, id_b)
        pa, pb = (produced_a, produced_b) if first == id_a else (produced_b, produced_a)
        return (config_hash, semantics, first, second, pa or "", pb or "")

    def path(self, config_hash: str, id_a: str, id_b: str) -> Path:
        first, second = pair_order(id_a, id_b)
        return self.root / config_hash / f"{first}__{second}.json"

    # -- read / write ---------------------------------------------------------

    def get(self, config_hash: str, id_a: str, id_b: str, produced_a: str | None,
            produced_b: str | None, *, with_edit_path: bool = False) -> dict[str, Any] | None:
        """The stored payload in sorted orientation, or None (miss, stale, corrupt).

        ``with_edit_path`` is what ``compare`` needs and ``rank`` does not.
        The light in-memory entry cannot answer it, so such a call falls
        through to the file - and, for a weight set that was never persisted,
        to a miss, which is a recomputation of one pair and the same number.
        """
        key = self.key(config_hash, id_a, id_b, produced_a, produced_b, self.semantics)
        with self._lock:
            store = self._full if with_edit_path else self._memory
            hit = store.get(key)
            if hit is not None:
                store.move_to_end(key)
        if hit is not None:
            return hit
        path = self.path(config_hash, id_a, id_b)
        payload = self._read(path)
        if payload is None:
            return None
        if not self._matches(payload, key):
            self._note_unusable(path, payload, key)
            return None
        self._remember(key, payload)
        return payload

    def put(
        self,
        config_hash: str,
        id_a: str,
        id_b: str,
        produced_a: str | None,
        produced_b: str | None,
        payload: dict[str, Any],
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Store ``payload`` (any orientation); returns it in sorted orientation.

        The cache stamps the entry with its own semantics: the writer does
        not get to claim a version the reader would then trust."""
        first, second = pair_order(id_a, id_b)
        stored = orient(payload, first, second)
        stored["semantics"] = self.semantics
        key = self.key(config_hash, id_a, id_b, produced_a, produced_b, self.semantics)
        self._remember(key, stored)
        if persist:
            self._write(self.path(config_hash, id_a, id_b), stored)
        return stored

    def clear_memory(self) -> None:
        with self._lock:
            self._memory.clear()
            self._full.clear()

    def memory_size(self) -> tuple[int, int]:
        """``(light entries, full entries)`` - what this process is holding."""
        with self._lock:
            return len(self._memory), len(self._full)

    # -- helpers --------------------------------------------------------------

    def _remember(self, key: tuple, payload: dict[str, Any]) -> None:
        """Keep it, bounded, and without repeating the edit path everywhere."""
        light = {name: value for name, value in payload.items() if name != "edit_path"}
        with self._lock:
            self._memory[key] = light
            self._memory.move_to_end(key)
            while len(self._memory) > MEMORY_ENTRIES:
                self._memory.popitem(last=False)
            if "edit_path" in payload:
                self._full[key] = payload
                self._full.move_to_end(key)
                while len(self._full) > FULL_ENTRIES:
                    self._full.popitem(last=False)

    def _note_unusable(self, path: Path, payload: dict[str, Any], key: tuple) -> None:
        """Say once, per file, that a readable entry could not be used.

        A stale ``producedAt`` or a superseded semantics stamp is the design
        working and says nothing; a *breakdown this reader cannot build* means
        the file was written by different code, and the operator should hear
        it once rather than wonder why nothing is ever cached.
        """
        if _breakdown_is_readable(payload.get("breakdown")):
            return
        name = str(path)
        with self._lock:
            if name in self._unusable:
                return
            self._unusable.add(name)
        print(f"[ged] ignoring a pair cache entry this version cannot read "
              f"(unexpected breakdown keys): {path.name}", file=sys.stderr)

    @staticmethod
    def _matches(payload: dict[str, Any], key: tuple) -> bool:
        config_hash, semantics, first, second, pa, pb = key
        config = payload.get("config")
        stored_hash = config.get("hash") if isinstance(config, dict) else None
        produced = payload.get("producedAt") or {}
        # An entry written before the stamp existed reports "" and misses,
        # which is what it should do: it was priced by unknown semantics.
        return (
            stored_hash == config_hash
            and str(payload.get("semantics") or "") == semantics
            and payload.get("graph_id_a") == first
            and payload.get("graph_id_b") == second
            and (produced.get("a") or "") == pa
            and (produced.get("b") or "") == pb
            and _breakdown_is_readable(payload.get("breakdown"))
        )

    @staticmethod
    def _read(path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _write(path: Path, payload: dict[str, Any]) -> None:
        _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic, fsynced, and silent about a disk that will not take it.

    A cache that cannot be written is a slower cache, not an error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
