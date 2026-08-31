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

import json
import os
import threading
from pathlib import Path
from typing import Any

from contract.registry import LINE_OCC
from ged.result import GedBreakdown, GedResult
from ged.semantics import SEMANTICS_STAMP, semantics_stamp


CACHE_DIR = ("ged", "cache")


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
        self._memory: dict[tuple, dict[str, Any]] = {}
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

    def get(self, config_hash: str, id_a: str, id_b: str, produced_a: str | None, produced_b: str | None) -> dict[str, Any] | None:
        """The stored payload in sorted orientation, or None (miss, stale, corrupt)."""
        key = self.key(config_hash, id_a, id_b, produced_a, produced_b, self.semantics)
        with self._lock:
            hit = self._memory.get(key)
        if hit is not None:
            return hit
        payload = self._read(self.path(config_hash, id_a, id_b))
        if payload is None or not self._matches(payload, key):
            return None
        with self._lock:
            self._memory[key] = payload
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
        with self._lock:
            self._memory[key] = stored
        if persist:
            self._write(self.path(config_hash, id_a, id_b), stored)
        return stored

    def clear_memory(self) -> None:
        with self._lock:
            self._memory.clear()

    # -- helpers --------------------------------------------------------------

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
            and isinstance(payload.get("breakdown"), dict)
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
            # A cache that cannot be written is a slower cache, not an error.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
