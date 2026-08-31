"""Similarity service: the one object the CLI and the viewer server both call.

It owns nothing the GED does not already have — it only wires together the
projection (``ged.contract``), the ranking (``retrieval.query``), the solver
(``ged.solver``), the cost presets (``ged.cost.presets``) and the pair cache
(``retrieval.cache``) over a ``runs/<tag>`` store, and returns dictionaries
that are ready to be written as JSON.  Because ``ged rank --json`` and
``GET /api/similarity/rank`` go through the same method, the two cannot drift
apart.

One service instance serves **one kernel line** (node 21'): the graphs it
projects, the representation it prices them with and the cache it stores them
in all come from that line, and two instances never share anything but the
weights.  ``SimilarityService(run_root)`` still means the open-source line, so
every existing caller is unchanged.

Part versions come from that line's block of ``record.json``
(``producedAt``) - a rebuild of the other line must not invalidate cached
distances that never looked at it; a part directory
without a record (a run older than the store) is versioned by the mtime of
its ``result.json`` so it still works, just without the store's guarantees.
Projected graphs are cached per ``(partId, producedAt)``; pair results go
through ``PairCache``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable

from ged.contract.projection import load_part
from ged.contract.representation import for_line
from ged.contract.schema import GedGraph
from ged.cost.config import CostConfig
from ged.cost.presets import DEFAULT_PRESET, WEIGHT_SPECS, list_presets, resolve_config
from ged.result import GedResult
from ged.retrieval.cache import PairCache, orient, result_from_payload
from ged.retrieval.corpus import CORPUS_LINE
from ged.retrieval.query import rank as rank_graphs
from ged.solver.bp import ged


class PartNotFound(LookupError):
    """No such part in the run (no directory, no result)."""


class ProjectionFailed(RuntimeError):
    """The part exists but cannot be projected to a GED graph (or its record says FAIL)."""


@dataclass(frozen=True)
class PartRef:
    part_id: str
    produced_at: str
    status: str | None
    error: str | None = None


def _config_block(info: dict[str, Any]) -> dict[str, Any]:
    """``{name, hash, weights, preset}`` — the same description ``configs``
    and ``rank`` return for a configuration."""
    return {
        "name": info["name"],
        "hash": info["hash"],
        "weights": dict(info["weights"]),
        "preset": bool(info.get("preset")),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SimilarityService:
    def __init__(self, run_root: str | Path, *, cache: PairCache | None = None,
                 line: str = CORPUS_LINE) -> None:
        self.run_root = Path(run_root)
        self.parts_dir = self.run_root / "parts"
        self.line = line
        self.representation = for_line(line)
        self.cache = cache or PairCache(self.run_root, line=line,
                                        representation=self.representation.name)
        self._graphs: dict[str, tuple[str, GedGraph]] = {}  # part_id -> (produced_at, graph)
        self._lock = threading.RLock()

    def _resolve(self, config_name, weights):
        return resolve_config(config_name, weights, representation=self.representation)

    # ------------------------------------------------------------ configs

    def configs(self) -> dict[str, Any]:
        return {
            "configs": list_presets(),
            "default": DEFAULT_PRESET,
            "weights": {key: dict(spec) for key, spec in WEIGHT_SPECS.items()},
            "line": self.line,
            "representation": self.representation.name,
        }

    # -------------------------------------------------------------- parts

    def ref(self, part_id: str) -> PartRef:
        """Version stamp and status of one part; PartNotFound if absent."""
        part_dir = self.parts_dir / part_id
        record_path = part_dir / "record.json"
        result_path = part_dir / self.line / "result.json"
        block = None
        if record_path.is_file():
            try:
                loaded = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                candidate = (loaded.get("lines") or {}).get(self.line)
                block = candidate if isinstance(candidate, dict) else None
        if block is not None:
            produced = str(block.get("producedAt") or block.get("builtAt") or "")
            if not produced:
                produced = self._mtime_stamp(result_path)
            return PartRef(part_id, produced, block.get("status"), block.get("error"))
        if result_path.is_file():
            return PartRef(part_id, self._mtime_stamp(result_path), "PASS")
        raise PartNotFound(f"{part_id}: not in {self.run_root}")

    @staticmethod
    def _mtime_stamp(path: Path) -> str:
        try:
            return f"mtime:{path.stat().st_mtime_ns}"
        except OSError:
            return ""

    def part_ids(self) -> list[str]:
        """Every PASS part of the run, sorted."""
        if not self.parts_dir.is_dir():
            return []
        ids: list[str] = []
        for entry in sorted(self.parts_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            try:
                ref = self.ref(entry.name)
            except PartNotFound:
                continue
            if ref.status == "PASS":
                ids.append(entry.name)
        return ids

    def graph(self, ref: PartRef) -> GedGraph:
        """The projected graph of ``ref`` (cached per version)."""
        if ref.status != "PASS":
            detail = ref.error or f"record status {ref.status}"
            raise ProjectionFailed(f"{ref.part_id}: {detail}")
        with self._lock:
            hit = self._graphs.get(ref.part_id)
            if hit is not None and hit[0] == ref.produced_at:
                return hit[1]
        try:
            graph = load_part(self.parts_dir / ref.part_id / self.line, graph_id=ref.part_id,
                              representation=self.representation)
        except Exception as error:  # noqa: BLE001 - the reason travels to the caller
            raise ProjectionFailed(f"{ref.part_id}: {type(error).__name__}: {error}") from error
        with self._lock:
            self._graphs[ref.part_id] = (ref.produced_at, graph)
        return graph

    # ------------------------------------------------------------ results

    def _lookup(self, info: dict[str, Any], a: PartRef, b: PartRef) -> dict[str, Any] | None:
        payload = self.cache.get(info["hash"], a.part_id, b.part_id, a.produced_at, b.produced_at)
        return None if payload is None else orient(payload, a.part_id, b.part_id)

    def _store(self, result: GedResult, info: dict[str, Any], a: PartRef, b: PartRef) -> dict[str, Any]:
        payload = result.to_dict(include_edit_path=True)
        payload["config"] = _config_block(info)
        payload["line"] = self.line
        payload["representation"] = self.representation.name
        payload["producedAt"] = {"a": a.produced_at, "b": b.produced_at}
        payload["computedAt"] = _utc_now()
        # Only preset configurations are worth a file on disk; a custom weight
        # set is an experiment and lives in memory for the life of the server.
        stored = self.cache.put(
            info["hash"], a.part_id, b.part_id, a.produced_at, b.produced_at, payload,
            persist=bool(info.get("preset")),
        )
        return orient(stored, a.part_id, b.part_id)

    def compare(
        self,
        id_a: str,
        id_b: str,
        *,
        config_name: str | None = None,
        weights: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        """The full pair result (with edit path), oriented ``a`` -> ``b``."""
        config, info = self._resolve(config_name, weights)
        ref_a, ref_b = self.ref(id_a), self.ref(id_b)
        graph_a, graph_b = self.graph(ref_a), self.graph(ref_b)
        started = time.perf_counter()
        payload = self._lookup(info, ref_a, ref_b)
        cached = payload is not None
        if payload is None:
            payload = self._store(ged(graph_a, graph_b, config), info, ref_a, ref_b)
        # The cache key is the configuration hash, so the description is the
        # same whichever side wrote the file; restating it here also upgrades
        # pair files written before the block carried ``preset``.
        payload["config"] = _config_block(info)
        payload["line"] = self.line
        payload["representation"] = self.representation.name
        payload["cached"] = cached
        payload["timingMs"] = round((time.perf_counter() - started) * 1000.0, 1)
        return payload

    def rank(
        self,
        query_id: str,
        *,
        config_name: str | None = None,
        weights: dict[str, object] | None = None,
        candidate_ids: Iterable[str] | None = None,
        top: int | None = None,
        prefilter_keep: int | None = None,
    ) -> dict[str, Any]:
        """k-NN ranking of ``query_id`` against the candidates (default: every
        other PASS part of the run)."""
        started = time.perf_counter()
        config, info = self._resolve(config_name, weights)
        query_ref = self.ref(query_id)
        query_graph = self.graph(query_ref)

        ids = list(candidate_ids) if candidate_ids is not None else self.part_ids()
        refs: dict[str, PartRef] = {query_id: query_ref}
        graphs: list[GedGraph] = []
        failed: dict[str, str] = {}
        for part_id in ids:
            if part_id == query_id:
                continue
            try:
                ref = self.ref(part_id)
                graphs.append(self.graph(ref))
                refs[part_id] = ref
            except (PartNotFound, ProjectionFailed) as error:
                failed[part_id] = str(error)

        hits: set[str] = set()

        def solve(left: GedGraph, right: GedGraph, cfg: CostConfig) -> GedResult:
            ref_l, ref_r = refs[left.graph_id], refs[right.graph_id]
            payload = self._lookup(info, ref_l, ref_r)
            if payload is not None:
                hits.add(right.graph_id)
                return result_from_payload(payload)
            result = ged(left, right, cfg)
            self._store(result, info, ref_l, ref_r)
            return result

        ranked = rank_graphs(
            query_graph, graphs, config=config, top=top, prefilter_keep=prefilter_keep, solve=solve,
        )
        rows = []
        for position, item in enumerate(ranked, 1):
            result = item.result
            rows.append({
                "rank": position,
                "partId": item.graph_id,
                "distance": result.normalized_distance,
                "similarity": result.similarity_percent,
                "rawCost": result.raw_cost,
                "upperBound": result.upper_bound,
                "breakdown": result.to_dict(include_edit_path=False)["breakdown"],
                "prefilterScore": item.prefilter_score,
                "cached": item.graph_id in hits,
                "producedAt": refs[item.graph_id].produced_at,
            })
        return {
            "query": query_id,
            "queryProducedAt": query_ref.produced_at,
            "line": self.line,
            "representation": self.representation.name,
            "config": info,
            "corpus": {"loaded": 1 + len(graphs), "failed": failed},
            "candidates": len(graphs),
            "top": top,
            "prefilterKeep": prefilter_keep,
            "ranked": rows,
            "timingMs": round((time.perf_counter() - started) * 1000.0, 1),
        }
