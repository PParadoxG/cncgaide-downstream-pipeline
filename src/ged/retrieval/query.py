"""k-NN ranking with an optional cheap pre-filter."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from ged.contract.schema import GedGraph
from ged.cost.config import CostConfig
from ged.cost.default import default_config
from ged.result import GedResult
from ged.solver.bp import ged


@dataclass(frozen=True)
class Ranked:
    graph_id: str
    result: GedResult
    prefilter_score: float


def prefilter_score(query: GedGraph, candidate: GedGraph) -> float:
    """Leaf-histogram L1 (normalised) plus an L_ref ratio term, in [0, 2].

    Reads only ``type_histogram()`` and ``l_ref``; a caller that can answer
    those without projecting may pass anything that does (node 26 FP-2, where
    the service passes a lazy candidate and lets ``solve`` decide whether a
    graph is needed at all).
    """
    hq = query.type_histogram()
    hc = candidate.type_histogram()
    keys = set(hq) | set(hc)
    total = sum(hq.values()) + sum(hc.values())
    l1 = sum(abs(hq.get(k, 0) - hc.get(k, 0)) for k in keys) / total if total else 0.0
    scale = math.tanh(abs(math.log(query.l_ref / candidate.l_ref)) / math.log(2.0))
    return l1 + scale


def rank(
    query: GedGraph,
    candidates: Iterable[GedGraph],
    *,
    config: CostConfig | None = None,
    top: int | None = None,
    prefilter_keep: int | None = None,
    exclude_self: bool = True,
    solve: Callable[[GedGraph, GedGraph, CostConfig], GedResult] = ged,
) -> list[Ranked]:
    """Rank ``candidates`` by GED distance to ``query`` (ascending).

    ``solve`` computes one pair; the default is the solver itself, a caller
    with a result cache passes a lookup-then-solve wrapper.  Pre-filter,
    ordering and truncation stay here either way.
    """
    config = config or default_config()
    pool = [c for c in candidates if not (exclude_self and c.graph_id == query.graph_id)]
    scored = sorted(((prefilter_score(query, c), c) for c in pool), key=lambda item: (item[0], item[1].graph_id))
    if prefilter_keep is not None:
        scored = scored[:prefilter_keep]
    ranked = [Ranked(c.graph_id, solve(query, c, config), score) for score, c in scored]
    ranked.sort(key=lambda item: (item.result.normalized_distance, item.graph_id))
    return ranked[:top] if top is not None else ranked
