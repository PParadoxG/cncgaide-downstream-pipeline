"""Symmetric Hungarian BP-GED (bipartite graph edit distance approximation).

Ported from the v6 feature-level GED.  This module knows only abstract
nodes, edges, fixed nodes and the callbacks supplied by ``CostConfig``; it
contains no feature names, relation names or attribute names.

Per direction:

1. ``linear_sum_assignment`` over the (n+m)×(n+m) node cost matrix chooses
   substitutions, deletions and insertions.
2. The induced node mapping classifies every edge as preserved (possibly
   with a label / metric mismatch), deleted or inserted.
3. Mandatory fixed nodes are substituted 1:1 by type.
4. A global scale term compares ``L_ref``.

``raw = node + fixed + edge + scale``, ``upper = n·C_DEL + m·C_INS +
W_REL·(Σ base_A + Σ base_B) + W_SCALE + fixed_bound``; the normalised
distance is ``raw / upper`` averaged over both directions.

Nodes are assigned before edges are scored.  With ``config.w_local == 0``
edges never influence which equal-cost assignment Hungarian picks (the v6
limitation).  With ``w_local > 0`` the assignment matrix is enriched
Riesen–Bunke style with the edit cost of each node's incident-edge multiset,
so structurally different but attribute-identical nodes are told apart.  The
enrichment steers the assignment only; the reported cost is always the cost
of the induced edit path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from ged.contract.schema import GedEdge, GedGraph, GedNode
from ged.cost.config import CostConfig
from ged.cost.node_cost import substitution_cost
from ged.result import GedBreakdown, GedResult


EPS = 1.0e-9
BIG_M = 1.0e6


@dataclass(frozen=True)
class _Assignment:
    cost: float
    mapping: dict[str, str]
    substitutions: int
    type_changes: int
    deletions: int
    insertions: int
    edits: tuple[dict[str, Any], ...]


def _incident_profiles(graph: GedGraph, config: CostConfig) -> dict[str, dict[tuple[str, str | None], list[float]]]:
    """Per node: incident edges grouped by (type, label) with their base weights."""
    profiles: dict[str, dict[tuple[str, str | None], list[float]]] = {node.id: {} for node in graph.nodes}
    for edge in graph.edges:
        weight = config.edge_base_weight(edge)
        key = (edge.type_key, edge.label)
        for endpoint in (edge.u, edge.v):
            profiles.setdefault(endpoint, {}).setdefault(key, []).append(weight)
    return profiles


def _local_structure_cost(
    left_profile: dict[tuple[str, str | None], list[float]],
    right_profile: dict[tuple[str, str | None], list[float]],
) -> float:
    """Edit cost of matching two incident-edge multisets by (type, label).

    Matched edges cost nothing; the surplus on either side is deleted or
    inserted at its base weight.  Halved because every edge is seen from
    both endpoints.
    """
    total = 0.0
    # Sorted, not set order: the per-key sums are added in a fixed order, so
    # the enriched matrix — and with it Hungarian's choice among equal-cost
    # assignments — is the same in every process (set order of str tuples
    # follows the per-process hash seed, and the resulting 1e-16 differences
    # were enough to flip ties and move the reported edge cost).
    for key in sorted(set(left_profile) | set(right_profile), key=lambda item: (item[0], item[1] or "")):
        lw = sorted(left_profile.get(key, ()), reverse=True)
        rw = sorted(right_profile.get(key, ()), reverse=True)
        matched = min(len(lw), len(rw))
        total += sum(lw[matched:]) + sum(rw[matched:])
    return 0.5 * total


def _node_assignment(left: GedGraph, right: GedGraph, config: CostConfig) -> _Assignment:
    n_left = len(left.nodes)
    n_right = len(right.nodes)
    size = n_left + n_right
    if size == 0:
        return _Assignment(0.0, {}, 0, 0, 0, 0, ())

    costs = np.zeros((size, size), dtype=float)
    cache: dict[tuple[int, int], tuple[float, float, Any]] = {}
    for i, left_node in enumerate(left.nodes):
        for j, right_node in enumerate(right.nodes):
            parts = substitution_cost(left_node, right_node, config)
            cache[(i, j)] = parts
            costs[i, j] = parts[0]
    if n_left:
        costs[:n_left, n_right:] = BIG_M
        for i in range(n_left):
            costs[i, n_right + i] = config.c_del
    if n_right:
        costs[n_left:, :n_right] = BIG_M
        for j in range(n_right):
            costs[n_left + j, j] = config.c_ins

    matrix = costs
    if config.w_local > 0:
        # Steering copy: the true node costs above are what the edit path
        # reports; this enriched matrix only decides the assignment.
        matrix = costs.copy()
        left_profiles = _incident_profiles(left, config)
        right_profiles = _incident_profiles(right, config)
        empty: dict[tuple[str, str | None], list[float]] = {}
        factor = config.w_local * config.w_rel
        for i, left_node in enumerate(left.nodes):
            lp = left_profiles.get(left_node.id, empty)
            for j, right_node in enumerate(right.nodes):
                matrix[i, j] += factor * _local_structure_cost(lp, right_profiles.get(right_node.id, empty))
            matrix[i, n_right + i] += factor * _local_structure_cost(lp, empty)
        for j, right_node in enumerate(right.nodes):
            matrix[n_left + j, j] += factor * _local_structure_cost(empty, right_profiles.get(right_node.id, empty))

    row_ind, col_ind = linear_sum_assignment(matrix)
    selected = float(costs[row_ind, col_ind].sum())
    if selected >= BIG_M:
        raise AssertionError("Hungarian assignment selected a forbidden edit")

    mapping: dict[str, str] = {}
    substitutions = type_changes = deletions = insertions = 0
    edits: list[dict[str, Any]] = []
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        if row < n_left:
            source = left.nodes[row]
            if col < n_right:
                target = right.nodes[col]
                cost, type_part, geometry = cache[(row, col)]
                mapping[source.id] = target.id
                substitutions += 1
                operation = "substitute_same_type"
                if type_part > 0:
                    type_changes += 1
                    operation = "substitute_type_change"
                edits.append(
                    {
                        "operation": operation,
                        "source": source.id,
                        "target": target.id,
                        "source_type": source.type_key,
                        "target_type": target.type_key,
                        "type_cost": type_part,
                        "geometry_distance": geometry.distance,
                        "attributes_compared": geometry.compared,
                        "attributes": list(geometry.details),
                        "cost": cost,
                    }
                )
            else:
                if col != n_right + row:
                    raise AssertionError("decoded invalid deletion")
                deletions += 1
                edits.append(
                    {
                        "operation": "delete",
                        "source": source.id,
                        "source_type": source.type_key,
                        "cost": config.c_del,
                    }
                )
        elif col < n_right:
            if row != n_left + col:
                raise AssertionError("decoded invalid insertion")
            target = right.nodes[col]
            insertions += 1
            edits.append(
                {
                    "operation": "insert",
                    "target": target.id,
                    "target_type": target.type_key,
                    "cost": config.c_ins,
                }
            )
    return _Assignment(selected, mapping, substitutions, type_changes, deletions, insertions, tuple(edits))


def _edge_key(u: str, v: str, edge_type: str) -> tuple[str, str, str]:
    a, b = sorted((u, v))
    return a, b, edge_type


def _edge_slots(edges: tuple[GedEdge, ...]) -> dict[tuple[str, str, str], GedEdge]:
    slots: dict[tuple[str, str, str], GedEdge] = {}
    for edge in edges:
        key = _edge_key(edge.u, edge.v, edge.type_key)
        if key in slots:
            raise ValueError(f"duplicate typed edge slot: {key}")
        slots[key] = edge
    return slots


@dataclass(frozen=True)
class _EdgeScore:
    cost: float
    preserved: int
    changed: int
    deleted: int
    inserted: int
    edits: tuple[dict[str, Any], ...]


def _edge_cost(left: GedGraph, right: GedGraph, mapping: dict[str, str], config: CostConfig) -> _EdgeScore:
    right_slots = _edge_slots(right.edges)
    preserved_right: set[tuple[str, str, str]] = set()
    cost = 0.0
    preserved = changed = deleted = 0
    edits: list[dict[str, Any]] = []
    for source in left.edges:
        target = None
        target_key = None
        if source.u in mapping and source.v in mapping:
            target_key = _edge_key(mapping[source.u], mapping[source.v], source.type_key)
            target = right_slots.get(target_key)
        if target is not None:
            assert target_key is not None
            preserved_right.add(target_key)
            base = config.edge_base_weight(source)
            mismatch, details = config.edge_mismatch(source, target, base)
            if mismatch > 0:
                changed += 1
                edit_cost = mismatch * config.w_rel
                cost += edit_cost
                edits.append(
                    {
                        "operation": "change",
                        "type": source.type_key,
                        "source_edge": [source.u, source.v],
                        "target_edge": [target.u, target.v],
                        "details": details,
                        "cost": edit_cost,
                    }
                )
            else:
                preserved += 1
                edits.append(
                    {
                        "operation": "preserve",
                        "type": source.type_key,
                        "source_edge": [source.u, source.v],
                        "target_edge": [target.u, target.v],
                        "cost": 0.0,
                    }
                )
        else:
            deleted += 1
            base = config.edge_base_weight(source)
            edit_cost = base * config.w_rel
            cost += edit_cost
            edits.append(
                {
                    "operation": "delete",
                    "type": source.type_key,
                    "source_edge": [source.u, source.v],
                    "base_weight": base,
                    "group_size": source.metadata.get("group_size"),
                    "cost": edit_cost,
                }
            )
    inserted = 0
    for key, target in right_slots.items():
        if key in preserved_right:
            continue
        inserted += 1
        base = config.edge_base_weight(target)
        edit_cost = base * config.w_rel
        cost += edit_cost
        edits.append(
            {
                "operation": "insert",
                "type": target.type_key,
                "target_edge": [target.u, target.v],
                "base_weight": base,
                "group_size": target.metadata.get("group_size"),
                "cost": edit_cost,
            }
        )
    return _EdgeScore(cost, preserved, changed, deleted, inserted, tuple(edits))


def _fixed_cost(left: GedGraph, right: GedGraph, config: CostConfig) -> tuple[float, tuple[dict[str, Any], ...]]:
    left_by_type = {node.type_key: node for node in left.fixed_nodes}
    right_by_type = {node.type_key: node for node in right.fixed_nodes}
    if set(left_by_type) != set(right_by_type):
        raise ValueError(f"fixed node type sets differ: {sorted(left_by_type)} vs {sorted(right_by_type)}")
    total = 0.0
    edits: list[dict[str, Any]] = []
    for source in left.fixed_nodes:
        target = right_by_type[source.type_key]
        parts = config.fixed_node_cost(source, target)
        total += parts.cost
        edits.append(
            {
                "operation": "substitute_fixed_1_to_1",
                "type": source.type_key,
                "source": dict(source.raw),
                "target": dict(target.raw),
                "differences": dict(parts.differences),
                "cost": parts.cost,
            }
        )
    return total, tuple(edits)


def _scale_cost(left: GedGraph, right: GedGraph, config: CostConfig) -> float:
    if left.l_ref <= 0 or right.l_ref <= 0:
        raise ValueError("L_ref must be positive")
    return config.w_scale * math.tanh(abs(math.log(left.l_ref / right.l_ref)) / math.log(2.0))


def _upper_bound(left: GedGraph, right: GedGraph, config: CostConfig) -> float:
    node_bound = len(left.nodes) * config.c_del + len(right.nodes) * config.c_ins
    edge_bound = config.w_rel * (
        sum(config.edge_base_weight(edge) for edge in left.edges)
        + sum(config.edge_base_weight(edge) for edge in right.edges)
    )
    return node_bound + edge_bound + config.w_scale + config.fixed_cost_upper_bound


@dataclass(frozen=True)
class _Directed:
    raw: float
    upper: float
    normalized: float
    breakdown: GedBreakdown
    edit_path: dict[str, Any]


def _directed(left: GedGraph, right: GedGraph, config: CostConfig) -> _Directed:
    assignment = _node_assignment(left, right, config)
    edges = _edge_cost(left, right, assignment.mapping, config)
    fixed_cost, fixed_edits = _fixed_cost(left, right, config)
    scale_cost = _scale_cost(left, right, config)
    raw = assignment.cost + fixed_cost + edges.cost + scale_cost
    upper = _upper_bound(left, right, config)
    normalized = raw / upper if upper > 0 else 0.0
    if normalized < -1.0e-9 or normalized > 1.0 + 1.0e-9:
        raise AssertionError(f"normalized GED outside [0,1]: {normalized}")
    normalized = min(1.0, max(0.0, normalized))
    return _Directed(
        raw,
        upper,
        normalized,
        GedBreakdown(
            node_cost=assignment.cost,
            fixed_cost=fixed_cost,
            edge_cost=edges.cost,
            scale_cost=scale_cost,
            n_sub=assignment.substitutions,
            n_type_change=assignment.type_changes,
            n_del=assignment.deletions,
            n_ins=assignment.insertions,
            edges_preserved=edges.preserved,
            edges_changed=edges.changed,
            edges_del=edges.deleted,
            edges_ins=edges.inserted,
        ),
        {
            "nodes": list(assignment.edits),
            "fixed_nodes": list(fixed_edits),
            "edges": list(edges.edits),
            "scale": {"source_L_ref": left.l_ref, "target_L_ref": right.l_ref, "cost": scale_cost},
        },
    )


def _avg(a: float, b: float) -> float:
    value = 0.5 * (a + b)
    rounded = round(value)
    return float(rounded) if abs(value - rounded) <= 1.0e-12 else value


def ged(left: GedGraph, right: GedGraph, config: CostConfig) -> GedResult:
    """Symmetric BP-GED between two graphs under one cost configuration."""
    ab = _directed(left, right, config)
    ba = _directed(right, left, config)
    breakdown = GedBreakdown(
        node_cost=0.5 * (ab.breakdown.node_cost + ba.breakdown.node_cost),
        fixed_cost=0.5 * (ab.breakdown.fixed_cost + ba.breakdown.fixed_cost),
        edge_cost=0.5 * (ab.breakdown.edge_cost + ba.breakdown.edge_cost),
        scale_cost=0.5 * (ab.breakdown.scale_cost + ba.breakdown.scale_cost),
        n_sub=_avg(ab.breakdown.n_sub, ba.breakdown.n_sub),
        n_type_change=_avg(ab.breakdown.n_type_change, ba.breakdown.n_type_change),
        n_del=_avg(ab.breakdown.n_del, ba.breakdown.n_ins),
        n_ins=_avg(ab.breakdown.n_ins, ba.breakdown.n_del),
        edges_preserved=_avg(ab.breakdown.edges_preserved, ba.breakdown.edges_preserved),
        edges_changed=_avg(ab.breakdown.edges_changed, ba.breakdown.edges_changed),
        edges_del=_avg(ab.breakdown.edges_del, ba.breakdown.edges_ins),
        edges_ins=_avg(ab.breakdown.edges_ins, ba.breakdown.edges_del),
    )
    normalized = 0.5 * (ab.normalized + ba.normalized)
    return GedResult(
        graph_id_a=left.graph_id,
        graph_id_b=right.graph_id,
        raw_cost=0.5 * (ab.raw + ba.raw),
        upper_bound=0.5 * (ab.upper + ba.upper),
        normalized_distance=normalized,
        similarity_percent=100.0 * (1.0 - normalized),
        breakdown=breakdown,
        edit_path={"a_to_b": ab.edit_path, "b_to_a": ba.edit_path},
        config_name=config.name,
    )
