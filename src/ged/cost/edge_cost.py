"""Relation edge base weights and preserved-edge mismatch."""

from __future__ import annotations

from typing import Any

from ged.contract.projection import GROUPED_RELATIONS
from ged.contract.schema import GedEdge
from .node_cost import bounded_diff


# Insert / delete base weight for relations that are not clique-normalised.
STATIC_BASE_WEIGHT: dict[str, float] = {
    "DIRECT_ADJACENT": 0.6,
    "LOCATED_ON": 0.6,
    "INTERSECTS": 1.0,
}
DEFAULT_BASE_WEIGHT = 0.6

LABEL_MISMATCH_BASE = 0.2
ATTRIBUTE_MISMATCH_BASE = 0.2


def edge_base_weight(edge: GedEdge) -> float:
    """Clique-normalised relations cost ``1/(k-1)`` per edge so that detaching
    one member of a ``k``-group costs exactly one ``W_REL`` in total."""
    if edge.type_key in GROUPED_RELATIONS:
        size = int(edge.metadata.get("group_size", 2) or 2)
        return 1.0 / (size - 1) if size >= 2 else 1.0
    return STATIC_BASE_WEIGHT.get(edge.type_key, DEFAULT_BASE_WEIGHT)


def edge_mismatch(left: GedEdge, right: GedEdge, base_weight: float) -> tuple[float, dict[str, Any]]:
    """Mismatch between two preserved edges of the same type (label, metrics).

    Capped at the edge's own base weight: changing an edge must never cost
    more than deleting and re-inserting it.
    """
    details: dict[str, Any] = {}
    total = 0.0
    if left.label != right.label and (left.label is not None or right.label is not None):
        total += LABEL_MISMATCH_BASE
        details["label"] = {"from": left.label, "to": right.label}
    shared = sorted(set(left.attributes) & set(right.attributes))
    if shared:
        distances = {name: bounded_diff(left.attributes[name], right.attributes[name]) for name in shared}
        mean = sum(distances.values()) / len(distances)
        total += ATTRIBUTE_MISMATCH_BASE * mean
        details["attributes"] = {
            name: {
                "from": left.raw_attributes.get(name),
                "to": right.raw_attributes.get(name),
                "distance": distances[name],
            }
            for name in shared
        }
    return min(total, base_weight), details
