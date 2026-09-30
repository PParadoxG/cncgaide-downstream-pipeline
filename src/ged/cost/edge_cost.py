"""Relation edge base weights and preserved-edge mismatch."""

from __future__ import annotations

from typing import Any

from ged.contract.representation import Representation
from ged.contract.schema import GedEdge
from .node_cost import bounded_diff


# What an edge of an unpriced relation type is worth.  Not a representation
# decision: it is what the cost layer charges when the ontology named a
# relation but no price for it.
DEFAULT_BASE_WEIGHT = 0.6

LABEL_MISMATCH_BASE = 0.2
ATTRIBUTE_MISMATCH_BASE = 0.2


def make_edge_base_weight(representation: Representation):
    """The insert / delete base weight of one edge, for one representation.

    Which relations are clique-normalised and what the others are worth are
    the *ontology's* answers, so they are read off the representation rather
    than imported from v5.1 (node 26 F-18: the cost layer used to reach for
    ``projection.GROUPED_RELATIONS``, which is v5.1's, whichever ontology the
    graph was written in).  Same shape as ``make_geometry_distance``: bound
    once per dialect in ``cost.default.DIALECTS``.
    """
    grouped = frozenset(representation.grouped_relations)
    static = dict(representation.static_edge_base_weight)

    def edge_base_weight(edge: GedEdge) -> float:
        """Clique-normalised relations cost ``1/(k-1)`` per edge so that
        detaching one member of a ``k``-group costs exactly one ``W_REL``."""
        if edge.type_key in grouped:
            size = int(edge.metadata.get("group_size", 2) or 2)
            return 1.0 / (size - 1) if size >= 2 else 1.0
        return static.get(edge.type_key, DEFAULT_BASE_WEIGHT)

    return edge_base_weight


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
