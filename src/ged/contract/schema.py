"""Immutable graph contract consumed by the GED solver.

Everything the solver sees is defined here.  The solver itself knows nothing
about manufacturing features: ``type_key`` strings and attribute names are
opaque to it and only interpreted by the cost callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping


AttributeKind = Literal["length", "area", "angle", "category", "count"]
AttributeState = Literal["valid", "null"]


@dataclass(frozen=True)
class GedAttribute:
    """One ontology attribute of a node after normalisation.

    ``value`` is the normalised comparison value (lengths divided by
    ``L_ref``, areas by ``L_ref²``, angles left in degrees, categories as
    strings).  ``raw`` keeps the original value from ``result.json`` so the
    edit path can report millimetres.  ``state == "null"`` means the
    attribute belongs to the class but the pipeline could not measure it;
    ``reason`` carries the pipeline's explanation verbatim.
    """

    name: str
    kind: AttributeKind
    value: float | str | None
    raw: object
    state: AttributeState
    reason: str | None = None


@dataclass(frozen=True)
class GedNode:
    id: str
    type_key: str
    family: str
    attributes: Mapping[str, GedAttribute]
    # Retained for explanation and for a later edge-level relative-angle
    # term.  Not used by any v0 cost (design outline §2.3).
    feature_axis: tuple[float, float, float] | None = None
    face_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GedEdge:
    u: str
    v: str
    type_key: str
    label: str | None = None
    # Normalised relation-valued quantities (e.g. axisSpacing / L_ref).
    attributes: Mapping[str, float] = field(default_factory=dict)
    raw_attributes: Mapping[str, object] = field(default_factory=dict)
    # Solver-visible context such as ``group_size`` for clique-normalised
    # relation types, or the semantic direction of LOCATED_ON.
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GedFixedNode:
    """A mandatory 1:1 aggregate (Surface background layer, unclaimed residue)."""

    id: str
    type_key: str
    attributes: Mapping[str, float]
    raw: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GedGraph:
    graph_id: str
    nodes: tuple[GedNode, ...]
    edges: tuple[GedEdge, ...]
    fixed_nodes: tuple[GedFixedNode, ...]
    l_ref: float
    meta: Mapping[str, object] = field(default_factory=dict)

    def node_by_id(self) -> dict[str, GedNode]:
        return {node.id: node for node in self.nodes}

    def type_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.type_key] = counts.get(node.type_key, 0) + 1
        return counts
