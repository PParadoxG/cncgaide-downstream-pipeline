"""Graph-agnostic cost configuration contract for the BP-GED solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ged.contract.representation import Representation, V5_1
from ged.contract.schema import GedEdge, GedFixedNode, GedNode


@dataclass(frozen=True)
class GeometryTerm:
    """Attribute distance between two nodes plus the per-attribute audit."""

    distance: float
    compared: int
    details: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class FixedCostParts:
    cost: float
    differences: Mapping[str, float]


TypeCost = Callable[[str, str], float]
GeometryDistance = Callable[[GedNode, GedNode], GeometryTerm]
EdgeBaseWeight = Callable[[GedEdge], float]
# (left_edge, right_edge, base_weight) -> (mismatch cost base, details)
EdgeMismatch = Callable[[GedEdge, GedEdge, float], tuple[float, dict[str, Any]]]
FixedNodeCost = Callable[[GedFixedNode, GedFixedNode], FixedCostParts]


@dataclass(frozen=True)
class CostConfig:
    name: str
    w_sem: float
    w_geom: float
    w_rel: float
    w_scale: float
    w_res: float
    c_del: float
    c_ins: float
    type_cost: TypeCost
    geometry_distance: GeometryDistance
    edge_base_weight: EdgeBaseWeight
    edge_mismatch: EdgeMismatch
    fixed_node_cost: FixedNodeCost
    fixed_cost_upper_bound: float
    # Riesen–Bunke local-structure enrichment of the assignment matrix:
    # each node substitution / deletion / insertion entry is increased by
    # ``w_local`` × the edit cost of the node's incident-edge multiset.  This
    # only steers which assignment Hungarian picks among (near-)equal node
    # costs; the reported cost is always the induced edit path.  0 disables.
    w_local: float = 0.0
    # Which ontology the graphs this config prices are written in.  Not part
    # of the configuration *hash* - that is the eight weights and nothing
    # else - but it is what the pair cache's semantics stamp is taken from,
    # so a v5_1 entry can never be served to an nx_v1 query.
    representation: Representation = V5_1

    def describe(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "W_SEM": self.w_sem,
            "W_GEOM": self.w_geom,
            "W_REL": self.w_rel,
            "W_SCALE": self.w_scale,
            "W_RES": self.w_res,
            "W_LOCAL": self.w_local,
            "C_DEL": self.c_del,
            "C_INS": self.c_ins,
        }
