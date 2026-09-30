"""The v0 placeholder cost configuration.

Constants are ported from the v6 feature-level GED (locked there after the
Rv2 variant study); they are a starting point, not a result.
"""

from __future__ import annotations

from ged.contract.representation import NX_V1, Representation, V5_1

from .config import CostConfig
from .edge_cost import edge_mismatch, make_edge_base_weight
from .fixed_cost import fixed_cost_upper_bound, make_fixed_node_cost
from .node_cost import make_geometry_distance
from .nx_type_matrix import type_cost as nx_type_cost
from .type_matrix import type_cost


W_SEM = 1.0
W_GEOM = 0.3
W_REL = 0.5
W_SCALE = 0.1
W_RES = 0.12
C_INS = 0.5
C_DEL = 0.5
# Local-structure steering of the assignment (see CostConfig.w_local).  The
# F1 variant family showed why it is needed: six attribute-identical
# TriangularSlots were renumbered between variants and plain BP mapped them
# by id order, turning one preserved SLOT_CLEARANCE edge into a delete+insert.
W_LOCAL = 1.0


# What is *not* adjustable, per representation: the class substitution cost
# and how the attribute mean is taken.  The eight weights above are shared -
# node 21' reuses one weight vocabulary and one preset table on both sides,
# because "how much does a type mismatch matter" is the same question in
# either ontology.  What differs is what a type mismatch *is*.
DIALECTS: dict[str, dict[str, object]] = {
    V5_1.name: {
        "representation": V5_1,
        "type_cost": type_cost,
        # v5.1's leaves publish a handful of attributes each and every one of
        # them is a measurement, so every shared pair is compared.
        "geometry_distance": make_geometry_distance(V5_1),
        "edge_base_weight": make_edge_base_weight(V5_1),
    },
    NX_V1.name: {
        "representation": NX_V1,
        # 1 - Wu-Palmer over NX's own tree (19' §1.4); OVERRIDES empty (D8=b).
        "type_cost": nx_type_cost,
        # Double-zero skip (19' D7 = a): an NX class inherits every slot its
        # chain declares and the unused ones read 0.
        "geometry_distance": make_geometry_distance(NX_V1, skip_both_zero=True),
        # nx_v1 publishes six relations and prices the two it does not group;
        # DIRECT_ADJACENT, which v5.1 prices, does not exist on this side.
        "edge_base_weight": make_edge_base_weight(NX_V1),
    },
}


def default_config(name: str = "v0_default", *, w_local: float = W_LOCAL,
                   representation: Representation = V5_1) -> CostConfig:
    dialect = DIALECTS[representation.name]
    return CostConfig(
        name=name,
        w_sem=W_SEM,
        w_geom=W_GEOM,
        w_rel=W_REL,
        w_scale=W_SCALE,
        w_res=W_RES,
        c_del=C_DEL,
        c_ins=C_INS,
        type_cost=dialect["type_cost"],
        geometry_distance=dialect["geometry_distance"],
        edge_base_weight=dialect["edge_base_weight"],
        edge_mismatch=edge_mismatch,
        fixed_node_cost=make_fixed_node_cost(
            W_RES, shares=representation.fixed_budget_shares,
            attribute_weights=representation.fixed_attribute_weights),
        fixed_cost_upper_bound=fixed_cost_upper_bound(
            W_RES, shares=representation.fixed_budget_shares),
        w_local=w_local,
        representation=representation,
    )
