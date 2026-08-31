"""Fixed 1:1 aggregate substitution (Surface background layer, unclaimed residue)."""

from __future__ import annotations

from ged.contract.schema import GedFixedNode
from .config import FixedCostParts
from .node_cost import bounded_diff


# Share of the W_RES budget each aggregate may consume; shares sum to 1 so
# the combined raw-cost ceiling is exactly W_RES.
BUDGET_SHARE: dict[str, float] = {
    "PlanarFaceAggregate": 0.5,
    "FreeformFaceAggregate": 0.2,
    "UnclaimedAggregate": 0.3,
}

ATTRIBUTE_WEIGHTS: dict[str, dict[str, float]] = {
    "PlanarFaceAggregate": {"face_count": 0.4, "area": 0.4, "normal_classes": 0.2},
    "FreeformFaceAggregate": {"face_count": 0.5, "area": 0.5},
    "UnclaimedAggregate": {"face_count": 0.5, "area": 0.5},
}


def make_fixed_node_cost(w_res: float, *, shares=None, attribute_weights=None):
    """The aggregate cost for one representation's background layer.

    The two tables are a parameter because the background layer is a
    representation decision: v5.1 has a planar and a freeform aggregate, nx_v1
    has one per class of NX's ``SURFACES`` family.  Both spend the same total
    ``W_RES``, because the shares sum to 1 either way.
    """
    shares = BUDGET_SHARE if shares is None else dict(shares)
    attribute_weights = ATTRIBUTE_WEIGHTS if attribute_weights is None else dict(attribute_weights)

    def fixed_node_cost(left: GedFixedNode, right: GedFixedNode) -> FixedCostParts:
        if left.type_key != right.type_key:
            raise ValueError(f"fixed node type mismatch: {left.type_key} vs {right.type_key}")
        share = shares.get(left.type_key)
        weights = attribute_weights.get(left.type_key)
        if share is None or weights is None:
            raise ValueError(f"no fixed-node cost registered for {left.type_key}")
        differences: dict[str, float] = {}
        total = 0.0
        weight_sum = 0.0
        for name, weight in weights.items():
            if name not in left.attributes or name not in right.attributes:
                continue
            distance = bounded_diff(float(left.attributes[name]), float(right.attributes[name]))
            differences[name] = distance
            total += weight * distance
            weight_sum += weight
        mixture = total / weight_sum if weight_sum else 0.0
        return FixedCostParts(cost=w_res * share * mixture, differences=differences)

    return fixed_node_cost


def fixed_cost_upper_bound(w_res: float, *, shares=None) -> float:
    return w_res * sum((BUDGET_SHARE if shares is None else shares).values())
