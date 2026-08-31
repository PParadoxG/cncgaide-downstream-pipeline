"""Leaf-class substitution cost.

Catalogue 1.1: the substitution cost is carried by an explicit matrix, not
by tree distance in the class hierarchy.  The hierarchy only provides the
*defaults* below; ``OVERRIDES`` is where manufacturing-semantic exceptions go
and is the place the thesis' "ontology-driven cost" iteration edits.
"""

from __future__ import annotations

from ged.contract.attribute_policy import FAMILY_OF, LEAF_TYPES, SUBFAMILY_OF


SAME_LEAF = 0.0
SAME_FAMILY = 0.25
CROSS_FAMILY = 1.0

# Recess sub-families differ more than leaves within one sub-family.
RECESS_CROSS: dict[frozenset[str], float] = {
    frozenset(("Slot", "Pocket")): 0.6,
    frozenset(("Pocket", "ThroughCut")): 0.5,
    frozenset(("Slot", "ThroughCut")): 0.8,
}

# Explicit pair overrides (cost layer).  Empty in v0; kept as the single
# documented place to encode exceptions such as "Counterbore ↔ Countersink
# should be cheaper than BlindHole ↔ ThroughHole".
OVERRIDES: dict[frozenset[str], float] = {}


def type_cost(left: str, right: str) -> float:
    if left == right:
        return SAME_LEAF
    key = frozenset((left, right))
    if key in OVERRIDES:
        return OVERRIDES[key]
    left_family = FAMILY_OF.get(left)
    right_family = FAMILY_OF.get(right)
    if left_family is None or right_family is None or left_family != right_family:
        return CROSS_FAMILY
    if left_family == "Recess":
        left_sub = SUBFAMILY_OF[left]
        right_sub = SUBFAMILY_OF[right]
        if left_sub == right_sub:
            return SAME_FAMILY
        return RECESS_CROSS[frozenset((left_sub, right_sub))]
    return SAME_FAMILY


def type_matrix() -> dict[str, dict[str, float]]:
    """Full leaf × leaf matrix, for reports and tests."""
    leaves = sorted(LEAF_TYPES)
    return {a: {b: type_cost(a, b) for b in leaves} for a in leaves}
