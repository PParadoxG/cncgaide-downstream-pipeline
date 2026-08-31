"""H11 (nx_v1) - remaining floor thickness below a milled pocket (hard, single feature)."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H11(FeatureRule):
    """Tier E: the value is an ``nxd:`` kernel measurement, not an NX word.

    The session that built the part measured it into the node's ``derived``
    block; the central gate reads that block through ``derived_state``, so a
    refused or missing measurement arrives here already turned into a named
    NOT_EVALUABLE and this rule only ever compares.
    """

    h_id = "H11"
    USES_ATTRIBUTES = ("derived_floor_thickness",)
    USES_THRESHOLDS = ("t_floor",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "derived_floor_thickness"),
            context.threshold("t_floor"),
            "derived_floor_thickness",
            "t_floor",
        )
