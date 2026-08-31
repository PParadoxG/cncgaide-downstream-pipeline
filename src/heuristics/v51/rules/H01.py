"""H01 - hole depth-to-diameter ratio (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_most
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H01(FeatureRule):
    """Depth over base-hole diameter, one formula for all four Hole leaves."""

    h_id = "H01"
    USES_ATTRIBUTES = ("derived_ratio_depth_to_diameter",)
    USES_THRESHOLDS = ("t_hole_ld",)

    def evaluate_feature(self, node, context):
        return compare_at_most(
            context.attribute(node, "derived_ratio_depth_to_diameter"),
            context.threshold("t_hole_ld"),
            "derived_ratio_depth_to_diameter",
            "t_hole_ld",
        )
