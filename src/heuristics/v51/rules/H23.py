"""H23 - slot depth-to-width ratio (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_most
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H23(FeatureRule):
    """RectangularSlot only: on converging sections the ratio is not free."""

    h_id = "H23"
    USES_ATTRIBUTES = ("derived_ratio_depth_to_min_width",)
    USES_THRESHOLDS = ("t_slot_ld",)

    def evaluate_feature(self, node, context):
        return compare_at_most(
            context.attribute(node, "derived_ratio_depth_to_min_width"),
            context.threshold("t_slot_ld"),
            "derived_ratio_depth_to_min_width",
            "t_slot_ld",
        )
