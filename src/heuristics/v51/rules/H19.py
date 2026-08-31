"""H19 - corner-radius to depth ratio (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H19(FeatureRule):
    """About tool deflection: a small corner radius forces a small, long tool."""

    h_id = "H19"
    USES_ATTRIBUTES = ("derived_ratio_min_corner_radius_to_depth",)
    USES_THRESHOLDS = ("t_corner_depth_ratio",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "derived_ratio_min_corner_radius_to_depth"),
            context.threshold("t_corner_depth_ratio"),
            "derived_ratio_min_corner_radius_to_depth",
            "t_corner_depth_ratio",
        )
