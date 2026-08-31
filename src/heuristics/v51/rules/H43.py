"""H43 - through-cut corner radius (hard, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H43(FeatureRule):
    """Same attribute as H10 on ThroughCut, with its own, larger threshold."""

    h_id = "H43"
    USES_ATTRIBUTES = ("min_corner_radius",)
    USES_THRESHOLDS = ("t_throughcut_corner_radius",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "min_corner_radius"),
            context.threshold("t_throughcut_corner_radius"),
            "min_corner_radius",
            "t_throughcut_corner_radius",
        )
