"""H10 - minimum pocket corner radius (hard, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H10(FeatureRule):
    """Both Pocket leaves; a radius of zero cannot be milled by a rotating tool."""

    h_id = "H10"
    USES_ATTRIBUTES = ("min_corner_radius",)
    USES_THRESHOLDS = ("t_corner_radius",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "min_corner_radius"),
            context.threshold("t_corner_radius"),
            "min_corner_radius",
            "t_corner_radius",
        )
