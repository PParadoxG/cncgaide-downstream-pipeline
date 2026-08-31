"""H41 - minimum internal corner angle (hard, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H41(FeatureRule):
    """A null here means every concave corner is a full reversal, not a corner."""

    h_id = "H41"
    USES_ATTRIBUTES = ("min_corner_angle",)
    USES_THRESHOLDS = ("t_corner_angle",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "min_corner_angle"),
            context.threshold("t_corner_angle"),
            "min_corner_angle",
            "t_corner_angle",
        )
