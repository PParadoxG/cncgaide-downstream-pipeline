"""H38 - rectangular-boss slenderness (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_most
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H38(FeatureRule):
    """The ratio already uses the most restrictive thickness station."""

    h_id = "H38"
    USES_ATTRIBUTES = ("derived_ratio_height_to_thickness",)
    USES_THRESHOLDS = ("t_boss_slenderness",)

    def evaluate_feature(self, node, context):
        return compare_at_most(
            context.attribute(node, "derived_ratio_height_to_thickness"),
            context.threshold("t_boss_slenderness"),
            "derived_ratio_height_to_thickness",
            "t_boss_slenderness",
        )
