"""H07 - circular-boss height-to-diameter ratio (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_most
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H07(FeatureRule):
    """Same station choice as H06: slenderness is bounded by the thinnest one."""

    h_id = "H07"
    USES_ATTRIBUTES = ("height", "base_diameter", "top_diameter")
    USES_THRESHOLDS = ("t_boss_hd",)

    def evaluate_feature(self, node, context):
        height = context.attribute(node, "height")
        base = context.attribute(node, "base_diameter")
        top = context.attribute(node, "top_diameter")
        factor = context.threshold("t_boss_hd")
        return compare_at_most(
            height,
            factor * min(base, top),
            "height",
            "t_boss_hd * min(base_diameter, top_diameter)",
            {"base_diameter": base, "top_diameter": top, "t_boss_hd": factor},
        )
