"""H37 - machining clearance around a boss (hard, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H37(FeatureRule):
    """Feature-level scalar; no pairwise distance relation is required."""

    h_id = "H37"
    USES_ATTRIBUTES = ("derived_min_clearance",)
    USES_THRESHOLDS = ("t_boss_clearance",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "derived_min_clearance"),
            context.threshold("t_boss_clearance"),
            "derived_min_clearance",
            "t_boss_clearance",
        )
