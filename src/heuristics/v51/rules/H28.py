"""H28 - remaining blind-hole floor thickness (hard, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H28(FeatureRule):
    """Dual of H11; a shared threshold symbol keeps the two numbers from drifting."""

    h_id = "H28"
    USES_ATTRIBUTES = ("derived_floor_thickness",)
    USES_THRESHOLDS = ("t_floor",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "derived_floor_thickness"),
            context.threshold("t_floor"),
            "derived_floor_thickness",
            "t_floor",
        )
