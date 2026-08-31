"""H51 - remaining wall thickness around a feature (hard, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H51(FeatureRule):
    """Revived v3.1 H29; complements H12, which only sees hole-to-hole webs."""

    h_id = "H51"
    USES_ATTRIBUTES = ("derived_min_wall_thickness",)
    USES_THRESHOLDS = ("t_min_wall",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "derived_min_wall_thickness"),
            context.threshold("t_min_wall"),
            "derived_min_wall_thickness",
            "t_min_wall",
        )
