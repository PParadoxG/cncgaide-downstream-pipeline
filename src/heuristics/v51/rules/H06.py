"""H06 - minimum circular-boss diameter (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H06(FeatureRule):
    """Stiffness bound on the *thinnest* station of a possibly tapered boss."""

    h_id = "H06"
    USES_ATTRIBUTES = ("base_diameter", "top_diameter")
    USES_THRESHOLDS = ("t_boss_diameter",)

    def evaluate_feature(self, node, context):
        base = context.attribute(node, "base_diameter")
        top = context.attribute(node, "top_diameter")
        return compare_at_least(
            min(base, top),
            context.threshold("t_boss_diameter"),
            "min(base_diameter, top_diameter)",
            "t_boss_diameter",
            {"base_diameter": base, "top_diameter": top},
        )
