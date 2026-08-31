"""H05 - minimum fillet radius (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H05(FeatureRule):
    """A radius below the bound needs a smaller edge-rounding tool than stock."""

    h_id = "H05"
    USES_ATTRIBUTES = ("fillet_radius",)
    USES_THRESHOLDS = ("t_fillet_radius",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "fillet_radius"),
            context.threshold("t_fillet_radius"),
            "fillet_radius",
            "t_fillet_radius",
        )
