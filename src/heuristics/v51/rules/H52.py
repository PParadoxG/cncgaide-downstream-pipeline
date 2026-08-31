"""H52 - standard countersink angle (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import member_of
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H52(FeatureRule):
    """DIN 335 cone angles; matching one of them keeps a stock countersink usable."""

    h_id = "H52"
    USES_ATTRIBUTES = ("derived_countersink_angle",)
    USES_THRESHOLDS = ("T_countersink_angles", "tol_angle")

    def evaluate_feature(self, node, context):
        return member_of(
            context.attribute(node, "derived_countersink_angle"),
            list(context.threshold("T_countersink_angles")),
            context.threshold("tol_angle"),
            "derived_countersink_angle",
            "T_countersink_angles",
        )
