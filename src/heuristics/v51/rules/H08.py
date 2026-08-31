"""H08 - available drill diameter (advisory, single feature, needs the magazine)."""

from __future__ import annotations

from heuristics.v51._shared import external_missing, member_of
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H08(FeatureRule):
    """Geometry side is complete; only the magazine list is external.

    The pipeline never invents a tool list, so without ``EXT.drill_diameters``
    every instance is NOT_EVALUABLE for an external reason - which the report
    keeps apart from a geometric gap.
    """

    h_id = "H08"
    USES_ATTRIBUTES = ("diameter",)
    USES_THRESHOLDS = ("tol_diameter",)
    EXTERNAL_INPUT = "drill_diameters"

    def evaluate_feature(self, node, context):
        magazine = context.external_inputs.get(self.EXTERNAL_INPUT)
        if not magazine:
            return external_missing(self.EXTERNAL_INPUT)
        return member_of(
            context.attribute(node, "diameter"),
            list(magazine),
            context.threshold("tol_diameter"),
            "diameter",
            "EXT.drill_diameters",
        )
