"""H53 - standard V-slot tool angle (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import member_of
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H53(FeatureRule):
    """A V groove is formed in one pass, so the tool point angle must exist."""

    h_id = "H53"
    USES_ATTRIBUTES = ("v_included_angle",)
    USES_THRESHOLDS = ("T_v_tool_angles", "tol_angle")

    def evaluate_feature(self, node, context):
        return member_of(
            context.attribute(node, "v_included_angle"),
            list(context.threshold("T_v_tool_angles")),
            context.threshold("tol_angle"),
            "v_included_angle",
            "T_v_tool_angles",
        )
