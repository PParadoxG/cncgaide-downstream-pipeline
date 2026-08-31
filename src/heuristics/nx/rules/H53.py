"""H53 (nx_v1) - standard V-slot included angle."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import member_of
from heuristics.nx.registry import register


@register
class H53(FeatureRule):
    """The three SLOT classes that declare ANGLE.

    ``SLOT_90_DEGREE`` declares none, because its class name fixes the angle
    at 90 degrees - which is already a standard form-tool angle, so it meets
    the rule by construction while sitting outside its domain.
    """

    h_id = "H53"
    USES_ATTRIBUTES = ("ANGLE",)
    USES_THRESHOLDS = ("T_v_tool_angles", "tol_angle")

    def evaluate_feature(self, node, context):
        return member_of(
            context.attribute(node, "ANGLE"),
            list(context.threshold("T_v_tool_angles")),
            context.threshold("tol_angle"),
            "ANGLE", "T_v_tool_angles",
        )
