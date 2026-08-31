"""H23 (nx_v1) - slot depth-to-width ratio."""

from __future__ import annotations

from heuristics.base import FeatureRule, not_evaluable
from heuristics.nx._shared import compare_at_most
from heuristics.nx.registry import register


@register
class H23(FeatureRule):
    """The same nine classes as H17.

    On ``SLOT_90_DEGREE`` the ratio is fixed at 0.5 by the 90-degree section
    itself, so the rule is met there by construction.  It is still evaluated
    rather than special-cased: a rule that is satisfied by the shape of the
    class is a fact worth publishing, and hiding it would make the class look
    as if it were out of the rule's domain.
    """

    h_id = "H23"
    USES_ATTRIBUTES = ("DEPTH", "WIDTH")
    USES_THRESHOLDS = ("t_slot_ld",)

    def evaluate_feature(self, node, context):
        depth = context.attribute(node, "DEPTH")
        width = context.attribute(node, "WIDTH")
        if width <= 0.0:
            return not_evaluable(
                f"WIDTH {width} is not positive, so no ratio exists",
                {"DEPTH": depth, "WIDTH": width}, {"blocked": "attributeGap"})
        return compare_at_most(
            round(depth / width, 6), context.threshold("t_slot_ld"),
            "depthToWidth", "t_slot_ld", {"DEPTH": depth, "WIDTH": width},
        )
