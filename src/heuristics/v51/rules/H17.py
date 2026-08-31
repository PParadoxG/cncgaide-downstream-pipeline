"""H17 - minimum slot width (hard, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H17(FeatureRule):
    """Both width stations bound the tool, so the minimum of the two decides.

    RectangularSlot only: the triangular and circular sections are not formed
    by an end mill at all, so the end-mill premise of this rule does not hold
    for them (they carry H53 / their own form heuristics instead).
    """

    h_id = "H17"
    USES_ATTRIBUTES = ("min_opening_width", "min_bottom_width")
    USES_THRESHOLDS = ("t_slot_width",)

    def evaluate_feature(self, node, context):
        opening = context.attribute(node, "min_opening_width")
        bottom = context.attribute(node, "min_bottom_width")
        return compare_at_least(
            min(opening, bottom),
            context.threshold("t_slot_width"),
            "min(min_opening_width, min_bottom_width)",
            "t_slot_width",
            {"min_opening_width": opening, "min_bottom_width": bottom},
        )
