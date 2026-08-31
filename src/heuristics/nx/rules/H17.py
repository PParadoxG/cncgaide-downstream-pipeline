"""H17 (nx_v1) - minimum slot width."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H17(FeatureRule):
    """The nine SLOT classes that declare a plain WIDTH.

    On those nine, WIDTH is the width at the opening - one meaning, the
    V-section ones included.  ``SLOT_ROUND`` and ``SLOT_U_SHAPED`` publish no
    width at all, and ``SLOT_DOVE_TAIL`` / ``SLOT_T_SHAPED`` publish WIDTH_1
    and WIDTH_2 without saying which is which; node 19a left all four out
    rather than guess.
    """

    h_id = "H17"
    USES_ATTRIBUTES = ("WIDTH",)
    USES_THRESHOLDS = ("t_slot_width",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "WIDTH"), context.threshold("t_slot_width"),
            "WIDTH", "t_slot_width",
        )
