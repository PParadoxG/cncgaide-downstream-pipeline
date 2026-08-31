"""H43 (nx_v1) - minimum corner radius of a through cut."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H43(FeatureRule):
    """A through-cavity, at its own threshold.

    The through / blind split here is NX's own (node 19a D2): all six HOLES
    display names end in "Through" and none of the seven POCKETS ones do.
    """

    h_id = "H43"
    USES_ATTRIBUTES = ("MIN_RADIUS",)
    USES_THRESHOLDS = ("t_throughcut_corner_radius",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "MIN_RADIUS"),
            context.threshold("t_throughcut_corner_radius"),
            "MIN_RADIUS", "t_throughcut_corner_radius",
        )
