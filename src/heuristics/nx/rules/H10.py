"""H10 (nx_v1) - minimum concave corner radius of a pocket."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H10(FeatureRule):
    """Three of the five classes that declare MIN_RADIUS.

    ``HOLE_FREE_SHAPED_STRAIGHT`` is a through-cavity and belongs to H43 at
    its own threshold; ``WEDM_FREE_SHAPED_STRAIGHT`` is wire-cut, where the
    corner radius follows the wire and not a cutter.
    """

    h_id = "H10"
    USES_ATTRIBUTES = ("MIN_RADIUS",)
    USES_THRESHOLDS = ("t_corner_radius",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "MIN_RADIUS"), context.threshold("t_corner_radius"),
            "MIN_RADIUS", "t_corner_radius",
        )
