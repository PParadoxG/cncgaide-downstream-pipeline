"""H03 (nx_v1) - pocket depth over its most restrictive width."""

from __future__ import annotations

from heuristics.base import FeatureRule, not_evaluable
from heuristics.nx._shared import compare_at_most
from heuristics.nx.registry import register


@register
class H03(FeatureRule):
    """``POCKET_RECTANGULAR_STRAIGHT`` is the one POCKET class with a WIDTH.

    It declares LENGTH too, and the station a cutter is restricted by is the
    smaller of the two - which is why the condition takes a minimum rather
    than reading WIDTH alone.
    """

    h_id = "H03"
    USES_ATTRIBUTES = ("DEPTH", "WIDTH", "LENGTH")
    USES_THRESHOLDS = ("t_pocket_ld",)

    def evaluate_feature(self, node, context):
        depth = context.attribute(node, "DEPTH")
        width = context.attribute(node, "WIDTH")
        length = context.attribute(node, "LENGTH")
        station = min(width, length)
        if station <= 0.0:
            return not_evaluable(
                f"min(WIDTH {width}, LENGTH {length}) is not positive, so no ratio exists",
                {"WIDTH": width, "LENGTH": length}, {"blocked": "attributeGap"})
        return compare_at_most(
            round(depth / station, 6), context.threshold("t_pocket_ld"),
            "depthToMinWidth", "t_pocket_ld",
            {"DEPTH": depth, "WIDTH": width, "LENGTH": length, "restrictiveWidth": station},
        )
