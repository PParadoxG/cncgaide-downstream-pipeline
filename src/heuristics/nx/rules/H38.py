"""H38 (nx_v1) - rectangular-boss slenderness."""

from __future__ import annotations

from heuristics.base import FeatureRule, not_evaluable
from heuristics.nx._shared import compare_at_most
from heuristics.nx.registry import register


@register
class H38(FeatureRule):
    """A tall boss on a thin footprint chatters; the thinner station governs."""

    h_id = "H38"
    USES_ATTRIBUTES = ("DEPTH", "WIDTH", "LENGTH")
    USES_THRESHOLDS = ("t_boss_slenderness",)

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
            round(depth / station, 6), context.threshold("t_boss_slenderness"),
            "heightToMinThickness", "t_boss_slenderness",
            {"DEPTH": depth, "WIDTH": width, "LENGTH": length, "minThickness": station},
        )
