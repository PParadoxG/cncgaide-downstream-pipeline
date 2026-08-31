"""H07 (nx_v1) - round-boss height against its diameter."""

from __future__ import annotations

from heuristics.base import FeatureRule, not_evaluable
from heuristics.nx._shared import compare_at_most
from heuristics.nx.registry import register


@register
class H07(FeatureRule):
    """DEPTH is the boss height on this class; the limit scales with DIAMETER."""

    h_id = "H07"
    USES_ATTRIBUTES = ("DEPTH", "DIAMETER")
    USES_THRESHOLDS = ("t_boss_hd",)

    def evaluate_feature(self, node, context):
        depth = context.attribute(node, "DEPTH")
        diameter = context.attribute(node, "DIAMETER")
        if diameter <= 0.0:
            return not_evaluable(
                f"DIAMETER {diameter} is not positive, so the limit has no value",
                {"DEPTH": depth, "DIAMETER": diameter}, {"blocked": "attributeGap"})
        factor = context.threshold("t_boss_hd")
        return compare_at_most(
            depth, round(factor * diameter, 6), "DEPTH", "t_boss_hd * DIAMETER",
            {"DIAMETER": diameter, "heightToDiameter": round(depth / diameter, 6)},
        )
