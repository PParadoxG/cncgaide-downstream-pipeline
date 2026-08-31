"""H37 (nx_v1) - machining clearance around a boss (hard, single feature)."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H37(FeatureRule):
    """Tier E: an ``nxd:`` kernel measurement read off the derived block.

    The rays are cast *outward* from the boss's own walls - the quantity is
    the free space a cutter has around the boss, and a boss no surface is
    reached from is a named refusal, not an infinite clearance.
    """

    h_id = "H37"
    USES_ATTRIBUTES = ("derived_min_clearance",)
    USES_THRESHOLDS = ("t_boss_clearance",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "derived_min_clearance"),
            context.threshold("t_boss_clearance"),
            "derived_min_clearance",
            "t_boss_clearance",
        )
