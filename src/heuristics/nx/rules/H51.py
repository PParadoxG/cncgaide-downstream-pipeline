"""H51 (nx_v1) - remaining wall thickness around a removal feature (hard, single feature)."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H51(FeatureRule):
    """Tier E: an ``nxd:`` kernel measurement read off the derived block.

    Complements H12, which only sees hole-to-hole webs along an edge; this
    one is against whatever material happens to be nearest a wall.
    """

    h_id = "H51"
    USES_ATTRIBUTES = ("derived_min_wall_thickness",)
    USES_THRESHOLDS = ("t_min_wall",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "derived_min_wall_thickness"),
            context.threshold("t_min_wall"),
            "derived_min_wall_thickness",
            "t_min_wall",
        )
