"""H12 - minimum web between two holes (hard, feature pair)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least, relation_missing
from heuristics.base import PairRule
from heuristics.v51.registry import register


@register
class H12(PairRule):
    """Remaining material, not axis spacing.

    v4.0 compared an axis spacing against a diameter - same dimension,
    different physical meaning.  Subtracting both radii is what leaves the
    material web the rule is actually about.
    """

    h_id = "H12"
    edge_type = "HOLE_SPACING"
    USES_ATTRIBUTES = ("diameter",)
    USES_RELATIONS = ("HOLE_SPACING",)
    USES_THRESHOLDS = ("t_hole_web",)

    def evaluate_pair(self, edge, source, target, context):
        spacing = (edge.properties or {}).get("axisSpacing")
        if spacing is None:
            return relation_missing(
                "HOLE_SPACING edge carries no axisSpacing", {"edgeId": edge.edgeId}
            )
        left = context.attribute(source, "diameter")
        right = context.attribute(target, "diameter")
        web = spacing - (left + right) / 2.0
        return compare_at_least(
            round(web, 6),
            context.threshold("t_hole_web"),
            "web",
            "t_hole_web",
            {"axisSpacing": spacing, "diameterA": left, "diameterB": right},
        )
