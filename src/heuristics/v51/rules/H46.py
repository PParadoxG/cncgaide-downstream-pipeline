"""H46 - intersecting-hole crossing angle (hard, feature pair)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least, relation_missing
from heuristics.base import PairRule
from heuristics.v51.registry import register


@register
class H46(PairRule):
    """INTERSECTS requires proven principal-wall contact, not merely crossing axes."""

    h_id = "H46"
    edge_type = "INTERSECTS"
    USES_RELATIONS = ("INTERSECTS",)
    USES_THRESHOLDS = ("t_crossing_angle",)

    def evaluate_pair(self, edge, source, target, context):
        value = (edge.properties or {}).get("crossingAngle")
        if value is None:
            return relation_missing(
                "INTERSECTS edge carries no crossingAngle", {"edgeId": edge.edgeId}
            )
        return compare_at_least(value, context.threshold("t_crossing_angle"), "crossingAngle", "t_crossing_angle")
