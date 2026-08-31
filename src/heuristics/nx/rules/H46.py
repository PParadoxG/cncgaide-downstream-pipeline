"""H46 (nx_v1) - crossing angle of two intersecting holes."""

from __future__ import annotations

from heuristics.base import PairRule
from heuristics.nx._shared import compare_at_least, relation_missing
from heuristics.nx.registry import register


@register
class H46(PairRule):
    """INTERSECTS requires a shared wall edge, not merely crossing axes."""

    h_id = "H46"
    edge_type = "INTERSECTS"
    USES_RELATIONS = ("INTERSECTS",)
    USES_THRESHOLDS = ("t_crossing_angle",)

    def evaluate_pair(self, edge, source, target, context):
        value = (edge.properties or {}).get("crossingAngle")
        if value is None:
            return relation_missing("INTERSECTS edge carries no crossingAngle",
                                    {"edgeId": edge.edgeId})
        return compare_at_least(value, context.threshold("t_crossing_angle"),
                                "crossingAngle", "t_crossing_angle")
