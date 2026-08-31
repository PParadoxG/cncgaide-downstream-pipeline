"""H15 - pocket-to-pocket clearance (hard, feature pair)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least, relation_missing
from heuristics.base import PairRule
from heuristics.v51.registry import register


@register
class H15(PairRule):
    """One unit per materialized edge; a missing edge is not a violation."""

    h_id = "H15"
    edge_type = "POCKET_CLEARANCE"
    USES_RELATIONS = ("POCKET_CLEARANCE",)
    USES_THRESHOLDS = ("t_pocket_gap",)

    def evaluate_pair(self, edge, source, target, context):
        value = (edge.properties or {}).get("pocketClearance")
        if value is None:
            return relation_missing(
                "POCKET_CLEARANCE edge carries no pocketClearance", {"edgeId": edge.edgeId}
            )
        return compare_at_least(value, context.threshold("t_pocket_gap"), "pocketClearance", "t_pocket_gap")
