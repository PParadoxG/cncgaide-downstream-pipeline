"""H16 - slot-to-slot clearance (hard, feature pair)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least, relation_missing
from heuristics.base import PairRule
from heuristics.v51.registry import register


@register
class H16(PairRule):
    """One unit per materialized edge; a missing edge is not a violation."""

    h_id = "H16"
    edge_type = "SLOT_CLEARANCE"
    USES_RELATIONS = ("SLOT_CLEARANCE",)
    USES_THRESHOLDS = ("t_slot_gap",)

    def evaluate_pair(self, edge, source, target, context):
        value = (edge.properties or {}).get("slotClearance")
        if value is None:
            return relation_missing(
                "SLOT_CLEARANCE edge carries no slotClearance", {"edgeId": edge.edgeId}
            )
        return compare_at_least(value, context.threshold("t_slot_gap"), "slotClearance", "t_slot_gap")
