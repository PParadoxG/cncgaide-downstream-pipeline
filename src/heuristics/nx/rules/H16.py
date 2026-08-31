"""H16 (nx_v1) - clearance between two slots (hard, pair)."""

from __future__ import annotations

from heuristics.base import PairRule
from heuristics.nx._shared import compare_at_least, relation_missing
from heuristics.nx.registry import register


@register
class H16(PairRule):
    """One unit per SLOT_CLEARANCE edge - the relation layer measured it."""

    h_id = "H16"
    edge_type = "SLOT_CLEARANCE"
    USES_RELATIONS = ("SLOT_CLEARANCE",)
    USES_THRESHOLDS = ("t_slot_gap",)

    def evaluate_pair(self, edge, source, target, context):
        clearance = (edge.properties or {}).get("slotClearance")
        if clearance is None:
            return relation_missing("SLOT_CLEARANCE edge carries no slotClearance",
                                    {"edgeId": edge.edgeId})
        return compare_at_least(
            clearance, context.threshold("t_slot_gap"),
            "slotClearance", "t_slot_gap",
        )
