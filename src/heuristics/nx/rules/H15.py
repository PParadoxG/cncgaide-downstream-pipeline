"""H15 (nx_v1) - side-wall clearance between two pockets (hard, pair)."""

from __future__ import annotations

from heuristics.base import PairRule
from heuristics.nx._shared import compare_at_least, relation_missing
from heuristics.nx.registry import register


@register
class H15(PairRule):
    """One unit per POCKET_CLEARANCE edge - the relation layer measured it.

    The clearance rides on the edge because it is a fact about the *pair*,
    measured through the kernel in the session that built the part; a pair
    the relation layer could not measure has no edge and is not a unit, and
    the relation diagnostics already say why.
    """

    h_id = "H15"
    edge_type = "POCKET_CLEARANCE"
    USES_RELATIONS = ("POCKET_CLEARANCE",)
    USES_THRESHOLDS = ("t_pocket_gap",)

    def evaluate_pair(self, edge, source, target, context):
        clearance = (edge.properties or {}).get("pocketClearance")
        if clearance is None:
            return relation_missing("POCKET_CLEARANCE edge carries no pocketClearance",
                                    {"edgeId": edge.edgeId})
        return compare_at_least(
            clearance, context.threshold("t_pocket_gap"),
            "pocketClearance", "t_pocket_gap",
        )
