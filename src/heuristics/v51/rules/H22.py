"""H22 - entry blind hole for a deep pocket (advisory, conditional)."""

from __future__ import annotations

from heuristics.base import NA_FALSE_ANTECEDENT, FeatureRule, failed, not_applicable, passed
from heuristics.v51.registry import register


@register
class H22(FeatureRule):
    """The entry hole must open *on the pocket floor*, which is LOCATED_ON.

    DIRECT_ADJACENT would also accept a blind hole outside the pocket that
    happens to touch a wall, and such a hole is no help when plunging.  The
    closed-state half of the v4.0 premise is gone: the Pocket leaf criteria
    are a closed side-wall band, so the domain contains closed pockets only.
    """

    h_id = "H22"
    USES_ATTRIBUTES = ("depth",)
    USES_RELATIONS = ("LOCATED_ON",)
    USES_THRESHOLDS = ("t_pocket_entry_depth",)

    def evaluate_feature(self, node, context):
        depth = context.attribute(node, "depth")
        limit = context.threshold("t_pocket_entry_depth")
        values = {"depth": depth, "t_pocket_entry_depth": limit}
        if depth < limit:
            return not_applicable(
                f"depth {round(depth, 6)} < t_pocket_entry_depth {limit}: the pocket is not deep "
                "enough for the rule to say anything",
                NA_FALSE_ANTECEDENT,
                values,
            )
        entries = sorted({
            terminal["sourceNodeId"] for terminal in context.terminals_on(node.nodeId)
            if terminal["sourceFeatureType"] == "BlindHole"
        })
        values["entryBlindHoleCount"] = len(entries)
        if entries:
            return passed(
                f"{len(entries)} blind hole(s) are located on this pocket",
                values, {"entryBlindHoleNodeIds": entries},
            )
        return failed(
            "the pocket is deep enough to need a plunge entry but carries no blind hole", values
        )
