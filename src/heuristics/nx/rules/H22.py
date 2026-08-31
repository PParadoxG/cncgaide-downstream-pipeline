"""H22 (nx_v1) - entry hole for a deep pocket."""

from __future__ import annotations

from fnmatch import fnmatchcase

from heuristics.base import NA_FALSE_ANTECEDENT, FeatureRule, failed, not_applicable, passed
from heuristics.nx.registry import register

# NX's own name for a blind stepped bore: the STEP family's POCKET half.
ENTRY_PATTERN = "STEP*POCKET"


@register
class H22(FeatureRule):
    """The entry hole must open *on the pocket*, which is what LOCATED_ON says.

    Adjacency would also accept a hole outside the pocket that happens to
    touch a wall, and such a hole is no help when plunging.
    """

    h_id = "H22"
    USES_ATTRIBUTES = ("DEPTH",)
    USES_RELATIONS = ("LOCATED_ON",)
    USES_THRESHOLDS = ("t_pocket_entry_depth",)

    def evaluate_feature(self, node, context):
        depth = context.attribute(node, "DEPTH")
        limit = context.threshold("t_pocket_entry_depth")
        values = {"DEPTH": depth, "t_pocket_entry_depth": limit}
        if depth < limit:
            return not_applicable(
                f"DEPTH {round(depth, 6)} < t_pocket_entry_depth {limit}: the pocket is not "
                "deep enough for the rule to say anything",
                NA_FALSE_ANTECEDENT, values)
        entries = sorted({
            terminal["sourceNodeId"] for terminal in context.terminals_on(node.nodeId)
            if fnmatchcase(str(terminal["sourceFeatureType"]), ENTRY_PATTERN)
        })
        values["entryHoleCount"] = len(entries)
        if entries:
            return passed(f"{len(entries)} blind stepped hole(s) are located on this pocket",
                          values, {"entryHoleNodeIds": entries})
        return failed(
            "the pocket is deep enough to need a plunge entry but carries no blind hole", values)
