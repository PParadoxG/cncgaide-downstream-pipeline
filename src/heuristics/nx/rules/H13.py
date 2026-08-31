"""H13 (nx_v1) - a hole-bearing round boss carries a coaxial hole."""

from __future__ import annotations

from heuristics.base import NA_FALSE_ANTECEDENT, FeatureRule, failed, not_applicable, passed
from heuristics.nx._shared import is_hole_class, relation_missing
from heuristics.nx.registry import register


@register
class H13(FeatureRule):
    """The subject is the boss, and the premise is that it carries a hole.

    The consequent asks for *at least one* coaxial hole: a flange with one
    centre hole and four bolt holes is a normal design, and nothing in what NX
    published separates a centre hole from a bolt hole that merely opens on
    the same face.
    """

    h_id = "H13"
    USES_RELATIONS = ("LOCATED_ON", "COAXIAL")

    def required_attributes(self, feature_type):
        return ()

    def evaluate_feature(self, node, context):
        holes = [terminal for terminal in context.terminals_on(node.nodeId)
                 if is_hole_class(terminal["sourceFeatureType"])]
        if not holes:
            if context.rejected_terminals_for_host(node.nodeId):
                return relation_missing(
                    "a LOCATED_ON terminal naming this boss as a candidate host was not "
                    "proven, so whether it carries a hole is undecided",
                    {"bossNodeId": node.nodeId},
                )
            return not_applicable("no hole is located on this boss", NA_FALSE_ANTECEDENT,
                                  {"holeCount": 0})
        hole_ids = sorted({terminal["sourceNodeId"] for terminal in holes})
        coaxial = [hole for hole in hole_ids if context.has_edge(hole, node.nodeId, "COAXIAL")]
        values = {"holeCount": len(hole_ids), "coaxialHoleCount": len(coaxial)}
        evidence = {"holeNodeIds": hole_ids, "coaxialHoleNodeIds": coaxial}
        if coaxial:
            return passed(
                f"{len(coaxial)} of {len(hole_ids)} hole(s) on this boss are coaxial with it",
                values, evidence)
        return failed(
            f"the boss carries {len(hole_ids)} hole(s), none coaxial with it", values, evidence)
