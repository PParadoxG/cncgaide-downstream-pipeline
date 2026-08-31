"""H49 - limit the number of distinct hole diameters (advisory, part aggregate)."""

from __future__ import annotations

from heuristics.v51._shared import ATTR_OK, aggregate_count_limit, distinct
from heuristics.base import PartRule
from heuristics.v51.registry import register


@register
class H49(PartRule):
    """Domain deliberately unchanged from v4.0, so the threshold keeps its basis.

    Compound holes are drilled with a base drill too, but widening the domain
    would move the calibration of the limit; that is the collaborator's call."""

    h_id = "H49"
    USES_ATTRIBUTES = ("diameter",)
    USES_THRESHOLDS = ("t_diameter_variety", "tol_diameter")

    def evaluate_part(self, context):
        nodes = context.domain_nodes(self.h_id)
        if not nodes:
            return None
        values, gaps, members = [], 0, []
        for node in nodes:
            members.append(node.nodeId)
            state, _ = context.attribute_state(node, "diameter")
            if state != ATTR_OK:
                gaps += 1
                continue
            values.append(context.attribute(node, "diameter"))
        clusters = distinct(values, context.threshold("tol_diameter"))
        return aggregate_count_limit(
            len(clusters),
            context.threshold("t_diameter_variety"),
            gap_count=gaps,
            values={"instanceCount": len(nodes)},
            evidence={
                "memberNodeIds": members,
                "clusters": [[round(value, 6) for value in cluster] for cluster in clusters],
            },
            quantity="distinct through- and blind-hole diameters in the part",
        )
