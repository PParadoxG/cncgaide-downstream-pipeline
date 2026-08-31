"""H35 - one chamfer angle throughout the part (advisory, part aggregate)."""

from __future__ import annotations

from heuristics.v51._shared import ATTR_OK, aggregate_count_limit, distinct
from heuristics.base import PartRule
from heuristics.v51.registry import register


@register
class H35(PartRule):
    """Angle only: the width is a programming parameter, not a tool identity.

    For a fixed-angle chamfer tool the bevel width follows from the depth of
    cut, so counting (angle, width) pairs would report one tool walking eight
    edges as five different tools - the opposite of what the rule means."""

    h_id = "H35"
    USES_ATTRIBUTES = ("min_chamfer_angle",)
    USES_THRESHOLDS = ("t_chamfer_variety", "tol_chamfer_angle")

    def evaluate_part(self, context):
        nodes = context.domain_nodes(self.h_id)
        if not nodes:
            return None
        values, gaps, members = [], 0, []
        for node in nodes:
            members.append(node.nodeId)
            state, _ = context.attribute_state(node, "min_chamfer_angle")
            if state != ATTR_OK:
                gaps += 1
                continue
            values.append(context.attribute(node, "min_chamfer_angle"))
        clusters = distinct(values, context.threshold("tol_chamfer_angle"))
        return aggregate_count_limit(
            len(clusters),
            context.threshold("t_chamfer_variety"),
            gap_count=gaps,
            values={"instanceCount": len(nodes)},
            evidence={
                "memberNodeIds": members,
                "clusters": [[round(value, 6) for value in cluster] for cluster in clusters],
            },
            quantity="distinct chamfer angles in the part",
        )
