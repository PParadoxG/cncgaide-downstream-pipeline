"""H34 - limit the number of distinct fillet radii (advisory, part aggregate)."""

from __future__ import annotations

from heuristics.v51._shared import ATTR_OK, aggregate_count_limit, distinct
from heuristics.base import PartRule
from heuristics.v51.registry import register


@register
class H34(PartRule):
    """Clustered within a tolerance, and a null makes the count a lower bound.

    Exact float equality would split one radius into several; the aggregate
    missing-value rule then decides whether the lower bound is already
    enough to fail (it is monotone) or whether the count is unknown."""

    h_id = "H34"
    USES_ATTRIBUTES = ("fillet_radius",)
    USES_THRESHOLDS = ("t_fillet_variety", "tol_fillet_radius")

    def evaluate_part(self, context):
        nodes = context.domain_nodes(self.h_id)
        if not nodes:
            return None
        values, gaps, members = [], 0, []
        for node in nodes:
            members.append(node.nodeId)
            state, _ = context.attribute_state(node, "fillet_radius")
            if state != ATTR_OK:
                gaps += 1
                continue
            values.append(context.attribute(node, "fillet_radius"))
        clusters = distinct(values, context.threshold("tol_fillet_radius"))
        return aggregate_count_limit(
            len(clusters),
            context.threshold("t_fillet_variety"),
            gap_count=gaps,
            values={"instanceCount": len(nodes)},
            evidence={
                "memberNodeIds": members,
                "clusters": [[round(value, 6) for value in cluster] for cluster in clusters],
            },
            quantity="distinct fillet radii in the part",
        )
