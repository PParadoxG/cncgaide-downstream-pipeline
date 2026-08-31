"""H09 - uniform hole axis orientation (advisory, part-level aggregate)."""

from __future__ import annotations

from heuristics.v51._shared import ATTR_OK, aggregate_count_limit, distinct_axes, unit
from heuristics.base import PartRule
from heuristics.v51.registry import register


@register
class H09(PartRule):
    """Counts orientations rather than comparing pairs.

    The pairwise reading of this rule degenerates on real parts: same-host
    hole pairs are parallel by construction (H27 already guarantees it), so
    the informative quantity is how many different hole-axis orientations a
    part needs - that is what drives the number of setups.
    """

    h_id = "H09"
    USES_ATTRIBUTES = ("feature_axis",)
    USES_THRESHOLDS = ("t_hole_axis_variety", "tol_axis")

    def evaluate_part(self, context):
        nodes = context.domain_nodes(self.h_id)
        if not nodes:
            return None
        axes, gaps, members = [], 0, []
        for node in nodes:
            members.append(node.nodeId)
            state, _ = context.attribute_state(node, "feature_axis")
            axis = unit(context.attribute(node, "feature_axis")) if state == ATTR_OK else None
            if axis is None:
                gaps += 1
                continue
            axes.append(axis)
        clusters = distinct_axes(axes, context.threshold("tol_axis"))
        return aggregate_count_limit(
            len(clusters),
            context.threshold("t_hole_axis_variety"),
            gap_count=gaps,
            values={"holeCount": len(nodes)},
            evidence={
                "memberNodeIds": members,
                "orientations": [cluster[0] for cluster in clusters],
            },
            quantity="distinct hole axis orientations",
        )
