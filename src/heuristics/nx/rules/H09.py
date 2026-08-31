"""H09 (nx_v1) - how many hole-axis orientations a part needs."""

from __future__ import annotations

from heuristics.base import PartRule
from heuristics.nx._shared import ORIENTATION_D, ATTR_OK, aggregate_count_limit, distinct_axes
from heuristics.nx.registry import register


@register
class H09(PartRule):
    """The STEP family only: the HOLES family is milled, not drilled.

    Node 19a restated the domain because the frozen condition says "over holes
    in part" while the v5.1 domain also held bosses and pockets.  All six
    HOLES display names end in "Through" - they are milled through-cavities,
    and the number of setups a drill needs says nothing about them.
    """

    h_id = "H09"
    USES_ATTRIBUTES = ORIENTATION_D
    USES_THRESHOLDS = ("t_hole_axis_variety", "tol_axis")

    def evaluate_part(self, context):
        nodes = context.domain_nodes(self.h_id)
        if not nodes:
            return None
        axes, gaps, members = [], 0, []
        for node in nodes:
            members.append(node.nodeId)
            state, _name, _reason = context.gate_attributes(node, ORIENTATION_D)
            axis = context.axis(node) if state == ATTR_OK else None
            if axis is None:
                gaps += 1
                continue
            axes.append(axis)
        clusters = distinct_axes(axes, context.threshold("tol_axis"))
        return aggregate_count_limit(
            len(clusters), context.threshold("t_hole_axis_variety"),
            gap_count=gaps,
            values={"holeCount": len(nodes)},
            evidence={"memberNodeIds": members,
                      "orientations": [cluster[0] for cluster in clusters]},
            quantity="distinct hole axis orientations",
        )
