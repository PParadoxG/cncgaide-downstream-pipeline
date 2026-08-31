"""H48 - limit the number of tool-axis orientations (advisory, part aggregate)."""

from __future__ import annotations

from heuristics.v51._shared import ATTR_OK, aggregate_count_limit, distinct_axes, unit
from heuristics.base import PartRule
from heuristics.v51.registry import register


@register
class H48(PartRule):
    """A lower bound on the number of setups, and it says so.

    Only feature axes are counted.  Face-milling directions were tested as an
    addition and rejected: including planar-face normals makes every part in
    the corpus exceed the limit, which turns the rule into a constant.
    ThroughCut is excluded from the domain because which of its two opening
    rims defines the axis depends on traversal order, and that ambiguity would
    leak straight into the count.
    """

    h_id = "H48"
    USES_ATTRIBUTES = ("feature_axis",)
    USES_THRESHOLDS = ("t_axis_variety", "tol_axis")

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
            context.threshold("t_axis_variety"),
            gap_count=gaps,
            values={"axisBearingFeatureCount": len(nodes)},
            evidence={
                "memberNodeIds": members,
                "orientations": [cluster[0] for cluster in clusters],
            },
            quantity="distinct tool-axis orientations",
        )
