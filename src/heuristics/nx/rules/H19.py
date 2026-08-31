"""H19 (nx_v1) - corner radius against pocket depth."""

from __future__ import annotations

from heuristics.base import FeatureRule, not_evaluable
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H19(FeatureRule):
    """A corner too sharp for its depth needs a cutter too slender to survive."""

    h_id = "H19"
    USES_ATTRIBUTES = ("MIN_RADIUS", "DEPTH")
    USES_THRESHOLDS = ("t_corner_depth_ratio",)

    def evaluate_feature(self, node, context):
        radius = context.attribute(node, "MIN_RADIUS")
        depth = context.attribute(node, "DEPTH")
        if depth <= 0.0:
            return not_evaluable(
                f"DEPTH {depth} is not positive, so no ratio exists",
                {"MIN_RADIUS": radius, "DEPTH": depth}, {"blocked": "attributeGap"})
        return compare_at_least(
            round(radius / depth, 6), context.threshold("t_corner_depth_ratio"),
            "cornerRadiusToDepth", "t_corner_depth_ratio",
            {"MIN_RADIUS": radius, "DEPTH": depth},
        )
