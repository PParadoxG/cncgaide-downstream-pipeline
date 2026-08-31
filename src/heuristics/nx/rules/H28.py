"""H28 (nx_v1) - remaining floor thickness below a blind hole (hard, single feature)."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H28(FeatureRule):
    """Same predicate and threshold as H11 on the complementary domain.

    On this line the blind holes are NX's ``STEP*POCKET`` classes (node 19a
    D2: the through / blind split is NX's own naming), and the floor the
    measurement casts from may be a drill-point cone as well as a plane - the
    face classification goes by angle against the access direction, not by
    surface type.
    """

    h_id = "H28"
    USES_ATTRIBUTES = ("derived_floor_thickness",)
    USES_THRESHOLDS = ("t_floor",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "derived_floor_thickness"),
            context.threshold("t_floor"),
            "derived_floor_thickness",
            "t_floor",
        )
