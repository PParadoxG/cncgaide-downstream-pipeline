"""H54 - step included angle (hard, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_least
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H54(FeatureRule):
    """An acute step forms a wedge a rotating tool cannot reach into."""

    h_id = "H54"
    USES_ATTRIBUTES = ("step_included_angle",)
    USES_THRESHOLDS = ("t_step_angle",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "step_included_angle"),
            context.threshold("t_step_angle"),
            "step_included_angle",
            "t_step_angle",
        )
