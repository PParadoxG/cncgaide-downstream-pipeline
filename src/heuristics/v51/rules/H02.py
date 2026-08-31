"""H02 - standard drill-point blind-hole bottom (informational, single feature)."""

from __future__ import annotations

from heuristics.base import FeatureRule, failed, passed
from heuristics.v51.registry import register

# A controlled vocabulary, not a threshold: the two standard drill-point
# angles are part of the value space of hole_bottom_type in the ontology, so
# they are named here rather than in thresholds.yaml.
STANDARD_DRILL_POINTS = ("conical_118", "conical_135")


@register
class H02(FeatureRule):
    """Graded informational: a flat blind-hole bottom is a normal design."""

    h_id = "H02"
    USES_ATTRIBUTES = ("hole_bottom_type",)

    def evaluate_feature(self, node, context):
        value = context.attribute(node, "hole_bottom_type")
        values = {"hole_bottom_type": value}
        if value in STANDARD_DRILL_POINTS:
            return passed(f"hole_bottom_type {value} is a standard drill point", values)
        return failed(
            f"hole_bottom_type {value} is not in {list(STANDARD_DRILL_POINTS)}", values
        )
