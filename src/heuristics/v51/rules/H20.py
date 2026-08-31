"""H20 - flat pocket bottom (informational, single feature)."""

from __future__ import annotations

from heuristics.base import FeatureRule, failed, passed
from heuristics.v51.registry import register

# Controlled vocabulary of pocket_bottom_type; flat is the one value that
# lets face milling replace ball-nose scanning.
FLAT = "flat"


@register
class H20(FeatureRule):
    """A null here is a floor that could not be attributed, not a shape."""

    h_id = "H20"
    USES_ATTRIBUTES = ("pocket_bottom_type",)

    def evaluate_feature(self, node, context):
        value = context.attribute(node, "pocket_bottom_type")
        values = {"pocket_bottom_type": value}
        if value == FLAT:
            return passed("the pocket floor is flat", values)
        return failed(f"pocket_bottom_type is {value}, not {FLAT}", values)
