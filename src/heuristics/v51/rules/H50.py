"""H50 - feature extent within the usable tool length (hard, needs the magazine)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_most, external_missing
from heuristics.base import FeatureRule
from heuristics.v51.registry import register

# The axial extent a tool has to reach, per leaf class.  A step has no depth:
# its v4.0 carrier (height) depended on a non-unique local layer pair and was
# retired, and the extent of its two principal faces from the shared straight
# edge is the conservative upper bound on how far the tool must reach.
EXTENT_CARRIER = {"Step": "max_face_extent"}
DEFAULT_CARRIER = "depth"


@register
class H50(FeatureRule):
    """Two carriers, one criterion: the reach a standard tool must have."""

    h_id = "H50"
    USES_ATTRIBUTES = ("depth", "max_face_extent")
    EXTERNAL_INPUT = "max_usable_length"

    def required_attributes(self, feature_type):
        return (EXTENT_CARRIER.get(feature_type, DEFAULT_CARRIER),)

    def evaluate_feature(self, node, context):
        limit = context.external_inputs.get(self.EXTERNAL_INPUT)
        if limit is None:
            return external_missing(self.EXTERNAL_INPUT)
        carrier = EXTENT_CARRIER.get(node.featureType, DEFAULT_CARRIER)
        return compare_at_most(
            context.attribute(node, carrier), limit, carrier, "EXT.max_usable_length"
        )
