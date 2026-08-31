"""H21 - vertical walls, no draft (advisory, single feature)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_most
from heuristics.base import FeatureRule
from heuristics.v51.registry import register

# One physical quantity, two carriers.  On a circular boss the draft angle is
# exactly recoverable from the two diameters and the height, so the ontology
# publishes it as a derived attribute there; asserting it independently would
# over-define the class.  Splitting the rule in two would produce two
# word-for-word identical heuristics, so the carrier is selected per leaf.
DRAFT_CARRIER = {"CircularBoss": "derived_wall_draft_angle"}
DEFAULT_CARRIER = "max_wall_draft_angle"


@register
class H21(FeatureRule):
    """Zero draft compared within a tolerance, never by exact float equality."""

    h_id = "H21"
    USES_ATTRIBUTES = ("max_wall_draft_angle", "derived_wall_draft_angle")
    USES_THRESHOLDS = ("tol_draft_zero",)

    def required_attributes(self, feature_type):
        return (DRAFT_CARRIER.get(feature_type, DEFAULT_CARRIER),)

    def evaluate_feature(self, node, context):
        carrier = DRAFT_CARRIER.get(node.featureType, DEFAULT_CARRIER)
        return compare_at_most(
            context.attribute(node, carrier),
            context.threshold("tol_draft_zero"),
            carrier,
            "tol_draft_zero",
        )
