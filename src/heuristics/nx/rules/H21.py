"""H21 (nx_v1) - vertical walls, no draft."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_most
from heuristics.nx.registry import register


@register
class H21(FeatureRule):
    """NX publishes the wall-to-floor angle directly; 90 degrees is vertical.

    This is where NX is ahead of the open-source line, which has to recover a
    draft angle from face normals: ``MIN_SIDE_ANGLE`` is a number NX states.
    The condition subtracts it from 90 exactly as written, so a wall that
    leans the other way reads as a negative deviation rather than as a
    violation - which is the ontology's reading, not a rounding of it.
    """

    h_id = "H21"
    USES_ATTRIBUTES = ("MIN_SIDE_ANGLE",)
    USES_THRESHOLDS = ("tol_draft_zero",)

    def evaluate_feature(self, node, context):
        angle = context.attribute(node, "MIN_SIDE_ANGLE")
        return compare_at_most(
            round(90.0 - angle, 9), context.threshold("tol_draft_zero"),
            "draftFromVertical", "tol_draft_zero", {"MIN_SIDE_ANGLE": angle},
        )
