"""H18 - nearby holes share one base-hole diameter (advisory, feature pair)."""

from __future__ import annotations

from heuristics.v51._shared import distinct, relation_missing
from heuristics.base import NA_FALSE_ANTECEDENT, PairRule, failed, not_applicable, passed
from heuristics.v51.registry import register


@register
class H18(PairRule):
    """"Nearby" is a normalized gate on top of the HOLE_SPACING candidate set.

    HOLE_SPACING already requires a shared host and parallel axes - the
    geometric precondition for one drill reaching both in one setup.  The
    remaining freedom is scale, so the gate is normalized by the larger
    diameter: two 60 mm holes 100 mm apart belong together, two 3 mm holes at
    the same distance do not.
    """

    h_id = "H18"
    edge_type = "HOLE_SPACING"
    USES_ATTRIBUTES = ("diameter",)
    USES_RELATIONS = ("HOLE_SPACING",)
    USES_THRESHOLDS = ("t_hole_near", "tol_diameter")

    def evaluate_pair(self, edge, source, target, context):
        spacing = (edge.properties or {}).get("axisSpacing")
        if spacing is None:
            return relation_missing(
                "HOLE_SPACING edge carries no axisSpacing", {"edgeId": edge.edgeId}
            )
        left = context.attribute(source, "diameter")
        right = context.attribute(target, "diameter")
        gate = context.threshold("t_hole_near") * max(left, right)
        values = {
            "axisSpacing": spacing,
            "diameterA": left,
            "diameterB": right,
            "nearGate": round(gate, 6),
        }
        if spacing > gate:
            return not_applicable(
                f"axisSpacing {round(spacing, 6)} > t_hole_near * max(diameter) {round(gate, 6)}: "
                "the pair is not nearby, the rule says nothing about it",
                NA_FALSE_ANTECEDENT,
                values,
            )
        clusters = distinct([left, right], context.threshold("tol_diameter"))
        values["distinctCount"] = len(clusters)
        if len(clusters) == 1:
            return passed("both nearby holes share one base-hole diameter", values)
        return failed(
            f"nearby holes use {len(clusters)} different base-hole diameters", values
        )
