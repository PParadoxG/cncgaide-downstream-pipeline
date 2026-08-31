"""H18 (nx_v1) - nearby holes share one base-hole diameter."""

from __future__ import annotations

from heuristics.base import NA_FALSE_ANTECEDENT, PairRule, failed, not_applicable, passed
from heuristics.nx._shared import base_diameter_name, distinct, relation_missing
from heuristics.nx.registry import register


@register
class H18(PairRule):
    """The base hole is the last section, so this reads ``DIAMETER_n``.

    Reading the entry instead judges the counterbore: two identical
    counterbores over different base holes would pass, and two different
    counterbores over one base hole would fail, which is the rule's own
    intent inverted.

    "Nearby" is a scale-normalized gate on top of the HOLE_SPACING candidate
    set: two 60 mm holes 100 mm apart belong together, two 3 mm holes at the
    same distance do not.
    """

    h_id = "H18"
    edge_type = "HOLE_SPACING"
    USES_ATTRIBUTES = ("DIAMETER_n",)
    USES_RELATIONS = ("HOLE_SPACING",)
    USES_THRESHOLDS = ("t_hole_near", "tol_diameter")

    def required_attributes(self, feature_type):
        name = base_diameter_name(feature_type)
        return (name,) if name else ()

    def evaluate_pair(self, edge, source, target, context):
        spacing = (edge.properties or {}).get("axisSpacing")
        if spacing is None:
            return relation_missing("HOLE_SPACING edge carries no axisSpacing",
                                    {"edgeId": edge.edgeId})
        left = context.attribute(source, base_diameter_name(source.featureType))
        right = context.attribute(target, base_diameter_name(target.featureType))
        gate = context.threshold("t_hole_near") * max(left, right)
        values = {"axisSpacing": spacing, "baseDiameterA": left, "baseDiameterB": right,
                  "nearGate": round(gate, 6)}
        if spacing > gate:
            return not_applicable(
                f"axisSpacing {round(spacing, 6)} > t_hole_near * max(base diameter) "
                f"{round(gate, 6)}: the pair is not nearby, the rule says nothing about it",
                NA_FALSE_ANTECEDENT, values)
        clusters = distinct([left, right], context.threshold("tol_diameter"))
        values["distinctCount"] = len(clusters)
        if len(clusters) == 1:
            return passed("both nearby holes share one base-hole diameter", values)
        return failed(f"nearby holes use {len(clusters)} different base-hole diameters", values)
