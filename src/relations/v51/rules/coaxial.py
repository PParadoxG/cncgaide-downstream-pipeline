"""COAXIAL — infinite feature-axis line coincidence for the frozen pairs."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.v51.registry import register_rule
from relations.v51._shared import (
    AXIS_LINE_TOL_MM,
    COAXIAL_ANGLE_TOL_DEG,
    HOLE_TYPES,
    SPACING_BASIS,
    _feature_edge,
    _report,
    acute_angle_deg,
    axis,
    axis_origin,
    coaxial_axis_distance,
    feature_type,
)


def _allowed(left: str, right: str) -> bool:
    if left in HOLE_TYPES and right in HOLE_TYPES:
        return True
    if left == "CircularBoss" and right == "CircularBoss":
        return True
    return (left in HOLE_TYPES and right == "CircularBoss") or (right in HOLE_TYPES and left == "CircularBoss")


@register_rule
class Coaxial(RelationRule):
    rule_id = "coaxial_v1"
    edge_type = "COAXIAL"
    priority = 30
    enabled_by_default = True
    directed = False
    description = "Allowed Hole/CircularBoss pairs sharing one infinite 3D axis line."

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        nodes = list(context.nodes)
        for index, left in enumerate(nodes):
            left_type = feature_type(left)
            for right in nodes[index + 1:]:
                right_type = feature_type(right)
                if not _allowed(left_type, right_type):
                    continue
                left_axis, right_axis = axis(left), axis(right)
                left_origin, right_origin = axis_origin(context, left), axis_origin(context, right)
                if left_axis is None or right_axis is None or left_origin is None or right_origin is None:
                    _report(context, self.rule_id, "axisLineProofUnavailable")
                    continue
                angle = acute_angle_deg(left_axis, right_axis)
                # One tolerance pair, applied in one place: the angular gate
                # lives inside ``coaxial_axis_distance`` so COAXIAL and the
                # HOLE_SPACING degeneracy gate can never drift apart.
                distance = coaxial_axis_distance(left_origin, left_axis, right_origin, right_axis)
                if distance is None:
                    _report(context, self.rule_id, "nonParallelAxisRejected")
                    continue
                if distance > AXIS_LINE_TOL_MM:
                    _report(context, self.rule_id, "parallelOffsetRejected")
                    continue
                edges.append(
                    _feature_edge(
                        left,
                        right,
                        self.edge_type,
                        self.rule_id,
                        {
                            "relationBasis": "coincidentInfiniteFeatureAxisLines",
                            "featureA": {"nodeId": left.nodeId, "featureType": left_type},
                            "featureB": {"nodeId": right.nodeId, "featureType": right_type},
                            "axisA": left_axis,
                            "axisB": right_axis,
                            "axisPointA": left_origin,
                            "axisPointB": right_origin,
                            "acuteAngularDeviation": round(angle, 6),
                            "axisLineDistance": round(distance, 6),
                            "axisLineBasis": SPACING_BASIS,
                            "axisLineTolerance": AXIS_LINE_TOL_MM,
                            "angularTolerance": COAXIAL_ANGLE_TOL_DEG,
                        },
                    )
                )
        return edges

