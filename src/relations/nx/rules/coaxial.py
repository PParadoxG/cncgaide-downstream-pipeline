"""COAXIAL (nx_v1) — two feature axes on one infinite line, from NX's nine scalars."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.nx._shared import (
    AXIS_LINE_TOL_MM,
    COAXIAL_ANGLE_TOL_DEG,
    ROUND_BOSS,
    SPACING_BASIS,
    _report,
    acute_angle_deg,
    axis,
    axis_point,
    coaxial_axis_distance,
    feature_type,
    is_hole_like,
)
from relations.nx.registry import register_rule


def _allowed(left, right) -> bool:
    """The pairs ``relations.yaml`` allows: hole-family pairs, and either with
    a round boss.  Membership comes from NX's class tree, not from a list."""
    left_hole, right_hole = is_hole_like(left), is_hole_like(right)
    if left_hole and right_hole:
        return True
    if feature_type(left) == ROUND_BOSS and feature_type(right) == ROUND_BOSS:
        return True
    return ((left_hole and feature_type(right) == ROUND_BOSS)
            or (right_hole and feature_type(left) == ROUND_BOSS))


@register_rule
class NxCoaxial(RelationRule):
    rule_id = "coaxial_nx_v1"
    edge_type = "COAXIAL"
    priority = 30
    enabled_by_default = True
    directed = False
    description = "Allowed NX hole/round-boss pairs sharing one infinite 3D axis line."

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        from relations.nx._shared import nx_edge

        edges: list[FeatureEdge] = []
        nodes = list(context.nodes)
        for index, left in enumerate(nodes):
            for right in nodes[index + 1:]:
                if not _allowed(left, right):
                    continue
                left_axis, right_axis = axis(left), axis(right)
                left_point, right_point = axis_point(left), axis_point(right)
                if None in (left_axis, right_axis, left_point, right_point):
                    _report(context, self.rule_id, "axisLineProofUnavailable")
                    continue
                angle = acute_angle_deg(left_axis, right_axis)
                # The angular gate lives inside the distance helper, so this
                # relation and the HOLE_SPACING degeneracy gate can never drift.
                distance = coaxial_axis_distance(left_point, left_axis, right_point, right_axis)
                if distance is None:
                    _report(context, self.rule_id, "nonParallelAxisRejected")
                    continue
                if distance > AXIS_LINE_TOL_MM:
                    _report(context, self.rule_id, "parallelOffsetRejected")
                    continue
                edges.append(nx_edge(
                    left, right, self.edge_type, self.rule_id, "nx_arithmetic",
                    {
                        "relationBasis": "coincidentInfiniteFeatureAxisLines",
                        "nxBasis": "X/Y/Z_ORIENTATION_D and X/Y/Z_POSITION, as NX published them",
                        "featureA": {"nodeId": left.nodeId, "featureType": feature_type(left)},
                        "featureB": {"nodeId": right.nodeId, "featureType": feature_type(right)},
                        "axisA": left_axis,
                        "axisB": right_axis,
                        "axisPointA": left_point,
                        "axisPointB": right_point,
                        "acuteAngularDeviation": round(angle, 6),
                        "axisLineDistance": round(distance, 6),
                        "axisLineBasis": SPACING_BASIS,
                        "axisLineTolerance": AXIS_LINE_TOL_MM,
                        "angularTolerance": COAXIAL_ANGLE_TOL_DEG,
                    },
                ))
        return edges
