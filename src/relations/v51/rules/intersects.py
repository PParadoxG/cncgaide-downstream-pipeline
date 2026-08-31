"""INTERSECTS — principal-hole-wall crossing proven by shared B-Rep edges."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.v51.registry import register_rule
from relations.v51._shared import (
    HOLE_TYPES,
    _feature_edge,
    _report,
    acute_angle_deg,
    axis,
    direct_face_pairs,
    feature_type,
    parallel,
    principal_wall_face_ids,
)


@register_rule
class Intersects(RelationRule):
    rule_id = "intersects_v1"
    edge_type = "INTERSECTS"
    priority = 40
    enabled_by_default = True
    directed = False
    description = "Nonparallel Hole features with a principal-wall shared B-Rep edge."

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        nodes = list(context.nodes)
        for index, left in enumerate(nodes):
            if feature_type(left) not in HOLE_TYPES:
                continue
            for right in nodes[index + 1:]:
                if feature_type(right) not in HOLE_TYPES:
                    continue
                left_axis, right_axis = axis(left), axis(right)
                if left_axis is None or right_axis is None:
                    _report(context, self.rule_id, "axisProofUnavailable")
                    continue
                pairs = direct_face_pairs(context, left, right)
                left_walls = set(principal_wall_face_ids(context, left))
                right_walls = set(principal_wall_face_ids(context, right))
                wall_pairs = [
                    pair for pair in pairs
                    if pair.get("leftPartmodelFaceId") in left_walls
                    and pair.get("rightPartmodelFaceId") in right_walls
                    and pair.get("leftSurfaceType") == "CYLINDER"
                    and pair.get("rightSurfaceType") == "CYLINDER"
                ]
                if not wall_pairs:
                    _report(context, self.rule_id, "noPrincipalWallSharedEdge")
                    continue
                angle = acute_angle_deg(left_axis, right_axis)
                if parallel(left_axis, right_axis):
                    _report(context, self.rule_id, "parallelAxisRejected")
                    continue
                shared_edges = sorted({pair.get("sharedEdgeId") for pair in wall_pairs if pair.get("sharedEdgeId")})
                edges.append(
                    _feature_edge(
                        left,
                        right,
                        self.edge_type,
                        self.rule_id,
                        {
                            "relationBasis": "principalHoleWallSharedBRepEdge",
                            "featureA": {"nodeId": left.nodeId, "featureType": feature_type(left)},
                            "featureB": {"nodeId": right.nodeId, "featureType": feature_type(right)},
                            "principalWallFacePairs": wall_pairs,
                            "principalWallPairCount": len(wall_pairs),
                            "sharedEdgeIds": shared_edges,
                            "axisA": left_axis,
                            "axisB": right_axis,
                        },
                        properties={"crossingAngle": round(angle, 6)},
                    )
                )
        return edges

