"""INTERSECTS (nx_v1) — two hole walls that meet, proven by a shared B-Rep edge."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.nx._shared import (
    PARALLEL_ANGLE_TOL_DEG,
    _report,
    acute_angle_deg,
    axis,
    cylindrical_walls,
    direct_face_pairs,
    feature_type,
    is_hole_like,
    nx_edge,
    parallel,
)
from relations.nx.registry import register_rule


@register_rule
class NxIntersects(RelationRule):
    """Crossing is a topological claim, so it is proven topologically.

    Two axes that cross in space prove nothing: they may cross outside the
    material, or at different depths.  What proves the removal volumes meet is
    that the two features' own cylindrical walls share a B-Rep edge - the
    curve where one bore broke into the other.  The angle is then read from
    the axes NX published.
    """

    rule_id = "intersects_nx_v1"
    edge_type = "INTERSECTS"
    priority = 40
    enabled_by_default = True
    directed = False
    description = "Nonparallel NX hole-family features whose walls share a B-Rep edge."

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        holes = [node for node in context.nodes if is_hole_like(node)]
        for index, left in enumerate(holes):
            left_walls = set(cylindrical_walls(context, left))
            for right in holes[index + 1:]:
                left_axis, right_axis = axis(left), axis(right)
                if left_axis is None or right_axis is None:
                    _report(context, self.rule_id, "axisProofUnavailable")
                    continue
                right_walls = set(cylindrical_walls(context, right))
                wall_pairs = [
                    pair for pair in direct_face_pairs(context, left, right)
                    if pair.get("leftPartmodelFaceId") in left_walls
                    and pair.get("rightPartmodelFaceId") in right_walls
                ]
                if not wall_pairs:
                    _report(context, self.rule_id, "noWallSharedEdge")
                    continue
                if parallel(left_axis, right_axis):
                    _report(context, self.rule_id, "parallelAxisRejected")
                    continue
                angle = acute_angle_deg(left_axis, right_axis)
                shared = sorted({pair.get("sharedEdgeId") for pair in wall_pairs
                                 if pair.get("sharedEdgeId")})
                edges.append(nx_edge(
                    left, right, self.edge_type, self.rule_id, "nx_topology",
                    {
                        "relationBasis": "holeWallSharedBRepEdge",
                        "nxBasis": "PartModel adjacency for the meeting, "
                                   "X/Y/Z_ORIENTATION_D for the angle",
                        "featureA": {"nodeId": left.nodeId, "featureType": feature_type(left)},
                        "featureB": {"nodeId": right.nodeId, "featureType": feature_type(right)},
                        "wallFacePairs": wall_pairs,
                        "wallPairCount": len(wall_pairs),
                        "sharedEdgeIds": shared,
                        "axisA": left_axis,
                        "axisB": right_axis,
                        "angularTolerance": PARALLEL_ANGLE_TOL_DEG,
                    },
                    properties={"crossingAngle": round(angle, 6)},
                ))
        return edges
