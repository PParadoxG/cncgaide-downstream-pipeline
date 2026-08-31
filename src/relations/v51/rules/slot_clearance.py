"""SLOT_CLEARANCE — same-host opening/profile boundary minimum distance."""

from __future__ import annotations

from collections import defaultdict

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.v51.registry import register_rule
from relations.v51._shared import (
    PARALLEL_ANGLE_TOL_DEG,
    SLOT_TYPES,
    _feature_edge,
    _report,
    acute_angle_deg,
    axis,
    feature_type,
    located_hosts,
    polyline_distance,
    slot_boundary_sets,
    slot_course_direction,
)


@register_rule
class SlotClearance(RelationRule):
    rule_id = "slot_clearance_v1"
    edge_type = "SLOT_CLEARANCE"
    priority = 70
    enabled_by_default = True
    directed = False
    description = "Specified Slot opening/profile boundary distance on a shared host."

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        nodes = list(context.nodes)
        for index, left in enumerate(nodes):
            if feature_type(left) not in SLOT_TYPES:
                continue
            left_hosts = located_hosts(context, left)
            left_boundaries = slot_boundary_sets(context, left)
            for right in nodes[index + 1:]:
                if feature_type(right) not in SLOT_TYPES:
                    continue
                right_hosts = located_hosts(context, right)
                common_hosts = sorted(set(left_hosts) & set(right_hosts))
                if not common_hosts:
                    _report(context, self.rule_id, "noSharedLocatedOnHost")
                    continue
                right_boundaries = slot_boundary_sets(context, right)
                by_host_left = defaultdict(list)
                by_host_right = defaultdict(list)
                for boundary in left_boundaries:
                    owner = boundary.get("hostOwnerNodeId")
                    if owner in common_hosts:
                        by_host_left[owner].append(boundary)
                for boundary in right_boundaries:
                    owner = boundary.get("hostOwnerNodeId")
                    if owner in common_hosts:
                        by_host_right[owner].append(boundary)

                best = None
                for host in common_hosts:
                    for left_boundary in by_host_left.get(host, []):
                        for right_boundary in by_host_right.get(host, []):
                            distance, point_a, point_b = polyline_distance(
                                left_boundary["points"], right_boundary["points"]
                            )
                            if distance is None:
                                continue
                            candidate = (distance, host, left_boundary, right_boundary, point_a, point_b)
                            if best is None or candidate[0] < best[0]:
                                best = candidate
                if best is None:
                    _report(context, self.rule_id, "profileBoundaryProofUnavailable")
                    continue

                _, host, left_boundary, right_boundary, point_a, point_b = best
                # Two different angles, and conflating them is what made the
                # earlier baseline call this pair "non-parallel at 76.1 deg".
                # The course is where the slot runs; the feature axis is the
                # opening-to-depth direction.  Neither is a gate: a pair of
                # converging slots is exactly the thin-material case this
                # relation has to keep.
                left_course = slot_course_direction(left)
                right_course = slot_course_direction(right)
                course_angle = (
                    acute_angle_deg(left_course, right_course)
                    if left_course and right_course else None
                )
                left_axis, right_axis = axis(left), axis(right)
                transverse_angle = (
                    acute_angle_deg(left_axis, right_axis)
                    if left_axis and right_axis else None
                )
                edges.append(
                    _feature_edge(
                        left,
                        right,
                        self.edge_type,
                        self.rule_id,
                        {
                            "relationBasis": "sharedLocatedOnHostAndOpeningProfileBoundaryDistance",
                            "featureA": {"nodeId": left.nodeId, "featureType": feature_type(left)},
                            "featureB": {"nodeId": right.nodeId, "featureType": feature_type(right)},
                            "sharedHostOwnerNodeIds": common_hosts,
                            "sharedHostFaceIds": sorted({
                                str(witness.get("hostFaceId"))
                                for owner in common_hosts
                                for witness in [*left_hosts[owner], *right_hosts[owner]]
                                if witness.get("hostFaceId")
                            }),
                            "sharedHostWitnesses": [
                                {
                                    "hostOwnerNodeId": host,
                                    "hostFaceIds": sorted({
                                        str(witness.get("hostFaceId"))
                                        for witness in [*left_hosts[host], *right_hosts[host]]
                                        if witness.get("hostFaceId")
                                    }),
                                    "leftTerminalWitnesses": left_hosts[host],
                                    "rightTerminalWitnesses": right_hosts[host],
                                }
                            ],
                            "clearanceWitnesses": [{
                                "hostOwnerNodeId": host,
                                "hostFaceId": left_boundary.get("hostFaceId"),
                                "leftBoundaryEdgeId": left_boundary.get("edgeId"),
                                "rightBoundaryEdgeId": right_boundary.get("edgeId"),
                                "nearestPointA": [round(value, 6) for value in point_a],
                                "nearestPointB": [round(value, 6) for value in point_b],
                            }],
                            "clearanceWitnessCount": 1,
                            "courseDirectionA": left_course,
                            "courseDirectionB": right_course,
                        },
                        properties={
                            "slotClearance": round(best[0], 6),
                            "courseAngle": round(course_angle, 6) if course_angle is not None else None,
                            "courseParallel": (
                                None if course_angle is None
                                else bool(course_angle <= PARALLEL_ANGLE_TOL_DEG)
                            ),
                            "transverseAngle": (
                                round(transverse_angle, 6) if transverse_angle is not None else None
                            ),
                        },
                    )
                )
        return edges
