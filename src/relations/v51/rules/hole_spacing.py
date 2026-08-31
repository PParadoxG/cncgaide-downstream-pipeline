"""HOLE_SPACING — same-host parallel Hole axis-line spacing."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.v51.registry import register_rule
from relations.v51._shared import (
    AXIS_LINE_TOL_MM,
    HOLE_TYPES,
    PARALLEL_ANGLE_TOL_DEG,
    SPACING_BASIS,
    _feature_edge,
    _report,
    acute_angle_deg,
    axis,
    axis_origin,
    feature_type,
    located_hosts,
    parallel_axis_spacing,
)


@register_rule
class HoleSpacing(RelationRule):
    rule_id = "hole_spacing_v1"
    edge_type = "HOLE_SPACING"
    priority = 50
    enabled_by_default = True
    directed = False
    description = "Parallel Hole features sharing at least one effective LOCATED_ON host."

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        nodes = list(context.nodes)
        for index, left in enumerate(nodes):
            if feature_type(left) not in HOLE_TYPES:
                continue
            left_hosts = located_hosts(context, left)
            for right in nodes[index + 1:]:
                if feature_type(right) not in HOLE_TYPES:
                    continue
                right_hosts = located_hosts(context, right)
                common_hosts = sorted(set(left_hosts) & set(right_hosts))
                if not common_hosts:
                    _report(context, self.rule_id, "noSharedLocatedOnHost")
                    continue
                left_axis, right_axis = axis(left), axis(right)
                left_origin, right_origin = axis_origin(context, left), axis_origin(context, right)
                if left_axis is None or right_axis is None or left_origin is None or right_origin is None:
                    _report(context, self.rule_id, "axisLineProofUnavailable")
                    continue
                angle = acute_angle_deg(left_axis, right_axis)
                spacing = parallel_axis_spacing(left_origin, left_axis, right_origin, right_axis)
                if spacing is None:
                    _report(context, self.rule_id, "nonParallelAxisRejected")
                    continue
                # A coaxial pair has no spacing to speak of: the two axes are
                # the same line, so the number would be a degenerate zero and
                # H12 would read it as a violation.  Gate on the degenerate
                # metric rather than on the COAXIAL edge, so a future change
                # to that relation's allowed pairs cannot let a zero through.
                if spacing <= AXIS_LINE_TOL_MM:
                    _report(context, self.rule_id, "coaxialPairRejected")
                    continue
                host_witnesses = []
                shared_faces = sorted({
                    str(witness.get("hostFaceId"))
                    for host in common_hosts
                    for witness in [*left_hosts[host], *right_hosts[host]]
                    if witness.get("hostFaceId")
                })
                for host in common_hosts:
                    host_faces = sorted({
                        str(witness.get("hostFaceId"))
                        for witness in [*left_hosts[host], *right_hosts[host]]
                        if witness.get("hostFaceId")
                    })
                    host_witnesses.append({
                        "hostOwnerNodeId": host,
                        "hostFaceIds": host_faces,
                        "leftTerminalWitnesses": left_hosts[host],
                        "rightTerminalWitnesses": right_hosts[host],
                    })
                edges.append(
                    _feature_edge(
                        left,
                        right,
                        self.edge_type,
                        self.rule_id,
                        {
                            "relationBasis": "sharedLocatedOnHostAndParallelInfiniteAxisLines",
                            "featureA": {"nodeId": left.nodeId, "featureType": feature_type(left)},
                            "featureB": {"nodeId": right.nodeId, "featureType": feature_type(right)},
                            "sharedHostWitnesses": host_witnesses,
                            "sharedHostOwnerNodeIds": common_hosts,
                            "sharedHostFaceIds": shared_faces,
                            "axisA": left_axis,
                            "axisB": right_axis,
                            "axisPointA": left_origin,
                            "axisPointB": right_origin,
                            "acuteAngularDeviation": round(angle, 6),
                            "angularTolerance": PARALLEL_ANGLE_TOL_DEG,
                            "spacingBasis": SPACING_BASIS,
                        },
                        properties={"axisSpacing": round(spacing, 6)},
                    )
                )
        return edges
