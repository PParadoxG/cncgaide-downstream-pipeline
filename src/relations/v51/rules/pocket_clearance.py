"""POCKET_CLEARANCE — same-host effective side-wall minimum distance."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.v51.registry import register_rule
from relations.v51._shared import (
    POCKET_TYPES,
    _feature_edge,
    _report,
    feature_type,
    located_hosts,
    side_wall_clearance,
)


@register_rule
class PocketClearance(RelationRule):
    rule_id = "pocket_clearance_v1"
    edge_type = "POCKET_CLEARANCE"
    priority = 60
    enabled_by_default = True
    directed = False
    description = "Specified Pocket effective side-wall sets measured on a shared host."

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        nodes = list(context.nodes)
        for index, left in enumerate(nodes):
            if feature_type(left) not in POCKET_TYPES:
                continue
            left_hosts = located_hosts(context, left)
            for right in nodes[index + 1:]:
                if feature_type(right) not in POCKET_TYPES:
                    continue
                right_hosts = located_hosts(context, right)
                common_hosts = sorted(set(left_hosts) & set(right_hosts))
                if not common_hosts:
                    _report(context, self.rule_id, "noSharedLocatedOnHost")
                    continue
                value, solver = side_wall_clearance(context, left, right)
                if value is None:
                    _report(context, self.rule_id, "clearanceProofUnavailable")
                    continue
                witnesses = []
                for host in common_hosts:
                    host_faces = sorted({
                        str(witness.get("hostFaceId"))
                        for witness in [*left_hosts[host], *right_hosts[host]]
                        if witness.get("hostFaceId")
                    })
                    witnesses.append({
                        "hostOwnerNodeId": host,
                        "hostFaceIds": host_faces,
                        "leftTerminalWitnesses": left_hosts[host],
                        "rightTerminalWitnesses": right_hosts[host],
                    })
                shared_faces = sorted({
                    face_id
                    for witness in witnesses
                    for face_id in witness.get("hostFaceIds", [])
                })
                solver = dict(solver)
                solver["sharedHostOwnerNodeIds"] = common_hosts
                solver["sharedHostFaceIds"] = shared_faces
                solver["sharedHostWitnesses"] = witnesses
                edges.append(
                    _feature_edge(
                        left,
                        right,
                        self.edge_type,
                        self.rule_id,
                        {
                            "relationBasis": "sharedLocatedOnHostAndEffectiveSideWallBoundaryDistance",
                            "featureA": {"nodeId": left.nodeId, "featureType": feature_type(left)},
                            "featureB": {"nodeId": right.nodeId, "featureType": feature_type(right)},
                            "sharedHostOwnerNodeIds": common_hosts,
                            "sharedHostFaceIds": shared_faces,
                            "sharedHostWitnesses": witnesses,
                            "clearanceWitnesses": [solver],
                            "clearanceWitnessCount": 1,
                        },
                        properties={"pocketClearance": value},
                    )
                )
        return edges
