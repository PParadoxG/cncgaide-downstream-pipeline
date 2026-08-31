"""POCKET_CLEARANCE (nx_v1) — remaining material between two pockets' walls."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.nx._shared import (
    _report,
    axis,
    feature_type,
    in_any_family,
    SEPARATION_TOL_MM,
    kernel_face_set_clearance,
    lateral_faces,
    nx_edge,
)
from relations.nx.registry import register_rule


@register_rule
class NxPocketClearance(RelationRule):
    """A kernel measurement, because no attribute NX publishes can answer it.

    ``relations.yaml`` defines the quantity as the shortest remaining material
    distance *between the walls* of two POCKETS-family features, so what is
    measured is each pocket's lateral face set — the faces running along its
    own access direction — not its floor.

    There is no shared-host gate, and ``relations.yaml`` says so: host
    identity is not comparable across the two lines (see the note on
    HOLE_SPACING, and node 21' §6 for the measurement that settled it).

    What the ontology *does* state, on this side as on v5.1's, is that the
    two wall sets must be **separated** - a distance of zero is two pockets
    touching, which is a contact and not a clearance.  That is the one clause
    node 19' dropped which is sayable in NX's vocabulary, so it is back, and
    a zero reading is refused with a named ``contactingFaceSets``.
    """

    rule_id = "pocket_clearance_nx_v1"
    edge_type = "POCKET_CLEARANCE"
    priority = 60
    enabled_by_default = True
    directed = False
    description = "NX POCKETS-family pair, side-wall face sets measured through KernelServices."

    family = "POCKETS"
    property_name = "pocketClearance"

    def _measurable_faces(self, context: RelationContext, node) -> list[str]:
        direction = axis(node)
        if direction is None:
            return []
        return lateral_faces(context.model, node, direction)

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        members = [node for node in context.nodes
                   if in_any_family(feature_type(node), (self.family,))]
        for index, left in enumerate(members):
            left_faces = self._measurable_faces(context, left)
            for right in members[index + 1:]:
                right_faces = self._measurable_faces(context, right)
                value, details = kernel_face_set_clearance(context, left_faces, right_faces)
                if value is None:
                    _report(context, self.rule_id, details["status"])
                    continue
                if value <= SEPARATION_TOL_MM:
                    details["status"] = "contacting"
                    _report(context, self.rule_id, "contactingFaceSets")
                    continue
                edges.append(nx_edge(
                    left, right, self.edge_type, self.rule_id, "nx_kernel",
                    {
                        "relationBasis": "kernelMinimumDistanceBetweenSeparatedSideWallFaceSets",
                        "nxBasis": "KernelServices face_set_distance; "
                                   "X/Y/Z_ORIENTATION_D selects each feature's side walls",
                        "featureA": {"nodeId": left.nodeId, "featureType": feature_type(left)},
                        "featureB": {"nodeId": right.nodeId, "featureType": feature_type(right)},
                        "separationTolerance": SEPARATION_TOL_MM,
                        "clearanceWitness": details,
                    },
                    properties={self.property_name: value},
                ))
        return edges
