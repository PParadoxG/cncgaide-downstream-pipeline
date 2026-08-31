"""SLOT_CLEARANCE (nx_v1) — the gap between two slots' opening boundaries."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.nx._shared import (
    SEPARATION_TOL_MM,
    _report,
    boundary_set_clearance,
    feature_type,
    in_any_family,
    nx_edge,
    opening_boundary_polylines,
)
from relations.nx.registry import register_rule


@register_rule
class NxSlotClearance(RelationRule):
    """Measured where ``cnc:SLOT_CLEARANCE`` measures: at the openings.

    Node 20'b read ``relations.yaml``'s "between two slots" as the two whole
    owned face sets and handed them to the kernel.  That is a different
    quantity from the one the open-source line reports, and a weaker one: a
    slot's floor is part of its owned set, so two slots on opposite faces of a
    web could be called close because their floors nearly meet.  Node 21'
    moved it onto the opening boundaries, which is the basis v5.1 uses, and
    which this line can now find - ``face_set_openings`` walks the boundary of
    a feature's own patch, and that walk is the same one LOCATED_ON runs.

    It needs no kernel any more, so ``relations.yaml`` files it as
    ``nx_topology``; it stays a node 20'b relation because that is when the
    edge first appeared, not because it still costs a licence.

    Two clauses, both from the ontology: the boundaries have to be findable,
    and they have to be **separated**.  A zero reading is two slots touching -
    which is what ``084_818_012_solid_001``'s three surviving pairs were under
    node 21's first attempt - and a contact is not a clearance.  There is no
    shared-host gate; host identity is not comparable across the two lines,
    and the note on HOLE_SPACING says why.
    """

    rule_id = "slot_clearance_nx_v1"
    edge_type = "SLOT_CLEARANCE"
    priority = 70
    enabled_by_default = True
    directed = False
    description = "NX SLOTS-family pair, opening boundaries measured on the PartModel."

    family = "SLOTS"
    property_name = "slotClearance"

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        members = [node for node in context.nodes
                   if in_any_family(feature_type(node), (self.family,))]
        for index, left in enumerate(members):
            left_boundaries = opening_boundary_polylines(context, left)
            for right in members[index + 1:]:
                right_boundaries = opening_boundary_polylines(context, right)
                value, details = boundary_set_clearance(left_boundaries, right_boundaries)
                if value is None:
                    _report(context, self.rule_id, details["status"])
                    continue
                if value <= SEPARATION_TOL_MM:
                    details["status"] = "contacting"
                    _report(context, self.rule_id, "contactingOpeningBoundaries")
                    continue
                edges.append(nx_edge(
                    left, right, self.edge_type, self.rule_id, "nx_topology",
                    {
                        "relationBasis": "shortestDistanceBetweenSeparatedOpeningBoundaries",
                        "nxBasis": "PartModel face-set boundary walk plus polyline distance; "
                                   "the same basis cnc:SLOT_CLEARANCE measures on",
                        "featureA": {"nodeId": left.nodeId, "featureType": feature_type(left)},
                        "featureB": {"nodeId": right.nodeId, "featureType": feature_type(right)},
                        "separationTolerance": SEPARATION_TOL_MM,
                        "clearanceWitness": details,
                    },
                    properties={self.property_name: value},
                ))
        return edges
