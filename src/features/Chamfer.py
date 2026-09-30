"""Chamfer recognition and ontology attribute extraction."""

from __future__ import annotations

from typing import Any

from contract.partmodel import PartModel

from core.models import FeatureNode
from features._shared.attributes import serialize_cad_face_ids, rounded_scalar
from features._shared.transition_evidence import TransitionEvidence, build_transition_evidence
from features._shared.transition_geometry import measure_transition, transition_regions


_FIELDS = ("cadFaceIds", "chamfer_width", "min_chamfer_angle", "posX", "posY", "posZ")


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None,
    claimed_face_ids: set[str] | None = None,
    *,
    transition: TransitionEvidence | None = None,
    kernel: Any | None = None,
    start_index: int = 1,
    extra_face_ids: set[str] | frozenset[str] | None = None,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    """Register connected planar/conical transition regions as Chamfers.

    AAGNet supplies only the transition proposal.  The region must be
    connected in the active PartModel and be geometrically planar/conical;
    no native AAGNet instance group is consulted.

    ``extra_face_ids`` adds faces that a geometric proof elsewhere has shown to be transitions
    even if AAGNet did not vote for them - today, the cones the hole grammar declined to absorb
    into a compound.  Only eligibility is widened: the region walk, the measurement and the node
    are unchanged, so a face admitted this way still has to measure as a chamfer to become one.
    Without it a released cone would belong to nobody, because the residual passes register
    planes and freeform surfaces and never cones.
    """

    claimed = set(claimed_face_ids or ())
    transition = transition or build_transition_evidence(model, evidence_document or {})
    if extra_face_ids:
        transition = transition.with_extra_faces(extra_face_ids)
    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    regions = transition_regions(model, transition, claimed)
    for region in regions:
        measurement = measure_transition(model, region.face_ids)
        if measurement.kind != "Chamfer":
            rejected.append({"faceIds": list(region.face_ids), "reason": measurement.reason or "not_a_planar_or_conical_chamfer", "surfaceTypes": list(region.surface_types)})
            continue
        node = _build_node(model, region.face_ids, measurement, region.verdict, start_index + len(nodes))
        nodes.append(node)
        claimed.update(region.face_ids)
    diagnostics = {
        "featureModule": "Chamfer",
        "method": "transition_evidence_connected_region_geometry_v1",
        "aagnetUsed": "transition_face_and_region_score_only",
        # Finding C-08: this used to be a second, unclaimed-blind walk of the
        # whole part, so the same key meant "every transition region on the
        # part" here and "the regions still unclaimed when I ran" in Fillet,
        # and the two modules' candidate-to-accepted ratios were not
        # comparable.  One meaning now: what this module saw.  Reusing the
        # walk above also retires the duplicate region walk (CP-4).
        "regionCandidateCount": len(regions),
        "acceptedCount": len(nodes),
        "claimedFaceCount": sum(len(node.partmodelFaceIds) for node in nodes),
        "rejectedRegions": rejected,
        "transitionEvidence": transition.diagnostics(),
    }
    return nodes, claimed, diagnostics


def _build_node(model: PartModel, face_ids, measurement, verdict, index: int) -> FeatureNode:
    centroid = measurement.centroid
    attributes = {
        "cadFaceIds": serialize_cad_face_ids(model, list(face_ids)),
        "chamfer_width": rounded_scalar(measurement.width),
        "min_chamfer_angle": rounded_scalar(measurement.angle),
        "posX": rounded_scalar(centroid[0]) if centroid else None,
        "posY": rounded_scalar(centroid[1]) if centroid else None,
        "posZ": rounded_scalar(centroid[2]) if centroid else None,
    }
    reasons = {
        "cadFaceIds": "no source CAD face ids were recoverable from the transition region",
        "chamfer_width": "two measurable transition boundaries or planar principal extents were unavailable",
        "min_chamfer_angle": "the two support faces did not provide a valid acute angle",
        "posX": "the transition region has no stable area centroid",
        "posY": "the transition region has no stable area centroid",
        "posZ": "the transition region has no stable area centroid",
    }
    return FeatureNode(
        nodeId=f"Chamfer_{index:03d}",
        featureType="Chamfer",
        partmodelFaceIds=list(face_ids),
        attributes=attributes,
        missingAttributes={field: reasons[field] for field in _FIELDS if attributes.get(field) is None},
        diagnostics={
            "method": "transition_evidence_connected_region_geometry_v1",
            "transitionRegionVerdict": verdict,
            "measurement": measurement.diagnostics,
            "measurementReason": measurement.reason,
            "featureAxisPolicy": "not_applicable_edge_treatment",
        },
    )
