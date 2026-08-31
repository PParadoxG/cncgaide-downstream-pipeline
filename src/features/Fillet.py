"""Fillet recognition and ontology attribute extraction."""

from __future__ import annotations

from typing import Any

from contract.partmodel import PartModel

from core.models import FeatureNode
from features._shared.attributes import rounded_scalar, serialize_cad_face_ids
from features._shared.transition_evidence import TransitionEvidence, build_transition_evidence
from features._shared.transition_geometry import measure_transition, transition_regions


_FIELDS = ("cadFaceIds", "fillet_radius", "posX", "posY", "posZ")


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None,
    claimed_face_ids: set[str] | None = None,
    *,
    transition: TransitionEvidence | None = None,
    kernel: Any | None = None,
    start_index: int = 1,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    claimed = claimed_face_ids if claimed_face_ids is not None else set()
    transition = transition or build_transition_evidence(model, evidence_document or {})
    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    regions = transition_regions(model, transition, claimed)
    for region in regions:
        measurement = measure_transition(model, region.face_ids)
        if measurement.kind != "Fillet":
            rejected.append({"faceIds": list(region.face_ids), "reason": measurement.reason or "not_a_smooth_blend_region", "surfaceTypes": list(region.surface_types)})
            continue
        node = _build_node(model, region.face_ids, measurement, region.verdict, start_index + len(nodes))
        nodes.append(node)
        claimed.update(region.face_ids)
    diagnostics = {
        "featureModule": "Fillet",
        "method": "transition_evidence_connected_region_geometry_v1",
        "aagnetUsed": "transition_face_and_region_score_only",
        "regionCandidateCount": len(regions),
        "acceptedCount": len(nodes),
        "claimedFaceCount": sum(len(node.partmodelFaceIds) for node in nodes),
        "rejectedRegions": rejected,
        "variableRadiusWithheldCount": sum(1 for node in nodes if node.attributes.get("fillet_radius") is None),
        "transitionEvidence": transition.diagnostics(),
    }
    return nodes, claimed, diagnostics


def _build_node(model: PartModel, face_ids, measurement, verdict, index: int) -> FeatureNode:
    centroid = measurement.centroid
    attributes = {
        "cadFaceIds": serialize_cad_face_ids(model, list(face_ids)),
        "fillet_radius": rounded_scalar(measurement.radius),
        "posX": rounded_scalar(centroid[0]) if centroid else None,
        "posY": rounded_scalar(centroid[1]) if centroid else None,
        "posZ": rounded_scalar(centroid[2]) if centroid else None,
    }
    reasons = {
        "cadFaceIds": "no source CAD face ids were recoverable from the transition region",
        "fillet_radius": "the blend is variable-radius/freeform or analytic radius candidates disagree; no average was guessed",
        "posX": "the transition region has no stable area centroid",
        "posY": "the transition region has no stable area centroid",
        "posZ": "the transition region has no stable area centroid",
    }
    return FeatureNode(
        nodeId=f"Fillet_{index:03d}",
        featureType="Fillet",
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
