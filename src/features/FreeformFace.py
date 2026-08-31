"""Residual non-analytic FreeformFace registration."""

from __future__ import annotations

import math
from typing import Any

from contract.partmodel import PartModel, SurfaceType

from core.models import FeatureNode
from features._shared.attributes import rounded_scalar, serialize_cad_face_ids
from features._shared.hole_topology import unit_vector
from features._shared.transition_geometry import face_sort_key, region_centroid


_FREEFORM_TYPES = {SurfaceType.BSPLINE, SurfaceType.REVOLUTION, SurfaceType.EXTRUSION}
_FIELDS = ("cadFaceIds", "area", "posX", "posY", "posZ")


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None = None,
    claimed_face_ids: set[str] | None = None,
    *,
    kernel: Any | None = None,
    start_index: int = 1,
    **_: Any,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    claimed = claimed_face_ids if claimed_face_ids is not None else set()
    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    for face_id in sorted(model.faces, key=face_sort_key):
        face = model.faces[face_id]
        if face_id in claimed:
            continue
        if face.surface_type == SurfaceType.OTHER:
            rejected.append({"faceId": face_id, "reason": "unsupported_or_unclassified_surface_is_not_proven_freeform", "surfaceType": face.surface_type.value})
            continue
        if face.surface_type not in _FREEFORM_TYPES:
            continue
        eligibility = _freeform_eligibility(face.surface_type, face.parameters)
        if not eligibility["eligible"]:
            rejected.append({"faceId": face_id, "reason": eligibility["reason"], "surfaceType": face.surface_type.value, "parameters": eligibility.get("parameters", {})})
            continue
        if not isinstance(face.area, (int, float)) or float(face.area) <= 1.0e-9:
            rejected.append({"faceId": face_id, "reason": "invalid_or_zero_area_freeform_face"})
            continue
        node = _build_node(model, face_id, kernel, start_index + len(nodes))
        nodes.append(node)
        claimed.add(face_id)
    diagnostics = {
        "featureModule": "FreeformFace",
        "method": "residual_nonanalytic_surface_registration_v1",
        "acceptedCount": len(nodes),
        "claimedFaceCount": len(nodes),
        "surfaceTypes": sorted(surface.value for surface in _FREEFORM_TYPES),
        "eligibilityPolicy": "BSPLINE accepted; REVOLUTION/EXTRUSION require parser basis_curve_type BSPLINE or BEZIER; OTHER and unproven fallback classifications withheld",
        "rejectedFaces": rejected,
        "analyticSurfacePolicy": "PLANE/CYLINDER/CONE/SPHERE/TORUS are never reclassified as FreeformFace",
    }
    return nodes, claimed, diagnostics


def _freeform_eligibility(surface_type: SurfaceType, parameters: dict) -> dict[str, Any]:
    if surface_type == SurfaceType.BSPLINE:
        return {"eligible": True, "reason": "bspline_surface"}
    if surface_type in {SurfaceType.REVOLUTION, SurfaceType.EXTRUSION}:
        basis = parameters.get("basis_curve_type") if isinstance(parameters, dict) else None
        if str(basis).upper() in {"BSPLINE", "BEZIER"}:
            return {"eligible": True, "reason": "non_elementary_bspline_or_bezier_generatrix", "parameters": {"basis_curve_type": basis}}
        return {"eligible": False, "reason": "non_elementary_generatrix_not_proven_by_parser_parameters", "parameters": {"basis_curve_type": basis}}
    return {"eligible": False, "reason": "unsupported_or_unclassified_surface_is_not_proven_freeform"}


def _build_node(model: PartModel, face_id: str, kernel: Any | None, index: int) -> FeatureNode:
    face = model.faces[face_id]
    centroid = region_centroid(model, [face_id])
    area_valid = isinstance(face.area, (int, float)) and math.isfinite(float(face.area)) and float(face.area) > 1.0e-9
    attributes = {
        "cadFaceIds": serialize_cad_face_ids(model, [face_id]),
        # Bounded face area (mm²), aggregated by the GED Surface background layer.
        "area": rounded_scalar(face.area) if area_valid else None,
        "posX": rounded_scalar(centroid[0]) if centroid else None,
        "posY": rounded_scalar(centroid[1]) if centroid else None,
        "posZ": rounded_scalar(centroid[2]) if centroid else None,
    }
    reasons = {
        "cadFaceIds": "no source CAD face ids were recoverable from the freeform face",
        "area": "the freeform face has no valid positive area",
        "posX": "the freeform face has no stable area centroid",
        "posY": "the freeform face has no stable area centroid",
        "posZ": "the freeform face has no stable area centroid",
    }
    return FeatureNode(
        nodeId=f"FreeformFace_{index:03d}",
        featureType="FreeformFace",
        partmodelFaceIds=[face_id],
        attributes=attributes,
        missingAttributes={field: reasons[field] for field in _FIELDS if attributes.get(field) is None},
        diagnostics={
            "method": "residual_nonanalytic_surface_registration_v1",
            "surfaceType": face.surface_type.value,
            "area": rounded_scalar(face.area),
            "eligibility": _freeform_eligibility(face.surface_type, face.parameters),
        },
    )
