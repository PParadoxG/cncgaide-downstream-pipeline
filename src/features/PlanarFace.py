"""Residual PlanarFace registration and ontology attribute extraction."""

from __future__ import annotations

import math
from typing import Any

from contract.partmodel import PartModel, SurfaceType

from core.models import FeatureNode
from features._shared.attributes import rounded_scalar, serialize_cad_face_ids
from features._shared.transition_geometry import face_sort_key, region_centroid
from features._shared.hole_topology import unit_vector


_FIELDS = ("cadFaceIds", "surface_normal", "area", "posX", "posY", "posZ")


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
        if face_id in claimed or face.surface_type != SurfaceType.PLANE:
            continue
        if not _valid_face(face):
            rejected.append({"faceId": face_id, "reason": "invalid_or_zero_area_planar_face"})
            continue
        node = _build_node(model, face_id, kernel, start_index + len(nodes))
        nodes.append(node)
        claimed.add(face_id)
    diagnostics = {
        "featureModule": "PlanarFace",
        "method": "residual_partmodel_planar_face_registration_v1",
        "acceptedCount": len(nodes),
        "claimedFaceCount": len(nodes),
        "rejectedFaces": rejected,
        "surfaceNormalPolicy": "signed_outward_face_normal; feature_axis_not_applicable",
        "instanceBoundaryPolicy": "one active PartModel face/region per node; coplanar faces are not globally merged",
    }
    return nodes, claimed, diagnostics


def _valid_face(face) -> bool:
    return isinstance(face.area, (int, float)) and math.isfinite(float(face.area)) and float(face.area) > 1.0e-9


def _build_node(model: PartModel, face_id: str, kernel: Any | None, index: int) -> FeatureNode:
    face = model.faces[face_id]
    raw_normal = unit_vector(face.parameters.get("normal"))
    if raw_normal is not None and face.is_reversed:
        raw_normal = [-value for value in raw_normal]
    centroid = region_centroid(model, [face_id])
    attributes = {
        "cadFaceIds": serialize_cad_face_ids(model, [face_id]),
        "surface_normal": [round(value, 6) + 0.0 for value in raw_normal] if raw_normal is not None else None,
        # Bounded face area (mm²).  The Surface background layer of the GED
        # aggregates it; the bare PartModel area is the only source.
        "area": rounded_scalar(face.area) if _valid_face(face) else None,
        "posX": rounded_scalar(centroid[0]) if centroid else None,
        "posY": rounded_scalar(centroid[1]) if centroid else None,
        "posZ": rounded_scalar(centroid[2]) if centroid else None,
    }
    reasons = {
        "cadFaceIds": "no source CAD face ids were recoverable from the planar face",
        "surface_normal": "the planar face has no valid normal or outward orientation",
        "area": "the planar face has no valid positive area",
        "posX": "the planar face has no stable area centroid",
        "posY": "the planar face has no stable area centroid",
        "posZ": "the planar face has no stable area centroid",
    }
    return FeatureNode(
        nodeId=f"PlanarFace_{index:03d}",
        featureType="PlanarFace",
        partmodelFaceIds=[face_id],
        attributes=attributes,
        missingAttributes={field: reasons[field] for field in _FIELDS if attributes.get(field) is None},
        diagnostics={
            "method": "residual_partmodel_planar_face_registration_v1",
            "surfaceType": face.surface_type.value,
            "area": rounded_scalar(face.area),
            "outwardNormalSource": "parameters.normal flipped by FaceData.is_reversed",
        },
    )
