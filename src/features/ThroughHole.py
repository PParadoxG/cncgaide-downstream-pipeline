"""ThroughHole recognition and ontology property extraction.

This is the complete maintenance unit for the ontology ``ThroughHole``:
semantic seeding, topology validation, feature-face ownership, direct
property extraction, null reasons, and final ``FeatureNode`` construction.
"""

from __future__ import annotations

from typing import Any

from contract.partmodel import PartModel
from features._shared.measurement import min_wall_thickness

from core.models import FeatureNode
from features._shared.attributes import (
    canonical_axis,
    cad_face_ids,
    first_face_bbox_center,
    rounded_scalar,
    serialize_cad_face_ids,
)
from features._shared.closed_wall_band import face_sort_key
from features._shared.evidence import best_hint, grouped_evidence, votes
from features._shared.hole_topology import (
    SignatureView,
    classify_rim_end,
    cylindrical_wall_axial_extent,
    is_reversed_cylindrical_wall,
    unit_vector,
    wall_rims,
)


_HOLE_LABELS = {"through_hole", "blind_hole"}
_FIELDS = (
    "cadFaceIds",
    "depth",
    "derived_ratio_depth_to_diameter",
    "derived_min_wall_thickness",
    "diameter",
    "posX",
    "posY",
    "posZ",
    "feature_axis",
)


def recognize_and_build(
    model: PartModel,
    evidence_document: dict,
    claimed_face_ids: set[str] | None = None,
    *,
    kernel=None,
    start_index: int = 1,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    """Recognize only complete open-to-open cylindrical holes.

    The caller explicitly invokes this module before ``BlindHole``.  This
    function only claims a candidate after it has proven to be ThroughHole,
    so it cannot consume a blind-hole wall while running first.
    """
    claimed = set(claimed_face_ids or ())
    evidence_by_face = grouped_evidence(evidence_document)
    # One signature table and one signature incidence index for this model,
    # shared by every rim walk below.
    signatures = SignatureView(model)
    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    seed_count = 0

    # Numeric face order, as every other recogniser walks it: plain string
    # order puts face_10 before face_9 and numbers the holes by that
    # (finding C-15).
    for wall_id in sorted(model.faces, key=face_sort_key):
        wall = model.faces[wall_id]
        if wall_id in claimed or not is_reversed_cylindrical_wall(wall):
            continue
        hint = best_hint(wall_id, evidence_by_face)
        if not _has_hole_seed(hint):
            continue
        seed_count += 1
        rims = wall_rims(model, wall_id, signatures)
        if len(rims) != 2:
            rejected.append(_rejection(wall_id, hint, "invalid_rim_group_count_or_nonperiodic_wall"))
            continue
        ends = [classify_rim_end(model, wall_id, rim, signatures) for rim in rims]
        if any(end.kind == "unknown" for end in ends):
            rejected.append(_rejection(wall_id, hint, "unknown_endpoint_roles"))
            continue
        if [end.kind for end in ends].count("open") != 2:
            # It is a valid cylindrical candidate but belongs to a different
            # explicit feature module (normally BlindHole), so do not claim it.
            continue

        owned_faces = [wall_id]
        if any(face_id in claimed for face_id in owned_faces):
            continue
        node = _build_node(model, wall_id, owned_faces, rims, hint, evidence_by_face, kernel, start_index + len(nodes))
        nodes.append(node)
        claimed.update(owned_faces)

    diagnostics = {
        "featureModule": "ThroughHole",
        "seedCount": seed_count,
        "acceptedCount": len(nodes),
        "rejectedCandidates": rejected,
    }
    return nodes, claimed, diagnostics


def _has_hole_seed(hint: dict | None) -> bool:
    if not isinstance(hint, dict):
        return False
    if hint.get("top_label") in _HOLE_LABELS:
        return True
    return any(item.get("label") in _HOLE_LABELS for item in hint.get("top_k", [])[:3] if isinstance(item, dict))


def _build_node(
    model: PartModel,
    wall_id: str,
    owned_faces: list[str],
    rims,
    hint: dict | None,
    evidence_by_face: dict[str, list[dict]],
    kernel,
    index: int,
) -> FeatureNode:
    wall = model.faces[wall_id]
    radius = wall.parameters.get("radius")
    diameter = rounded_scalar(2.0 * float(radius)) if isinstance(radius, (int, float)) and float(radius) > 0.0 else None
    positions = sorted({float(rim.axis_position) for rim in rims})
    extent = cylindrical_wall_axial_extent(model, wall_id)
    depth = rounded_scalar(extent) if extent is not None else None
    ratio = rounded_scalar(depth / diameter) if depth is not None and diameter is not None and diameter > 0.0 else None
    axis = _signed_axis(wall.parameters.get("axis_direction"))
    position = first_face_bbox_center(model, owned_faces, wall_id)
    wall_diagnostics: dict[str, Any] = {}
    wall_thickness = min_wall_thickness(
        kernel,
        cad_face_ids(model, owned_faces),
        cad_face_ids(model, [wall_id]),
        material_side=True,
        diagnostics=wall_diagnostics,
    )
    attributes = {
        "cadFaceIds": serialize_cad_face_ids(model, owned_faces),
        "depth": depth,
        "derived_ratio_depth_to_diameter": ratio,
        "derived_min_wall_thickness": wall_thickness,
        "diameter": diameter,
        "posX": position[0] if position else None,
        "posY": position[1] if position else None,
        "posZ": position[2] if position else None,
        "feature_axis": axis,
    }
    missing = _missing(attributes)
    return FeatureNode(
        nodeId=f"ThroughHole_{index:03d}",
        featureType="ThroughHole",
        partmodelFaceIds=list(owned_faces),
        attributes=attributes,
        missingAttributes=missing,
        diagnostics={
            "seed": _seed_diagnostic(wall_id, hint, evidence_by_face),
            "topology": {
                "classification": "two_open_complete_rims",
                "rimAxisPositions": positions,
                "ownedPartmodelFaceIds": list(owned_faces),
                "derivedMinWallThicknessMethod": (
                    kernel.provenance("min_wall_thickness") if kernel is not None
                    else "sampled_wall_normals_material_side_excluding_self_hits"
                ),
            },
            "wallThickness": wall_diagnostics,
        },
    )


def _signed_axis(value: object) -> list[float] | None:
    """Retain the legacy ontology attribute convention for v8 baseline parity."""
    return canonical_axis(value)


def _missing(attributes: dict[str, Any]) -> dict[str, str]:
    reasons = {
        "cadFaceIds": "no source CAD face ids could be recovered from the recognized faces",
        "depth": "the cylindrical wall face has no persisted axial extent",
        "derived_ratio_depth_to_diameter": "positive depth and diameter were not both available",
        "derived_min_wall_thickness": "no non-self material surface was reached from sampled hole-wall normals",
        "diameter": "the cylindrical wall has no positive numeric radius",
        "posX": "no feature face with a valid persisted bounding box was available",
        "posY": "no feature face with a valid persisted bounding box was available",
        "posZ": "no feature face with a valid persisted bounding box was available",
        "feature_axis": "the cylindrical wall has no valid numeric axis direction",
    }
    return {field: reasons[field] for field in _FIELDS if attributes.get(field) is None}


def _seed_diagnostic(wall_id: str, hint: dict | None, evidence_by_face: dict[str, list[dict]]) -> dict[str, Any]:
    return {
        "partmodelFaceId": wall_id,
        "topLabel": hint.get("top_label") if isinstance(hint, dict) else None,
        "topK": hint.get("top_k", [])[:3] if isinstance(hint, dict) else [],
        "memberVotes": votes(wall_id, evidence_by_face),
    }


def _rejection(wall_id: str, hint: dict | None, reason: str) -> dict[str, Any]:
    return {"partmodelFaceId": wall_id, "topLabel": hint.get("top_label") if isinstance(hint, dict) else None, "reason": reason}
