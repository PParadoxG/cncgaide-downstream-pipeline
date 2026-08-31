"""BlindHole recognition and ontology property extraction.

This file is the complete maintenance unit for ``BlindHole``.  It performs
its own detection, ownership, direct ontology-property extraction, null-reason
generation, and node construction; shared modules only provide generic CAD,
topology, and numerical primitives.
"""

from __future__ import annotations

import math
from typing import Any

from contract.partmodel import PartModel, SurfaceType

from features._shared.measurement import min_wall_thickness, remaining_floor_thickness
from core.models import FeatureNode
from features._shared.attributes import (
    cad_face_ids,
    canonical_axis,
    rounded_scalar,
    serialize_cad_face_ids,
)
from features._shared.evidence import best_hint, grouped_evidence, votes
from features._shared.hole_topology import (
    classify_rim_end,
    cylindrical_wall_axial_extent,
    is_reversed_cylindrical_wall,
    unit_vector,
    wall_rims,
)


_HOLE_LABELS = {"through_hole", "blind_hole"}
_FIELDS = (
    "hole_bottom_type",
    "cadFaceIds",
    "depth",
    "derived_floor_thickness",
    "derived_min_wall_thickness",
    "derived_ratio_depth_to_diameter",
    "diameter",
    "machining_direction",
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
    """Recognize only complete open-to-capped cylindrical holes."""
    claimed = claimed_face_ids if claimed_face_ids is not None else set()
    evidence_by_face = grouped_evidence(evidence_document)
    nodes: list[FeatureNode] = []
    rejected: list[dict[str, Any]] = []
    seed_count = 0

    for wall_id, wall in sorted(model.faces.items()):
        if wall_id in claimed or not is_reversed_cylindrical_wall(wall):
            continue
        hint = best_hint(wall_id, evidence_by_face)
        if not _has_hole_seed(hint):
            continue
        seed_count += 1
        rims = wall_rims(model, wall_id)
        if len(rims) != 2:
            rejected.append(_rejection(wall_id, hint, "invalid_rim_group_count_or_nonperiodic_wall"))
            continue
        ends = [classify_rim_end(model, wall_id, rim) for rim in rims]
        if any(end.kind == "unknown" for end in ends):
            rejected.append(_rejection(wall_id, hint, "unknown_endpoint_roles"))
            continue
        open_ends = [end for end in ends if end.kind == "open"]
        capped_ends = [end for end in ends if end.kind == "capped"]
        if len(open_ends) != 1 or len(capped_ends) != 1:
            continue
        bottom_faces = sorted({face_id for end in capped_ends for face_id in end.face_ids})
        owned_faces = [wall_id, *[face_id for face_id in bottom_faces if face_id != wall_id]]
        if any(face_id in claimed for face_id in owned_faces):
            continue
        node = _build_node(
            model,
            wall_id,
            owned_faces,
            bottom_faces,
            open_ends[0].axis_position,
            capped_ends[0].axis_position,
            hint,
            evidence_by_face,
            kernel,
            start_index + len(nodes),
        )
        nodes.append(node)
        claimed.update(owned_faces)

    diagnostics = {
        "featureModule": "BlindHole",
        "seedCount": seed_count,
        "acceptedCount": len(nodes),
        "rejectedCandidates": rejected,
    }
    return nodes, claimed, diagnostics


def _has_hole_seed(hint: dict | None) -> bool:
    if not isinstance(hint, dict):
        return False
    return hint.get("top_label") in _HOLE_LABELS or any(
        item.get("label") in _HOLE_LABELS for item in hint.get("top_k", [])[:3] if isinstance(item, dict)
    )


def _build_node(
    model: PartModel,
    wall_id: str,
    owned_faces: list[str],
    bottom_faces: list[str],
    opening_position: float,
    capped_position: float,
    hint: dict | None,
    evidence_by_face: dict[str, list[dict]],
    kernel,
    index: int,
) -> FeatureNode:
    wall = model.faces[wall_id]
    radius = wall.parameters.get("radius")
    diameter = rounded_scalar(2.0 * float(radius)) if isinstance(radius, (int, float)) and float(radius) > 0.0 else None
    # Rim positions are projected onto the parser's signed cylinder axis, so
    # the floor-thickness ray must use that same raw orientation.  Depth is
    # the shared cylindrical-wall axial extent and does not involve rims or
    # bottom geometry.  The externally emitted ontology attribute retains the
    # legacy canonical undirected representation as a separate value.
    raw_axis = unit_vector(wall.parameters.get("axis_direction"))
    drilling_direction = _drilling_direction(opening_position, capped_position, raw_axis)
    # ``feature_axis`` remains the canonical undirected cylinder axis.  The
    # signed drilling direction is now a published property as well; it is
    # the same direction already used by the floor-thickness ray.
    axis = canonical_axis(raw_axis)
    extent = cylindrical_wall_axial_extent(model, wall_id)
    depth = rounded_scalar(extent) if extent is not None else None
    floor_thickness = _floor_thickness(model, owned_faces, bottom_faces, drilling_direction, kernel)
    ratio = rounded_scalar(depth / diameter) if depth is not None and diameter is not None and diameter > 0.0 else None
    wall_diagnostics: dict[str, Any] = {}
    wall_thickness = _wall_thickness(model, kernel, owned_faces, [wall_id], diagnostics=wall_diagnostics)
    position = _bottom_reference_point(
        model,
        bottom_faces,
        raw_axis,
        wall.parameters.get("axis_location"),
    )
    attributes = {
        "hole_bottom_type": _hole_bottom_type(model, bottom_faces),
        "cadFaceIds": serialize_cad_face_ids(model, owned_faces),
        "depth": depth,
        "derived_floor_thickness": floor_thickness,
        "derived_min_wall_thickness": wall_thickness,
        "derived_ratio_depth_to_diameter": ratio,
        "diameter": diameter,
        "machining_direction": drilling_direction,
        "posX": position[0] if position else None,
        "posY": position[1] if position else None,
        "posZ": position[2] if position else None,
        "feature_axis": axis,
    }
    return FeatureNode(
        nodeId=f"BlindHole_{index:03d}",
        featureType="BlindHole",
        partmodelFaceIds=list(owned_faces),
        attributes=attributes,
        missingAttributes=_missing(attributes),
        diagnostics={
            "seed": _seed_diagnostic(wall_id, hint, evidence_by_face),
            "topology": {
                "classification": "one_open_one_capped_complete_rims",
                "openingAxisPosition": opening_position,
                "cappedAxisPosition": capped_position,
                "bottomPartmodelFaceIds": bottom_faces,
                "ownedPartmodelFaceIds": list(owned_faces),
                "machiningDirection": drilling_direction,
            },
            "wallThickness": wall_diagnostics,
        },
    )


def _floor_thickness(model: PartModel, owned_faces: list[str], bottom_faces: list[str], drilling_direction: list[float] | None, kernel) -> float | None:
    if kernel is None or drilling_direction is None:
        return None
    return remaining_floor_thickness(
        kernel,
        cad_face_ids(model, owned_faces),
        cad_face_ids(model, bottom_faces),
        drilling_direction,
    )


def _wall_thickness(
    model: PartModel,
    kernel,
    owned_faces: list[str],
    wall_face_ids: list[str],
    *,
    diagnostics: dict[str, Any] | None = None,
) -> float | None:
    """Measure the nearest material wall from the hole wall's material side."""
    return min_wall_thickness(
        kernel,
        cad_face_ids(model, owned_faces),
        cad_face_ids(model, wall_face_ids),
        material_side=True,
        diagnostics=diagnostics,
    )


def _hole_bottom_type(model: PartModel, bottom_face_ids: list[str]) -> str | None:
    """Extract the BlindHole-only ontology bottom-type property."""
    faces = [model.faces.get(face_id) for face_id in bottom_face_ids]
    if not faces or any(face is None for face in faces):
        return None
    surface_types = {face.surface_type for face in faces}
    if surface_types == {SurfaceType.PLANE}:
        return "flat"
    if surface_types <= {SurfaceType.SPHERE, SurfaceType.TORUS}:
        return "rounded"
    if surface_types != {SurfaceType.CONE}:
        return None
    angles = []
    for face in faces:
        semi_angle = face.parameters.get("semi_angle")
        if not isinstance(semi_angle, (int, float)) or not math.isfinite(float(semi_angle)):
            return None
        angles.append(2.0 * math.degrees(abs(float(semi_angle))))
    if all(abs(angle - 118.0) <= 1.0 for angle in angles):
        return "conical_118"
    if all(abs(angle - 135.0) <= 1.0 for angle in angles):
        return "conical_135"
    return None


def _bottom_reference_point(
    model: PartModel,
    bottom_face_ids: list[str],
    axis: list[float] | None,
    axis_origin: list[float] | None,
) -> list[float] | None:
    """Return the axis/bottom intersection or the deepest conical apex.

    A persisted bounding-box centre is not a geometric reference for a blind
    hole: it can be off-axis on a trimmed or merged face.  Planar bottoms are
    intersected with the cylindrical wall axis.  A conical bottom publishes
    its parser-provided apex, which is the deepest point on that axis.  Other
    mixed or under-specified bottom geometry remains unavailable rather than
    receiving a guessed point.
    """
    if not bottom_face_ids or axis is None:
        return None
    faces = [model.faces.get(face_id) for face_id in bottom_face_ids]
    if any(face is None for face in faces):
        return None
    surface_types = {face.surface_type for face in faces}
    if surface_types == {SurfaceType.CONE}:
        apexes = []
        for face in faces:
            value = face.parameters.get("apex")
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                return None
            try:
                apexes.append([float(component) for component in value])
            except (TypeError, ValueError):
                return None
        reference = apexes[0]
        if any(math.dist(reference, apex) > 1.0e-5 for apex in apexes[1:]):
            return None
        return [round(component, 6) + 0.0 for component in reference]
    if surface_types != {SurfaceType.PLANE} or axis_origin is None:
        return None
    if not isinstance(axis_origin, (list, tuple)) or len(axis_origin) != 3:
        return None
    try:
        origin = [float(component) for component in axis_origin]
    except (TypeError, ValueError):
        return None
    intersections = []
    for face in faces:
        normal = unit_vector(face.parameters.get("normal"))
        point = face.parameters.get("location")
        if normal is None or not isinstance(point, (list, tuple)) or len(point) != 3:
            return None
        try:
            plane_point = [float(component) for component in point]
        except (TypeError, ValueError):
            return None
        denominator = sum(axis[index] * normal[index] for index in range(3))
        if abs(denominator) <= 1.0e-12:
            return None
        distance = sum((plane_point[index] - origin[index]) * normal[index] for index in range(3)) / denominator
        intersections.append([origin[index] + distance * axis[index] for index in range(3)])
    reference = intersections[0]
    if any(math.dist(reference, point) > 1.0e-5 for point in intersections[1:]):
        return None
    return [round(component, 6) + 0.0 for component in reference]


def _drilling_direction(opening_position: float, capped_position: float, axis: list[float] | None) -> list[float] | None:
    if axis is None:
        return None
    sign = 1.0 if capped_position > opening_position else -1.0
    return [component * sign for component in axis]


def _missing(attributes: dict[str, Any]) -> dict[str, str]:
    reasons = {
        "hole_bottom_type": "the recognized bottom faces did not establish flat, conical_118, conical_135, or rounded geometry",
        "cadFaceIds": "no source CAD face ids could be recovered from the recognized faces",
        "depth": "the cylindrical wall face has no persisted axial extent",
        "derived_floor_thickness": "no reliable first opposite-surface hit was found below the blind-hole bottom",
        "derived_min_wall_thickness": "no non-self material surface was reached from sampled blind-hole wall normals",
        "derived_ratio_depth_to_diameter": "positive depth and diameter were not both available",
        "diameter": "the cylindrical wall has no positive numeric radius",
        "machining_direction": "the blind-hole opening and bottom did not establish a signed drilling direction",
        "posX": "the bottom geometry did not establish a unique axis reference point (requires consistent all-planar axis intersections or a common all-conical apex)",
        "posY": "the bottom geometry did not establish a unique axis reference point (requires consistent all-planar axis intersections or a common all-conical apex)",
        "posZ": "the bottom geometry did not establish a unique axis reference point (requires consistent all-planar axis intersections or a common all-conical apex)",
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
