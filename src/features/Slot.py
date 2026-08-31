"""Formal ontology ``Slot`` recognition and property extraction.

The structural proof is supplied by shared Open Sweep analysis.  This module
only consumes a candidate after the shared recessed-terminal gate, AAGNet
family decision, and the mandatory two-ended exterior proof have passed.  A
profile classifier is deliberately measurement-only: a valid Slot whose
profile is not one of the strict common forms remains a Slot with ``other``
diagnostics and null profile-dependent measurements.
"""

from __future__ import annotations

import math
from typing import Any

from contract.partmodel import CurveType, PartModel, SurfaceType

from features._shared.measurement import min_wall_thickness
from core.models import FeatureNode
from features._shared.attributes import canonical_axis, cad_face_ids, rounded_scalar, serialize_cad_face_ids
from features._shared.open_sweep import OpenSweepAnalysis, OpenSweepCandidate
from features._shared.slot_step_resolution import (
    CandidateResolution,
    resolve_open_sweep_candidates,
)
from features._shared.transition_evidence import TransitionEvidence


RECTANGULAR = "RectangularSlot"
TRIANGULAR = "TriangularSlot"
CIRCULAR = "CircularSlot"
GENERAL = "GeneralSlot"
_COMMON_FIELDS = ("cadFaceIds", "feature_axis", "machining_direction", "depth", "course_length", "derived_min_wall_thickness", "posX", "posY", "posZ")
_ATTRIBUTE_FIELDS = {
    RECTANGULAR: (*_COMMON_FIELDS, "min_opening_width", "min_bottom_width", "max_wall_draft_angle", "derived_ratio_depth_to_min_width"),
    TRIANGULAR: (*_COMMON_FIELDS, "min_opening_width", "v_included_angle", "derived_ratio_depth_to_min_width"),
    CIRCULAR: (*_COMMON_FIELDS, "min_opening_width", "profile_radius", "derived_ratio_depth_to_min_width"),
    GENERAL: _COMMON_FIELDS,
}

_GEOMETRY_TOL = 1.0e-4
_VECTOR_TOL = 1.0e-12


def recognize_and_build(
    model: PartModel,
    evidence_document: dict | None,
    claimed_face_ids: set[str] | None = None,
    *,
    sweeps: OpenSweepAnalysis,
    transition: TransitionEvidence | None = None,
    kernel: Any | None = None,
    resolutions: list[CandidateResolution] | None = None,
    start_index: int = 1,
    **_: Any,
) -> tuple[list[FeatureNode], set[str], dict[str, Any]]:
    """Build Slot nodes from shared semantic resolutions."""

    claimed = claimed_face_ids if claimed_face_ids is not None else set()
    if resolutions is None:
        if transition is None:
            transition = TransitionEvidence(model, evidence_document)
        resolutions = resolve_open_sweep_candidates(
            model,
            evidence_document,
            sweeps,
            transition=transition,
            kernel=kernel,
            claimed_face_ids=claimed,
        )

    nodes: list[FeatureNode] = []
    counters = {RECTANGULAR: 0, TRIANGULAR: 0, CIRCULAR: 0, GENERAL: 0}
    rejected: list[dict[str, Any]] = []
    semantic_records: list[dict[str, Any]] = []
    for resolution in resolutions:
        record = resolution.diagnostics()
        semantic_records.append(record)
        if resolution.decision != "Slot":
            if resolution.reject_reason or resolution.decision == "Step":
                rejected.append({
                    "candidateKey": resolution.candidate.candidate_key,
                    "reason": resolution.reject_reason or "resolved_as_step",
                    "detail": resolution.reject_detail,
                    "semanticDecision": resolution.decision,
                })
            continue
        principal = list(resolution.candidate.principal_face_ids)
        overlap = sorted(set(principal) & claimed)
        if overlap:
            rejected.append({
                "candidateKey": resolution.candidate.candidate_key,
                "reason": "principal_faces_claimed_before_slot_registration",
                "detail": ",".join(overlap),
            })
            continue
        profile = classify_profile(model, resolution.candidate)
        geometry = measure_profile(model, resolution.candidate, profile)
        feature_type = {
            "rectangular": RECTANGULAR,
            "triangular_v": TRIANGULAR,
            "semicircular": CIRCULAR,
            "rounded_u": CIRCULAR,
            "other": GENERAL,
        }.get(profile.get("class"), GENERAL)
        counters[feature_type] += 1
        node = _build_node(
            model,
            resolution,
            profile,
            geometry,
            feature_type,
            counters[feature_type],
            kernel,
        )
        nodes.append(node)
        claimed.update(principal)

    diagnostics = {
        "featureModule": "Slot",
        "ontologyFeature": True,
        "method": "open_sweep_terminal_resolution_aagnet_family_resolution_slot_v1",
        "candidateCount": len(sweeps.candidates),
        "semanticResolverCandidateCount": len(resolutions),
        "acceptedCount": len(nodes),
        "acceptedCountByLeafClass": {key: value for key, value in sorted(counters.items()) if value},
        "claimedFaceCount": sum(len(node.partmodelFaceIds) for node in nodes),
        "ownershipPolicy": (
            "only OS principal faces; transverse supports, longitudinal end supports, "
            "and transition faces remain unclaimed"
        ),
        "leafClassification": "rectangular, triangular_v, circular and other profiles map to the four Slot leaves",
        "referencePointConvention": (
            "the deepest transverse profile point; rectangular floors use the root centre, "
            "not the mouth geometric centre"
        ),
        "semanticCandidates": semantic_records,
        "rejectedCandidates": rejected,
    }
    return nodes, claimed, diagnostics


def classify_profile(model: PartModel, candidate: OpenSweepCandidate) -> dict[str, Any]:
    """Strictly classify only the common transverse profiles used for metrics."""

    faces = candidate.ordered_profile_face_ids
    surfaces = [model.faces[face_id].surface_type.value for face_id in faces if face_id in model.faces]
    if candidate.discovery_mode == "single_cylindrical_profile":
        if _strict_semicircle(model, candidate):
            return {"class": "semicircular", "surfaces": surfaces, "method": "single_cylinder_arc_and_half_turn"}
        return {
            "class": "other",
            "surfaces": surfaces,
            "method": "single_cylinder_arc_not_strict_semicircle",
            "reason": "cylindrical arc was not proved to have a complete semicircular transverse profile",
        }

    if len(faces) == 2 and all(surface == SurfaceType.PLANE.value for surface in surfaces):
        normals = [_face_normal(model.faces[face_id]) for face_id in faces]
        if all(value is not None for value in normals) and not _parallel(normals[0], normals[1]):
            return {"class": "triangular_v", "surfaces": surfaces, "method": "two_non_coplanar_planar_faces"}

    if len(faces) == 3 and surfaces == [SurfaceType.PLANE.value, SurfaceType.PLANE.value, SurfaceType.PLANE.value]:
        normals = [_face_normal(model.faces[face_id]) for face_id in faces]
        if all(value is not None for value in normals):
            mouth = _mouth_reference(model, candidate)
            deepest = _deepest_profile_point(model, candidate, "rectangular", mouth)
            axis = _depth_axis(model, candidate, mouth, deepest)
            floor_perpendicular = (
                axis is not None
                and abs(abs(_dot(normals[1], axis)) - 1.0) <= _GEOMETRY_TOL
            )
            # R-1: each wall may carry its own stable in-plane draft.  The
            # former parallel-wall gate rejected exactly the drafted
            # rectangular slots that the two station widths describe.
            if floor_perpendicular:
                return {
                    "class": "rectangular",
                    "surfaces": surfaces,
                    "method": "three_planar_faces_stable_side_drafts_perpendicular_floor",
                    "sideWallParallel": _parallel(normals[0], normals[2]),
                }

    if (
        len(faces) == 3
        and len(surfaces) == 3
        and surfaces[0] == SurfaceType.PLANE.value
        and surfaces[1] == SurfaceType.CYLINDER.value
        and surfaces[2] == SurfaceType.PLANE.value
    ):
        normals = [_face_normal(model.faces[face_id]) for face_id in (faces[0], faces[2])]
        cylinder = model.faces[faces[1]]
        axis = _unit(cylinder.parameters.get("axis_direction"))
        course = _unit(candidate.course_direction)
        radius = cylinder.parameters.get("radius")
        if (
            all(value is not None for value in normals)
            and _parallel(normals[0], normals[1])
            and axis is not None
            and course is not None
            and _circular_side_walls_are_undrafted(model, candidate, normals[0], normals[1])
            and _parallel(axis, course)
            and isinstance(radius, (int, float))
            and float(radius) > 0.0
            and len(candidate.internal_rail_connections) == 2
        ):
            return {
                "class": "rounded_u",
                "surfaces": surfaces,
                "method": "plane_cylinder_plane_with_opposite_side_walls",
            }

    return {
        "class": "other",
        "surfaces": surfaces,
        "method": "strict_common_profile_classifier",
        "reason": "transverse profile did not meet a strict rectangular, V, semicircular, or rounded-U proof",
    }


def _circular_side_walls_are_undrafted(model, candidate, left_normal, right_normal):
    """Require plane side walls to stay in the mouth-chord direction.

    CircularSlot has no station-wise draft property.  A plane--cylinder--plane
    section whose side normals tilt away from the terminal mouth chord is a
    drafted arc slot and therefore belongs to GeneralSlot.  The two joins
    themselves need not be smooth: a non-tangent partial arc is still a
    CircularSlot when its walls remain undrafted.
    """
    course = _unit(candidate.course_direction)
    chord = _mouth_chord(model, candidate, course)
    if chord is None:
        return False
    return all(abs(abs(_dot(normal, chord)) - 1.0) <= _GEOMETRY_TOL for normal in (left_normal, right_normal))


def measure_profile(
    model: PartModel,
    candidate: OpenSweepCandidate,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Extract Slot properties without making extraction a recognition gate."""

    class_name = profile.get("class", "other")
    mouth = _mouth_reference(model, candidate)
    deepest = _deepest_profile_point(model, candidate, class_name, mouth)
    axis = _depth_axis(model, candidate, mouth, deepest)
    depth = _depth(axis, mouth, deepest)
    mouth_width = _mouth_width(model, candidate)
    width = _width(model, candidate, class_name, mouth, axis, depth)
    bottom_width = _width_at_depth(model, candidate, mouth, axis, depth) if class_name == "rectangular" else None
    if class_name == "triangular_v":
        width = mouth_width
    if class_name == "semicircular":
        radius = _cylinder_radius(model, candidate)
        if radius is not None:
            width = 2.0 * radius
            depth = radius
            axis, deepest = _semicircle_axis_and_point(model, candidate, mouth, axis, deepest, radius)

    # A valid Slot with an unsupported profile is still registered, but no
    # profile-dependent direction/depth/width is guessed from an arbitrary
    # internal rail or face centroid.
    if class_name == "other":
        deepest = _deepest_profile_point(model, candidate, class_name, mouth)
        axis = _depth_axis(model, candidate, mouth, deepest)
        depth = _depth(axis, mouth, deepest)

    ratio = None
    if depth is not None and width is not None and width > _VECTOR_TOL:
        ratio = depth / width

    v_angle = _v_included_angle(model, candidate) if class_name == "triangular_v" else None
    max_draft = _max_wall_draft_angle(model, candidate, axis) if class_name == "rectangular" else None

    reasons = {
        "feature_axis": "the transverse mouth-to-deepest-point direction could not be established",
        "machining_direction": "the transverse mouth-to-deepest-point direction could not be established",
        "depth": "the profile deepest point could not be measured",
        "min_opening_width": "the slot mouth chord could not be measured",
        "min_bottom_width": "the rectangular slot bottom station could not be measured",
        "derived_ratio_depth_to_min_width": "depth and a positive minimum width were not both available",
        "posX": "the two terminal rails did not provide geometric midpoints",
        "posY": "the two terminal rails did not provide geometric midpoints",
        "posZ": "the two terminal rails did not provide geometric midpoints",
        "cadFaceIds": "no source CAD face ids could be recovered from the recognized principal faces",
        "course_length": "the open sweep has no finite course span",
        "derived_min_wall_thickness": "no non-self material surface was reached from sampled slot-wall normals",
        "v_included_angle": "the two V walls have no valid included angle",
        "profile_radius": "the circular slot has no positive cylindrical radius",
        "max_wall_draft_angle": "the rectangular slot walls have no stable sampled drafts",
    }
    if class_name == "triangular_v":
        reasons["min_opening_width"] = "the V mouth chord could not be measured"
    elif class_name == "other":
        reasons["min_opening_width"] = "unsupported or unclassified transverse slot profile"
        reasons["feature_axis"] = "the general profile deepest point could not be measured"
        reasons["machining_direction"] = "the general profile deepest point could not be measured"
        reasons["depth"] = "the general profile deepest point could not be measured"

    return {
        "profileClass": class_name,
        "feature_axis": canonical_axis(axis),
        "machining_direction": _vector(axis),
        "depth": rounded_scalar(depth),
        "minWidth": rounded_scalar(width),
        "minBottomWidth": rounded_scalar(bottom_width),
        "vIncludedAngle": rounded_scalar(v_angle),
        "profileRadius": rounded_scalar(_cylinder_radius(model, candidate)) if class_name in {"semicircular", "rounded_u"} else None,
        "maxWallDraftAngle": rounded_scalar(max_draft),
        "ratio": rounded_scalar(ratio),
        "courseLength": rounded_scalar(candidate.course_length),
        "mouthReferencePoint": mouth,
        "deepestPoint": deepest,
        "missingReasons": reasons,
    }


def _build_node(
    model: PartModel,
    resolution: CandidateResolution,
    profile: dict[str, Any],
    geometry: dict[str, Any],
    feature_type: str,
    leaf_index: int,
    kernel,
) -> FeatureNode:
    principal = list(resolution.candidate.principal_face_ids)
    fields = _ATTRIBUTE_FIELDS[feature_type]
    attributes: dict[str, Any] = {name: None for name in fields}
    cad_ids = serialize_cad_face_ids(model, principal)
    attributes["cadFaceIds"] = cad_ids
    attributes["feature_axis"] = geometry["feature_axis"]
    attributes["machining_direction"] = geometry["machining_direction"]
    attributes["depth"] = geometry["depth"]
    attributes["course_length"] = geometry["courseLength"]
    wall_face_ids = _wall_sampling_face_ids(resolution.candidate, feature_type)
    wall_diagnostics: dict[str, Any] = {}
    attributes["derived_min_wall_thickness"] = _wall_thickness(
        model,
        kernel,
        principal,
        wall_face_ids,
        diagnostics=wall_diagnostics,
    )
    if feature_type == RECTANGULAR:
        attributes["min_opening_width"] = geometry["minWidth"]
        attributes["min_bottom_width"] = geometry["minBottomWidth"]
        attributes["max_wall_draft_angle"] = geometry.get("maxWallDraftAngle")
        attributes["derived_ratio_depth_to_min_width"] = _ratio(
            attributes["depth"], _min_positive(attributes["min_opening_width"], attributes["min_bottom_width"])
        )
    elif feature_type == TRIANGULAR:
        attributes["min_opening_width"] = geometry["minWidth"]
        attributes["v_included_angle"] = geometry["vIncludedAngle"]
        attributes["derived_ratio_depth_to_min_width"] = _ratio(attributes["depth"], attributes["min_opening_width"])
    elif feature_type == CIRCULAR:
        attributes["min_opening_width"] = geometry["minWidth"]
        attributes["profile_radius"] = geometry["profileRadius"]
        attributes["derived_ratio_depth_to_min_width"] = _ratio(attributes["depth"], attributes["min_opening_width"])
    # All four leaves use the deepest cross-section point as their reference.
    # For a rectangular slot this is the geometric centre of the floor; for a
    # V/circular/general slot it is the apex/lowest generator/farthest point.
    reference = geometry.get("deepestPoint")
    for name, component in zip(("posX", "posY", "posZ"), reference or (None, None, None)):
        attributes[name] = rounded_scalar(component)

    missing = {}
    for name in fields:
        if attributes.get(name) is None:
            reason = geometry["missingReasons"].get(name, "property was not reliably measurable")
            missing[name] = reason

    return FeatureNode(
        nodeId=f"{feature_type}_{leaf_index:03d}",
        featureType=feature_type,
        partmodelFaceIds=principal,
        attributes=attributes,
        missingAttributes=missing,
        diagnostics={
            "method": "open_sweep_terminal_resolution_aagnet_family_resolution_slot_v1",
            "semanticResolution": resolution.diagnostics(),
            "slotProfile": profile,
            "propertyExtraction": geometry,
            "ownership": "principal OS faces only; all supports and transitions remain unclaimed",
            "referencePointConvention": "deepest transverse profile point; rectangular floor centre where applicable",
            "leafPropertySet": list(fields),
            "wallThickness": wall_diagnostics,
        },
    )


def _mouth_reference(model: PartModel, candidate: OpenSweepCandidate) -> list[float] | None:
    points = []
    for edge_id in candidate.profile_terminal_rail_edge_ids:
        midpoint = _edge_midpoint(model, edge_id)
        if midpoint is not None:
            points.append(midpoint)
    if len(points) != 2:
        return None
    return [sum(point[index] for point in points) / 2.0 for index in range(3)]


def _deepest_profile_point(model, candidate, profile_class: str, mouth):
    if mouth is None:
        return None
    if profile_class == "semicircular":
        points = _cross_section_points(model, candidate)
        if points:
            return max(points, key=lambda point: _chord_distance_sq(model, candidate, point))
    if profile_class == "rounded_u":
        return _cylindrical_bottom_point(model, candidate, mouth)
    if profile_class in {"rectangular", "triangular_v"}:
        internal = [_edge_midpoint(model, edge_id) for edge_id in candidate.internal_rail_edge_ids]
        internal = [point for point in internal if point is not None]
        if internal:
            return [sum(point[index] for point in internal) / len(internal) for index in range(3)]
    # GeneralSlot has no named rail, but its legal transverse section still
    # has a unique point furthest from the mouth chord.  Use sampled boundary
    # points from the principal profile faces, normalized to the common course
    # level, and select that point geometrically.  This is deliberately a
    # boundary extremum rather than an arbitrary face centroid.
    points = []
    for face_id in candidate.ordered_profile_face_ids:
        points.extend(_face_points(model, face_id))
    points.extend(_cross_section_points(model, candidate))
    if points:
        course = _unit(candidate.course_direction)
        if course is not None:
            mouth_level = _dot(mouth, course)
            normalized = [
                [point[index] + (mouth_level - _dot(point, course)) * course[index] for index in range(3)]
                for point in points
            ]
            return list(max(normalized, key=lambda point: _chord_distance_sq(model, candidate, point)))
    return None


def _depth_axis(model, candidate, mouth, deepest):
    if mouth is None or deepest is None:
        return None
    course = _unit(candidate.course_direction)
    if course is None:
        return None
    deepest_transverse = [
        deepest[index] - _dot(deepest, course) * course[index]
        for index in range(3)
    ]
    delta = [deepest_transverse[index] - mouth[index] for index in range(3)]
    chord = _mouth_chord(model, candidate, course)
    if chord is not None:
        # Project the deepest point to the mouth chord in the transverse
        # section.  The remaining vector is the perpendicular depth vector;
        # using mouth-midpoint -> apex would be wrong for unequal terminals.
        terminal = _mouth_points(model, candidate)
        if terminal:
            anchor = [terminal[0][index] - _dot(terminal[0], course) * course[index] for index in range(3)]
            projection = _dot(
                [deepest_transverse[index] - anchor[index] for index in range(3)],
                chord,
            )
            foot = [anchor[index] + projection * chord[index] for index in range(3)]
            delta = [
                deepest_transverse[index] - foot[index]
                for index in range(3)
            ]
    if _norm(delta) <= _VECTOR_TOL:
        return None
    axis = _unit(delta)
    return axis


def _depth(axis, mouth, deepest):
    if axis is None or mouth is None or deepest is None:
        return None
    value = _dot([deepest[index] - mouth[index] for index in range(3)], axis)
    return value if value > _GEOMETRY_TOL else None


def _mouth_width(model, candidate):
    """Length of the chord joining the two transverse terminal-rail centres."""
    points = _mouth_points(model, candidate)
    course = _unit(candidate.course_direction)
    if len(points) != 2 or course is None:
        return None
    delta = [points[1][index] - points[0][index] for index in range(3)]
    delta = [delta[index] - _dot(delta, course) * course[index] for index in range(3)]
    value = _norm(delta)
    return value if value > _GEOMETRY_TOL else None


def _width(model, candidate, profile_class, mouth, axis, depth):
    if profile_class == "triangular_v":
        return _mouth_width(model, candidate)
    if profile_class == "other":
        return None
    if profile_class == "rectangular":
        return _mouth_width(model, candidate)
    if profile_class == "rounded_u":
        # The opening is still the terminal chord; the cylinder radius is a
        # separate, independent property of CircularSlot.
        return _mouth_width(model, candidate)
    if profile_class == "semicircular":
        radius = _cylinder_radius(model, candidate)
        return 2.0 * radius if radius is not None else _mouth_width(model, candidate)
    return _mouth_width(model, candidate)


def _width_at_depth(model, candidate, mouth, axis, depth):
    """Measure the rectangular station width at a transverse depth.

    Each drafted side wall is intersected with the line through the station
    centre in the mouth-chord direction.  Parallel-wall distance is retained
    as a conservative fallback for the old, perfectly prismatic case.
    """
    if mouth is None or axis is None or depth is None:
        return None
    profile_ids = candidate.ordered_profile_face_ids
    if len(profile_ids) < 3:
        return None
    left = _plane_geometry(model, profile_ids[0])
    right = _plane_geometry(model, profile_ids[-1])
    chord = _mouth_chord(model, candidate, _unit(candidate.course_direction))
    if left is None or right is None or chord is None:
        return None
    station = [mouth[index] + float(depth) * axis[index] for index in range(3)]
    values = []
    for normal, offset in (left, right):
        denominator = _dot(normal, chord)
        if abs(denominator) <= _VECTOR_TOL:
            values = []
            break
        values.append((offset - _dot(normal, station)) / denominator)
    if len(values) == 2:
        value = abs(values[1] - values[0])
        if value > _GEOMETRY_TOL:
            return value
    # A stable parallel-wall fallback also handles a station that is exactly
    # at the opening and avoids fabricating a width from a failed intersection.
    if _parallel(left[0], right[0]):
        return _parallel_plane_distance(model, profile_ids[0], profile_ids[-1])
    return None


def _v_included_angle(model, candidate):
    profile_ids = candidate.ordered_profile_face_ids
    if len(profile_ids) != 2:
        return None
    normals = [_face_normal(model.faces.get(face_id)) for face_id in profile_ids]
    if any(normal is None for normal in normals):
        return None
    # The unified outward normals describe the supplement of the V interior
    # angle.  Keep the signed dot product: abs() would fold a wide V into its
    # acute complement and violate the v5 definition.
    dot = max(-1.0, min(1.0, _dot(normals[0], normals[1])))
    angle = 180.0 - math.degrees(math.acos(dot))
    return angle if _GEOMETRY_TOL < angle < 180.0 - _GEOMETRY_TOL else None


def _max_wall_draft_angle(model, candidate, axis):
    if axis is None or len(candidate.ordered_profile_face_ids) < 3:
        return None
    values = []
    for face_id in (candidate.ordered_profile_face_ids[0], candidate.ordered_profile_face_ids[-1]):
        normal = _face_normal(model.faces.get(face_id))
        if normal is None:
            return None
        dot = min(1.0, max(0.0, abs(_dot(normal, axis))))
        values.append(math.degrees(math.asin(dot)))
    return max(values) if values else None


def _strict_semicircle(model, candidate):
    radius = _cylinder_radius(model, candidate)
    if radius is None:
        return False
    records = candidate.longitudinal_start_edge_ids_by_face + candidate.longitudinal_end_edge_ids_by_face
    arcs = []
    for record in records:
        for edge_id in record.get("edgeIds", []):
            edge = model.edges.get(edge_id)
            if edge is None or edge.curve_type != CurveType.CIRCLE:
                continue
            edge_radius = edge.parameters.get("radius")
            if isinstance(edge_radius, (int, float)):
                arcs.append((edge, float(edge_radius)))
    if len(arcs) != 2:
        return False
    for edge, edge_radius in arcs:
        if abs(edge_radius - radius) > max(_GEOMETRY_TOL, radius * _GEOMETRY_TOL):
            return False
        if abs(float(edge.length) / edge_radius - math.pi) > 1.0e-3:
            return False
    return True


def _semicircle_axis_and_point(model, candidate, mouth, axis, deepest, radius):
    if mouth is None:
        return axis, deepest
    points = _cross_section_points(model, candidate)
    if not points:
        return axis, deepest
    if axis is None:
        deepest = max(points, key=lambda point: _chord_distance_sq(model, candidate, point))
        axis = _depth_axis(model, candidate, mouth, deepest)
    if axis is not None:
        deepest = [mouth[index] + radius * axis[index] for index in range(3)]
    return axis, deepest


def _cylinder_radius(model, candidate):
    face_ids = list(candidate.ordered_profile_face_ids) + list(candidate.principal_face_ids)
    for face_id in face_ids:
        face = model.faces.get(face_id)
        if face is None or face.surface_type != SurfaceType.CYLINDER:
            continue
        value = face.parameters.get("radius")
        if isinstance(value, (int, float)) and float(value) > 0.0:
            return float(value)
    return None


def _cylindrical_bottom_point(model, candidate, mouth):
    if len(candidate.ordered_profile_face_ids) < 2:
        return None
    cylinder_id = candidate.ordered_profile_face_ids[1]
    points = _face_points(model, cylinder_id)
    if not points:
        return None
    course = _unit(candidate.course_direction)
    chord = _mouth_chord(model, candidate, course)
    terminal = _mouth_points(model, candidate)
    if course is None or chord is None or not terminal:
        return None
    anchor = [terminal[0][index] - _dot(terminal[0], course) * course[index] for index in range(3)]
    candidates = []
    for point in points:
        transverse = [point[index] - _dot(point, course) * course[index] for index in range(3)]
        projection = _dot([transverse[index] - anchor[index] for index in range(3)], chord)
        foot = [anchor[index] + projection * chord[index] for index in range(3)]
        perpendicular = [transverse[index] - foot[index] for index in range(3)]
        magnitude = _norm(perpendicular)
        if magnitude > _GEOMETRY_TOL:
            candidates.append((magnitude, point))
    if not candidates:
        return None
    # The deepest point is the cylindrical bottom furthest from the mouth
    # chord.  Tangent rails are closer to that chord and therefore cannot win.
    bottom = list(max(candidates, key=lambda item: item[0])[1])
    # This is a sampled cylindrical arc/bottom-generator point, not a claim
    # that a B-Rep longitudinal rail exists.  Report it at the common mouth
    # reference course level rather than at whichever boundary sample
    # happened to carry the largest transverse distance.
    mouth_level = _dot(mouth, course)
    bottom_level = _dot(bottom, course)
    return [bottom[index] + (mouth_level - bottom_level) * course[index] for index in range(3)]


def _mouth_points(model, candidate):
    points = [_edge_midpoint(model, edge_id) for edge_id in candidate.profile_terminal_rail_edge_ids]
    return [point for point in points if point is not None] if len([point for point in points if point is not None]) == 2 else []


def _mouth_chord(model, candidate, course):
    if course is None:
        return None
    points = _mouth_points(model, candidate)
    if len(points) != 2:
        return None
    delta = [points[1][index] - points[0][index] for index in range(3)]
    delta = [delta[index] - _dot(delta, course) * course[index] for index in range(3)]
    return _unit(delta)


def _chord_distance_sq(model, candidate, point):
    course = _unit(candidate.course_direction)
    chord = _mouth_chord(model, candidate, course)
    terminal = _mouth_points(model, candidate)
    if course is None or chord is None or not terminal:
        return 0.0
    anchor = [terminal[0][index] - _dot(terminal[0], course) * course[index] for index in range(3)]
    transverse = [point[index] - _dot(point, course) * course[index] for index in range(3)]
    delta = [transverse[index] - anchor[index] for index in range(3)]
    projection = _dot(delta, chord)
    foot = [anchor[index] + projection * chord[index] for index in range(3)]
    return _distance_sq(transverse, foot)


def _parallel_plane_distance(model, left_id, right_id):
    left = _plane_geometry(model, left_id)
    right = _plane_geometry(model, right_id)
    if left is None or right is None or not _parallel(left[0], right[0]):
        return None
    normal, left_offset = left
    right_normal, right_offset = right
    if _dot(normal, right_normal) < 0.0:
        right_offset = -right_offset
    value = abs(left_offset - right_offset)
    return value if value > _GEOMETRY_TOL else None


def _plane_geometry(model, face_id):
    face = model.faces.get(face_id)
    if face is None or face.surface_type != SurfaceType.PLANE:
        return None
    normal = _face_normal(face)
    if normal is None:
        return None
    point = face.parameters.get("location")
    if not isinstance(point, (list, tuple)) or len(point) != 3:
        points = _face_points(model, face_id)
        if not points:
            return None
        point = [sum(item[index] for item in points) / len(points) for index in range(3)]
    try:
        point = [float(item) for item in point]
    except (TypeError, ValueError):
        return None
    normal = _unit(normal)
    if normal is None:
        return None
    for component in normal:
        if abs(component) <= _VECTOR_TOL:
            continue
        if component < 0.0:
            normal = [-item for item in normal]
        break
    return normal, _dot(normal, point)


def _cross_section_points(model, candidate):
    course = _unit(candidate.course_direction)
    if course is None:
        return []
    points = []
    for record in candidate.longitudinal_start_edge_ids_by_face:
        for edge_id in record.get("edgeIds", []):
            edge = model.edges.get(edge_id)
            if edge is None:
                continue
            values = getattr(edge, "sample_points", None) or []
            if values:
                level = sum(_dot(point, course) for point in values) / len(values)
                points.extend([[point[index] - (_dot(point, course) - level) * course[index] for index in range(3)] for point in values])
    return points


def _edge_midpoint(model, edge_id):
    edge = model.edges.get(edge_id)
    if edge is None:
        return None
    points = [list(point) for point in (getattr(edge, "sample_points", None) or [])]
    if not points:
        points = [list(model.vertices[vertex_id].coordinates) for vertex_id in getattr(edge, "vertex_ids", []) if vertex_id in model.vertices]
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    return [(points[0][index] + points[-1][index]) * 0.5 for index in range(3)]


def _face_points(model, face_id):
    face = model.faces.get(face_id)
    if face is None:
        return []
    points = []
    for loop_id in face.loop_ids:
        loop = model.loops.get(loop_id)
        if loop is None or not loop.is_outer:
            continue
        for edge_id in loop.edge_ids:
            edge = model.edges.get(edge_id)
            if edge is not None:
                points.extend([list(point) for point in (getattr(edge, "sample_points", None) or [])])
    return points


def _face_normal(face):
    value = face.parameters.get("normal") if face is not None else None
    unit = _unit(value)
    if unit is not None and face.is_reversed:
        unit = [-value for value in unit]
    return unit


def _parallel(left, right):
    if left is None or right is None:
        return False
    return abs(_dot(left, right)) >= 1.0 - _GEOMETRY_TOL


def _vector(value):
    if value is None:
        return None
    return [rounded_scalar(item) + 0.0 for item in value]


def _unit(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    norm = math.sqrt(sum(item * item for item in values))
    if norm <= _VECTOR_TOL or not all(math.isfinite(item) for item in values):
        return None
    return [item / norm for item in values]


def _dot(left, right):
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _norm(value):
    return math.sqrt(sum(float(item) ** 2 for item in value))


def _distance_sq(left, right):
    return sum((float(left[index]) - float(right[index])) ** 2 for index in range(3))


def _min_positive(*values):
    usable = [float(value) for value in values if isinstance(value, (int, float)) and float(value) > 0.0]
    return min(usable) if len(usable) == len(values) and usable else None


def _ratio(numerator, denominator):
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if float(denominator) <= _VECTOR_TOL:
        return None
    return rounded_scalar(float(numerator) / float(denominator))


def _wall_sampling_face_ids(candidate, feature_type):
    """Return only faces that are walls for the slot leaf's thickness ray.

    Rectangular and General profiles carry an intermediate floor station in
    the ordered transverse profile; that face is a terminal floor, not a wall
    sample.  Triangular profiles have only their two side walls.  Circular
    profiles retain their complete principal face family because a side wall
    and bottom arc may be one indivisible cylindrical face.
    """
    profile_ids = list(getattr(candidate, "ordered_profile_face_ids", ()) or ())
    if feature_type in {RECTANGULAR, GENERAL} and len(profile_ids) >= 3:
        profile_ids = [profile_ids[0], profile_ids[-1]]
    if not profile_ids:
        profile_ids = list(getattr(candidate, "principal_face_ids", ()) or ())
    return list(dict.fromkeys(profile_ids))


def _wall_thickness(model, kernel, principal_face_ids, wall_face_ids, *, diagnostics=None):
    cad_ids = cad_face_ids(model, principal_face_ids)
    wall_cad_ids = cad_face_ids(model, wall_face_ids)
    if not cad_ids or not wall_cad_ids:
        return None
    return min_wall_thickness(
        kernel,
        cad_ids,
        wall_cad_ids,
        diagnostics=diagnostics,
    )
