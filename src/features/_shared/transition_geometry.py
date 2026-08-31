"""Geometry and grouping helpers for transition and residual features.

The parser deliberately keeps the useful analytic parameters on
``FaceData``.  This module consumes those parameters, plus the persisted edge
samples, so transition feature modules do not need to reach into the AAGNet
vendor tree or invent a second B-Rep representation.

The AAGNet transition score is only a proposal.  The public helpers here
always require a real connected ``PartModel`` region and report an explicit
reason when a geometric attribute cannot be measured reliably.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from contract.partmodel import CurveType, PartModel, SurfaceType

from features._shared.attributes import rounded_scalar
from features._shared.hole_topology import dot, unit_vector, valid_vector
from features._shared.transition_evidence import TransitionEvidence


_GEOM_TOL = 1.0e-5
_ANGLE_TOL_DEG = 1.0
_CIRCLE_COVERAGE_TOL = 0.15


@dataclass(frozen=True)
class TransitionRegion:
    """One connected transition region proposed by AAGNet evidence."""

    face_ids: tuple[str, ...]
    surface_types: tuple[str, ...]
    verdict: dict[str, Any]
    family: str = ""


@dataclass
class TransitionMeasurement:
    """Measured attributes and audit details for one transition region."""

    kind: str | None
    width: float | None = None
    angle: float | None = None
    radius: float | None = None
    centroid: list[float] | None = None
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


def transition_regions(
    model: PartModel,
    transition: TransitionEvidence,
    claimed_face_ids: set[str] | None = None,
) -> list[TransitionRegion]:
    """Return connected transition-face components in deterministic order.

    A region is accepted only when every member is an active PartModel face,
    carries a transition score, and is not already owned by an earlier
    manufacturing feature.  The evidence region verdict is retained so a
    caller can explain why a candidate was withheld.
    """

    claimed = set(claimed_face_ids or ())
    families = {
        face_id: _transition_family(model, face_id)
        for face_id in model.faces
        if face_id not in claimed and transition.is_transition_face(face_id)
    }
    eligible = {face_id for face_id, family in families.items() if family is not None}
    regions: list[TransitionRegion] = []
    visited: set[str] = set()
    for start in sorted(eligible, key=face_sort_key):
        if start in visited:
            continue
        component: set[str] = set()
        queue: deque[str] = deque([start])
        while queue:
            face_id = queue.popleft()
            if face_id in visited or face_id not in eligible:
                continue
            visited.add(face_id)
            component.add(face_id)
            face = model.faces.get(face_id)
            if face is None:
                continue
            for adjacency in face.adjacencies:
                neighbour = adjacency.adjacent_face_id
                if (
                    neighbour in eligible
                    and neighbour not in visited
                    and _transition_faces_compatible(model, face_id, neighbour, families[face_id])
                ):
                    queue.append(neighbour)
        ordered = tuple(sorted(component, key=face_sort_key))
        family = families[start]
        # Every member is already an eligible transition face.  The fraction
        # is therefore not a claim about coverage of an omitted wall band; use
        # the area-weighted score as the guard and retain the fraction only as
        # scoped diagnostics.
        verdict_obj = transition.region_verdict(ordered, apply_face_fraction_guard=False)
        if not verdict_obj.is_transition:
            continue
        verdict = verdict_obj.to_dict()
        verdict.update({
            "faceFractionScope": "eligible_transition_faces_only",
            "faceFractionUsedAsGuard": False,
            "geometryFamily": family,
        })
        regions.append(
            TransitionRegion(
                face_ids=ordered,
                surface_types=tuple(sorted({model.faces[f].surface_type.value for f in ordered})),
                verdict=verdict,
                family=family,
            )
        )
    return regions


def face_sort_key(face_id: str) -> tuple[int, str]:
    tail = str(face_id).rsplit("_", 1)[-1]
    return (int(tail), str(face_id)) if tail.isdigit() else (10**12, str(face_id))


def _transition_family(model: PartModel, face_id: str) -> str | None:
    """Map a transition surface to its manufacturing family.

    This is deliberately geometry-driven.  A cone/plane may propose a
    Chamfer, while smooth analytic/freeform blend surfaces propose Fillet;
    cylindrical blends are admitted only when their own topology has at
    least two smooth supports.
    """
    face = model.faces.get(face_id)
    if face is None:
        return None
    if face.surface_type in {SurfaceType.PLANE, SurfaceType.CONE}:
        return "Chamfer"
    if face.surface_type in {
        SurfaceType.TORUS,
        SurfaceType.SPHERE,
        SurfaceType.BSPLINE,
        SurfaceType.REVOLUTION,
        SurfaceType.EXTRUSION,
        SurfaceType.OTHER,
    }:
        return "Fillet"
    if face.surface_type == SurfaceType.CYLINDER:
        smooth = sum(1 for item in face.adjacencies if item.connection_type.value == "SMOOTH")
        return "Fillet" if smooth >= 2 else None
    return None


def _transition_faces_compatible(model: PartModel, left_id: str, right_id: str, family: str | None) -> bool:
    if family is None or _transition_family(model, right_id) != family:
        return False
    left, right = model.faces[left_id], model.faces[right_id]
    if left.surface_type == right.surface_type:
        if left.surface_type == SurfaceType.PLANE:
            ln = unit_vector(left.parameters.get("normal"))
            rn = unit_vector(right.parameters.get("normal"))
            return ln is None or rn is None or abs(abs(dot(ln, rn)) - 1.0) <= 1.0e-3
        if left.surface_type == SurfaceType.CONE:
            la = unit_vector(left.parameters.get("axis_direction"))
            ra = unit_vector(right.parameters.get("axis_direction"))
            ls = left.parameters.get("semi_angle")
            rs = right.parameters.get("semi_angle")
            if la is not None and ra is not None and abs(abs(dot(la, ra)) - 1.0) > 1.0e-3:
                return False
            if isinstance(ls, (int, float)) and isinstance(rs, (int, float)) and abs(float(ls) - float(rs)) > math.radians(1.0):
                return False
        if left.surface_type == SurfaceType.TORUS:
            la = unit_vector(left.parameters.get("axis"))
            ra = unit_vector(right.parameters.get("axis"))
            for key in ("major_radius", "minor_radius"):
                lv, rv = left.parameters.get(key), right.parameters.get(key)
                if isinstance(lv, (int, float)) and isinstance(rv, (int, float)) and abs(float(lv) - float(rv)) > max(_GEOM_TOL, 1.0e-3 * max(abs(float(lv)), abs(float(rv)), 1.0)):
                    return False
            if la is not None and ra is not None and abs(abs(dot(la, ra)) - 1.0) > 1.0e-3:
                return False
        if left.surface_type == SurfaceType.CYLINDER:
            la = unit_vector(left.parameters.get("axis_direction"))
            ra = unit_vector(right.parameters.get("axis_direction"))
            lr, rr = left.parameters.get("radius"), right.parameters.get("radius")
            if la is not None and ra is not None and abs(abs(dot(la, ra)) - 1.0) > 1.0e-3:
                return False
            if isinstance(lr, (int, float)) and isinstance(rr, (int, float)) and abs(float(lr) - float(rr)) > max(_GEOM_TOL, 1.0e-3 * max(abs(float(lr)), abs(float(rr)), 1.0)):
                return False
    # Different surface types inside the same family may form one treatment
    # only through an explicit B-Rep adjacency; never infer a bridge from
    # neighboring IDs or AAGNet grouping.
    return _shared_transition_adjacency(model, left_id, right_id)


def _shared_transition_adjacency(model: PartModel, left_id: str, right_id: str) -> bool:
    return any(item.adjacent_face_id == right_id for item in model.faces[left_id].adjacencies) or any(item.adjacent_face_id == left_id for item in model.faces[right_id].adjacencies)


def measure_transition(
    model: PartModel,
    face_ids: Iterable[str],
) -> TransitionMeasurement:
    """Classify and measure one connected transition region.

    Planar and conical regions are classified as ``Chamfer``.  TORUS,
    cylindrical and other smooth transition surfaces are classified as
    ``Fillet`` when the adjacency geometry supports a blend.  A Fillet may
    still be accepted with a missing radius for a variable-radius/freeform
    blend; fabricating an average is explicitly avoided.
    """

    ids = tuple(sorted(set(face_ids), key=face_sort_key))
    faces = [model.faces.get(face_id) for face_id in ids]
    faces = [face for face in faces if face is not None]
    if not faces:
        return TransitionMeasurement(None, reason="empty_transition_region")
    types = {face.surface_type for face in faces}
    centroid = region_centroid(model, ids)
    diagnostics: dict[str, Any] = {
        "faceIds": list(ids),
        "surfaceTypes": sorted(item.value for item in types),
        # The stored centroids *are* OCC surface mass centres on the OCC line
        # (they were kernel queries before they became parse-time data), so
        # the recorded method name is unchanged there; a model another line
        # filled names that line instead.  An old partmodel.json without the
        # field falls back to the bbox estimate, as a missing kernel used to.
        "centroidMethod": _stored_centroid_method(model) if _has_stored_centroids(model) else "area_weighted_face_bbox_centres",
    }

    proof = _transition_topology_proof(model, ids)
    diagnostics["topologyProof"] = proof
    if not proof["valid"]:
        return TransitionMeasurement(None, centroid=centroid, reason=proof["reason"], diagnostics=diagnostics)

    if types and types <= {SurfaceType.PLANE, SurfaceType.CONE}:
        width, angle, geometry = measure_chamfer(
            model,
            ids,
            support_face_ids=tuple(proof.get("supportFaceIds", ())),
        )
        diagnostics.update(geometry)
        reason = "" if width is not None or angle is not None else geometry.get("reason", "chamfer_geometry_unavailable")
        return TransitionMeasurement("Chamfer", width, angle, centroid=centroid, reason=reason, diagnostics=diagnostics)

    if _fillet_surface_candidate(model, ids):
        radius, geometry = measure_fillet_radius(model, ids)
        diagnostics.update(geometry)
        reason = "" if radius is not None else geometry.get("reason", "unique_constant_blend_radius_unavailable")
        return TransitionMeasurement("Fillet", radius=radius, centroid=centroid, reason=reason, diagnostics=diagnostics)

    return TransitionMeasurement(
        None,
        centroid=centroid,
        reason="transition_surface_family_not_supported_as_chamfer_or_fillet",
        diagnostics=diagnostics,
    )


def _fillet_surface_candidate(model: PartModel, face_ids: tuple[str, ...]) -> bool:
    """Require smooth/blend geometry, not merely a ``round`` top label."""
    proof = _transition_topology_proof(model, face_ids)
    if not proof["valid"] or proof["family"] != "Fillet":
        return False
    if len(proof["smoothSupportFaceIds"]) < 2:
        return False

    for face_id in face_ids:
        face = model.faces[face_id]
        if face.surface_type in {SurfaceType.TORUS, SurfaceType.SPHERE, SurfaceType.BSPLINE, SurfaceType.REVOLUTION, SurfaceType.EXTRUSION, SurfaceType.OTHER}:
            return True
        if face.surface_type == SurfaceType.CYLINDER:
            smooth = sum(1 for item in face.adjacencies if item.connection_type.value == "SMOOTH")
            if smooth >= 2:
                return True
    return False


def _transition_topology_proof(model: PartModel, face_ids: tuple[str, ...]) -> dict[str, Any]:
    """Check active B-Rep boundary/support evidence before promotion."""
    region = set(face_ids)
    if not region:
        return {"valid": False, "family": None, "reason": "empty_transition_region"}
    families = {_transition_family(model, face_id) for face_id in region}
    family = next(iter(families)) if len(families) == 1 else None
    boundary_edges: set[str] = set()
    support_faces: set[str] = set()
    smooth_support_faces: set[str] = set()
    for face_id in region:
        face = model.faces.get(face_id)
        if face is None:
            continue
        for adjacency in face.adjacencies:
            if adjacency.adjacent_face_id in region:
                continue
            if adjacency.adjacent_face_id not in model.faces or adjacency.shared_edge_id not in model.edges:
                continue
            support_faces.add(adjacency.adjacent_face_id)
            boundary_edges.add(adjacency.shared_edge_id)
            if adjacency.connection_type.value == "SMOOTH":
                smooth_support_faces.add(adjacency.adjacent_face_id)
    if family is None:
        return {"valid": False, "family": None, "reason": "mixed_or_unknown_transition_family", "boundaryEdgeCount": len(boundary_edges), "supportFaceIds": sorted(support_faces)}
    if family == "Chamfer" and 0 < len(support_faces) < 2:
        return {
            "valid": False,
            "family": family,
            "reason": "chamfer_requires_two_distinct_support_faces",
            "boundaryEdgeCount": len(boundary_edges),
            "supportFaceIds": sorted(support_faces),
            "smoothSupportFaceIds": sorted(smooth_support_faces),
        }
    if len(boundary_edges) < 2 or not support_faces:
        return {"valid": False, "family": family, "reason": "transition_boundary_or_support_topology_unavailable", "boundaryEdgeCount": len(boundary_edges), "supportFaceIds": sorted(support_faces), "smoothSupportFaceIds": sorted(smooth_support_faces)}
    if family == "Fillet" and len(smooth_support_faces) < 2:
        return {"valid": False, "family": family, "reason": "fillet_requires_two_smooth_support_faces", "boundaryEdgeCount": len(boundary_edges), "supportFaceIds": sorted(support_faces), "smoothSupportFaceIds": sorted(smooth_support_faces)}
    return {
        "valid": True,
        "family": family,
        "boundaryEdgeCount": len(boundary_edges),
        "boundaryEdgeIds": sorted(boundary_edges),
        "supportFaceIds": sorted(support_faces),
        "smoothSupportFaceIds": sorted(smooth_support_faces),
    }


def measure_chamfer(
    model: PartModel,
    face_ids: tuple[str, ...],
    *,
    support_face_ids: Iterable[str] | None = None,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Measure slant width and reference-surface angle for a Chamfer."""

    types = {model.faces[face_id].surface_type for face_id in face_ids}
    if SurfaceType.CONE in types:
        width, angle, detail = _measure_conical_chamfer(
            model,
            face_ids,
            support_face_ids=support_face_ids,
        )
        return width, angle, detail
    width = _planar_region_width(model, face_ids)
    angle, angle_detail = _planar_reference_angle(
        model,
        face_ids,
        support_face_ids=support_face_ids,
    )
    detail = {
        "geometryMethod": "planar_boundary_principal_extent_and_support_dihedral",
        "planarWidth": width,
        "referenceAngle": angle,
        "referenceAngleDiagnostics": angle_detail,
    }
    if width is None and angle is None:
        detail["reason"] = "planar_boundary_extent_and_reference_surface_angle_unavailable"
    return width, angle, detail


def _measure_conical_chamfer(
    model: PartModel,
    face_ids: tuple[str, ...],
    *,
    support_face_ids: Iterable[str] | None = None,
) -> tuple[float | None, float | None, dict[str, Any]]:
    circles = _region_circle_groups(model, face_ids)
    detail: dict[str, Any] = {
        "geometryMethod": "two_distinct_circular_boundaries_on_conical_region",
        "circleGroups": circles,
    }
    if len(circles) != 2:
        detail["reason"] = "conical_measurement_requires_exactly_two_boundary_groups"
        return None, None, detail
    if any(not item.get("complete", False) for item in circles):
        detail["reason"] = "conical_boundary_group_is_incomplete"
        return None, None, detail
    first, second = min(circles, key=lambda item: item["radius"]), max(circles, key=lambda item: item["radius"])
    radius_delta = abs(float(second["radius"]) - float(first["radius"]))
    axis_delta = abs(float(second["axisPosition"]) - float(first["axisPosition"]))
    width = math.hypot(radius_delta, axis_delta) if radius_delta > _GEOM_TOL or axis_delta > _GEOM_TOL else None
    angle, angle_detail = _reference_angle_from_boundary(
        model,
        face_ids,
        first,
        second,
        support_face_ids=support_face_ids,
    )
    detail.update({"smallBoundary": first, "largeBoundary": second, "referenceAngleDiagnostics": angle_detail})
    if width is None:
        detail["reason"] = "conical_boundaries_have_no_measurable_separation"
    return rounded_scalar(width), rounded_scalar(angle), detail


def _region_circle_groups(model: PartModel, face_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    axis = _common_axis(model, face_ids)
    if axis is None:
        return []
    grouped: dict[tuple, dict[str, Any]] = {}
    for face_id in face_ids:
        face = model.faces[face_id]
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None:
                continue
            for edge_id in loop.edge_ids:
                edge = model.edges.get(edge_id)
                if edge is None or edge.curve_type != CurveType.CIRCLE:
                    continue
                radius = edge.parameters.get("radius")
                center = edge.parameters.get("center")
                if not isinstance(radius, (int, float)) or float(radius) <= 0.0 or not valid_vector(center):
                    continue
                key = (round(float(radius), 5), tuple(round(float(value), 5) for value in center))
                item = grouped.setdefault(key, {
                    "edgeIds": [],
                    "radius": round(float(radius), 6),
                    "center": [round(float(value), 6) for value in center],
                    "axisPosition": round(dot(center, axis), 6),
                    "adjacentFaceIds": [],
                    "samplePoints": [],
                    "angularCoverage": 0.0,
                })
                if edge_id in item["edgeIds"]:
                    continue
                item["edgeIds"].append(edge_id)
                item["adjacentFaceIds"].extend(_edge_neighbours(model, face_ids, edge_id))
                samples = [list(point) for point in (edge.sample_points or []) if valid_vector(point)]
                if not samples:
                    samples = [
                        list(model.vertices[vertex_id].coordinates)
                        for vertex_id in edge.vertex_ids
                        if vertex_id in model.vertices
                    ]
                item["samplePoints"].extend(samples)
                length = float(edge.length) if isinstance(edge.length, (int, float)) else 0.0
                sampled_coverage = _sample_angular_coverage(samples, center, axis)
                length_coverage = length / float(radius) if length > 0.0 and float(radius) > 0.0 else 0.0
                item["angularCoverage"] += max(length_coverage, sampled_coverage)
    groups = list(grouped.values())
    for item in groups:
        item["edgeIds"] = sorted(set(item["edgeIds"]))
        item["adjacentFaceIds"] = sorted(set(item["adjacentFaceIds"]))
        item["complete"] = float(item["angularCoverage"]) >= 2.0 * math.pi - _CIRCLE_COVERAGE_TOL
    groups.sort(key=lambda item: (item["radius"], item["axisPosition"], item["edgeIds"]))
    return groups


def _sample_angular_coverage(samples: object, centre: object, axis: list[float]) -> float:
    if not isinstance(samples, list) or len(samples) < 2 or not valid_vector(centre):
        return 0.0
    basis = _perpendicular_basis(axis)
    angles: list[float] = []
    for point in samples:
        if not valid_vector(point):
            continue
        delta = [float(point[index]) - float(centre[index]) for index in range(3)]
        axial = dot(delta, axis)
        radial = [delta[index] - axial * axis[index] for index in range(3)]
        angles.append(math.atan2(dot(radial, basis[1]), dot(radial, basis[0])))
    if len(angles) < 2:
        return 0.0
    total = 0.0
    for left, right in zip(angles, angles[1:]):
        delta = abs(right - left)
        while delta > math.pi:
            delta = abs(delta - 2.0 * math.pi)
        total += delta
    return total


def _common_axis(model: PartModel, face_ids: Iterable[str]) -> list[float] | None:
    axes = []
    for face_id in face_ids:
        face = model.faces[face_id]
        axis = face.parameters.get("axis_direction")
        if axis is None and face.surface_type == SurfaceType.CONE:
            axis = face.parameters.get("axis_direction")
        unit = unit_vector(axis)
        if unit is not None:
            axes.append(unit)
    if not axes:
        return None
    base = axes[0]
    for axis in axes[1:]:
        if abs(abs(dot(base, axis)) - 1.0) > 1.0e-3:
            return None
    return base


def _reference_angle_from_boundary(
    model: PartModel,
    face_ids: tuple[str, ...],
    first: dict,
    second: dict,
    *,
    support_face_ids: Iterable[str] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    axis = _common_axis(model, face_ids)
    if axis is None:
        return None, {"reason": "conical_boundary_axis_unavailable"}
    delta = [float(second["center"][i]) - float(first["center"][i]) for i in range(3)]
    axial_delta = dot(delta, axis)
    radial_delta = float(second["radius"]) - float(first["radius"])
    if abs(axial_delta) <= _GEOM_TOL and abs(radial_delta) <= _GEOM_TOL:
        return None, {"reason": "conical_boundary_generator_unavailable", "axialDelta": axial_delta, "radialDelta": radial_delta}

    region = set(face_ids)
    supports = set(support_face_ids or ())
    boundary_candidates: list[dict[str, Any]] = []
    generator_candidates: list[dict[str, Any]] = []
    for side_name, boundary in (("small", first), ("large", second)):
        for edge_id in boundary.get("edgeIds", []):
            radial = _edge_boundary_radial_unit(model, edge_id, boundary, axis) or _boundary_radial_unit(boundary, axis)
            if radial is None:
                continue
            unit_generator = unit_vector([
                axial_delta * axis[index] + radial_delta * radial[index]
                for index in range(3)
            ])
            if unit_generator is None:
                continue
            generator = [round(value, 9) for value in unit_generator]
            generator_candidates.append({"side": side_name, "edgeId": edge_id, "generator": generator})
            for neighbour_id in _edge_neighbours(model, face_ids, edge_id):
                if neighbour_id in region or neighbour_id not in model.faces:
                    continue
                if supports and neighbour_id not in supports:
                    continue
                face = model.faces[neighbour_id]
                normal = None
                if face.surface_type == SurfaceType.PLANE:
                    normal = unit_vector(face.parameters.get("normal"))
                elif face.surface_type in {SurfaceType.CYLINDER, SurfaceType.CONE}:
                    point = _boundary_sample_for_direction(boundary, radial, axis)
                    normal = _radial_normal(face, point) if point is not None else None
                if normal is None:
                    continue
                value = math.degrees(math.asin(min(1.0, abs(dot(unit_generator, normal)))))
                boundary_candidates.append({
                    "side": side_name,
                    "edgeId": edge_id,
                    "faceId": neighbour_id,
                    "angleDeg": value,
                    "generator": generator,
                })

    large_candidates = [item for item in boundary_candidates if item["side"] == "large"]
    large_faces = {item["faceId"] for item in large_candidates}
    if len(large_faces) == 1:
        chosen_candidates = large_candidates
        selection = "unique_large_boundary_support"
    else:
        chosen_candidates = boundary_candidates
        selection = "all_boundary_local_supports_consensus"
    chosen = _stable_angle([(item["faceId"], item["angleDeg"]) for item in chosen_candidates])
    detail = {
        "axialDelta": round(axial_delta, 9),
        "radialDelta": round(radial_delta, 9),
        "generatorCandidates": generator_candidates,
        "selection": selection,
        "candidates": [
            {
                "side": item["side"],
                "edgeId": item["edgeId"],
                "faceId": item["faceId"],
                "angleDeg": round(item["angleDeg"], 6),
                "generator": item["generator"],
            }
            for item in boundary_candidates
        ],
    }
    if chosen is None:
        detail["reason"] = "reference_side_ambiguous_or_unavailable"
    return rounded_scalar(chosen), detail


def _edge_boundary_radial_unit(model, edge_id: str, boundary: dict[str, Any], axis: list[float]) -> list[float] | None:
    edge = model.edges.get(edge_id)
    center = boundary.get("center")
    if edge is None or not valid_vector(center):
        return None
    points = [list(point) for point in (edge.sample_points or []) if valid_vector(point)]
    if not points:
        points = [
            list(model.vertices[vertex_id].coordinates)
            for vertex_id in edge.vertex_ids
            if vertex_id in model.vertices and valid_vector(model.vertices[vertex_id].coordinates)
        ]
    for point in points:
        delta = [float(point[index]) - float(center[index]) for index in range(3)]
        axial = dot(delta, axis)
        value = unit_vector([delta[index] - axial * axis[index] for index in range(3)])
        if value is not None:
            return value
    return None


def _boundary_radial_unit(boundary: dict[str, Any], axis: list[float]) -> list[float] | None:
    center = boundary.get("center")
    for point in boundary.get("samplePoints", []):
        if not valid_vector(point) or not valid_vector(center):
            continue
        delta = [float(point[index]) - float(center[index]) for index in range(3)]
        axial = dot(delta, axis)
        radial = [delta[index] - axial * axis[index] for index in range(3)]
        value = unit_vector(radial)
        if value is not None:
            return value
    return None


def _boundary_sample_for_direction(boundary: dict[str, Any], radial: list[float], axis: list[float]) -> list[float] | None:
    center = boundary.get("center")
    if not valid_vector(center):
        return None
    radius = boundary.get("radius")
    if not isinstance(radius, (int, float)) or float(radius) <= 0.0:
        return None
    return [float(center[index]) + float(radius) * radial[index] for index in range(3)]


def _radial_normal(face, point: list[float]) -> list[float] | None:
    axis = unit_vector(face.parameters.get("axis_direction"))
    origin = face.parameters.get("axis_location") or face.parameters.get("apex")
    if axis is None or not valid_vector(origin):
        return None
    delta = [float(point[index]) - float(origin[index]) for index in range(3)]
    axial = dot(delta, axis)
    radial = [delta[index] - axial * axis[index] for index in range(3)]
    return unit_vector(radial)


def _stable_angle(candidates: list[tuple[str, float]]) -> float | None:
    if not candidates:
        return None
    values = [value for _, value in candidates if 0.0 < value < 90.0]
    if not values:
        return None
    # Chamfer semantics use the smaller valid acute angle among the two
    # support references.  A long transition band may touch many segmented
    # faces; callers constrain the candidate set to the topology proof's
    # supportFaceIds before this aggregation.
    return min(values)


def _planar_region_width(model: PartModel, face_ids: tuple[str, ...]) -> float | None:
    points = _boundary_points(model, face_ids)
    if len(points) < 3:
        return None
    normal = _plane_normal(model, face_ids)
    if normal is None:
        return None
    u, v = _perpendicular_basis(normal)
    projected = [(dot(point, u), dot(point, v)) for point in points]
    mean_u = sum(item[0] for item in projected) / len(projected)
    mean_v = sum(item[1] for item in projected) / len(projected)
    c_uu = sum((item[0] - mean_u) ** 2 for item in projected) / len(projected)
    c_vv = sum((item[1] - mean_v) ** 2 for item in projected) / len(projected)
    c_uv = sum((item[0] - mean_u) * (item[1] - mean_v) for item in projected) / len(projected)
    trace = c_uu + c_vv
    det_term = math.sqrt(max(0.0, (c_uu - c_vv) ** 2 + 4.0 * c_uv * c_uv))
    largest = 0.5 * (trace + det_term)
    if largest <= _GEOM_TOL:
        return None
    theta = 0.5 * math.atan2(2.0 * c_uv, c_uu - c_vv)
    major = [math.cos(theta), math.sin(theta)]
    minor = [-major[1], major[0]]
    extent_major = max(item[0] * major[0] + item[1] * major[1] for item in projected) - min(item[0] * major[0] + item[1] * major[1] for item in projected)
    extent_minor = max(item[0] * minor[0] + item[1] * minor[1] for item in projected) - min(item[0] * minor[0] + item[1] * minor[1] for item in projected)
    width = min(abs(extent_major), abs(extent_minor))
    return rounded_scalar(width) if width > _GEOM_TOL else None


def _planar_reference_angle(
    model: PartModel,
    face_ids: tuple[str, ...],
    *,
    support_face_ids: Iterable[str] | None = None,
) -> tuple[float | None, dict[str, Any]]:
    candidates: list[tuple[str, float]] = []
    region = set(face_ids)
    supports = set(support_face_ids or ())
    for face_id in face_ids:
        face = model.faces[face_id]
        normal = unit_vector(face.parameters.get("normal"))
        if normal is None:
            continue
        for adjacency in face.adjacencies:
            if adjacency.adjacent_face_id in region:
                continue
            if supports and adjacency.adjacent_face_id not in supports:
                continue
            neighbour = model.faces.get(adjacency.adjacent_face_id)
            if neighbour is None:
                continue
            ref = unit_vector(neighbour.parameters.get("normal"))
            if ref is None and neighbour.surface_type in {SurfaceType.CYLINDER, SurfaceType.CONE}:
                points = _boundary_points(model, [face_id])
                ref = _radial_normal(neighbour, points[0]) if points else None
            if ref is None:
                continue
            angle = math.degrees(math.acos(max(-1.0, min(1.0, abs(dot(normal, ref))))))
            if 0.0 < angle < 90.0:
                candidates.append((neighbour.id, angle))
    chosen = _stable_angle(candidates)
    detail = {"candidates": [{"faceId": key, "angleDeg": round(value, 6)} for key, value in candidates]}
    if chosen is None:
        detail["reason"] = "reference_side_ambiguous_or_unavailable"
    return rounded_scalar(chosen), detail


def _plane_normal(model: PartModel, face_ids: Iterable[str]) -> list[float] | None:
    normals = [unit_vector(model.faces[face_id].parameters.get("normal")) for face_id in face_ids]
    normals = [normal for normal in normals if normal is not None]
    if not normals:
        return None
    base = normals[0]
    if any(abs(abs(dot(base, normal)) - 1.0) > 1.0e-3 for normal in normals[1:]):
        return None
    return base


def _boundary_points(model: PartModel, face_ids: Iterable[str]) -> list[list[float]]:
    points: list[list[float]] = []
    seen: set[tuple[float, float, float]] = set()
    for face_id in face_ids:
        face = model.faces.get(face_id)
        if face is None:
            continue
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None:
                continue
            for edge_id in loop.edge_ids:
                edge = model.edges.get(edge_id)
                if edge is None:
                    continue
                samples = edge.sample_points or []
                for point in samples:
                    if not valid_vector(point):
                        continue
                    key = tuple(round(float(value), 6) for value in point)
                    if key not in seen:
                        seen.add(key)
                        points.append([float(value) for value in point])
    return points


def _edge_neighbours(model: PartModel, face_ids: Iterable[str], edge_id: str) -> list[str]:
    region = set(face_ids)
    neighbours: set[str] = set()
    for face_id, face in model.faces.items():
        if face_id in region:
            continue
        if any(adjacency.shared_edge_id == edge_id for source_id in region for adjacency in model.faces[source_id].adjacencies if adjacency.adjacent_face_id == face_id):
            neighbours.add(face_id)
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is not None and edge_id in loop.edge_ids:
                neighbours.add(face_id)
    return sorted(neighbours, key=face_sort_key)


def region_centroid(model: PartModel, face_ids: Iterable[str]) -> list[float] | None:
    """Return an area centroid without relying on guide/seam traversal starts."""

    ids = list(face_ids)
    exact = _stored_centroid(model, ids)
    if exact is not None:
        return [round(value, 6) for value in exact]
    weighted: list[tuple[float, list[float]]] = []
    for face_id in ids:
        face = model.faces.get(face_id)
        bbox = getattr(face, "bbox", None) if face is not None else None
        if face is None or not _valid_bbox(bbox):
            continue
        centre = [(float(bbox["min"][index]) + float(bbox["max"][index])) * 0.5 for index in range(3)]
        area = float(face.area) if isinstance(face.area, (int, float)) and float(face.area) > 0.0 else 0.0
        if area > 0.0:
            weighted.append((area, centre))
    if not weighted:
        points = _boundary_points(model, ids)
        if not points:
            return None
        return [round(sum(point[index] for point in points) / len(points), 6) for index in range(3)]
    total = sum(item[0] for item in weighted)
    return [round(sum(area * centre[index] for area, centre in weighted) / total, 6) for index in range(3)]


def _stored_centroid(model: PartModel, face_ids: list[str]) -> list[float] | None:
    """Exact area centroid from the stored per-face centroids.

    ``FaceData.centroid`` is the surface mass centre the kernel line measured
    at parse time and ``area`` is its pooling weight, so a region's centroid
    is their weighted mean.  ``None`` when no face carries one (a model
    stored before the field existed).
    """
    points: list[tuple[float, list[float]]] = []
    for face_id in face_ids:
        face = model.faces.get(face_id)
        if face is None:
            continue
        centroid = getattr(face, "centroid", None)
        if not valid_vector(centroid):
            continue
        area = float(face.area) if isinstance(face.area, (int, float)) and float(face.area) > 0.0 else 0.0
        if area <= 0.0:
            continue
        points.append((area, [float(value) for value in centroid]))
    if not points:
        return None
    total = sum(item[0] for item in points)
    return [sum(mass * centre[index] for mass, centre in points) / total for index in range(3)]


def _has_stored_centroids(model: PartModel) -> bool:
    return any(getattr(face, "centroid", None) is not None for face in model.faces.values())


def _stored_centroid_method(model: PartModel) -> str:
    provider = getattr(model.metadata, "geometry_provider", "") or ""
    if provider in ("", "occline"):
        return "occ_surface_mass_centres"
    return f"{provider}_surface_mass_centres"


def _valid_bbox(value: object) -> bool:
    if not isinstance(value, dict) or not valid_vector(value.get("min")) or not valid_vector(value.get("max")):
        return False
    return all(float(value["min"][index]) <= float(value["max"][index]) for index in range(3))


def _perpendicular_basis(axis: list[float]) -> tuple[list[float], list[float]]:
    reference = [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0]
    u = [reference[1] * axis[2] - reference[2] * axis[1], reference[2] * axis[0] - reference[0] * axis[2], reference[0] * axis[1] - reference[1] * axis[0]]
    magnitude = math.sqrt(sum(value * value for value in u))
    if magnitude <= 1.0e-12:
        reference = [0.0, 1.0, 0.0]
        u = [reference[1] * axis[2] - reference[2] * axis[1], reference[2] * axis[0] - reference[0] * axis[2], reference[0] * axis[1] - reference[1] * axis[0]]
        magnitude = math.sqrt(sum(value * value for value in u))
    u = [value / magnitude for value in u]
    v = [axis[1] * u[2] - axis[2] * u[1], axis[2] * u[0] - axis[0] * u[2], axis[0] * u[1] - axis[1] * u[0]]
    return u, v


def measure_fillet_radius(model: PartModel, face_ids: tuple[str, ...]) -> tuple[float | None, dict[str, Any]]:
    values: list[tuple[str, float]] = []
    types: list[str] = []
    unproven: list[str] = []
    for face_id in face_ids:
        face = model.faces[face_id]
        types.append(face.surface_type.value)
        if face.surface_type == SurfaceType.TORUS:
            radius = face.parameters.get("minor_radius")
        elif face.surface_type in {SurfaceType.CYLINDER, SurfaceType.SPHERE}:
            radius = face.parameters.get("radius")
        else:
            radius = None
        if isinstance(radius, (int, float)) and math.isfinite(float(radius)) and float(radius) > 0.0:
            values.append((face_id, float(radius)))
        else:
            unproven.append(face_id)
    detail: dict[str, Any] = {
        "geometryMethod": "analytic_constant_radius_consensus",
        "surfaceTypes": sorted(set(types)),
        "radiusCandidates": [{"faceId": key, "radius": round(value, 6)} for key, value in values],
        "unprovenFaceIds": list(unproven),
    }
    if unproven:
        detail["reason"] = "blend_region_contains_unproven_or_nonconstant_radius_faces"
        return None, detail
    if not values:
        detail["reason"] = "transition_is_freeform_or_has_no_unique_analytic_radius"
        return None, detail
    radii = [value for _, value in values]
    tolerance = max(1.0e-4, max(radii) * 1.0e-3)
    if max(radii) - min(radii) > tolerance:
        detail["reason"] = "blend_radius_candidates_disagree_variable_or_mixed_radius"
        return None, detail
    return rounded_scalar(sum(radii) / len(radii)), detail
