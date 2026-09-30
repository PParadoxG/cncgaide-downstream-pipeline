"""Formal ``Step`` recognition and geometry extraction.

Step is an un-axised pair of planar principal faces sharing one straight
internal guide edge. Its semantic measurements are the two face extents and
the angle between the principal planes; the obsolete axis/height pair is not
part of the Step attribute set.
"""

from __future__ import annotations

import math
from typing import Any

from contract.partmodel import PartModel, SurfaceType

from core.models import FeatureNode
from features._shared.attributes import rounded_scalar, serialize_cad_face_ids
from features._shared.open_sweep import OpenSweepAnalysis
from features._shared.slot_step_resolution import (
    CandidateResolution,
    resolve_open_sweep_candidates,
)
from features._shared.transition_evidence import TransitionEvidence


_ATTRIBUTE_FIELDS = (
    "cadFaceIds",
    "posX",
    "posY",
    "posZ",
    "min_face_extent",
    "max_face_extent",
    "step_included_angle",
)

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
    """Build Step nodes from shared semantic resolutions."""

    claimed = set(claimed_face_ids or ())
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
    rejected: list[dict[str, Any]] = []
    semantic_records: list[dict[str, Any]] = []
    for resolution in resolutions:
        semantic_records.append(resolution.diagnostics())
        if resolution.decision != "Step":
            if resolution.reject_reason or resolution.decision == "Slot":
                rejected.append({
                    "candidateKey": resolution.candidate.candidate_key,
                    "reason": resolution.reject_reason or "resolved_as_slot",
                    "detail": resolution.reject_detail,
                    "semanticDecision": resolution.decision,
                })
            continue
        principal = list(resolution.candidate.principal_face_ids)
        overlap = sorted(set(principal) & claimed)
        if overlap:
            rejected.append({
                "candidateKey": resolution.candidate.candidate_key,
                "reason": "principal_faces_claimed_before_step_registration",
                "detail": ",".join(overlap),
            })
            continue
        geometry = measure_step(model, resolution)
        node = _build_node(model, resolution, geometry, start_index + len(nodes))
        nodes.append(node)
        claimed.update(principal)

    diagnostics = {
        "featureModule": "Step",
        "ontologyFeature": True,
        "method": "open_sweep_terminal_resolution_aagnet_family_resolution_step_v2",
        "candidateCount": len(sweeps.candidates),
        "semanticResolverCandidateCount": len(resolutions),
        "acceptedCount": len(nodes),
        "claimedFaceCount": sum(len(node.partmodelFaceIds) for node in nodes),
        "ownershipPolicy": (
            "only OS principal faces; transverse supports, longitudinal end supports, "
            "and transition faces remain unclaimed"
        ),
        "referencePointConvention": "midpoint of the shared straight internal guide edge",
        "semanticCandidates": semantic_records,
        "rejectedCandidates": rejected,
    }
    return nodes, claimed, diagnostics


def measure_step(model: PartModel, resolution: CandidateResolution) -> dict[str, Any]:
    """Measure the shared-edge midpoint, two face extents, and included angle."""

    candidate = resolution.candidate
    guide_edge = candidate.internal_rail_edge_ids[0] if candidate.internal_rail_edge_ids else None
    midpoint = _edge_midpoint(model, guide_edge)
    principal = list(candidate.principal_face_ids)
    extents = [_face_extent_from_edge(model, face_id, guide_edge) for face_id in principal[:2]]
    usable_extents = [value for value in extents if value is not None and value > _VECTOR_TOL]
    min_extent = min(usable_extents) if len(usable_extents) == 2 else None
    max_extent = max(usable_extents) if len(usable_extents) == 2 else None
    angle = _included_angle(model, principal[:2])

    extent_reason = (
        "both principal faces need measurable boundary samples away from the shared straight guide edge"
    )
    angle_reason = "the two principal faces do not provide two valid planar normals"
    return {
        "guideRailEdgeId": guide_edge,
        "guideRailMidpoint": midpoint,
        "minFaceExtent": rounded_scalar(min_extent),
        "maxFaceExtent": rounded_scalar(max_extent),
        "stepIncludedAngle": rounded_scalar(angle),
        "missingReasons": {
            "cadFaceIds": "no source CAD face ids could be recovered from the recognized principal faces",
            "posX": "the shared straight guide edge has no recoverable midpoint",
            "posY": "the shared straight guide edge has no recoverable midpoint",
            "posZ": "the shared straight guide edge has no recoverable midpoint",
            "min_face_extent": extent_reason,
            "max_face_extent": extent_reason,
            "step_included_angle": angle_reason,
        },
    }


def _build_node(model, resolution, geometry, index):
    principal = list(resolution.candidate.principal_face_ids)
    attributes = {name: None for name in _ATTRIBUTE_FIELDS}
    attributes["cadFaceIds"] = serialize_cad_face_ids(model, principal)
    point = geometry.get("guideRailMidpoint")
    for name, component in zip(("posX", "posY", "posZ"), point or (None, None, None)):
        attributes[name] = rounded_scalar(component)
    attributes["min_face_extent"] = geometry.get("minFaceExtent")
    attributes["max_face_extent"] = geometry.get("maxFaceExtent")
    attributes["step_included_angle"] = geometry.get("stepIncludedAngle")
    missing = {
        name: geometry["missingReasons"].get(name, "property was not reliably measurable")
        for name in _ATTRIBUTE_FIELDS
        if attributes.get(name) is None
    }
    return FeatureNode(
        nodeId=f"Step_{index:03d}",
        featureType="Step",
        partmodelFaceIds=principal,
        attributes=attributes,
        missingAttributes=missing,
        diagnostics={
            "method": "open_sweep_terminal_resolution_aagnet_family_resolution_step_v2",
            "semanticResolution": resolution.diagnostics(),
            "propertyExtraction": geometry,
            "ownership": "principal OS faces only; all supports and transitions remain unclaimed",
            "referencePointConvention": "midpoint of shared internal straight guide edge",
        },
    )


def _edge_points(model: PartModel, edge_id: str | None) -> list[list[float]]:
    if edge_id is None:
        return []
    edge = model.edges.get(edge_id)
    if edge is None:
        return []
    points = [list(point) for point in (getattr(edge, "sample_points", None) or [])]
    if not points:
        points = [
            list(model.vertices[vertex_id].coordinates)
            for vertex_id in getattr(edge, "vertex_ids", ())
            if vertex_id in model.vertices
        ]
    return points


def _edge_midpoint(model: PartModel, edge_id: str | None) -> list[float] | None:
    points = _edge_points(model, edge_id)
    if not points:
        return None
    if len(points) == 1:
        return points[0]
    return [0.5 * (points[0][index] + points[-1][index]) for index in range(3)]


def _face_extent_from_edge(model: PartModel, face_id: str, edge_id: str | None) -> float | None:
    points = _face_points(model, face_id)
    edge_points = _edge_points(model, edge_id)
    if len(edge_points) < 2 or not points:
        return None
    origin = edge_points[0]
    direction = _unit([edge_points[-1][index] - origin[index] for index in range(3)])
    if direction is None:
        return None
    distances = []
    for point in points:
        delta = [point[index] - origin[index] for index in range(3)]
        projection = _dot(delta, direction)
        perpendicular = [delta[index] - projection * direction[index] for index in range(3)]
        distances.append(_norm(perpendicular))
    return max(distances) if distances else None


def _included_angle(model: PartModel, principal: list[str]) -> float | None:
    if len(principal) != 2:
        return None
    faces = [model.faces.get(face_id) for face_id in principal]
    if any(face is None or face.surface_type != SurfaceType.PLANE for face in faces):
        return None
    normals = [_face_normal(face) for face in faces]
    if any(normal is None for normal in normals):
        return None
    dot = max(-1.0, min(1.0, _dot(normals[0], normals[1])))
    # The unified outward normals describe the supplement of the interior
    # material angle.  Publish the internal Step angle, not the raw normal
    # separation, so a nearly antiparallel pair is a shallow step.
    angle = 180.0 - math.degrees(math.acos(dot))
    return angle if _VECTOR_TOL < angle < 180.0 - _VECTOR_TOL else None


def _face_points(model: PartModel, face_id: str) -> list[list[float]]:
    face = model.faces.get(face_id)
    if face is None:
        return []
    points: list[list[float]] = []
    for loop_id in face.loop_ids:
        loop = model.loops.get(loop_id)
        if loop is None or not loop.is_outer:
            continue
        for edge_id in loop.edge_ids:
            edge = model.edges.get(edge_id)
            if edge is not None:
                samples = getattr(edge, "sample_points", None) or []
                if samples:
                    points.extend([list(point) for point in samples])
                else:
                    points.extend(
                        [
                            list(model.vertices[vertex_id].coordinates)
                            for vertex_id in getattr(edge, "vertex_ids", ())
                            if vertex_id in model.vertices
                        ]
                    )
    return points


def _face_normal(face):
    value = face.parameters.get("normal") if face is not None else None
    normal = _unit(value)
    if normal is not None and face.is_reversed:
        normal = [-value for value in normal]
    return normal


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
