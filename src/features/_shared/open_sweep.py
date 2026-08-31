"""Topology/geometry-only open linear sweep (OS) discovery.

This module is shared structural analysis, not Slot/Step recognition.  It
finds an ordered, open chain of analytic faces that is invariant along one
common course direction.  The chain is recovered from longitudinal B-Rep
rails; no AAGNet label, feature angle, slenderness, or Slot-vs-Step rule is
used here.

The v1 structural contract is intentionally conservative:

* multi-face principal paths use unclaimed ``PLANE`` or ``CYLINDER`` faces;
* every principal face is longitudinal for the same course direction;
* consecutive multi-face principals share one full-length linear rail whose
  material-side connection is ``CONCAVE`` or ``SMOOTH``;
* the multi-face principal adjacency graph is one simple open path;
* a separate singleton path admits only an unclaimed ``CYLINDER`` whose
  intrinsic axis is bounded by exactly two full-length terminal rails;
* every face has the same two axial levels and all remaining **outer-loop**
  boundary edges lie wholly on one of those two transverse end levels;
* each path endpoint has one full-length terminal rail leading to faces
  outside the candidate.

Inner loops are local interruptions rather than boundaries of the principal
swept strip.  They neither seed rails nor participate in rail/start-end/
oblique validation; they remain attached as explicit audit diagnostics.

Those rules recover a stable swept *profile path*.  They do not say whether
the path is a Slot or a Step.  External terminal supports and longitudinal
end supports are context only and are never principal/owned geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from contract.partmodel import ConcavityType, CurveType, PartModel, SurfaceType

from features._shared.closed_wall_band import EdgeIndex, build_edge_index, face_sort_key


_ANALYTIC_SURFACES = {SurfaceType.PLANE, SurfaceType.CYLINDER}
_PROFILE_CONNECTIONS = {ConcavityType.CONCAVE.value, ConcavityType.SMOOTH.value}
_PARALLEL_COS = math.cos(1.0e-4)
_IN_PLANE_SIN = math.sin(1.0e-4)
_MIN_COURSE_LENGTH = 1.0e-6
_MAX_REJECTIONS_IN_DIAGNOSTICS = 80

TERMINAL_RECESS_LIKE = "RECESS_LIKE"
TERMINAL_OUTWARD_PROTRUSION_LIKE = "OUTWARD_PROTRUSION_LIKE"
TERMINAL_SMOOTH_OR_TRANSITION = "SMOOTH_OR_TRANSITION"
TERMINAL_AMBIGUOUS = "AMBIGUOUS"
TERMINAL_UNRESOLVED = "UNRESOLVED"


@dataclass
class OpenSweepCandidate:
    candidate_key: str
    course_direction: list[float]
    course_length: float
    course_start: float
    course_end: float
    principal_face_ids: list[str]
    ordered_profile_face_ids: list[str]
    principal_surface_types: dict[str, str]
    internal_rail_edge_ids: list[str]
    internal_rail_connections: list[str]
    profile_terminal_face_ids: list[str]
    profile_terminal_rail_edge_ids: list[str]
    profile_terminal_support_face_ids: list[list[str]]
    profile_terminal_material_connections: list[list[dict[str, str]]]
    profile_terminal_effective_contexts: list[str]
    longitudinal_start_edge_ids_by_face: list[dict[str, Any]]
    longitudinal_end_edge_ids_by_face: list[dict[str, Any]]
    longitudinal_start_support_face_ids: list[str]
    longitudinal_end_support_face_ids: list[str]
    adjacent_context_face_ids: list[str]
    principal_inner_loop_diagnostics: dict[str, list[dict[str, Any]]]
    discovery_mode: str = "internal_rail_profile_chain"
    acceptance_reason: str = "stable_open_profile_linear_sweep"


@dataclass
class OpenSweepRejection:
    course_direction: list[float]
    face_ids: list[str]
    reason: str
    detail: str = ""


@dataclass
class OpenSweepAnalysis:
    candidates: list[OpenSweepCandidate] = field(default_factory=list)
    rejections: list[OpenSweepRejection] = field(default_factory=list)
    claimed_input_face_count: int = 0
    analytic_unclaimed_face_count: int = 0
    course_direction_count: int = 0
    structural_component_count: int = 0


def discover_open_sweeps(
    model: PartModel,
    claimed_face_ids: set[str] | None = None,
) -> OpenSweepAnalysis:
    """Discover conservative open-sweep candidates in the unclaimed residue."""
    claimed = set(claimed_face_ids or ())
    available = {
        face_id
        for face_id, face in model.faces.items()
        if face_id not in claimed and face.surface_type in _ANALYTIC_SURFACES
    }
    analysis = OpenSweepAnalysis(
        claimed_input_face_count=len(claimed),
        analytic_unclaimed_face_count=len(available),
    )
    if not available:
        return analysis

    edge_index = build_edge_index(model)
    outer_sigs = {
        face_id: set(_face_boundary_sigs(model, edge_index, face_id))
        for face_id in available
    }
    directions = _course_direction_seeds(model, edge_index, available, outer_sigs)
    discovered_directions = list(directions)
    seen_face_sets: set[frozenset[str]] = set()

    for direction in directions:
        longitudinal = {
            face_id for face_id in available
            if _face_supports_course(model, face_id, direction)
        }
        graph, pair_to_sigs, pair_connections = _profile_graph(
            model, edge_index, longitudinal, direction, outer_sigs
        )
        for component in _components(graph):
            analysis.structural_component_count += 1
            result = _analyze_component(
                model,
                edge_index,
                direction,
                component,
                graph,
                pair_to_sigs,
                pair_connections,
            )
            if isinstance(result, OpenSweepRejection):
                analysis.rejections.append(result)
                continue
            signature = frozenset(result.principal_face_ids)
            if signature in seen_face_sets:
                analysis.rejections.append(
                    OpenSweepRejection(
                        direction,
                        sorted(component, key=face_sort_key),
                        "duplicate_principal_face_set",
                        "the same principal face set was recovered from an equivalent course direction",
                    )
                )
                continue
            seen_face_sets.add(signature)
            result.candidate_key = f"OS{len(analysis.candidates):03d}"
            analysis.candidates.append(result)

    # A transverse profile made from one cylindrical arc has no internal rail
    # from which the multi-face path can seed.  Analyze such cylinders only
    # after the existing graph path, and never split a face already accepted
    # as part of a multi-face candidate into a singleton fragment.
    multi_face_ids = {
        face_id
        for candidate in analysis.candidates
        for face_id in candidate.principal_face_ids
    }
    for face_id in sorted(available - multi_face_ids, key=face_sort_key):
        if model.faces[face_id].surface_type != SurfaceType.CYLINDER:
            continue
        result = _analyze_single_cylinder(model, edge_index, face_id)
        if isinstance(result, OpenSweepRejection):
            analysis.rejections.append(result)
            continue
        signature = frozenset(result.principal_face_ids)
        if signature in seen_face_sets:
            continue
        seen_face_sets.add(signature)
        if not any(_parallel(result.course_direction, known) for known in discovered_directions):
            discovered_directions.append(result.course_direction)
        result.candidate_key = f"OS{len(analysis.candidates):03d}"
        analysis.candidates.append(result)

    analysis.course_direction_count = len(discovered_directions)
    return analysis


def analysis_diagnostics(analysis: OpenSweepAnalysis) -> dict[str, Any]:
    """Whole-part diagnostics for ``result.diagnostics.openSweeps``."""
    return {
        "role": "shared_structural_analysis_not_slot_or_step_semantics",
        "ownsFaces": False,
        "aagnetUsed": False,
        "method": "open_linear_profile_sweep_v1",
        "surfaceTypePolicy": ["PLANE", "CYLINDER"],
        "profileConnectionPolicy": sorted(_PROFILE_CONNECTIONS),
        "semanticExclusions": ["Slot-vs-Step", "feature angle", "slenderness", "AAGNet Slot/Step labels"],
        "claimedInputFaceCount": analysis.claimed_input_face_count,
        "analyticUnclaimedFaceCount": analysis.analytic_unclaimed_face_count,
        "courseDirectionCount": analysis.course_direction_count,
        "structuralComponentCount": analysis.structural_component_count,
        "candidateCount": len(analysis.candidates),
        "candidates": [candidate_diagnostics(item) for item in analysis.candidates],
        "rejectionSummary": _rejection_summary(analysis.rejections),
        "rejectedComponents": [
            {
                "courseDirection": item.course_direction,
                "faceIds": item.face_ids,
                "reason": item.reason,
                "detail": item.detail,
            }
            for item in analysis.rejections[:_MAX_REJECTIONS_IN_DIAGNOSTICS]
        ],
    }


def candidate_diagnostics(candidate: OpenSweepCandidate) -> dict[str, Any]:
    """Render one accepted OS record for part and audit-feature diagnostics."""
    return {
        "method": "open_linear_profile_sweep_v1",
        "candidateKey": candidate.candidate_key,
        "discoveryMode": candidate.discovery_mode,
        "acceptanceReason": candidate.acceptance_reason,
        "aagnetUsed": False,
        "semanticClassification": None,
        "course": {
            "direction": list(candidate.course_direction),
            "length": round(candidate.course_length, 6),
            "startProjection": round(candidate.course_start, 6),
            "endProjection": round(candidate.course_end, 6),
            "orientationPolicy": "deterministic undirected-axis canonicalization; no machining-direction semantics",
        },
        "principalSweepGeometry": {
            "faceIds": list(candidate.principal_face_ids),
            "surfaceTypes": dict(candidate.principal_surface_types),
            "ownership": (
                "OS owns none; formal Slot/Step may claim these faces only "
                "after their semantic and ownership gates pass"
            ),
        },
        "orderedOpenProfilePath": {
            "faceIds": list(candidate.ordered_profile_face_ids),
            "internalRailEdgeIds": list(candidate.internal_rail_edge_ids),
            "internalRailConnections": list(candidate.internal_rail_connections),
            "orderingPolicy": "deterministic endpoint first; reverse path is structurally equivalent",
        },
        "profileTerminals": [
            {
                "principalFaceId": candidate.profile_terminal_face_ids[index],
                "railEdgeId": candidate.profile_terminal_rail_edge_ids[index],
                "supportFaceIds": list(candidate.profile_terminal_support_face_ids[index]),
                "materialSideConnections": list(candidate.profile_terminal_material_connections[index]),
                "effectiveContext": candidate.profile_terminal_effective_contexts[index],
                "supportOwned": False,
            }
            for index in range(2)
        ],
        "longitudinalEnds": {
            "start": {
                "profileEdgesByFace": candidate.longitudinal_start_edge_ids_by_face,
                "supportFaceIds": list(candidate.longitudinal_start_support_face_ids),
                "supportOwned": False,
            },
            "end": {
                "profileEdgesByFace": candidate.longitudinal_end_edge_ids_by_face,
                "supportFaceIds": list(candidate.longitudinal_end_support_face_ids),
                "supportOwned": False,
            },
        },
        "adjacentContextFaceIds": list(candidate.adjacent_context_face_ids),
        "principalInnerLoops": candidate.principal_inner_loop_diagnostics,
        "innerLoopPolicy": (
            "recorded for audit but excluded from principal outer-boundary "
            "rail/start-end/oblique validation"
        ),
    }


def _course_direction_seeds(
    model: PartModel,
    index: EdgeIndex,
    available: set[str],
    outer_sigs: dict[str, set[tuple]],
) -> list[list[float]]:
    seeds: list[list[float]] = []
    for sig in sorted(index.sig_incidence):
        inside = index.sig_incidence[sig] & available
        if len(inside) != 2:
            continue
        left, right = sorted(inside, key=face_sort_key)
        if sig not in outer_sigs.get(left, set()) or sig not in outer_sigs.get(right, set()):
            continue
        if _connection_type(model, index, left, right, sig) not in _PROFILE_CONNECTIONS:
            continue
        direction = _edge_direction(model, index, sig)
        if direction is None:
            continue
        direction = _canonical_axis(direction)
        if not any(_parallel(direction, known) for known in seeds):
            seeds.append(direction)
    return sorted(seeds, key=lambda value: tuple(round(component, 9) for component in value))


def _profile_graph(
    model: PartModel,
    index: EdgeIndex,
    face_ids: set[str],
    direction: list[float],
    outer_sigs: dict[str, set[tuple]],
) -> tuple[dict[str, set[str]], dict[tuple[str, str], list[tuple]], dict[tuple[str, str], list[str]]]:
    graph: dict[str, set[str]] = {}
    pair_to_sigs: dict[tuple[str, str], list[tuple]] = {}
    pair_connections: dict[tuple[str, str], list[str]] = {}
    for sig, incidence in index.sig_incidence.items():
        inside = incidence & face_ids
        if len(inside) != 2:
            continue
        left, right = sorted(inside, key=face_sort_key)
        if sig not in outer_sigs.get(left, set()) or sig not in outer_sigs.get(right, set()):
            continue
        edge_direction = _edge_direction(model, index, sig)
        if edge_direction is None or not _parallel(edge_direction, direction):
            continue
        connection = _connection_type(model, index, left, right, sig)
        if connection not in _PROFILE_CONNECTIONS:
            continue
        pair = (left, right)
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
        pair_to_sigs.setdefault(pair, []).append(sig)
        pair_connections.setdefault(pair, []).append(connection)
    return graph, pair_to_sigs, pair_connections


def _analyze_component(
    model: PartModel,
    index: EdgeIndex,
    direction: list[float],
    component: set[str],
    graph: dict[str, set[str]],
    pair_to_sigs: dict[tuple[str, str], list[tuple]],
    pair_connections: dict[tuple[str, str], list[str]],
) -> OpenSweepCandidate | OpenSweepRejection:
    face_ids = sorted(component, key=face_sort_key)

    def reject(reason: str, detail: str = "") -> OpenSweepRejection:
        return OpenSweepRejection(list(direction), face_ids, reason, detail)

    if len(component) < 2:
        return reject("profile_has_fewer_than_two_faces")
    # Keyed off the sorted list rather than the set: three rejections below
    # stringify this dict into a stored diagnostic, and a set's iteration
    # order follows the process hash seed - the same part would otherwise
    # produce a different result.json on every run.
    degrees = {face_id: len(graph.get(face_id, set()) & component) for face_id in face_ids}
    endpoints = sorted((face_id for face_id, degree in degrees.items() if degree == 1), key=face_sort_key)
    if any(degree > 2 for degree in degrees.values()):
        return reject("profile_path_branches", str(degrees))
    if len(endpoints) == 0:
        return reject("profile_path_is_closed", str(degrees))
    if len(endpoints) != 2 or any(degree == 0 for degree in degrees.values()):
        return reject("profile_path_is_not_one_simple_open_chain", str(degrees))

    ordered = _ordered_path(graph, component, endpoints[0])
    if len(ordered) != len(component):
        return reject("profile_path_is_disconnected_after_ordering")

    internal_sigs: list[tuple] = []
    internal_connections: list[str] = []
    for left, right in zip(ordered, ordered[1:]):
        pair = tuple(sorted((left, right), key=face_sort_key))
        sigs = pair_to_sigs.get(pair, [])
        if len(sigs) != 1:
            return reject("profile_neighbors_do_not_share_exactly_one_internal_rail", f"{left},{right}: {len(sigs)} rails")
        internal_sigs.append(sigs[0])
        values = pair_connections.get(pair, [])
        internal_connections.append(values[0] if values else ConcavityType.UNKNOWN.value)

    extents: dict[str, tuple[float, float]] = {}
    for face_id in ordered:
        extent = _face_course_extent(model, index, face_id, direction)
        if extent is None or extent[1] - extent[0] <= _MIN_COURSE_LENGTH:
            return reject("principal_face_course_extent_unavailable", face_id)
        extents[face_id] = extent
    start = sum(value[0] for value in extents.values()) / len(extents)
    end = sum(value[1] for value in extents.values()) / len(extents)
    tolerance = _length_tolerance(end - start)
    unstable = [
        face_id for face_id, extent in extents.items()
        if abs(extent[0] - start) > tolerance or abs(extent[1] - end) > tolerance
    ]
    if unstable:
        return reject("principal_faces_do_not_share_two_longitudinal_ends", ",".join(unstable))

    boundary_by_face: dict[str, dict[str, list[tuple]]] = {}
    all_internal = set(internal_sigs)
    for face_id in ordered:
        classified = _classify_face_boundary(
            model, index, face_id, direction, start, end, tolerance
        )
        if classified["oblique"]:
            return reject(
                "principal_face_has_non_sweep_boundary_edge",
                f"{face_id}: " + ",".join(index.representative(sig) for sig in classified["oblique"]),
            )
        if not classified["start"] or not classified["end"]:
            return reject("principal_face_missing_transverse_end_profile", face_id)
        boundary_by_face[face_id] = classified

    # Every verified internal connection must be a complete course rail.
    for sig in internal_sigs:
        if not _sig_is_full_rail(model, index, sig, direction, start, end, tolerance):
            return reject("internal_rail_does_not_span_both_longitudinal_ends", index.representative(sig))

    terminal_sigs: list[tuple] = []
    terminal_supports: list[list[str]] = []
    terminal_material_connections: list[list[dict[str, str]]] = []
    terminal_effective_contexts: list[str] = []
    for endpoint in (ordered[0], ordered[-1]):
        candidates = [
            sig for sig in boundary_by_face[endpoint]["rails"]
            if sig not in all_internal
        ]
        if len(candidates) != 1:
            return reject("profile_terminal_rail_count_not_one", f"{endpoint}: {len(candidates)}")
        sig = candidates[0]
        supports = sorted(index.sig_incidence.get(sig, set()) - component, key=face_sort_key)
        if not supports:
            return reject("profile_terminal_has_no_external_support", endpoint)
        connections = [
            {
                "supportFaceId": support,
                "connectionType": _terminal_connection_type(
                    model, index, endpoint, support, sig
                ),
            }
            for support in supports
        ]
        terminal_sigs.append(sig)
        terminal_supports.append(supports)
        terminal_material_connections.append(connections)
        terminal_effective_contexts.append(
            _terminal_effective_context([item["connectionType"] for item in connections])
        )

    # Interior faces may not leak another complete longitudinal rail to an
    # outside face: that would make the chosen component a branch fragment.
    for face_id in ordered[1:-1]:
        leaked = [sig for sig in boundary_by_face[face_id]["rails"] if sig not in all_internal]
        if leaked:
            return reject("interior_profile_face_has_external_full_rail", face_id)

    start_edges = [
        {"faceId": face_id, "edgeIds": [index.representative(sig) for sig in boundary_by_face[face_id]["start"]]}
        for face_id in ordered
    ]
    end_edges = [
        {"faceId": face_id, "edgeIds": [index.representative(sig) for sig in boundary_by_face[face_id]["end"]]}
        for face_id in ordered
    ]
    start_sigs = {sig for face_id in ordered for sig in boundary_by_face[face_id]["start"]}
    end_sigs = {sig for face_id in ordered for sig in boundary_by_face[face_id]["end"]}
    start_supports = _outside_faces(index, start_sigs, component)
    end_supports = _outside_faces(index, end_sigs, component)
    if not start_supports or not end_supports:
        return reject("longitudinal_end_has_no_external_support", f"start={start_supports} end={end_supports}")

    boundary_sigs = {
        sig
        for face_id in ordered
        for group in ("rails", "start", "end")
        for sig in boundary_by_face[face_id][group]
    }
    context = _outside_faces(index, boundary_sigs, component)
    return OpenSweepCandidate(
        candidate_key="",
        course_direction=[round(value, 6) + 0.0 for value in direction],
        course_length=end - start,
        course_start=start,
        course_end=end,
        principal_face_ids=sorted(component, key=face_sort_key),
        ordered_profile_face_ids=ordered,
        principal_surface_types={face_id: model.faces[face_id].surface_type.value for face_id in ordered},
        internal_rail_edge_ids=[index.representative(sig) for sig in internal_sigs],
        internal_rail_connections=internal_connections,
        profile_terminal_face_ids=[ordered[0], ordered[-1]],
        profile_terminal_rail_edge_ids=[index.representative(sig) for sig in terminal_sigs],
        profile_terminal_support_face_ids=terminal_supports,
        profile_terminal_material_connections=terminal_material_connections,
        profile_terminal_effective_contexts=terminal_effective_contexts,
        longitudinal_start_edge_ids_by_face=start_edges,
        longitudinal_end_edge_ids_by_face=end_edges,
        longitudinal_start_support_face_ids=start_supports,
        longitudinal_end_support_face_ids=end_supports,
        adjacent_context_face_ids=context,
        principal_inner_loop_diagnostics=_inner_loop_diagnostics(
            model, index, ordered, direction
        ),
        discovery_mode="internal_rail_profile_chain",
    )


def _analyze_single_cylinder(
    model: PartModel,
    index: EdgeIndex,
    face_id: str,
) -> OpenSweepCandidate | OpenSweepRejection:
    """Prove one cylindrical transverse-profile arc as an open sweep.

    This path is deliberately independent of the internal-rail profile graph.
    It admits no planar singleton and uses only the cylinder's intrinsic axis
    plus its outer-loop boundary.  Inner loops remain audit-only context.
    """
    face_ids = [face_id]
    face = model.faces[face_id]
    axis = _unit(face.parameters.get("axis_direction"))

    def reject(reason: str, detail: str = "") -> OpenSweepRejection:
        return OpenSweepRejection(
            list(_canonical_axis(axis)) if axis is not None else [],
            face_ids,
            reason,
            detail,
        )

    if face.surface_type != SurfaceType.CYLINDER:
        return reject("single_profile_face_is_not_cylinder", face_id)
    if axis is None:
        return reject("single_cylinder_axis_unavailable", face_id)
    direction = _canonical_axis(axis)
    extent = _face_course_extent(model, index, face_id, direction)
    if extent is None or extent[1] - extent[0] <= _MIN_COURSE_LENGTH:
        return reject("single_cylinder_course_extent_unavailable", face_id)
    start, end = extent
    tolerance = _length_tolerance(end - start)
    classified = _classify_face_boundary(
        model, index, face_id, direction, start, end, tolerance
    )
    if classified["oblique"]:
        return reject(
            "single_cylinder_has_non_sweep_outer_boundary_edge",
            ",".join(index.representative(sig) for sig in classified["oblique"]),
        )
    terminal_sigs = sorted(
        classified["rails"], key=lambda sig: index.representative(sig)
    )
    if len(terminal_sigs) != 2:
        return reject(
            "single_cylinder_terminal_rail_count_not_two",
            str(len(terminal_sigs)),
        )
    if not classified["start"] or not classified["end"]:
        return reject("single_cylinder_missing_transverse_longitudinal_end", face_id)

    component = {face_id}
    terminal_supports: list[list[str]] = []
    terminal_material_connections: list[list[dict[str, str]]] = []
    terminal_effective_contexts: list[str] = []
    for sig in terminal_sigs:
        supports = sorted(
            index.sig_incidence.get(sig, set()) - component,
            key=face_sort_key,
        )
        if not supports:
            return reject(
                "single_cylinder_terminal_has_no_external_support",
                index.representative(sig),
            )
        connections = [
            {
                "supportFaceId": support,
                "connectionType": _terminal_connection_type(
                    model, index, face_id, support, sig
                ),
            }
            for support in supports
        ]
        terminal_supports.append(supports)
        terminal_material_connections.append(connections)
        terminal_effective_contexts.append(
            _terminal_effective_context(
                [item["connectionType"] for item in connections]
            )
        )

    start_sigs = set(classified["start"])
    end_sigs = set(classified["end"])
    start_supports = _outside_faces(index, start_sigs, component)
    end_supports = _outside_faces(index, end_sigs, component)
    if not start_supports or not end_supports:
        return reject(
            "single_cylinder_longitudinal_end_has_no_external_support",
            f"start={start_supports} end={end_supports}",
        )

    boundary_sigs = set(terminal_sigs) | start_sigs | end_sigs
    return OpenSweepCandidate(
        candidate_key="",
        course_direction=[round(value, 6) + 0.0 for value in direction],
        course_length=end - start,
        course_start=start,
        course_end=end,
        principal_face_ids=[face_id],
        ordered_profile_face_ids=[face_id],
        principal_surface_types={face_id: SurfaceType.CYLINDER.value},
        internal_rail_edge_ids=[],
        internal_rail_connections=[],
        profile_terminal_face_ids=[face_id, face_id],
        profile_terminal_rail_edge_ids=[
            index.representative(sig) for sig in terminal_sigs
        ],
        profile_terminal_support_face_ids=terminal_supports,
        profile_terminal_material_connections=terminal_material_connections,
        profile_terminal_effective_contexts=terminal_effective_contexts,
        longitudinal_start_edge_ids_by_face=[{
            "faceId": face_id,
            "edgeIds": [
                index.representative(sig) for sig in classified["start"]
            ],
        }],
        longitudinal_end_edge_ids_by_face=[{
            "faceId": face_id,
            "edgeIds": [
                index.representative(sig) for sig in classified["end"]
            ],
        }],
        longitudinal_start_support_face_ids=start_supports,
        longitudinal_end_support_face_ids=end_supports,
        adjacent_context_face_ids=_outside_faces(index, boundary_sigs, component),
        principal_inner_loop_diagnostics=_inner_loop_diagnostics(
            model, index, [face_id], direction
        ),
        discovery_mode="single_cylindrical_profile",
    )


def _classify_face_boundary(
    model: PartModel,
    index: EdgeIndex,
    face_id: str,
    direction: list[float],
    start: float,
    end: float,
    tolerance: float,
) -> dict[str, list[tuple]]:
    result: dict[str, list[tuple]] = {"rails": [], "start": [], "end": [], "oblique": []}
    for sig in _face_boundary_sigs(model, index, face_id):
        if sig in index.self_seam_sigs:
            continue
        extent = _edge_projection_extent(model, index, sig, direction)
        if extent is None:
            result["oblique"].append(sig)
            continue
        lower, upper = extent
        edge_direction = _edge_direction(model, index, sig)
        if (
            edge_direction is not None
            and _parallel(edge_direction, direction)
            and abs(lower - start) <= tolerance
            and abs(upper - end) <= tolerance
        ):
            result["rails"].append(sig)
        elif upper - lower <= tolerance and abs((lower + upper) * 0.5 - start) <= tolerance:
            result["start"].append(sig)
        elif upper - lower <= tolerance and abs((lower + upper) * 0.5 - end) <= tolerance:
            result["end"].append(sig)
        else:
            result["oblique"].append(sig)
    return result


def _face_supports_course(model: PartModel, face_id: str, direction: list[float]) -> bool:
    face = model.faces[face_id]
    if face.surface_type == SurfaceType.PLANE:
        normal = _unit(face.parameters.get("normal"))
        return normal is not None and abs(_dot(normal, direction)) <= _IN_PLANE_SIN
    if face.surface_type == SurfaceType.CYLINDER:
        axis = _unit(face.parameters.get("axis_direction"))
        return axis is not None and _parallel(axis, direction)
    return False


def _connection_type(model: PartModel, index: EdgeIndex, left: str, right: str, sig: tuple) -> str:
    found = _connection_values(model, index, left, right, sig)
    if not found:
        return ConcavityType.UNKNOWN.value
    for preferred in (ConcavityType.CONCAVE.value, ConcavityType.SMOOTH.value, ConcavityType.CONVEX.value):
        if preferred in found:
            return preferred
    return found[0]


def _terminal_connection_type(
    model: PartModel,
    index: EdgeIndex,
    principal_face_id: str,
    support_face_id: str,
    sig: tuple,
) -> str:
    values = set(_connection_values(
        model, index, principal_face_id, support_face_id, sig
    ))
    if not values:
        return ConcavityType.UNKNOWN.value
    if len(values) == 1:
        return next(iter(values))
    return TERMINAL_AMBIGUOUS


def _connection_values(
    model: PartModel,
    index: EdgeIndex,
    left: str,
    right: str,
    sig: tuple,
) -> list[str]:
    found: list[str] = []
    for face_id, other_id in ((left, right), (right, left)):
        face = model.faces.get(face_id)
        if face is None:
            continue
        for adjacency in face.adjacencies:
            if adjacency.adjacent_face_id != other_id:
                continue
            edge_sig = index.edge_sig.get(adjacency.shared_edge_id)
            if edge_sig == sig:
                value = adjacency.connection_type
                found.append(value.value if hasattr(value, "value") else str(value))
    return found


def _face_boundary_sigs(model: PartModel, index: EdgeIndex, face_id: str) -> list[tuple]:
    """Canonical signatures from a face's outer loops only.

    Inner loops are local interruptions, not boundaries of the principal
    swept strip.  They are rendered separately by ``_inner_loop_diagnostics``
    and deliberately do not seed rails or participate in outer-boundary
    rail/start-end/oblique validation.
    """
    face = model.faces[face_id]
    seen: set[tuple] = set()
    result: list[tuple] = []
    for loop_id in face.loop_ids:
        loop = model.loops.get(loop_id)
        if loop is None or not loop.is_outer:
            continue
        for edge_id in loop.edge_ids:
            sig = index.edge_sig.get(edge_id)
            if sig is not None and sig not in seen:
                seen.add(sig)
                result.append(sig)
    return result


def _inner_loop_diagnostics(
    model: PartModel,
    index: EdgeIndex,
    face_ids: list[str],
    direction: list[float],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for face_id in face_ids:
        records: list[dict[str, Any]] = []
        face = model.faces[face_id]
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None or loop.is_outer:
                continue
            edges: list[dict[str, Any]] = []
            adjacent: set[str] = set()
            for edge_id in loop.edge_ids:
                sig = index.edge_sig.get(edge_id)
                if sig is None:
                    continue
                representative = index.representative(sig)
                edge = model.edges[representative]
                extent = _edge_projection_extent(model, index, sig, direction)
                adjacent.update(index.sig_incidence.get(sig, set()) - {face_id})
                edges.append({
                    "edgeId": representative,
                    "curveType": edge.curve_type.value,
                    "courseProjectionExtent": (
                        [round(extent[0], 6), round(extent[1], 6)]
                        if extent is not None else None
                    ),
                })
            records.append({
                "loopId": loop_id,
                "edgeIds": [item["edgeId"] for item in edges],
                "edges": edges,
                "adjacentFaceIds": sorted(adjacent, key=face_sort_key),
                "ignoredForPrincipalSweepBoundaryValidation": True,
            })
        if records:
            result[face_id] = records
    return result


def _terminal_effective_context(connection_types: list[str]) -> str:
    values = set(connection_types)
    if not values or ConcavityType.UNKNOWN.value in values:
        return TERMINAL_UNRESOLVED if len(values) <= 1 else TERMINAL_AMBIGUOUS
    if values == {ConcavityType.CONVEX.value}:
        return TERMINAL_RECESS_LIKE
    if values == {ConcavityType.CONCAVE.value}:
        return TERMINAL_OUTWARD_PROTRUSION_LIKE
    if values == {ConcavityType.SMOOTH.value}:
        return TERMINAL_SMOOTH_OR_TRANSITION
    return TERMINAL_AMBIGUOUS


def _face_course_extent(
    model: PartModel,
    index: EdgeIndex,
    face_id: str,
    direction: list[float],
) -> tuple[float, float] | None:
    values: list[float] = []
    for sig in _face_boundary_sigs(model, index, face_id):
        edge = model.edges[index.representative(sig)]
        for point in _edge_points(model, edge):
            values.append(_dot(point, direction))
    if not values:
        face = model.faces[face_id]
        bbox = face.bbox
        if isinstance(bbox, dict):
            lower, upper = bbox.get("min"), bbox.get("max")
            if _vector(lower) and _vector(upper):
                values = [
                    _dot([upper[i] if mask & (1 << i) else lower[i] for i in range(3)], direction)
                    for mask in range(8)
                ]
    return (min(values), max(values)) if values else None


def _edge_projection_extent(
    model: PartModel,
    index: EdgeIndex,
    sig: tuple,
    direction: list[float],
) -> tuple[float, float] | None:
    edge = model.edges[index.representative(sig)]
    values = [_dot(point, direction) for point in _edge_points(model, edge)]
    return (min(values), max(values)) if values else None


def _sig_is_full_rail(model, index, sig, direction, start, end, tolerance) -> bool:
    extent = _edge_projection_extent(model, index, sig, direction)
    edge_direction = _edge_direction(model, index, sig)
    return bool(
        extent is not None
        and edge_direction is not None
        and _parallel(edge_direction, direction)
        and abs(extent[0] - start) <= tolerance
        and abs(extent[1] - end) <= tolerance
    )


def _edge_direction(model: PartModel, index: EdgeIndex, sig: tuple) -> list[float] | None:
    edge = model.edges[index.representative(sig)]
    if edge.curve_type != CurveType.LINE:
        return None
    direction = _unit(edge.parameters.get("direction"))
    if direction is not None:
        return direction
    points = _edge_points(model, edge)
    if len(points) >= 2:
        best = max(
            ((math.dist(left, right), left, right) for left in points for right in points),
            key=lambda item: item[0],
        )
        if best[0] > _MIN_COURSE_LENGTH:
            return _unit([best[2][axis] - best[1][axis] for axis in range(3)])
    return None


def _edge_points(model: PartModel, edge) -> list[list[float]]:
    points = [list(point) for point in (getattr(edge, "sample_points", None) or []) if _vector(point)]
    if points:
        return points
    return [
        list(model.vertices[vertex_id].coordinates)
        for vertex_id in (getattr(edge, "vertex_ids", None) or [])
        if vertex_id in model.vertices
    ]


def _components(graph: dict[str, set[str]]) -> list[set[str]]:
    result: list[set[str]] = []
    seen: set[str] = set()
    for start in sorted(graph, key=face_sort_key):
        if start in seen:
            continue
        component = {start}
        stack = [start]
        seen.add(start)
        while stack:
            for neighbor in sorted(graph.get(stack.pop(), set()), key=face_sort_key):
                if neighbor not in seen:
                    seen.add(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        result.append(component)
    return result


def _ordered_path(graph: dict[str, set[str]], component: set[str], start: str) -> list[str]:
    ordered: list[str] = []
    previous: str | None = None
    current: str | None = start
    while current is not None:
        ordered.append(current)
        choices = sorted((graph.get(current, set()) & component) - ({previous} if previous else set()), key=face_sort_key)
        previous, current = current, choices[0] if choices else None
    return ordered


def _outside_faces(index: EdgeIndex, sigs: set[tuple], component: set[str]) -> list[str]:
    found: set[str] = set()
    for sig in sigs:
        found.update(index.sig_incidence.get(sig, set()) - component)
    return sorted(found, key=face_sort_key)


def _rejection_summary(rejections: list[OpenSweepRejection]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rejections:
        counts[item.reason] = counts.get(item.reason, 0) + 1
    return dict(sorted(counts.items()))


def _length_tolerance(length: float) -> float:
    return max(1.0e-4, abs(float(length)) * 1.0e-5)


def _parallel(left: list[float], right: list[float]) -> bool:
    return abs(_dot(left, right)) >= _PARALLEL_COS


def _canonical_axis(value: list[float]) -> list[float]:
    axis = _unit(value) or [1.0, 0.0, 0.0]
    for component in axis:
        if abs(component) <= 1.0e-12:
            continue
        if component < 0.0:
            axis = [-item for item in axis]
        break
    return axis


def _unit(value: object) -> list[float] | None:
    if not _vector(value):
        return None
    magnitude = math.sqrt(sum(float(item) ** 2 for item in value))
    if magnitude <= 1.0e-12:
        return None
    return [float(item) / magnitude for item in value]


def _vector(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def _dot(left, right) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))
