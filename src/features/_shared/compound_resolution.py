"""Second-layer compound-hole resolution.

The resolver consumes already recognised low-level cylindrical-hole evidence;
it never acts as a first-layer face scanner.  A successful resolution returns
one ontology-facing compound node and a list of component node ids to remove
from the final result.  When the coaxial, adjacency and axial checks are not
all satisfied, the low-level nodes remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from contract.partmodel import PartModel, SurfaceType

from core.models import FeatureNode
from features._shared.attributes import canonical_axis, cad_face_ids, rounded_scalar, serialize_cad_face_ids
from features._shared.measurement import min_wall_thickness
from features._shared.evidence import grouped_evidence
from features._shared.hole_topology import (
    _edge_signature,
    classify_rim_end,
    dot,
    unit_vector,
    valid_vector,
    wall_rims,
)


_AXIS_TOL = 1.0e-3
_AXIS_DISTANCE_TOL = 1.0e-3
_RADIUS_TOL = 1.0e-3
_AXIAL_TOL = 1.0e-3
_CIRCLE_COVERAGE_TOL = 0.15  # radians; permits split circle edges
_COMPOUND_COMPONENT_TYPES = {"ThroughHole", "BlindHole", "Chamfer", "Fillet"}


@dataclass
class CompoundResolution:
    nodes: list[FeatureNode] = field(default_factory=list)
    claimed: set[str] = field(default_factory=set)
    consumed_node_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


def resolve_counterbore(
    model: PartModel,
    evidence_document: dict | None,
    component_nodes: list[FeatureNode],
    claimed_face_ids: set[str] | None = None,
    *,
    start_index: int = 1,
    kernel=None,
    ladder=None,
) -> CompoundResolution:
    """Resolve a coaxial run of **exactly two** cylinders into one Counterbore.

    The run length is the whole gate.  Scanning for adjacent (smaller, larger) pairs without
    it is what produced chains of overlapping Counterbores on a multi-step bore; a run of
    three or more belongs to StepBore, and a run of one is a plain hole.
    """

    return _resolve(
        model,
        evidence_document,
        component_nodes,
        claimed_face_ids,
        kind="Counterbore",
        start_index=start_index,
        kernel=kernel,
        ladder=ladder,
    )


def resolve_countersink(
    model: PartModel,
    evidence_document: dict | None,
    component_nodes: list[FeatureNode],
    claimed_face_ids: set[str] | None = None,
    *,
    start_index: int = 1,
    kernel=None,
    ladder=None,
) -> CompoundResolution:
    """Resolve a **lone** cylinder plus the cone on its entry end into one Countersink.

    Cones are otherwise transparent (D-7 (1)/(2)): one attached to a rung of a ladder, or to
    the far end of a bore, is a Chamfer atom.  Without the lone-cylinder gate a through hole
    read from its far opening turns its exit chamfer into a countersink - the double reading
    that made 084 report two countersinks that are really chamfers.
    """

    return _resolve(
        model,
        evidence_document,
        component_nodes,
        claimed_face_ids,
        kind="Countersink",
        start_index=start_index,
        kernel=kernel,
        ladder=ladder,
    )


def _resolve(model, evidence_document, component_nodes, claimed_face_ids, *, kind: str, start_index: int,
             kernel=None, ladder=None) -> CompoundResolution:
    if ladder is None:
        # Imported here, not at module scope: hole_ladder builds on this module's geometry
        # helpers, so a top-level import would be circular.  Deriving it when a caller does not
        # supply one keeps every entry point on the same grammar instead of silently falling
        # back to the pairwise behaviour this gate replaces.
        from features._shared.hole_ladder import discover_hole_ladders

        ladder = discover_hole_ladders(model)
    claimed = set(claimed_face_ids or ())
    holes = [node for node in component_nodes if node.featureType in {"ThroughHole", "BlindHole"}]
    evidence_by_face = grouped_evidence(evidence_document or {})
    candidates: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    ambiguity_diagnostics: list[dict[str, Any]] = []

    for base_node in holes:
        base_faces = [face_id for face_id in base_node.partmodelFaceIds if _is_cylindrical(model, face_id)]
        for base_id in base_faces:
            base = model.faces[base_id]
            if not _is_hole_component_seed(base_node, base_id, evidence_by_face):
                withheld.append({"baseFaceId": base_id, "reason": "base_node_has_no_stable_hole_evidence"})
                continue
            gate = _ladder_gate(ladder, kind, base_id)
            if gate is not None:
                withheld.append({"baseFaceId": base_id, **gate})
                continue
            match_rejections: list[dict[str, Any]] = []
            if kind == "Counterbore":
                matches = _counterbore_matches(model, base_id, component_nodes, match_rejections)
            else:
                matches = _countersink_matches(model, base_id, component_nodes, match_rejections)
            matches = [match for match in matches if _ladder_admits(ladder, kind, base_id, match, withheld)]
            if not matches:
                if match_rejections:
                    withheld.extend(match_rejections)
                else:
                    withheld.append({"baseFaceId": base_id, "reason": "no_verified_coaxial_compound_section"})
                continue
            for match in matches:
                match["baseNodeId"] = base_node.nodeId
                match["baseFaceId"] = base_id
                component_face_seed = {base_id, match["sectionFaceId"]}
                if match.get("supportFaceId"):
                    component_face_seed.add(match["supportFaceId"])
                conflicts = sorted({
                    node.nodeId
                    for node in component_nodes
                    if node.featureType not in _COMPOUND_COMPONENT_TYPES
                    and set(node.partmodelFaceIds) & component_face_seed
                })
                if conflicts:
                    withheld.append({
                        "baseFaceId": base_id,
                        "sectionFaceId": match["sectionFaceId"],
                        "reason": "compound_faces_already_owned_by_non_hole_feature",
                        "conflictingNodeIds": conflicts,
                    })
                    continue
                candidates.append(match)

    selected, ambiguity_diagnostics = _deduplicate_candidates(candidates)
    for ambiguity in ambiguity_diagnostics:
        for candidate_key in ambiguity["candidates"]:
            withheld.append({
                "kind": ambiguity["kind"],
                "baseFaceId": candidate_key["baseFaceId"],
                "sectionFaceId": candidate_key["sectionFaceId"],
                "reason": "ambiguous_compound_candidates",
                "candidateSet": ambiguity["candidates"],
            })
    nodes: list[FeatureNode] = []
    consumed: list[str] = []
    for index, candidate in enumerate(selected, start=start_index):
        component_node_ids = sorted({
            node.nodeId
            for node in component_nodes
            if node.featureType in _COMPOUND_COMPONENT_TYPES
            and set(node.partmodelFaceIds)
            & {candidate["baseFaceId"], candidate["sectionFaceId"]}
        })
        # A BlindHole component can carry its terminal/floor face in addition
        # to the cylindrical wall.  A successful compound replaces the whole
        # low-level representation, so retain every face owned by the
        # consumed components on the compound node; otherwise those faces
        # would remain claimed but disappear from the final ownership map and
        # could not be registered by the residual surface pass.
        component_face_ids = [candidate["baseFaceId"], candidate["sectionFaceId"]]
        for component in component_nodes:
            if component.nodeId in component_node_ids:
                component_face_ids.extend(component.partmodelFaceIds)
        if candidate.get("supportFaceId"):
            component_face_ids.append(candidate["supportFaceId"])
        component_face_ids = list(dict.fromkeys(component_face_ids))
        node = _build_node(model, candidate, component_face_ids, component_node_ids, index, kernel)
        nodes.append(node)
        consumed.extend(component_node_ids)
        claimed.update(component_face_ids)

    diagnostics = {
        "featureModule": kind,
        "method": "two_layer_compound_hole_resolution_v1",
        "inputLowLevelNodeCount": len(holes),
        "candidateCount": len(candidates),
        "acceptedCount": len(nodes),
        "consumedNodeIds": sorted(set(consumed)),
        "withheldCandidates": withheld,
        "ambiguousCandidates": ambiguity_diagnostics,
        "checks": [
            "primary cylindrical hole evidence",
            "coaxial axis line",
            "adjacency or common support face",
            "shoulder adjacency at the ordered axial coordinate",
            "ordered serial axial section boundaries with no interior overlap",
            "complete two-rim cylinder or two-circle cone topology",
        ],
    }
    return CompoundResolution(nodes, claimed, sorted(set(consumed)), diagnostics)


def _ladder_gate(ladder, kind: str, base_id: str) -> dict[str, Any] | None:
    """Reject a base face whose coaxial run is the wrong shape for *kind*, with the reason.

    Returning the reason rather than dropping the candidate silently is the point: a
    "no compound here" decision has to be reviewable from ``result.json`` alone.
    """
    run = ladder.run_for_face(base_id)
    if run is None:
        return {"reason": "base_face_is_not_part_of_any_coaxial_cylinder_run"}
    if not run.valid:
        return {"reason": "coaxial_run_has_no_single_entry", "runReason": run.reason,
                "runFaceIds": [segment.faceId for segment in run.segments]}
    if kind == "Counterbore" and run.size != 2:
        return {"reason": "counterbore_requires_a_run_of_exactly_two_cylinders", "runSize": run.size,
                "runFaceIds": [segment.faceId for segment in run.segments]}
    if kind == "Countersink" and run.size != 1:
        return {"reason": "countersink_base_is_a_ladder_segment_not_a_lone_cylinder", "runSize": run.size,
                "runFaceIds": [segment.faceId for segment in run.segments]}
    return None


def _ladder_admits(ladder, kind: str, base_id: str, match: dict[str, Any], withheld: list[dict[str, Any]]) -> bool:
    """Second half of the gate: the matched *section* must be the one the run names."""
    run = ladder.run_for_face(base_id)
    section_id = match["sectionFaceId"]
    if kind == "Counterbore":
        members = {segment.faceId for segment in run.segments}
        if members != {base_id, section_id}:
            withheld.append({"baseFaceId": base_id, "sectionFaceId": section_id,
                             "reason": "counterbore_section_is_not_the_other_member_of_the_run",
                             "runFaceIds": sorted(members)})
            return False
        return True
    cone = ladder.countersink_cone(base_id)
    if cone is None:
        withheld.append({"baseFaceId": base_id, "sectionFaceId": section_id,
                         "reason": "no_cone_sits_on_the_entry_end_of_this_lone_cylinder"})
        return False
    if cone.faceId != section_id:
        withheld.append({"baseFaceId": base_id, "sectionFaceId": section_id,
                         "reason": "countersink_cone_is_not_at_the_large_diameter_entry_end",
                         "entryConeFaceId": cone.faceId})
        return False
    return True


def _is_hole_component_seed(node: FeatureNode, face_id: str, evidence_by_face: dict[str, list[dict]]) -> bool:
    # Existing ThroughHole/BlindHole nodes are already topology validated.  A
    # synthetic caller may provide a node without diagnostics; in that case
    # retain the explicit face-level AAGNet seed as the fallback guard.
    if node.featureType in {"ThroughHole", "BlindHole"} and node.partmodelFaceIds:
        topology = node.diagnostics.get("topology") if isinstance(node.diagnostics, dict) else None
        if isinstance(topology, dict):
            return True
    return any(item.get("top_label") in {"through_hole", "blind_hole"} for item in evidence_by_face.get(face_id, []))


def _counterbore_matches(
    model: PartModel,
    base_id: str,
    component_nodes: list[FeatureNode],
    rejection_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    base = model.faces[base_id]
    base_axis = unit_vector(base.parameters.get("axis_direction"))
    base_origin = base.parameters.get("axis_location")
    base_radius = _positive(base.parameters.get("radius"))
    base_interval = _face_interval(model, base, base_axis)
    if base_axis is None or not valid_vector(base_origin) or base_radius is None or base_interval is None:
        return []
    results: list[dict[str, Any]] = []
    for section_id, section in model.faces.items():
        if section_id == base_id or section.surface_type != SurfaceType.CYLINDER:
            continue
        radius = _positive(section.parameters.get("radius"))
        axis = unit_vector(section.parameters.get("axis_direction"))
        origin = section.parameters.get("axis_location")
        interval = _face_interval(model, section, base_axis)
        if radius is None or radius <= base_radius + _RADIUS_TOL or axis is None or not valid_vector(origin) or interval is None:
            continue
        if _axis_angle(base_axis, axis) > _AXIS_TOL or _line_distance(base_origin, origin, base_axis) > _AXIS_DISTANCE_TOL:
            continue
        boundary = _ordered_section_boundaries(base_interval, interval)
        if boundary is None:
            continue
        outer, shoulder, terminal, inward = boundary
        if bool(section.is_reversed) != bool(base.is_reversed):
            _record_rejection(rejection_log, base_id, section_id, "compound_section_orientation_mismatch")
            continue
        topology = _complete_cylindrical_section(model, section_id, base_axis)
        if not topology["valid"]:
            _record_rejection(
                rejection_log,
                base_id,
                section_id,
                topology.get("reason", "counterbore_section_topology_unavailable"),
            )
            continue
        shoulder_proof = _validate_counterbore_shoulder(
            model,
            base_id,
            section_id,
            base_axis,
            shoulder,
            base_radius,
            radius,
        )
        if shoulder_proof is None:
            _record_rejection(rejection_log, base_id, section_id, "counterbore_internal_annular_shoulder_unproven")
            continue
        support = shoulder_proof["supportFaceId"]
        depth = abs(terminal - outer)
        counterbore_depth = abs(shoulder - outer)
        if depth <= _AXIAL_TOL or counterbore_depth <= _AXIAL_TOL or counterbore_depth >= depth - _AXIAL_TOL:
            continue
        centre = _axis_point_at_coordinate(base_origin, base_axis, terminal)
        results.append({
            "kind": "Counterbore",
            "sectionFaceId": section_id,
            "baseRadius": base_radius,
            "sectionRadius": radius,
            "baseInterval": base_interval,
            "sectionInterval": interval,
            "outerCoordinate": outer,
            "shoulderCoordinate": shoulder,
            "terminalCoordinate": terminal,
            "machiningDirection": [value * inward for value in base_axis],
            "baseAxis": list(base_axis),
            "baseOrigin": list(base_origin),
            "depth": depth,
            "counterboreDepth": counterbore_depth,
            "position": centre,
            "supportFaceId": support,
            "adjacencyMethod": "common_support_face",
            "shoulderProof": shoulder_proof,
            "sectionTopology": topology,
        })
    return results


def _countersink_matches(
    model: PartModel,
    base_id: str,
    component_nodes: list[FeatureNode],
    rejection_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    base = model.faces[base_id]
    base_axis = unit_vector(base.parameters.get("axis_direction"))
    base_origin = base.parameters.get("axis_location")
    base_radius = _positive(base.parameters.get("radius"))
    base_interval = _face_interval(model, base, base_axis)
    if base_axis is None or not valid_vector(base_origin) or base_radius is None or base_interval is None:
        return []
    results: list[dict[str, Any]] = []
    for section_id, section in model.faces.items():
        if section.surface_type != SurfaceType.CONE:
            continue
        axis = unit_vector(section.parameters.get("axis_direction"))
        apex = section.parameters.get("apex")
        if axis is None or not valid_vector(apex) or _axis_angle(base_axis, axis) > _AXIS_TOL:
            continue
        circles = _cone_circles(model, section_id, base_axis)
        if len(circles) != 2:
            if len(circles) > 2:
                _record_rejection(rejection_log, base_id, section_id, "countersink_requires_exactly_two_circular_boundaries")
            continue
        if any(not item.get("complete") for item in circles):
            _record_rejection(rejection_log, base_id, section_id, "countersink_has_incomplete_circular_boundary")
            continue
        small = min(circles, key=lambda item: item["radius"])
        large = max(circles, key=lambda item: item["radius"])
        if abs(float(small["radius"]) - base_radius) > max(_RADIUS_TOL, base_radius * 1.0e-3):
            continue
        if float(large["radius"]) <= base_radius + _RADIUS_TOL:
            continue
        if _line_distance(base_origin, apex, base_axis) > _AXIS_DISTANCE_TOL:
            continue
        if bool(section.is_reversed) != bool(base.is_reversed):
            _record_rejection(rejection_log, base_id, section_id, "countersink_section_orientation_mismatch")
            continue
        section_positions = [float(small["axisPosition"]), float(large["axisPosition"])]
        section_interval = (min(section_positions), max(section_positions))
        if section_interval[1] - section_interval[0] <= _AXIAL_TOL:
            _record_rejection(rejection_log, base_id, section_id, "countersink_circular_boundaries_not_axially_distinct")
            continue
        boundary = _ordered_section_boundaries(base_interval, section_interval)
        if boundary is None:
            continue
        outer, shoulder, terminal, inward = boundary
        # The small cone rim is the shoulder; a large circle somewhere else
        # on the same cone is not enough to prove a countersink junction.
        if abs(float(small["axisPosition"]) - shoulder) > _AXIAL_TOL:
            continue
        if not _direct_rim_junction(model, base_id, section_id, small, base_axis, shoulder):
            _record_rejection(rejection_log, base_id, section_id, "countersink_small_rim_junction_unproven")
            continue
        depth = abs(terminal - outer)
        if depth <= _AXIAL_TOL:
            continue
        semi = section.parameters.get("semi_angle")
        angle = 2.0 * math.degrees(abs(float(semi))) if isinstance(semi, (int, float)) and math.isfinite(float(semi)) else None
        if angle is None or not 0.0 < angle < 180.0:
            continue
        major_centre = _axis_point_at_coordinate(base_origin, base_axis, terminal)
        results.append({
            "kind": "Countersink",
            "sectionFaceId": section_id,
            "baseRadius": base_radius,
            "sectionRadius": float(large["radius"]),
            "baseInterval": base_interval,
            "sectionInterval": section_interval,
            "outerCoordinate": outer,
            "shoulderCoordinate": shoulder,
            "terminalCoordinate": terminal,
            "machiningDirection": [value * inward for value in base_axis],
            "baseAxis": list(base_axis),
            "baseOrigin": list(base_origin),
            "depth": depth,
            "countersinkDepth": abs(float(large["axisPosition"]) - float(small["axisPosition"])),
            "position": major_centre,
            "smallBoundary": small,
            "largeBoundary": large,
            "adjacencyMethod": "direct_small_rim_junction",
            "sectionTopology": {
                "valid": True,
                "kind": "cone_two_complete_circular_boundaries",
                "boundaries": [
                    {
                        "radius": item["radius"],
                        "axisPosition": item["axisPosition"],
                        "edgeIds": item.get("edgeIds", []),
                        "angularCoverage": item.get("angularCoverage"),
                    }
                    for item in circles
                ],
            },
        })
    return results


def _ordered_section_boundaries(base_interval: tuple[float, float], section_interval: tuple[float, float]) -> tuple[float, float, float, float] | None:
    """Return ``(outer, shoulder, terminal, inward_sign)`` for a serial join.

    A compound treatment is a one-dimensional serial arrangement: the
    enlarged/conical section lies wholly outside the primary wall and touches
    exactly one primary endpoint at the shoulder.  Merely finding the nearest
    pair of endpoints is unsafe because an embedded section (for example
    ``[28, 48]`` inside ``[0, 48]``) can otherwise look like a valid shoulder.
    """
    try:
        b0, b1 = sorted((float(base_interval[0]), float(base_interval[1])))
        s0, s1 = sorted((float(section_interval[0]), float(section_interval[1])))
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(value) for value in (b0, b1, s0, s1)):
        return None
    if b1 - b0 <= _AXIAL_TOL or s1 - s0 <= _AXIAL_TOL:
        return None
    scale = max(abs(b1 - b0), abs(s1 - s0), 1.0)
    tolerance = max(_AXIAL_TOL, 1.0e-3 * scale)

    # The only allowed intersection is a tolerance-sized endpoint contact.
    overlap = max(0.0, min(b1, s1) - max(b0, s0))
    if overlap > tolerance:
        return None
    below = s1 <= b0 + tolerance and abs(s1 - b0) <= tolerance
    above = s0 >= b1 - tolerance and abs(s0 - b1) <= tolerance
    if not (below or above):
        return None

    if below:
        outer, shoulder = s0, s1
        terminal = b1
    else:
        outer, shoulder = s1, s0
        terminal = b0
    inward_sign = 1.0 if terminal > outer else -1.0
    return outer, shoulder, terminal, inward_sign


def _build_node(model: PartModel, candidate: dict[str, Any], component_face_ids: list[str], component_node_ids: list[str], index: int, kernel=None) -> FeatureNode:
    kind = candidate["kind"]
    fields = (
        "cadFaceIds", "depth", "derived_ratio_depth_to_diameter", "diameter",
        "posX", "posY", "posZ", "feature_axis", "machining_direction",
        "counterbore_diameter", "counterbore_depth", "derived_min_wall_thickness",
    ) if kind == "Counterbore" else (
        "cadFaceIds", "depth", "derived_ratio_depth_to_diameter", "diameter",
        "posX", "posY", "posZ", "feature_axis", "machining_direction",
        "countersink_diameter", "countersink_depth", "derived_countersink_angle",
        "derived_min_wall_thickness",
    )
    position = candidate.get("position")
    attributes: dict[str, Any] = {name: None for name in fields}
    attributes["cadFaceIds"] = serialize_cad_face_ids(model, component_face_ids)
    attributes["depth"] = rounded_scalar(candidate.get("depth"))
    attributes["diameter"] = rounded_scalar(2.0 * candidate["baseRadius"])
    attributes["derived_ratio_depth_to_diameter"] = _ratio(attributes["depth"], attributes["diameter"])
    attributes["feature_axis"] = canonical_axis(candidate.get("baseAxis"))
    attributes["machining_direction"] = [round(float(value), 6) + 0.0 for value in candidate["machiningDirection"]]
    wall_diagnostics: dict[str, Any] = {}
    if kind == "Counterbore":
        attributes["counterbore_diameter"] = rounded_scalar(2.0 * candidate["sectionRadius"])
        attributes["counterbore_depth"] = rounded_scalar(candidate.get("counterboreDepth"))
        attributes["derived_min_wall_thickness"] = _compound_wall_thickness(
            model,
            kernel,
            component_face_ids,
            [candidate["baseFaceId"], candidate["sectionFaceId"]],
            diagnostics=wall_diagnostics,
        )
    else:
        attributes["countersink_diameter"] = rounded_scalar(2.0 * candidate["sectionRadius"])
        attributes["countersink_depth"] = rounded_scalar(candidate.get("countersinkDepth"))
        attributes["derived_countersink_angle"] = _derived_countersink_angle(
            attributes["countersink_diameter"], attributes["diameter"], attributes["countersink_depth"]
        )
        attributes["derived_min_wall_thickness"] = _compound_wall_thickness(
            model,
            kernel,
            component_face_ids,
            [candidate["baseFaceId"], candidate["sectionFaceId"]],
            diagnostics=wall_diagnostics,
        )
    reasons = {
        "cadFaceIds": "no source CAD face ids were recoverable from the compound faces",
        "depth": "the ordered compound axial extent was unavailable",
        "derived_ratio_depth_to_diameter": "positive compound depth and base diameter were not both available",
        "diameter": "the primary cylindrical wall has no positive radius",
        "posX": "the outer compound opening centre was unavailable",
        "posY": "the outer compound opening centre was unavailable",
        "posZ": "the outer compound opening centre was unavailable",
        "machining_direction": "the compound opening-to-material axis could not be oriented",
        "feature_axis": "the compound base axis could not be oriented",
        "counterbore_diameter": "the enlarged cylindrical section has no positive radius",
        "counterbore_depth": "the enlarged section shoulder could not be ordered from the outer opening",
        "countersink_diameter": "the countersink major boundary has no positive radius",
        "countersink_depth": "the conical section boundary positions were unavailable",
        "derived_countersink_angle": "the conical section diameters and depth did not establish a valid angle",
        "derived_min_wall_thickness": "no non-self material surface was reached from sampled compound wall normals",
    }
    for index_name, coordinate in enumerate(("posX", "posY", "posZ")):
        attributes[coordinate] = rounded_scalar(position[index_name]) if valid_vector(position) else None
    return FeatureNode(
        nodeId=f"{kind}_{index:03d}",
        featureType=kind,
        partmodelFaceIds=list(component_face_ids),
        attributes=attributes,
        missingAttributes={name: reasons[name] for name in fields if attributes.get(name) is None},
        diagnostics={
            "method": "two_layer_compound_hole_resolution_v1",
            "baseFaceId": candidate["baseFaceId"],
            "sectionFaceId": candidate["sectionFaceId"],
            "componentNodeIds": component_node_ids,
            "componentFaceIds": component_face_ids,
            "supportFaceId": candidate.get("supportFaceId"),
            "adjacencyMethod": candidate.get("adjacencyMethod"),
            "sectionTopology": candidate.get("sectionTopology"),
            "shoulderProof": candidate.get("shoulderProof"),
            "baseInterval": list(candidate.get("baseInterval", ())),
            "sectionInterval": list(candidate.get("sectionInterval", ())),
            "outerCoordinate": candidate.get("outerCoordinate"),
            "shoulderCoordinate": candidate.get("shoulderCoordinate"),
            "terminalCoordinate": candidate.get("terminalCoordinate"),
            "countersinkBoundaries": {
                "small": candidate.get("smallBoundary"),
                "large": candidate.get("largeBoundary"),
            } if kind == "Countersink" else None,
            "wallThickness": wall_diagnostics,
        },
    )


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove exact duplicates and withhold unresolved overlap ambiguity.

    Ordering is only an output stability rule.  A candidate conflict graph is
    built from shared base or section faces; every connected component with
    more than one distinct candidate is withheld rather than guessed.
    """
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in sorted(candidates, key=lambda item: (item["baseFaceId"], item["sectionFaceId"], item["kind"])):
        key = (candidate["kind"], candidate["baseFaceId"], candidate["sectionFaceId"])
        unique.setdefault(key, candidate)
    keys = list(unique)
    parent = list(range(len(keys)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, left in enumerate(keys):
        for other, right in enumerate(keys[index + 1:], start=index + 1):
            if left[0] == right[0] and (left[1] == right[1] or left[2] == right[2]):
                union(index, other)

    groups: dict[int, list[int]] = {}
    for index in range(len(keys)):
        groups.setdefault(find(index), []).append(index)
    ambiguous: list[dict[str, Any]] = []
    withheld_keys: set[tuple[str, str, str]] = set()
    for indexes in groups.values():
        if len(indexes) <= 1:
            continue
        group_keys = [keys[index] for index in indexes]
        kind = group_keys[0][0]
        candidate_set = [
            {"kind": item[0], "baseFaceId": item[1], "sectionFaceId": item[2]}
            for item in sorted(group_keys)
        ]
        ambiguous.append({"kind": kind, "candidates": candidate_set, "reason": "competing_candidates_share_base_or_section"})
        withheld_keys.update(group_keys)

    selected = [unique[key] for key in keys if key not in withheld_keys]
    return selected, ambiguous


def _is_cylindrical(model: PartModel, face_id: str) -> bool:
    return face_id in model.faces and model.faces[face_id].surface_type == SurfaceType.CYLINDER


def _positive(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0 else None


def _axis_angle(left: list[float], right: list[float]) -> float:
    value = max(-1.0, min(1.0, abs(dot(left, right))))
    return math.acos(value)


def _line_distance(left: object, right: object, axis: list[float]) -> float:
    if not valid_vector(left) or not valid_vector(right):
        return math.inf
    delta = [float(right[index]) - float(left[index]) for index in range(3)]
    projected = dot(delta, axis)
    radial = [delta[index] - projected * axis[index] for index in range(3)]
    return math.sqrt(sum(value * value for value in radial))


def _face_interval(model: PartModel, face, axis: list[float] | None) -> tuple[float, float] | None:
    if axis is None:
        return None
    # The normalizer persists true axial extrema from the B-Rep boundary.
    # They are authoritative, especially for a tilted axis where projecting an
    # axis-aligned bounding box can substantially overestimate the interval.
    minimum = face.parameters.get("axial_min")
    maximum = face.parameters.get("axial_max")
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
        minimum, maximum = float(minimum), float(maximum)
        if math.isfinite(minimum) and math.isfinite(maximum) and maximum - minimum > _AXIAL_TOL:
            stored_axis = unit_vector(face.parameters.get("axis_direction"))
            orientation = dot(stored_axis, axis) if stored_axis is not None else 1.0
            if abs(abs(orientation) - 1.0) <= _AXIS_TOL:
                converted = (minimum * (1.0 if orientation >= 0.0 else -1.0), maximum * (1.0 if orientation >= 0.0 else -1.0))
                return min(converted), max(converted)
            # A persisted axial range with an unrelated axis is not reliable
            # for this query; continue to the rim/AABB fallbacks.

    # For a cylindrical wall with complete rims, use their actual projected
    # positions before falling back to the coarse AABB projection.  This keeps
    # seam/split geometry accurate without inventing a second rim detector.
    if getattr(face, "surface_type", None) == SurfaceType.CYLINDER:
        rims = wall_rims(model, face.id)
        positions = [float(rim.axis_position) for rim in rims if math.isfinite(float(rim.axis_position))]
        if len(positions) >= 2 and max(positions) - min(positions) > _AXIAL_TOL:
            return min(positions), max(positions)

    bbox = face.bbox
    if isinstance(bbox, dict) and valid_vector(bbox.get("min")) and valid_vector(bbox.get("max")):
        values = []
        for mask in range(8):
            point = [bbox["max"][index] if mask & (1 << index) else bbox["min"][index] for index in range(3)]
            values.append(dot(point, axis))
        return min(values), max(values)
    return None


def _complete_cylindrical_section(model: PartModel, face_id: str, axis: list[float]) -> dict[str, Any]:
    """Require exactly two complete closed wall rims on a cylinder section.

    ``wall_rims`` performs the existing winding, radius and seam/split-aware
    graph validation.  Requiring its two-rim result prevents a pair of
    arbitrary ``CIRCLE`` edges or a local half-cylinder wire from becoming a
    Counterbore section.
    """
    rims = wall_rims(model, face_id)
    if len(rims) != 2:
        return {
            "valid": False,
            "kind": "cylinder_two_complete_wall_rims",
            "reason": "section_does_not_have_exactly_two_complete_wall_rims",
            "rimCount": len(rims),
        }
    positions = sorted(float(rim.axis_position) for rim in rims)
    if positions[1] - positions[0] <= _AXIAL_TOL:
        return {
            "valid": False,
            "kind": "cylinder_two_complete_wall_rims",
            "reason": "section_rims_are_not_axially_distinct",
            "rimCount": len(rims),
        }
    section_edges = {
        edge_id
        for loop_id in model.faces[face_id].loop_ids
        if loop_id in model.loops
        for edge_id in model.loops[loop_id].edge_ids
    }
    section_edge_signatures = {
        _edge_signature(model.edges[edge_id])
        for edge_id in section_edges
        if edge_id in model.edges
    }
    unmatched_rim_edges = [
        [
            edge_id
            for edge_id in rim.edge_ids
            if edge_id not in model.edges or _edge_signature(model.edges[edge_id]) not in section_edge_signatures
        ]
        for rim in rims
    ]
    if any(unmatched_rim_edges):
        return {
            "valid": False,
            "kind": "cylinder_two_complete_wall_rims",
            "reason": "complete_rims_are_not_on_section_boundary",
            "rimCount": len(rims),
            "unmatchedRimEdgeIds": unmatched_rim_edges,
        }
    rim_ends = [classify_rim_end(model, face_id, rim) for rim in rims]
    end_kinds = [end.kind for end in rim_ends]
    if sorted(end_kinds) != ["capped", "open"]:
        return {
            "valid": False,
            "kind": "cylinder_two_complete_wall_rims",
            "reason": "section_rims_do_not_have_one_open_and_one_capped_end",
            "rimCount": len(rims),
            "rimEnds": [
                {"kind": end.kind, "axisPosition": round(float(end.axis_position), 6), "faceIds": list(end.face_ids)}
                for end in rim_ends
            ],
        }
    return {
        "valid": True,
        "kind": "cylinder_two_complete_wall_rims",
        "rimCount": len(rims),
        "rimEnds": [
            {"kind": end.kind, "axisPosition": round(float(end.axis_position), 6), "faceIds": list(end.face_ids)}
            for end in rim_ends
        ],
        "rims": [
            {
                "axisPosition": round(float(rim.axis_position), 6),
                "edgeIds": list(rim.edge_ids),
                "sampleCount": len(rim.sample_points),
            }
            for rim in rims
        ],
    }


def _edge_axis_range(model: PartModel, edge_id: str, axis: list[float]) -> tuple[float, float] | None:
    edge = model.edges.get(edge_id)
    if edge is None:
        return None
    points = [point for point in (edge.sample_points or []) if valid_vector(point)]
    if not points:
        points = [
            list(model.vertices[vertex_id].coordinates)
            for vertex_id in edge.vertex_ids
            if vertex_id in model.vertices
        ]
    centre = edge.parameters.get("center")
    if valid_vector(centre):
        points.append(list(centre))
    if not points:
        return None
    values = [dot(point, axis) for point in points]
    return min(values), max(values)


def _edge_at_coordinate(model: PartModel, edge_id: str, axis: list[float], coordinate: float) -> bool:
    span = _edge_axis_range(model, edge_id, axis)
    if span is None:
        return False
    tolerance = max(_AXIAL_TOL, 1.0e-3 * max(abs(coordinate), 1.0))
    return span[1] - span[0] <= tolerance and abs((span[0] + span[1]) * 0.5 - coordinate) <= tolerance


def _validate_counterbore_shoulder(
    model: PartModel,
    base_id: str,
    section_id: str,
    axis: list[float],
    shoulder: float,
    base_radius: float,
    section_radius: float,
) -> dict[str, Any] | None:
    """Prove that the common face is the internal annular shoulder.

    A direct cylinder-to-cylinder adjacency is not enough: the real shoulder
    is a planar annulus with one base rim and one enlarged-wall rim.  The
    opposite concavity signs are the material-side signal observed on the
    accepted corpus (base convex, enlarged wall concave).
    """
    left = model.faces.get(base_id)
    right = model.faces.get(section_id)
    if left is None or right is None:
        return None
    left_neighbours = {item.adjacent_face_id for item in left.adjacencies}
    right_neighbours = {item.adjacent_face_id for item in right.adjacencies}
    for support_id in sorted(left_neighbours & right_neighbours):
        support = model.faces.get(support_id)
        if support is None or support.surface_type != SurfaceType.PLANE:
            continue
        base_adj = [item for item in left.adjacencies if item.adjacent_face_id == support_id]
        section_adj = [item for item in right.adjacencies if item.adjacent_face_id == support_id]
        base_edges = [item.shared_edge_id for item in base_adj if _edge_at_coordinate(model, item.shared_edge_id, axis, shoulder)]
        section_edges = [item.shared_edge_id for item in section_adj if _edge_at_coordinate(model, item.shared_edge_id, axis, shoulder)]
        if not base_edges or not section_edges:
            continue
        base_signs = {item.connection_type.value for item in base_adj if item.shared_edge_id in base_edges}
        section_signs = {item.connection_type.value for item in section_adj if item.shared_edge_id in section_edges}
        if "CONVEX" not in base_signs or "CONCAVE" not in section_signs:
            continue
        loop_info: dict[str, dict[str, Any]] = {}
        for loop_id in support.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None:
                continue
            loop_info[loop_id] = {"isOuter": bool(loop.is_outer), "edgeIds": set(loop.edge_ids)}
        if not loop_info:
            continue
        base_loop_ids = [loop_id for loop_id, info in loop_info.items() if set(base_edges) & info["edgeIds"]]
        section_loop_ids = [loop_id for loop_id, info in loop_info.items() if set(section_edges) & info["edgeIds"]]
        pair: tuple[str, str] | None = None
        for base_loop_id in base_loop_ids:
            for section_loop_id in section_loop_ids:
                if base_loop_id == section_loop_id:
                    continue
                if loop_info[base_loop_id]["isOuter"] == loop_info[section_loop_id]["isOuter"]:
                    continue
                pair = (base_loop_id, section_loop_id)
                break
            if pair is not None:
                break
        if pair is None:
            continue
        base_edge_radii = [_edge_radius(model, edge_id) for edge_id in base_edges]
        section_edge_radii = [_edge_radius(model, edge_id) for edge_id in section_edges]
        if not any(_close_radius(value, base_radius) for value in base_edge_radii):
            continue
        if not any(_close_radius(value, section_radius) for value in section_edge_radii):
            continue
        return {
            "valid": True,
            "supportFaceId": support_id,
            "baseEdgeIds": sorted(set(base_edges)),
            "sectionEdgeIds": sorted(set(section_edges)),
            "baseLoopId": pair[0],
            "sectionLoopId": pair[1],
            "baseConnection": "CONVEX",
            "sectionConnection": "CONCAVE",
            "shoulderCoordinate": round(float(shoulder), 6),
        }
    return None


def _edge_radius(model: PartModel, edge_id: str) -> float | None:
    edge = model.edges.get(edge_id)
    value = edge.parameters.get("radius") if edge is not None else None
    return _positive(value)


def _close_radius(value: float | None, expected: float) -> bool:
    return value is not None and abs(value - expected) <= max(_RADIUS_TOL, abs(expected) * 1.0e-3)


def _direct_rim_junction(
    model: PartModel,
    base_id: str,
    section_id: str,
    small_boundary: dict[str, Any],
    axis: list[float],
    shoulder: float,
) -> bool:
    small_edges = set(small_boundary.get("edgeIds", ()))
    if not small_edges:
        return False
    for item in model.faces[base_id].adjacencies:
        if item.adjacent_face_id == section_id and item.shared_edge_id in small_edges and _edge_at_coordinate(model, item.shared_edge_id, axis, shoulder):
            return True
    for item in model.faces[section_id].adjacencies:
        if item.adjacent_face_id == base_id and item.shared_edge_id in small_edges and _edge_at_coordinate(model, item.shared_edge_id, axis, shoulder):
            return True
    return False


def _record_rejection(
    rejection_log: list[dict[str, Any]] | None,
    base_id: str,
    section_id: str,
    reason: str,
) -> None:
    if rejection_log is None:
        return
    rejection_log.append({"baseFaceId": base_id, "sectionFaceId": section_id, "reason": reason})


def _axis_point_at_coordinate(origin: object, axis: list[float], coordinate: float) -> list[float] | None:
    """Project a signed axial coordinate onto the base-hole axis line."""
    if not valid_vector(origin) or not valid_vector(axis):
        return None
    try:
        values = [float(value) for value in origin]
        unit = unit_vector(axis)
        if unit is None or not math.isfinite(float(coordinate)):
            return None
        current = dot(values, unit)
        return [round(values[index] + (float(coordinate) - current) * unit[index], 6) + 0.0 for index in range(3)]
    except (TypeError, ValueError):
        return None


def _derived_countersink_angle(countersink_diameter: float | None, diameter: float | None, depth: float | None) -> float | None:
    if countersink_diameter is None or diameter is None or depth is None or depth <= 0.0:
        return None
    delta = float(countersink_diameter) - float(diameter)
    if delta <= 0.0:
        return None
    value = 2.0 * math.degrees(math.atan(delta / (2.0 * float(depth))))
    return rounded_scalar(value) if 0.0 < value < 180.0 else None


def _compound_wall_thickness(
    model,
    kernel,
    component_face_ids: list[str],
    wall_face_ids: list[str],
    *,
    diagnostics: dict[str, Any] | None = None,
) -> float | None:
    return min_wall_thickness(
        kernel,
        cad_face_ids(model, component_face_ids),
        cad_face_ids(model, wall_face_ids),
        material_side=True,
        diagnostics=diagnostics,
    )


def _cone_circles(model: PartModel, face_id: str, axis: list[float]) -> list[dict[str, Any]]:
    face = model.faces[face_id]
    grouped: dict[tuple, dict[str, Any]] = {}
    for loop_id in face.loop_ids:
        loop = model.loops.get(loop_id)
        if loop is None:
            continue
        for edge_id in loop.edge_ids:
            edge = model.edges.get(edge_id)
            if edge is None or edge.curve_type.value != "CIRCLE":
                continue
            radius, centre = edge.parameters.get("radius"), edge.parameters.get("center")
            if not isinstance(radius, (int, float)) or not valid_vector(centre):
                continue
            key = (round(float(radius), 5), tuple(round(float(value), 5) for value in centre))
            item = grouped.setdefault(key, {
                "edgeIds": [],
                "radius": float(radius),
                "center": [float(value) for value in centre],
                "axisPosition": dot(centre, axis),
                "angularCoverage": 0.0,
            })
            if edge_id in item["edgeIds"]:
                continue
            item["edgeIds"].append(edge_id)
            length = float(edge.length) if isinstance(edge.length, (int, float)) else 0.0
            if length > 0.0 and float(radius) > 0.0:
                item["angularCoverage"] += length / float(radius)
            else:
                item["angularCoverage"] += _sample_angular_coverage(edge.sample_points, centre, axis)
    result = []
    for item in grouped.values():
        item["complete"] = float(item["angularCoverage"]) >= 2.0 * math.pi - _CIRCLE_COVERAGE_TOL
        result.append(item)
    return sorted(result, key=lambda item: (item["radius"], item["axisPosition"], item["edgeIds"]))


def _sample_angular_coverage(samples: object, centre: object, axis: list[float]) -> float:
    """Conservative angular coverage fallback for edges without length."""
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


def _perpendicular_basis(axis: list[float]) -> tuple[list[float], list[float]]:
    reference = [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0]
    cross = [
        reference[1] * axis[2] - reference[2] * axis[1],
        reference[2] * axis[0] - reference[0] * axis[2],
        reference[0] * axis[1] - reference[1] * axis[0],
    ]
    magnitude = math.sqrt(sum(value * value for value in cross))
    if magnitude <= 1.0e-12:
        return [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    u = [value / magnitude for value in cross]
    v = [
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
    ]
    return u, v


def _ratio(numerator: object, denominator: object) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)) or float(denominator) <= 0.0:
        return None
    return rounded_scalar(float(numerator) / float(denominator))
