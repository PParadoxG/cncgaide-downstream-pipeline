"""The v5.1 dialect's conservative, relation-specific geometry witnesses.

Everything here reads v5.1 leaf classes, v5.1 attribute names and the
diagnostics the v5.1 feature recognizers publish — terminal resolution starts
from the evidence those recognizers already emitted (rims, endpoint paths,
rails and side-wall bands), which is exactly why another ontology's rules
cannot reuse it.

The parts that name no class — vector arithmetic, the tolerance policy, face
ownership, shared-edge enumeration, the edge constructor — are machinery and
live in ``relations._shared``; a v5.1 rule imports both through this module.
"""

from __future__ import annotations

from collections import defaultdict

from contract.kernel import DISTANCE_MEASURED
from core.models import FeatureNode
from features._shared.attributes import cad_face_ids
from features._shared.hole_topology import RimGroup, classify_rim_end, wall_rims

from relations._shared import (  # noqa: F401
    AXIS_LINE_TOL_MM,
    COAXIAL_ANGLE_TOL_DEG,
    PARALLEL_ANGLE_TOL_DEG,
    SPACING_BASIS,
    _cached,
    _edge_points,
    _face_normal,
    _face_owner,
    _face_resolution_record,
    _face_surface_type,
    _feature_edge,
    _report,
    _set_report,
    acute_angle_deg,
    attributes,
    coaxial_axis_distance,
    direct_face_pairs,
    dot,
    feature_type,
    is_coaxial_pair,
    mean_axis_direction,
    norm,
    parallel,
    parallel_axis_spacing,
    point_value,
    polyline_distance,
    vector,
)

HOLE_TYPES = frozenset({"ThroughHole", "BlindHole", "Counterbore", "Countersink", "StepBore"})
SLOT_TYPES = frozenset({"RectangularSlot", "TriangularSlot", "CircularSlot", "GeneralSlot"})
POCKET_TYPES = frozenset({"RectangularPocket", "GeneralPocket"})
TRANSITION_TYPES = frozenset({"Chamfer", "Fillet"})
LOCATED_SOURCE_TYPES = frozenset({
    *HOLE_TYPES,
    *POCKET_TYPES,
    "CircularBoss",
    "RectangularBoss",
    "GeneralBoss",
    "ThroughCut",
    *SLOT_TYPES,
})


def axis(node: FeatureNode) -> list[float] | None:
    return vector(attributes(node).get("feature_axis"))


def _transition_support_faces(context, transition_node_id: str) -> list[str]:
    node = context.node_by_id.get(transition_node_id)
    if node is None:
        return []
    diagnostics = getattr(node, "diagnostics", {}) or {}
    measurement = diagnostics.get("measurement") or {}
    topology = measurement.get("topologyProof") or {}
    values = topology.get("supportFaceIds") or []
    return [str(value) for value in values if str(value) in context.model.faces]


# A transition bypass walks from a terminal outward across Chamfer/Fillet
# faces.  Real stacks are one or two deep; anything longer is either a
# modelling pathology or a cycle, and both must end in a diagnosis rather
# than in a longer walk.
MAX_TRANSITION_HOPS = 4


def _effective_host_candidates(context, source: FeatureNode, face_ids: list[str]) -> dict:
    """Bypass transition owners and return proven effective host faces.

    A terminal is intentionally rejected when the bypass yields zero or more
    than one effective host.  Guessing among two support planes changes the
    meaning of LOCATED_ON, so the caller records diagnosis and emits no edge.

    Termination is explicit.  Faces are visited once and each transition node
    is expanded once: two fillets that list each other's faces as supports
    would otherwise walk forever, because a queue entry carrying its own path
    is never equal to the one before it.  Depth is capped as a second guard.
    """
    queue = [(str(face_id), []) for face_id in face_ids if str(face_id) in context.model.faces]
    effective: list[str] = []
    path: list[dict] = []
    resolution: list[dict] = []
    seen_resolution: set[str] = set()
    visited_faces: set[str] = set()
    expanded_transitions: set[str] = set()

    def record(face_id: str) -> None:
        if face_id not in seen_resolution:
            seen_resolution.add(face_id)
            resolution.append(_face_resolution_record(context, face_id, source.nodeId))

    # Resolve every candidate up front, not only the ones reached before an
    # abort.  A record that stops at the first blocking face cannot show that
    # a perfectly good host was sitting next to it.
    for face_id, _ in queue:
        record(face_id)

    def reject(reason: str, blocking: list[str], hops: list[dict]) -> dict:
        for face_id in blocking:
            record(face_id)
        return {
            "effective": [],
            "path": hops,
            "reason": reason,
            "faceResolution": resolution,
            "blockingFaceIds": sorted(set(blocking)),
        }

    while queue:
        face_id, hops = queue.pop(0)
        if face_id in visited_faces:
            continue
        visited_faces.add(face_id)
        record(face_id)
        if len(hops) >= MAX_TRANSITION_HOPS:
            return reject("transition_bypass_depth_exceeded", [face_id], [*hops])
        owner_id, owner_type, status = _face_owner(context, face_id, source.nodeId)
        if status != "resolved":
            return reject(status, [face_id], path)
        hop = {"faceId": face_id, "ownerNodeId": owner_id, "ownerNodeType": owner_type}
        next_path = [*hops, hop]
        if owner_type in TRANSITION_TYPES:
            if owner_id in expanded_transitions:
                continue
            expanded_transitions.add(owner_id)
            supports = _transition_support_faces(context, owner_id)
            supports = [support for support in supports if support not in visited_faces]
            if not supports:
                return reject("transition_bypass_unresolved", [face_id], next_path)
            for support in supports:
                queue.append((support, next_path))
            continue
        if face_id not in effective:
            effective.append(face_id)
            path.extend(next_path)

    if not effective:
        return reject("missing_effective_host", [], path)
    # Several PartModel faces can be the split/merged representation of one
    # semantic host node.  They are one effective host at relation level, but
    # every physical face ID remains in the witness.  Distinct owner nodes are
    # still ambiguous and are rejected conservatively.
    owner_ids = {
        owner_id
        for face_id in effective
        for owner_id in [
            next((owner for owner in context.face_owners.get(face_id, ()) if owner != source.nodeId), None)
        ]
        if owner_id is not None
    }
    if len(owner_ids) != 1:
        return reject("ambiguous_effective_host", list(effective), path)
    return {
        "effective": effective,
        "path": path,
        "reason": None,
        "faceResolution": resolution,
        "blockingFaceIds": [],
    }


def _record_located_rejection(
    context,
    source: FeatureNode,
    *,
    role: str,
    boundary_face_ids: list[str] | None,
    boundary_edge_ids: list[str] | None,
    host_face_ids: list[str] | None,
    transition_faces: list[str] | None,
    transition_path: list[dict] | None,
    reason: str,
    host_face_resolution: list[dict] | None = None,
    blocking_face_ids: list[str] | None = None,
) -> None:
    """Keep conservative terminal failures in the LOCATED_ON rule report.

    A missing edge is intentional for ambiguous proofs, but a bare counter is
    not sufficient for audit.  The report retains the source, terminal role,
    candidate faces/owners and bypass path without mutating feature nodes.
    """
    candidates = [str(value) for value in (host_face_ids or [])]
    candidate_owners = sorted({
        str(owner)
        for face_id in candidates
        for owner in context.face_owners.get(face_id, ())
        if owner != source.nodeId
    })
    path = [dict(item) for item in (transition_path or []) if isinstance(item, dict)]
    report = context.rule_diagnostics.setdefault("located_on_v1", {})
    records = report.setdefault("rejectedTerminals", [])
    record = {
        "semanticSourceFeatureId": source.nodeId,
        "semanticSourceFeatureType": source.featureType,
        "terminalRole": role,
        "terminalBoundaryFaceIds": sorted(set(str(value) for value in (boundary_face_ids or []))),
        "terminalBoundaryEdgeIds": sorted(set(str(value) for value in (boundary_edge_ids or []))),
        "candidateHostFaceIds": sorted(set(candidates)),
        "candidateHostOwnerNodeIds": candidate_owners,
        # ``candidateHostOwnerNodeIds`` only lists the owners that exist, so
        # on its own it cannot show that the terminal was rejected *because*
        # some other candidate face had no owner at all.  The per-face
        # resolution and the blocking face IDs make the record readable
        # without re-running the pipeline.
        "hostFaceResolution": [dict(item) for item in (host_face_resolution or [])],
        "blockingFaceIds": sorted(set(str(value) for value in (blocking_face_ids or []))),
        "transitionFaceIds": sorted(set(str(value) for value in (transition_faces or []))),
        "transitionBypassPath": path,
        "reason": reason,
    }
    if record not in records:
        records.append(record)
    report["rejectedTerminalCount"] = len(records)


def _terminal_witness(
    context,
    source: FeatureNode,
    *,
    role: str,
    boundary_face_ids: list[str],
    boundary_edge_ids: list[str] | None,
    host_face_ids: list[str],
    transition_faces: list[str] | None = None,
    transition_path: list[dict] | None = None,
) -> dict | None:
    resolved = _effective_host_candidates(context, source, host_face_ids)
    effective = resolved["effective"]
    bypass_path = resolved["path"]
    reason = resolved["reason"]
    if reason or not effective:
        _record_located_rejection(
            context,
            source,
            role=role,
            boundary_face_ids=boundary_face_ids,
            boundary_edge_ids=boundary_edge_ids,
            host_face_ids=host_face_ids,
            transition_faces=transition_faces,
            transition_path=transition_path or bypass_path,
            reason=reason or "effective_host_not_unique",
            host_face_resolution=resolved["faceResolution"],
            blocking_face_ids=resolved["blockingFaceIds"],
        )
        return None
    host_face_id = effective[0]
    owner_ids = {
        owner_id
        for face_id in effective
        for owner_id in context.face_owners.get(face_id, ())
        if owner_id != source.nodeId
    }
    owner_id = next(iter(owner_ids), None) if len(owner_ids) == 1 else None
    owner = context.node_by_id.get(owner_id) if owner_id else None
    if owner is None:
        _record_located_rejection(
            context,
            source,
            role=role,
            boundary_face_ids=boundary_face_ids,
            boundary_edge_ids=boundary_edge_ids,
            host_face_ids=host_face_ids,
            transition_faces=transition_faces,
            transition_path=transition_path or bypass_path,
            reason="missing_effective_host_owner",
            host_face_resolution=resolved["faceResolution"],
            blocking_face_ids=list(effective),
        )
        return None
    host_face = context.model.faces.get(host_face_id)
    return {
        "semanticSourceFeatureId": source.nodeId,
        "semanticSourceFeatureType": source.featureType,
        "terminalRole": role,
        "terminalBoundaryFaceIds": sorted(set(str(value) for value in boundary_face_ids)),
        "terminalBoundaryEdgeIds": sorted(set(str(value) for value in (boundary_edge_ids or []))),
        "transitionFaceIds": sorted(set(str(value) for value in (transition_faces or []))),
        "transitionBypassPath": transition_path or bypass_path,
        "hostFaceId": host_face_id,
        "hostFaceIds": sorted(set(effective)),
        "hostFaceDetails": [
            {
                "hostFaceId": face_id,
                "hostSurfaceType": _face_surface_type(context.model.faces.get(face_id)),
                "hostSurfaceNormal": _face_normal(context.model.faces.get(face_id)),
            }
            for face_id in sorted(set(effective))
        ],
        "hostSurfaceType": _face_surface_type(host_face),
        "hostSurfaceNormal": _face_normal(host_face),
        "hostOwnerNodeId": owner.nodeId,
        "hostOwnerNodeType": owner.featureType,
    }


def _injected_terminal_witnesses(context, node: FeatureNode) -> list[dict]:
    """Terminals supplied by the caller through the context, if any.

    Deliberately not read from ``node.diagnostics``: a hook living in feature
    data would let a recognizer publish a LOCATED_ON host directly, skipping
    the effective-host resolution and the ambiguity rejection that make this
    relation a proof rather than a claim.
    """
    values = (getattr(context, "terminal_overrides", None) or {}).get(node.nodeId)
    return [dict(value) for value in values if isinstance(value, dict)] if isinstance(values, list) else []


def _endpoint_witness(context, node: FeatureNode, endpoint: dict | None, role: str, boundary_edges: list[str]) -> dict | None:
    if not isinstance(endpoint, dict):
        _record_located_rejection(
            context,
            node,
            role=role,
            boundary_face_ids=list(node.partmodelFaceIds),
            boundary_edge_ids=boundary_edges,
            host_face_ids=[],
            transition_faces=[],
            transition_path=[],
            reason="missing_endpoint_resolution",
        )
        return None
    resolution = endpoint.get("resolution") if "resolution" in endpoint else endpoint
    host_faces = resolution.get("terminalFaceIds") if isinstance(resolution, dict) else None
    if not isinstance(host_faces, list):
        host_faces = endpoint.get("terminalFaceIds") if isinstance(endpoint, dict) else None
    if not isinstance(host_faces, list):
        _record_located_rejection(
            context,
            node,
            role=role,
            boundary_face_ids=list(node.partmodelFaceIds),
            boundary_edge_ids=boundary_edges,
            host_face_ids=[],
            transition_faces=[],
            transition_path=[],
            reason="missing_terminal_face_ids",
        )
        return None
    transition_faces = (resolution or {}).get("transitionFaceIds", []) if isinstance(resolution, dict) else []
    return _terminal_witness(
        context,
        node,
        role=role,
        boundary_face_ids=list(node.partmodelFaceIds),
        boundary_edge_ids=boundary_edges,
        host_face_ids=[str(value) for value in host_faces],
        transition_faces=transition_faces,
        transition_path=(resolution or {}).get("steps", []) if isinstance(resolution, dict) else [],
    )


def _band_edge_ids(node: FeatureNode, name: str) -> list[str]:
    diagnostics = getattr(node, "diagnostics", {}) or {}
    band = diagnostics.get("sideWallBand") or {}
    rim = band.get(name) or {}
    return [str(value) for value in (rim.get("edgeIds") or [])]


def principal_wall_face_ids(context, node: FeatureNode) -> list[str]:
    diagnostics = getattr(node, "diagnostics", {}) or {}
    selected: list[str] = []
    if node.featureType in HOLE_TYPES:
        for key in ("baseFaceId", "sectionFaceId"):
            value = diagnostics.get(key)
            if isinstance(value, str) and value in context.model.faces:
                selected.append(value)
        # For compound holes only the base cylindrical wall is the principal
        # wall for INTERSECTS; section/outer treatment faces are excluded.
        if selected and node.featureType in {"Counterbore", "Countersink"}:
            base = diagnostics.get("baseFaceId")
            if isinstance(base, str) and base in context.model.faces:
                selected = [base]
    if not selected:
        selected = [
            str(face_id)
            for face_id in node.partmodelFaceIds
            if _face_surface_type(context.model.faces.get(str(face_id))) == "CYLINDER"
        ]
    return sorted(set(selected))


def axis_origin(context, node: FeatureNode) -> list[float] | None:
    """Return a geometric point on a feature's axis, never posX/Y/Z."""
    for face_id in principal_wall_face_ids(context, node):
        face = context.model.faces.get(face_id)
        origin = point_value(getattr(face, "parameters", {}).get("axis_location")) if face else None
        if origin is not None:
            return origin
    diagnostics = getattr(node, "diagnostics", {}) or {}
    frame = diagnostics.get("featureFrame") or {}
    return point_value(frame.get("origin"))


def _hole_wall_for_terminal(context, node: FeatureNode) -> list[str]:
    diagnostics = getattr(node, "diagnostics", {}) or {}
    preferred = diagnostics.get("sectionFaceId")
    if isinstance(preferred, str) and preferred in context.model.faces:
        return [preferred]
    return principal_wall_face_ids(context, node)


def _compound_outer_opening(diagnostics: dict) -> tuple[list[str], list[str]] | None:
    """Reuse compound recognition's already-proven outer rim evidence."""
    topology = diagnostics.get("sectionTopology") or {}
    rim_ends = [item for item in (topology.get("rimEnds") or []) if isinstance(item, dict)]
    rims = [item for item in (topology.get("rims") or []) if isinstance(item, dict)]
    outer = diagnostics.get("outerCoordinate")
    if not rim_ends:
        return None
    ordered = sorted(
        rim_ends,
        key=lambda item: abs(float(item.get("axisPosition", 0.0)) - float(outer))
        if isinstance(outer, (int, float)) else float(item.get("axisPosition", 0.0)),
    )
    selected = next((item for item in ordered if item.get("kind") == "open"), None)
    if selected is None:
        return None
    axis_position = selected.get("axisPosition")
    matching = sorted(
        rims,
        key=lambda item: abs(float(item.get("axisPosition", 0.0)) - float(axis_position))
        if isinstance(axis_position, (int, float)) else 0.0,
    )
    edge_ids = [str(value) for value in (matching[0].get("edgeIds") or [])] if matching else []
    face_ids = [str(value) for value in (selected.get("faceIds") or [])]
    return (face_ids, edge_ids) if face_ids else None


def located_terminals(context, node: FeatureNode) -> list[dict]:
    """Resolve all semantic terminal -> effective host witnesses for a node."""
    return _cached(context, "locatedTerminals", node, lambda: _resolve_located_terminals(context, node))


def _resolve_located_terminals(context, node: FeatureNode) -> list[dict]:
    explicit = _injected_terminal_witnesses(context, node)
    if explicit:
        return explicit

    diagnostics = getattr(node, "diagnostics", {}) or {}
    records: list[dict] = []

    if node.featureType in HOLE_TYPES:
        if node.featureType == "Countersink":
            # The outer opening is on the cone section.  ``wall_rims`` is
            # intentionally cylinder-only, so reconstruct the already
            # recognised largest cone boundary and reuse the topology-only
            # rim classifier to find its external host.
            section_id = diagnostics.get("sectionFaceId")
            large = (diagnostics.get("countersinkBoundaries") or {}).get("large") or {}
            edge_ids = [str(value) for value in (large.get("edgeIds") or [])]
            end = None
            if isinstance(section_id, str) and section_id in context.model.faces and edge_ids:
                axis_position = large.get("axisPosition", diagnostics.get("outerCoordinate"))
                try:
                    rim = RimGroup(edge_ids=edge_ids, axis_position=float(axis_position), sample_points=[])
                    end = classify_rim_end(context.model, section_id, rim)
                except (TypeError, ValueError):
                    end = None
                if end is not None and end.kind == "open":
                    witness = _terminal_witness(
                        context,
                        node,
                        role="outer_opening",
                        boundary_face_ids=[section_id],
                        boundary_edge_ids=edge_ids,
                        host_face_ids=end.face_ids,
                    )
                    if witness is not None:
                        return [witness]
            _record_located_rejection(
                context,
                node,
                role="outer_opening",
                boundary_face_ids=[str(section_id)] if section_id else [],
                boundary_edge_ids=edge_ids,
                host_face_ids=list(end.face_ids) if end is not None else [],
                transition_faces=[],
                transition_path=[],
                reason=(
                    "countersink_outer_opening_not_open_or_unresolved"
                    if end is not None else "missing_countersink_large_boundary"
                ),
            )
            return []
        # The composite hole's outer/largest opening is represented by the
        # section wall; its internal shoulder/base stage is not a terminal.
        for wall_id in _hole_wall_for_terminal(context, node):
            rims = wall_rims(context.model, wall_id)
            if node.featureType == "Counterbore":
                # Compound resolution already proved the cylinder rim kind
                # and face set.  Prefer that frozen evidence when a second
                # wall-rim walk cannot classify a seam-split opening.
                diagnostic_opening = _compound_outer_opening(diagnostics)
                if diagnostic_opening is not None:
                    host_faces, edge_ids = diagnostic_opening
                    witness = _terminal_witness(
                        context,
                        node,
                        role="outer_opening",
                        boundary_face_ids=[wall_id],
                        boundary_edge_ids=edge_ids,
                        host_face_ids=host_faces,
                    )
                    if witness is not None:
                        return [witness]
            if not rims:
                _record_located_rejection(
                    context,
                    node,
                    role="opening",
                    boundary_face_ids=[wall_id],
                    boundary_edge_ids=[],
                    host_face_ids=[],
                    transition_faces=[],
                    transition_path=[],
                    reason="missing_terminal_rims",
                )
                continue
            outer_coordinate = diagnostics.get("outerCoordinate")
            if node.featureType in {"Counterbore", "Countersink"} and isinstance(outer_coordinate, (int, float)):
                rim_order = sorted(rims, key=lambda rim: abs(float(rim.axis_position) - float(outer_coordinate)))[:1]
            else:
                rim_order = rims
            for index, rim in enumerate(rim_order):
                end = classify_rim_end(context.model, wall_id, rim)
                if end.kind != "open":
                    continue
                if node.featureType == "BlindHole":
                    role = "opening"
                elif node.featureType == "ThroughHole":
                    role = f"opening_end_{index}"
                else:
                    role = "outer_opening"
                witness = _terminal_witness(
                    context,
                    node,
                    role=role,
                    boundary_face_ids=[wall_id],
                    boundary_edge_ids=rim.edge_ids,
                    host_face_ids=end.face_ids,
                )
                if witness is not None:
                    records.append(witness)
            if records and node.featureType in {"Counterbore", "Countersink"}:
                break
        return records

    if node.featureType in POCKET_TYPES or node.featureType in {"CircularBoss", "RectangularBoss", "GeneralBoss"}:
        endpoint = diagnostics.get("incomingEndpoint")
        role = "opening" if node.featureType in POCKET_TYPES else "base_support"
        witness = _endpoint_witness(context, node, endpoint, role, _band_edge_ids(node, "rimA"))
        return [witness] if witness is not None else []

    if node.featureType == "ThroughCut":
        endpoints = (("entry_opening", diagnostics.get("incomingEndpoint"), "rimA"), ("exit_opening", diagnostics.get("outgoingEndpoint"), "rimB"))
        for role, endpoint, rim_name in endpoints:
            witness = _endpoint_witness(context, node, endpoint, role, _band_edge_ids(node, rim_name))
            if witness is not None:
                records.append(witness)
        return records

    if node.featureType in SLOT_TYPES:
        semantic = diagnostics.get("semanticResolution") or {}
        terminal_resolution = semantic.get("terminalResolution") or {}
        terminal_values = terminal_resolution.get("terminals") or []
        if not terminal_values:
            _record_located_rejection(
                context,
                node,
                role="opening",
                boundary_face_ids=list(node.partmodelFaceIds),
                boundary_edge_ids=[],
                host_face_ids=[],
                transition_faces=[],
                transition_path=[],
                reason="missing_terminal_resolution",
            )
        for index, terminal in enumerate(terminal_values):
            resolution = terminal.get("resolution") or {}
            witness = _terminal_witness(
                context,
                node,
                role=f"transverse_rail_{index}",
                boundary_face_ids=[str(value) for value in node.partmodelFaceIds],
                boundary_edge_ids=[str(terminal.get("railEdgeId"))] if terminal.get("railEdgeId") else [],
                host_face_ids=[str(value) for value in (resolution.get("terminalFaceIds") or [])],
                transition_faces=resolution.get("transitionFaceIds") or [],
                transition_path=resolution.get("steps") or [],
            )
            if witness is not None:
                records.append(witness)
        return records

    return records


def located_hosts(context, node: FeatureNode) -> dict[str, list[dict]]:
    """Return LOCATED_ON witnesses grouped by host-owner node.

    A host owner is the semantic shared-host identity.  A single owner may
    expose more than one physical host face, so grouping by ``hostFaceId``
    would incorrectly reject valid same-owner pairs.  The individual witness
    records retain their effective face IDs for geometry and audit display.
    """
    def build() -> dict[str, list[dict]]:
        result: dict[str, list[dict]] = defaultdict(list)
        for witness in located_terminals(context, node):
            owner = witness.get("hostOwnerNodeId")
            if owner:
                result[str(owner)].append(witness)
        return dict(result)

    return _cached(context, "locatedHosts", node, build)


def side_wall_face_ids(node: FeatureNode) -> list[str]:
    diagnostics = getattr(node, "diagnostics", {}) or {}
    band = diagnostics.get("sideWallBand") or {}
    values = band.get("wallFaceIds") or diagnostics.get("sideWallFaceIds") or []
    return sorted(set(str(value) for value in values))


def slot_course_direction(node: FeatureNode) -> list[float] | None:
    """The slot's sweep (course) direction, as proven by recognition.

    Taken from the longitudinal open proof the Slot/OpenSweep recognizer
    already produced, not re-derived from a boundary polyline: a chord
    between the first and last sample point of whichever rail happened to be
    nearest is not the course (it is wrong outright for an arc rail), and it
    makes the published angle depend on which pair of rails won the distance
    search.
    """
    diagnostics = getattr(node, "diagnostics", {}) or {}
    proof = (diagnostics.get("semanticResolution") or {}).get("longitudinalOpenProof") or {}
    for end in proof.get("ends") or []:
        if not isinstance(end, dict):
            continue
        direction = vector(end.get("direction"))
        if direction is not None:
            return direction
    return None


def slot_boundary_sets(context, node: FeatureNode) -> list[dict]:
    return _cached(context, "slotBoundarySets", node, lambda: _resolve_slot_boundary_sets(context, node))


def _resolve_slot_boundary_sets(context, node: FeatureNode) -> list[dict]:
    diagnostics = getattr(node, "diagnostics", {}) or {}
    semantic = diagnostics.get("semanticResolution") or {}
    terminals = (semantic.get("terminalResolution") or {}).get("terminals") or []
    resolved_witnesses = located_terminals(context, node)
    result = []
    for index, terminal in enumerate(terminals):
        edge_id = terminal.get("railEdgeId")
        resolution = terminal.get("resolution") or {}
        if not edge_id:
            continue
        points = _edge_points(context, str(edge_id))
        if len(points) < 2:
            continue
        for host_face_id in resolution.get("terminalFaceIds") or []:
            role = f"transverse_rail_{index}"
            witness = next(
                (
                    value for value in resolved_witnesses
                    if value.get("terminalRole") == role
                    and str(edge_id) in {str(item) for item in value.get("terminalBoundaryEdgeIds", [])}
                ),
                None,
            )
            effective_face_id = str(witness.get("hostFaceId")) if witness and witness.get("hostFaceId") else str(host_face_id)
            owner_id = witness.get("hostOwnerNodeId") if witness else None
            if owner_id is None:
                owners = [
                    owner for owner in context.face_owners.get(effective_face_id, ())
                    if owner != node.nodeId
                ]
                if len(owners) == 1:
                    owner_id = owners[0]
            if owner_id is None:
                continue
            result.append({
                "terminalRole": role,
                "hostFaceId": effective_face_id,
                "hostOwnerNodeId": str(owner_id),
                "edgeId": str(edge_id),
                "points": points,
            })
    return result


def side_wall_clearance(context, left: FeatureNode, right: FeatureNode) -> tuple[float | None, dict]:
    """Measure two specified side-wall face sets against the kernel, if available."""
    left_ids = side_wall_face_ids(left)
    right_ids = side_wall_face_ids(right)
    kernel = getattr(context, "kernel", None)
    details = {
        # The measuring kernel names its own method (node 10 §5.3); the
        # stored literal remains the no-kernel fallback so replayed evidence
        # keeps its historical shape.
        "method": kernel.provenance("face_set_distance") if kernel is not None
        else "BRepExtrema_DistShapeShape_side_wall_compounds_v1",
        "leftPartmodelFaceIds": left_ids,
        "rightPartmodelFaceIds": right_ids,
        "status": "not_measured",
    }
    if not left_ids or not right_ids:
        details["status"] = "missing_side_wall_faces"
        return None, details
    if kernel is None:
        details["status"] = "no_occ_context"
        return None, details
    left_cad = sorted(cad_face_ids(context.model, left_ids))
    right_cad = sorted(cad_face_ids(context.model, right_ids))
    details["leftCadFaceIds"] = left_cad
    details["rightCadFaceIds"] = right_cad
    measured = kernel.face_set_distance(left_cad, right_cad)
    if measured.status != DISTANCE_MEASURED:
        details["status"] = measured.status
        if measured.error is not None:
            details["error"] = measured.error
        return None, details
    value = float(measured.distance)
    details.update({
        "status": "measured",
        "clearance": round(value, 6),
        "nearestPointA": [round(float(component), 6) for component in measured.point_a],
        "nearestPointB": [round(float(component), 6) for component in measured.point_b],
    })
    # The nearest points are always available; the faces they sit on are not,
    # and the first input face must not be passed off as the witness.
    if measured.witness_cad_face_ids is not None:
        details["witnessCadFaceIds"] = list(measured.witness_cad_face_ids)
    else:
        details["witnessFaceStatus"] = "unavailable"
        if measured.witness_reason is not None:
            details["witnessFaceReason"] = measured.witness_reason
    return round(value, 6), details


__all__ = [
    "AXIS_LINE_TOL_MM",
    "COAXIAL_ANGLE_TOL_DEG",
    "HOLE_TYPES",
    "LOCATED_SOURCE_TYPES",
    "PARALLEL_ANGLE_TOL_DEG",
    "POCKET_TYPES",
    "SLOT_TYPES",
    "SPACING_BASIS",
    "TRANSITION_TYPES",
    "_cached",
    "_edge_points",
    "_face_normal",
    "_face_owner",
    "_face_resolution_record",
    "_face_surface_type",
    "_feature_edge",
    "_report",
    "_set_report",
    "acute_angle_deg",
    "attributes",
    "axis",
    "axis_origin",
    "coaxial_axis_distance",
    "direct_face_pairs",
    "dot",
    "feature_type",
    "is_coaxial_pair",
    "located_hosts",
    "located_terminals",
    "mean_axis_direction",
    "norm",
    "parallel",
    "parallel_axis_spacing",
    "point_value",
    "polyline_distance",
    "principal_wall_face_ids",
    "side_wall_clearance",
    "side_wall_face_ids",
    "slot_boundary_sets",
    "slot_course_direction",
    "vector",
]
