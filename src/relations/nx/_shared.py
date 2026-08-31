"""The nx_v1 dialect's witnesses: NX's own attributes plus PartModel topology.

Two sources, and the rules keep them apart:

* **what NX published.**  The access direction and locating point of a feature
  arrive as nine scalars (``X_ORIENTATION_D`` … ``Z_POSITION``); a COAXIAL or
  a HOLE_SPACING edge is arithmetic over those and nothing else, which is why
  node 19' files them as ``nx_arithmetic`` and they need no kernel.
* **what the PartModel says.**  A LOCATED_ON terminal is found by walking the
  rims of a feature's own cylindrical walls and asking which face lies beyond
  an open one — pure topology, and the same two functions the v5.1 dialect
  walks its rims with (``wall_rims`` / ``classify_rim_end`` take a PartModel
  and a face id and know no leaf class).

What is deliberately *not* here is a translation.  Which classes are holes is
answered by NX's own family tree (``STEP``, ``HOLES`` are classes NX
declares), never by a list of names we chose.

Coverage, stated rather than hidden.  The terminal resolver has two proofs and
tries them in that order:

* a **cylindrical wall rim** - walk the two rims of each cylindrical wall the
  feature owns and take the open ones.  This is the hole case, and it is first
  because a hole's rim is a stronger statement than a patch boundary: it is
  validated against the wall's own axis and radius.
* a **face-set boundary** (node 21') - where the feature's own faces stop, the
  edges they share with faces outside the set form closed components, and a
  component whose far side is not a cap is a way out.  This is how a pocket, a
  slot or a boss opens, and it needs no leaf semantics: ``face_set_openings``
  takes a PartModel and a set of face ids.  Node 20'a recorded 36 such
  features as ``terminalResolverUnavailable``; this is the proof they were
  waiting for.

Whatever neither proves is still named rather than dropped, and a host is
still required to resolve to exactly one owner.
"""

from __future__ import annotations

from tbox import nx_v1 as source
from contract.kernel import DISTANCE_MEASURED
from core.models import FeatureEdge, FeatureNode
from features._shared.hole_topology import classify_rim_end, wall_rims

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
    face_set_openings,
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

# NX's own names for the two frames it publishes on every feature.
ORIENTATION_D = ("X_ORIENTATION_D", "Y_ORIENTATION_D", "Z_ORIENTATION_D")
POSITION = ("X_POSITION", "Y_POSITION", "Z_POSITION")

# NX's own family classes.  Named, not enumerated: which leaves belong to
# ``STEP`` is a fact in ``customization_base.xml``, transcribed into
# ``classes.yaml`` and checked against the installed file by the ontology's
# own S8, so restating the membership here would be a second answer to a
# question that already has one.
HOLE_FAMILIES = ("STEP", "HOLES")
ROUND_BOSS = "BOSS_ROUND_STRAIGHT"

# The source families ``relations.yaml`` allows a LOCATED_ON terminal on.
LOCATED_SOURCE_FAMILIES = (
    "STEP", "HOLES", "POCKETS", "BOSSES", "SLOTS",
    "CORNER_NOTCHES", "SIDE_NOTCHES", "GROOVES",
)

# How a face of a feature counts against the feature's own access direction,
# by the cosine of the angle between its outward normal and that axis.  A
# floor faces back toward the entry (a flat one at cos 1, a 118-degree drill
# cone at cos 31 deg ~ 0.86, a 135-degree one at ~0.92); a lateral wall runs
# along the axis (a cylinder at cos 0, the 45-degree flank of a V-slot at
# ~0.71).  0.75 sits between the steepest floor and the steepest wall the
# corpus and the NX catalogue actually contain; these are face-classification
# geometry, not process thresholds anyone tunes.
FLOOR_MIN_COS = 0.75
LATERAL_MAX_COS = 0.75


def in_any_family(name: str | None, families) -> bool:
    return bool(name) and any(source.in_family(name, family) for family in families)


def is_hole_like(node: FeatureNode | None) -> bool:
    """A member of one of NX's two drilled/milled hole families."""
    return in_any_family(feature_type(node), HOLE_FAMILIES)


def is_located_source(node: FeatureNode | None) -> bool:
    return in_any_family(feature_type(node), LOCATED_SOURCE_FAMILIES)


# ------------------------------------------------- what NX published

def axis(node: FeatureNode) -> list[float] | None:
    """The feature's access direction as a unit vector, from NX's three."""
    return vector([attributes(node).get(name) for name in ORIENTATION_D])


def axis_point(node: FeatureNode) -> list[float] | None:
    """The feature's locating point, from NX's three."""
    return point_value([attributes(node).get(name) for name in POSITION])


# --------------------------------------------- what the PartModel says

def outward_normal(face) -> list[float] | None:
    """The face's outward unit normal, by the contract's own reading first.

    For a plane that reading is ``parameters['normal']`` negated when
    ``is_reversed`` — the shared machine's ``_face_normal``, which node 21'
    made answer the same on both kernel lines by pinning the pairing in the
    contract and fixing the NX builder to obey it.  It is preferred because
    it is exact and always there; the interior samples are the fallback for
    the curved faces ``_face_normal`` declines (a drill-point cone floor has
    no single stored normal).  ``None`` when neither is available: a refusal,
    not a guess.
    """
    if face is None:
        return None
    stored = _face_normal(face)
    if stored is not None:
        return stored
    samples = list(getattr(face, "interior_samples", None) or [])
    total = [0.0, 0.0, 0.0]
    for sample in samples:
        direction = sample.get("normal")
        if isinstance(direction, (list, tuple)) and len(direction) == 3:
            total = [a + float(b) for a, b in zip(total, direction)]
    normal = vector(total)
    if normal is None:
        return None
    return [round(0.0 if abs(component) < 5.0e-7 else component, 6) for component in normal]


def _axis_cosine(face, axis: list[float]) -> float | None:
    normal = outward_normal(face)
    return dot(normal, axis) if normal is not None else None


def floor_faces(model, node: FeatureNode, axis_direction: list[float]) -> list[str]:
    """The node's own faces that face back toward the entry — its floor."""
    found = []
    for face_id in node.partmodelFaceIds:
        cosine = _axis_cosine(model.faces.get(str(face_id)), axis_direction)
        if cosine is not None and cosine <= -FLOOR_MIN_COS:
            found.append(str(face_id))
    return sorted(set(found), key=_face_order)


def lateral_faces(model, node: FeatureNode, axis_direction: list[float]) -> list[str]:
    """The node's own faces that run along its axis — its side walls."""
    found = []
    for face_id in node.partmodelFaceIds:
        cosine = _axis_cosine(model.faces.get(str(face_id)), axis_direction)
        if cosine is not None and abs(cosine) <= LATERAL_MAX_COS:
            found.append(str(face_id))
    return sorted(set(found), key=_face_order)


# Below this, a "clearance" is a contact.  ``relations.yaml`` states on both
# clearance relations that the two sides must be separated, and the number is
# the same order as the relation layer's other positional budget
# (``AXIS_LINE_TOL_MM``): it separates "these two touch" from "there is a
# measurable web here", which is the distinction the rule turns on, not a
# process threshold anyone tunes.
SEPARATION_TOL_MM = 1.0e-3


def opening_boundary_polylines(context, node: FeatureNode) -> list[dict]:
    """Each of the feature's opening boundaries, as the polylines of its edges.

    ``cnc:SLOT_CLEARANCE`` measures between two slots' opening boundaries, and
    node 21' gave this line a way to find those: the boundary of a feature's
    own face patch is exactly the walk ``face_set_openings`` performs for
    LOCATED_ON.  Measuring the same thing here is what makes the two lines'
    numbers answers to one question - the whole owned face set, which node
    20'b measured instead, lets a slot's floor stand in for its opening.
    """
    return _cached(context, "nxOpeningBoundaries", node,
                   lambda: _resolve_opening_boundaries(context, node))


def _resolve_opening_boundaries(context, node: FeatureNode) -> list[dict]:
    found = []
    for opening in face_set_openings(context.model, node.partmodelFaceIds):
        if not opening.far_face_ids:
            continue
        polylines = []
        for edge_id in opening.edge_ids:
            points = _edge_points(context, edge_id)
            if len(points) >= 2:
                polylines.append(points)
        if polylines:
            found.append({
                "boundaryEdgeIds": list(opening.edge_ids),
                "ownFaceIds": list(opening.own_face_ids),
                "polylines": polylines,
            })
    return found


def boundary_set_clearance(left: list[dict], right: list[dict]) -> tuple[float | None, dict]:
    """Shortest distance between two features' opening boundaries.

    Pure arithmetic over the PartModel, the same ``polyline_distance`` the
    v5.1 rule uses on its rails.  ``None`` always comes with a named status,
    so a no-edge decision can be re-argued from the diagnostics alone.
    """
    details: dict = {
        "method": "opening_boundary_polyline_distance_v1",
        "leftBoundaryEdgeIds": sorted({eid for item in left for eid in item["boundaryEdgeIds"]}),
        "rightBoundaryEdgeIds": sorted({eid for item in right for eid in item["boundaryEdgeIds"]}),
        "status": "not_measured",
    }
    if not left or not right:
        details["status"] = "missing_opening_boundary"
        return None, details
    best = None
    for left_item in left:
        for right_item in right:
            for left_line in left_item["polylines"]:
                for right_line in right_item["polylines"]:
                    distance, left_point, right_point = polyline_distance(left_line, right_line)
                    if distance is None:
                        continue
                    if best is None or distance < best[0]:
                        best = (distance, left_point, right_point)
    if best is None:
        details["status"] = "distance_solver_failed"
        return None, details
    details["status"] = "measured"
    details["clearance"] = round(float(best[0]), 6)
    details["witnessPoints"] = {"left": best[1], "right": best[2]}
    return round(float(best[0]), 6), details


def kernel_face_set_clearance(context, left_face_ids: list[str],
                              right_face_ids: list[str]) -> tuple[float | None, dict]:
    """Measure two PartModel face sets against the session kernel.

    Returns ``(value, details)``; a ``None`` value always comes with a named
    status in the details — no kernel, empty face set, or the kernel's own
    refusal vocabulary — so a no-edge decision can be re-argued from the
    diagnostics alone.
    """
    from features._shared.attributes import cad_face_ids

    kernel = getattr(context, "kernel", None)
    details: dict = {
        "method": kernel.provenance("face_set_distance") if kernel is not None
        else "face_set_distance",
        "leftPartmodelFaceIds": list(left_face_ids),
        "rightPartmodelFaceIds": list(right_face_ids),
        "status": "not_measured",
    }
    if not left_face_ids or not right_face_ids:
        details["status"] = "missing_measurable_faces"
        return None, details
    if kernel is None:
        details["status"] = "kernel_unavailable"
        return None, details
    left_cad = sorted(cad_face_ids(context.model, left_face_ids))
    right_cad = sorted(cad_face_ids(context.model, right_face_ids))
    details["leftCadFaceIds"] = left_cad
    details["rightCadFaceIds"] = right_cad
    measured = kernel.face_set_distance(left_cad, right_cad)
    if measured.status != DISTANCE_MEASURED:
        details["status"] = measured.status
        if measured.error is not None:
            details["error"] = measured.error
        return None, details
    value = round(float(measured.distance), 6)
    details["status"] = "measured"
    details["clearance"] = value
    if measured.witness_cad_face_ids is not None:
        details["witnessCadFaceIds"] = list(measured.witness_cad_face_ids)
    elif measured.witness_reason is not None:
        details["witnessFaceReason"] = measured.witness_reason
    return value, details


def _face_order(face_id: str) -> int:
    try:
        return int(str(face_id).rsplit("_", 1)[-1])
    except ValueError:
        return 0


def cylindrical_walls(context, node: FeatureNode) -> list[str]:
    """The node's own cylindrical faces, in the geometry provider's order."""
    walls = [
        str(face_id) for face_id in node.partmodelFaceIds
        if _face_surface_type(context.model.faces.get(str(face_id))) == "CYLINDER"
    ]
    return sorted(set(walls), key=_face_order)


def located_terminals(context, node: FeatureNode) -> list[dict]:
    """Every proven opening of *node*, resolved outward to one host owner."""
    return _cached(context, "nxLocatedTerminals", node,
                   lambda: _resolve_located_terminals(context, node))


class _Opening:
    """One candidate way out of a feature, whichever proof found it."""

    __slots__ = ("basis", "wall_face_id", "axis_position", "edge_ids", "far_face_ids")

    def __init__(self, basis, far_face_ids, *, wall_face_id=None, axis_position=None,
                 edge_ids=()):
        self.basis = basis
        self.far_face_ids = [str(face_id) for face_id in far_face_ids]
        self.wall_face_id = wall_face_id
        self.axis_position = axis_position
        self.edge_ids = tuple(edge_ids)


def _cylinder_rim_openings(context, node: FeatureNode) -> list[_Opening]:
    """The open rims of every cylindrical wall the feature owns."""
    ends: list[tuple[str, float, list[str]]] = []
    for wall_id in cylindrical_walls(context, node):
        for rim in wall_rims(context.model, wall_id):
            end = classify_rim_end(context.model, wall_id, rim)
            if end.kind != "open":
                continue
            ends.append((wall_id, float(end.axis_position),
                         [str(face_id) for face_id in end.face_ids]))
    ends.sort(key=lambda item: (_face_order(item[0]), item[1]))
    return [_Opening("cylindricalWallRim", face_ids,
                     wall_face_id=wall_id, axis_position=axis_position)
            for wall_id, axis_position, face_ids in ends]


def _face_set_boundary_openings(context, node: FeatureNode) -> list[_Opening]:
    """Where the feature's own patch of faces stops, and what lies across it.

    Node 21': this is the proof for the features that own no cylindrical wall
    — pockets, slots, bosses, notches — and it reads nothing but the
    PartModel.  A component whose far side is entirely capped is a closed end,
    not an opening, so ``face_set_openings`` has already dropped those faces
    and a fully capped component arrives here with nothing on the far side.
    """
    direction = axis(node)
    found: list[tuple[float, str, _Opening]] = []
    for opening in face_set_openings(context.model, node.partmodelFaceIds):
        if not opening.far_face_ids:
            continue
        position = _boundary_axis_position(context.model, opening.edge_ids, direction)
        found.append((
            position if position is not None else 0.0,
            opening.edge_ids[0] if opening.edge_ids else "",
            _Opening("faceSetBoundary", opening.far_face_ids,
                     axis_position=position, edge_ids=opening.edge_ids),
        ))
    found.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in found]


def _boundary_axis_position(model, edge_ids, direction) -> float | None:
    """Where a boundary component sits along the feature's access direction.

    The same kind of coordinate ``classify_rim_end`` reports for a cylinder
    rim, so the two proofs order their openings on one scale.  ``None`` when
    NX published no usable direction — the edge ids then order them instead.
    """
    if direction is None:
        return None
    total, count = 0.0, 0
    for edge_id in edge_ids:
        edge = model.edges.get(str(edge_id))
        for vertex_id in (getattr(edge, "vertex_ids", None) or []) if edge is not None else []:
            vertex = model.vertices.get(str(vertex_id))
            point = point_value(getattr(vertex, "coordinates", None))
            if point is None:
                continue
            total += dot(point, direction)
            count += 1
    return round(total / count, 6) if count else None


def _resolve_located_terminals(context, node: FeatureNode) -> list[dict]:
    rule_id = "located_on_nx_v1"
    # The context's own injection point, and the only one: reading a terminal
    # out of a node's diagnostics would let a recognizer assert a host without
    # going through the proof this layer exists to perform.
    injected = (getattr(context, "terminal_overrides", None) or {}).get(node.nodeId)
    if injected is not None:
        return [dict(witness) for witness in injected]

    openings = _cylinder_rim_openings(context, node)
    if not openings:
        # The rim walk found no wall to walk, or walked one and found no open
        # end.  Either way the boundary of the feature's own patch is a
        # different question, so it gets asked rather than assumed away.
        openings = _face_set_boundary_openings(context, node)
    if not openings:
        _record_rejection(context, rule_id, node, role=None, host_face_ids=[],
                          reason="terminalResolverUnavailable: neither the cylindrical "
                                 "wall rims nor the boundary of this feature's own faces "
                                 "yielded an opening")
        return []

    own = {str(face_id) for face_id in node.partmodelFaceIds}
    witnesses: list[dict] = []
    for index, opening in enumerate(openings):
        role = f"opening_end_{index}"
        candidates = [face_id for face_id in opening.far_face_ids if face_id not in own]
        owners: dict[str, list[str]] = {}
        for face_id in candidates:
            owner, _owner_type, status = _face_owner(context, face_id, node.nodeId)
            if status == "resolved" and owner:
                owners.setdefault(owner, []).append(face_id)
        if len(owners) != 1:
            _record_rejection(
                context, rule_id, node, role=role, host_face_ids=candidates,
                reason=("ambiguous_effective_host" if len(owners) > 1
                        else "no_single_effective_host"),
                wall_face_id=opening.wall_face_id, axis_position=opening.axis_position,
                basis=opening.basis,
            )
            continue
        owner_id, owner_faces = next(iter(owners.items()))
        host_face_id = sorted(owner_faces, key=_face_order)[0]
        host_face = context.model.faces.get(host_face_id)
        # The ontology declares this field as the host face's *outward*
        # normal.  Planar hosts only; a curved host has no single normal to
        # publish.  Node 20'b had to read this out of the interior samples
        # because the two lines disagreed about the stored one, and paid for
        # it on the faces NX never sampled; node 21' fixed the disagreement at
        # the source, so the stored normal is read again and those faces
        # answer.
        witness = {
            "semanticSourceFeatureId": node.nodeId,
            "semanticSourceFeatureType": node.featureType,
            "terminalRole": role,
            "terminalBasis": opening.basis,
            "hostFaceId": host_face_id,
            "hostFaceIds": sorted(owner_faces, key=_face_order),
            "hostOwnerNodeId": owner_id,
            "hostSurfaceType": _face_surface_type(host_face),
            "hostSurfaceNormal": (outward_normal(host_face)
                                  if _face_surface_type(host_face) == "PLANE" else None),
        }
        if opening.wall_face_id is not None:
            witness["wallFaceId"] = opening.wall_face_id
        if opening.edge_ids:
            witness["boundaryEdgeIds"] = list(opening.edge_ids)
        if opening.axis_position is not None:
            witness["rimAxisPosition"] = round(opening.axis_position, 6)
        witnesses.append(witness)
    return witnesses


def _record_rejection(context, rule_id: str, node: FeatureNode, *, role, host_face_ids,
                      reason: str, wall_face_id: str | None = None,
                      axis_position: float | None = None,
                      basis: str | None = None) -> None:
    """Why one opening did not become an edge, in enough detail to re-argue it."""
    report = context.rule_diagnostics.setdefault(rule_id, {})
    records = report.setdefault("rejectedTerminals", [])
    record = {
        "semanticSourceFeatureId": node.nodeId,
        "semanticSourceFeatureType": node.featureType,
        "terminalRole": role,
        "reason": reason,
        "candidateHostFaceIds": list(host_face_ids),
        "candidateHostOwnerNodeIds": sorted({
            owner for face_id in host_face_ids
            for owner in context.face_owners.get(str(face_id), ())
            if owner != node.nodeId
        }),
        "faceResolution": [_face_resolution_record(context, face_id, node.nodeId)
                           for face_id in host_face_ids],
    }
    if basis is not None:
        record["terminalBasis"] = basis
    if wall_face_id is not None:
        record["wallFaceId"] = wall_face_id
    if axis_position is not None:
        record["rimAxisPosition"] = round(axis_position, 6)
    records.append(record)
    _report(context, rule_id, "terminalRejected")


# --------------------------------------------------------------- the edge

def nx_edge(source_node: FeatureNode, target_node: FeatureNode, edge_type: str,
            rule_id: str, derived_by: str, evidence: dict,
            properties: dict | None = None) -> FeatureEdge:
    """One NX-dialect edge, carrying how it was derived.

    ``derivedBy`` goes in the diagnostics rather than in the properties: the
    properties are the quantities ``relations.yaml`` declares on the relation,
    and node 19' 6 is explicit that a value we derived must never be published
    where a value NX stated would be.
    """
    return _feature_edge(
        source_node, target_node, edge_type, rule_id,
        dict(evidence),
        {"derivedBy": derived_by},
        properties or {},
    )


__all__ = [
    "AXIS_LINE_TOL_MM",
    "COAXIAL_ANGLE_TOL_DEG",
    "FLOOR_MIN_COS",
    "HOLE_FAMILIES",
    "LATERAL_MAX_COS",
    "LOCATED_SOURCE_FAMILIES",
    "ORIENTATION_D",
    "PARALLEL_ANGLE_TOL_DEG",
    "POSITION",
    "ROUND_BOSS",
    "SPACING_BASIS",
    "_report",
    "acute_angle_deg",
    "axis",
    "axis_point",
    "coaxial_axis_distance",
    "cylindrical_walls",
    "direct_face_pairs",
    "feature_type",
    "floor_faces",
    "in_any_family",
    "is_hole_like",
    "is_located_source",
    "kernel_face_set_clearance",
    "lateral_faces",
    "located_terminals",
    "nx_edge",
    "outward_normal",
    "opening_boundary_polylines",
    "parallel",
    "parallel_axis_spacing",
]
