"""Relation machinery: the edge structure and the witnesses that name no class.

The relation layer reads the final ``FeatureNode`` objects and the normalized
``PartModel`` only.  This module holds the part of that work which is the same
whatever ontology the rules are written in: vector and axis-line arithmetic,
the tolerance policy, face-to-owner resolution, shared-edge enumeration, the
per-run memo and the ``FeatureEdge`` constructor.

It names no leaf class and no ontology attribute — deliberately.  A dialect's
own vocabulary lives beside that dialect's rules
(``relations.v51._shared``, ``relations.nx._shared``).
"""

from __future__ import annotations

import math
from typing import Any

from core.models import FeatureEdge, FeatureNode

# Relation-layer tolerance policy.  These are implementation parameters, not
# H thresholds.
#
# ``PARALLEL_ANGLE_TOL_DEG`` follows the one-degree budget the project already
# uses for local frame parallel checks; it answers "parallel enough that an
# axis-to-axis spacing is a meaningful number at all".
#
# ``COAXIAL_ANGLE_TOL_DEG`` is deliberately much tighter.  Claiming two axes
# are the *same line* while allowing a full degree of tilt is not self
# consistent: one degree over a 50 mm hole is a 0.87 mm deviation, three
# orders of magnitude past the positional budget below.  Coaxiality is
# therefore held to an angular tolerance of the same order as its distance
# tolerance.  The whole corpus sits at <= 1e-6 deg, so this leaves ample room.
PARALLEL_ANGLE_TOL_DEG = 1.0
COAXIAL_ANGLE_TOL_DEG = 0.01
AXIS_LINE_TOL_MM = 1.0e-3


def attributes(node: FeatureNode) -> dict[str, Any]:
    return getattr(node, "attributes", {}) or {}


def feature_type(node: FeatureNode | None) -> str | None:
    return getattr(node, "featureType", None) if node is not None else None


def vector(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    length = math.sqrt(sum(item * item for item in result))
    if length <= 1.0e-12:
        return None
    return [item / length for item in result]


def point_value(value: object) -> list[float] | None:
    """Validate a finite world-coordinate point without normalizing it."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def dot(left: list[float], right: list[float]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def norm(value: list[float]) -> float:
    return math.sqrt(sum(item * item for item in value))


def acute_angle_deg(left: list[float], right: list[float]) -> float:
    cosine = max(-1.0, min(1.0, abs(dot(left, right))))
    return math.degrees(math.acos(cosine))


def parallel(left: list[float], right: list[float], tolerance_deg: float = PARALLEL_ANGLE_TOL_DEG) -> bool:
    return acute_angle_deg(left, right) <= tolerance_deg


def mean_axis_direction(axis_a: list[float], axis_b: list[float]) -> list[float] | None:
    """One direction representing a pair of (undirected) near-parallel axes."""
    sign = 1.0 if dot(axis_a, axis_b) >= 0.0 else -1.0
    combined = [axis_a[index] + sign * axis_b[index] for index in range(3)]
    length = norm(combined)
    if length <= 1.0e-12:
        return None
    return [item / length for item in combined]


SPACING_BASIS = "perpendicular_offset_in_plane_normal_to_mean_axis"


def parallel_axis_spacing(
    origin_a: list[float],
    axis_a: list[float],
    origin_b: list[float],
    axis_b: list[float],
    *,
    tolerance_deg: float = PARALLEL_ANGLE_TOL_DEG,
) -> float | None:
    """Separation of two near-parallel infinite axis lines, or ``None``.

    Deliberately *not* the skew-line common perpendicular.  That formula is
    exact for two arbitrary lines, but the callers admit a pair on an angular
    gate, and for a pair that is only parallel *within a tolerance* the
    common perpendicular collapses: two holes 50 mm apart whose axes differ
    by 0.5 deg have infinite lines that nearly meet far outside the part, so
    the skew distance reports ~0 mm.  That produced both a COAXIAL false
    positive and an ``axisSpacing`` of zero.

    What is returned instead is the offset between the two axis reference
    points measured in the plane perpendicular to the pair's mean direction,
    which is exactly the line-line distance when the axes are truly parallel
    and degrades continuously when they are not.  ``None`` means the pair
    failed the angular gate and has no spacing worth publishing.
    """
    if acute_angle_deg(axis_a, axis_b) > tolerance_deg:
        return None
    direction = mean_axis_direction(axis_a, axis_b)
    if direction is None:
        return None
    delta = [origin_b[index] - origin_a[index] for index in range(3)]
    axial = dot(delta, direction)
    return norm([delta[index] - axial * direction[index] for index in range(3)])


def coaxial_axis_distance(
    origin_a: list[float], axis_a: list[float], origin_b: list[float], axis_b: list[float]
) -> float | None:
    """Axis-line offset under the stricter coaxial angular tolerance."""
    return parallel_axis_spacing(
        origin_a, axis_a, origin_b, axis_b, tolerance_deg=COAXIAL_ANGLE_TOL_DEG
    )


def is_coaxial_pair(
    origin_a: list[float], axis_a: list[float], origin_b: list[float], axis_b: list[float]
) -> bool:
    """Single definition of coaxiality, shared by every rule that needs it."""
    distance = coaxial_axis_distance(origin_a, axis_a, origin_b, axis_b)
    return distance is not None and distance <= AXIS_LINE_TOL_MM


def _report(context, rule_id: str, key: str, amount: int = 1) -> None:
    report = context.rule_diagnostics.setdefault(rule_id, {})
    report[key] = int(report.get(key, 0)) + amount


def _set_report(context, rule_id: str, key: str, value: Any) -> None:
    context.rule_diagnostics.setdefault(rule_id, {})[key] = value


def _face_surface_type(face) -> str | None:
    value = getattr(getattr(face, "surface_type", None), "value", None)
    return value if value is not None else (str(getattr(face, "surface_type", "")) or None)


def _face_normal(face) -> list[float] | None:
    if face is None or _face_surface_type(face) != "PLANE":
        return None
    normal = vector(getattr(face, "parameters", {}).get("normal"))
    if normal is None:
        return None
    if bool(getattr(face, "is_reversed", False)):
        normal = [-item for item in normal]
    return [round(0.0 if abs(item) < 5.0e-7 else item, 6) for item in normal]


def _face_owner(context, face_id: str, source_node_id: str) -> tuple[str | None, str | None, str]:
    """Resolve one face to its single non-source owner.

    The failure statuses are deliberately distinct.  "The face has no owner
    at all" and "the only owner is the feature we started from" and "two
    nodes claim it" are three different findings for whoever reads the
    rejection record, and collapsing them into one ``missing_host_owner``
    made a diagnosis that listed a perfectly good candidate owner look
    self contradictory.
    """
    all_owners = tuple(context.face_owners.get(str(face_id), ()))
    owners = [owner for owner in all_owners if owner != source_node_id]
    if len(owners) == 1:
        owner = context.node_by_id.get(owners[0])
        return owners[0], feature_type(owner), "resolved"
    if owners:
        return None, None, "ambiguous_host_owner"
    return None, None, "host_face_owned_only_by_source" if all_owners else "unowned_host_face"


def _face_resolution_record(context, face_id: str, source_node_id: str) -> dict:
    """One self-explanatory line per candidate face, for the audit record."""
    owner_ids = [str(owner) for owner in context.face_owners.get(str(face_id), ())]
    _, _, status = _face_owner(context, face_id, source_node_id)
    return {
        "faceId": str(face_id),
        "surfaceType": _face_surface_type(context.model.faces.get(str(face_id))),
        "ownerNodeIds": owner_ids,
        "ownerNodeTypes": [feature_type(context.node_by_id.get(owner)) for owner in owner_ids],
        "status": status,
    }


def _cached(context, bucket: str, node: FeatureNode, factory):
    """Memo one per-node resolution for the whole relation run.

    Also makes the conservative rejection records a single-shot side effect:
    they used to be re-appended (and deduplicated by value) once per candidate
    partner, from whichever rule happened to ask.
    """
    cache = getattr(context, "cache", None)
    if cache is None:
        return factory()
    store = cache.setdefault(bucket, {})
    if node.nodeId not in store:
        store[node.nodeId] = factory()
    return store[node.nodeId]


def direct_face_pairs(context, left: FeatureNode, right: FeatureNode) -> list[dict]:
    """Enumerate real shared-edge face witnesses between two final nodes."""
    left_faces = {str(value) for value in left.partmodelFaceIds}
    right_faces = {str(value) for value in right.partmodelFaceIds}
    records: dict[tuple, dict] = {}
    for face_id in sorted(left_faces):
        face = context.model.faces.get(face_id)
        if face is None:
            continue
        for adjacency in getattr(face, "adjacencies", []):
            target = str(getattr(adjacency, "adjacent_face_id", ""))
            if target not in right_faces:
                continue
            edge_id = str(getattr(adjacency, "shared_edge_id", "")) or None
            key = (face_id, target, edge_id)
            records[key] = {
                "leftPartmodelFaceId": face_id,
                "rightPartmodelFaceId": target,
                "sharedEdgeId": edge_id,
                "leftSurfaceType": _face_surface_type(face),
                "rightSurfaceType": _face_surface_type(context.model.faces.get(target)),
                "connectionType": getattr(getattr(adjacency, "connection_type", None), "value", getattr(adjacency, "connection_type", None)),
            }
    return [records[key] for key in sorted(records)]


def _edge_points(context, edge_id: str) -> list[list[float]]:
    edge = context.model.edges.get(str(edge_id))
    if edge is None:
        return []
    points = getattr(edge, "sample_points", None) or []
    if points:
        return [[float(point[index]) for index in range(3)] for point in points if isinstance(point, (list, tuple)) and len(point) == 3]
    result = []
    for vertex_id in getattr(edge, "vertex_ids", []) or []:
        vertex = context.model.vertices.get(vertex_id)
        if vertex is not None:
            result.append([float(value) for value in vertex.coordinates])
    return result


def _segment_distance(a0, a1, b0, b1) -> tuple[float, list[float], list[float]]:
    """Closest points between two finite 3D segments (Ericson-style)."""
    u = [a1[i] - a0[i] for i in range(3)]
    v = [b1[i] - b0[i] for i in range(3)]
    w = [a0[i] - b0[i] for i in range(3)]
    aa, bb, cc = dot(u, u), dot(u, v), dot(v, v)
    dd, ee = dot(u, w), dot(v, w)
    denominator = aa * cc - bb * bb
    if aa <= 1.0e-15 and cc <= 1.0e-15:
        pa, pb = a0, b0
    elif aa <= 1.0e-15:
        t = max(0.0, min(1.0, ee / cc))
        pb = [b0[i] + t * v[i] for i in range(3)]
        pa = list(a0)
    elif cc <= 1.0e-15:
        s = max(0.0, min(1.0, -dd / aa))
        pa = [a0[i] + s * u[i] for i in range(3)]
        pb = list(b0)
    else:
        if denominator <= 1.0e-15:
            s = 0.0
            t = max(0.0, min(1.0, ee / cc))
        else:
            s = max(0.0, min(1.0, (bb * ee - cc * dd) / denominator))
            t = (bb * s + ee) / cc
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -dd / aa))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (bb - dd) / aa))
        pa = [a0[i] + s * u[i] for i in range(3)]
        pb = [b0[i] + t * v[i] for i in range(3)]
    return math.dist(pa, pb), pa, pb


def polyline_distance(left: list[list[float]], right: list[list[float]]) -> tuple[float | None, list[float] | None, list[float] | None]:
    best = None
    for left_start, left_end in zip(left, left[1:]):
        for right_start, right_end in zip(right, right[1:]):
            candidate = _segment_distance(left_start, left_end, right_start, right_end)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best if best is not None else (None, None, None)


# ------------------------------------------------- face-set boundary walk

_DEGENERATE_EDGE_LENGTH = 1.0e-9


class FaceSetOpening:
    """One boundary component of a face set, and what lies across it.

    A feature's own faces form a patch; where that patch stops, its edges are
    shared with faces outside it.  Group those exit edges into connected
    components and each component is one way out of the patch - the mouth of
    a pocket, the base rim of a boss, one end of a slot.  ``far_face_ids``
    are the faces on the other side, minus the ones that *cap* the component
    (a face all of whose outer boundary lies on this rim closes the patch off
    rather than opening it, exactly as ``classify_rim_end`` reads a cylinder's
    end).  ``capped`` says one was found, so a caller can tell "this way out
    leads nowhere" from "this way out leads to more than one owner".
    """

    __slots__ = ("edge_ids", "own_face_ids", "far_face_ids", "capped_face_ids")

    def __init__(self, edge_ids, own_face_ids, far_face_ids, capped_face_ids):
        self.edge_ids = tuple(edge_ids)
        self.own_face_ids = tuple(own_face_ids)
        self.far_face_ids = tuple(far_face_ids)
        self.capped_face_ids = tuple(capped_face_ids)

    @property
    def capped(self) -> bool:
        return bool(self.capped_face_ids)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"FaceSetOpening(edges={len(self.edge_ids)}, "
                f"far={list(self.far_face_ids)}, capped={list(self.capped_face_ids)})")


def face_set_openings(model, own_face_ids) -> list[FaceSetOpening]:
    """Every boundary component of *own_face_ids*, as pure PartModel topology.

    Knows no leaf class and no ontology: it takes a model and a set of face
    ids and answers where that set stops.  The cylindrical-wall rim walk in
    ``features._shared.hole_topology`` answers the same question for the one
    shape it is specialised to; this answers it for a patch of any shape,
    which is what a pocket, a slot or a boss needs.
    """
    own = {str(face_id) for face_id in own_face_ids}
    own = {face_id for face_id in own if face_id in getattr(model, "faces", {})}
    if not own:
        return []

    exits: dict[str, dict[str, set]] = {}
    for face_id in sorted(own):
        for adjacency in getattr(model.faces[face_id], "adjacencies", None) or []:
            neighbour = str(getattr(adjacency, "adjacent_face_id", "") or "")
            edge_id = str(getattr(adjacency, "shared_edge_id", "") or "")
            if not neighbour or not edge_id or neighbour in own:
                continue
            record = exits.setdefault(edge_id, {"own": set(), "far": set()})
            record["own"].add(face_id)
            record["far"].add(neighbour)
    if not exits:
        return []

    # Two exit edges belong to one component when they meet at a vertex.
    by_vertex: dict[str, list[str]] = {}
    for edge_id in exits:
        edge = model.edges.get(edge_id)
        for vertex_id in (getattr(edge, "vertex_ids", None) or []) if edge is not None else []:
            by_vertex.setdefault(str(vertex_id), []).append(edge_id)

    components: list[list[str]] = []
    unvisited = set(exits)
    for start in sorted(exits):
        if start not in unvisited:
            continue
        stack, component = [start], []
        unvisited.discard(start)
        while stack:
            edge_id = stack.pop()
            component.append(edge_id)
            edge = model.edges.get(edge_id)
            for vertex_id in (getattr(edge, "vertex_ids", None) or []) if edge is not None else []:
                for neighbour in by_vertex.get(str(vertex_id), ()):
                    if neighbour in unvisited:
                        unvisited.discard(neighbour)
                        stack.append(neighbour)
        components.append(sorted(component))

    openings: list[FaceSetOpening] = []
    for component in sorted(components):
        edge_set = set(component)
        far = sorted({face_id for edge_id in component for face_id in exits[edge_id]["far"]})
        own_side = sorted({face_id for edge_id in component for face_id in exits[edge_id]["own"]})
        capped = [face_id for face_id in far if _caps_component(model, face_id, edge_set)]
        openings.append(FaceSetOpening(
            edge_ids=component,
            own_face_ids=own_side,
            far_face_ids=[face_id for face_id in far if face_id not in capped],
            capped_face_ids=capped,
        ))
    return openings


def _caps_component(model, face_id: str, edge_ids: set) -> bool:
    """True when the whole outer boundary of *face_id* lies on this component.

    Zero-length apex edges are ignored, the way ``classify_rim_end`` ignores
    the degenerate circle a cone stores at its tip.
    """
    face = model.faces.get(face_id)
    if face is None:
        return False
    saw_outer = False
    for loop_id in getattr(face, "loop_ids", None) or []:
        loop = model.loops.get(loop_id)
        if loop is None or not getattr(loop, "is_outer", False):
            continue
        members = {
            str(edge_id) for edge_id in (loop.edge_ids or [])
            if edge_id in model.edges
            and float(getattr(model.edges[edge_id], "length", 0.0)) > _DEGENERATE_EDGE_LENGTH
        }
        if not members:
            continue
        saw_outer = True
        if not members <= edge_ids:
            return False
    return saw_outer


def _feature_edge(
    source: FeatureNode,
    target: FeatureNode,
    edge_type: str,
    rule_id: str,
    evidence: dict,
    diagnostics: dict | None = None,
    properties: dict | None = None,
) -> FeatureEdge:
    return FeatureEdge(
        edgeId="",
        source=source.nodeId,
        target=target.nodeId,
        edgeType=edge_type,
        directed=False,
        sourceRule=rule_id,
        properties=properties or {},
        evidence=evidence,
        diagnostics=diagnostics or {},
    )


__all__ = [
    "AXIS_LINE_TOL_MM",
    "COAXIAL_ANGLE_TOL_DEG",
    "FaceSetOpening",
    "PARALLEL_ANGLE_TOL_DEG",
    "SPACING_BASIS",
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
    "coaxial_axis_distance",
    "direct_face_pairs",
    "dot",
    "face_set_openings",
    "feature_type",
    "is_coaxial_pair",
    "mean_axis_direction",
    "norm",
    "parallel",
    "parallel_axis_spacing",
    "point_value",
    "polyline_distance",
    "vector",
]
