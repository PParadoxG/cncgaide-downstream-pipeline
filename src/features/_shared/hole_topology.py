"""Pure topological facts used by ontology hole modules.

This module intentionally does not decide whether a candidate is ThroughHole
or BlindHole and does not construct ontology property dictionaries.  Those
feature-specific acceptance decisions stay in their respective modules.

Rim detection is geometry-driven: a *rim* is a complete closed boundary
curve of the cylindrical wall patch that winds once around the wall axis.
Rim curves may be arbitrary STEP curve types (circles on flat end planes,
BSPLINE cylinder/cylinder intersection curves, ...) as long as they lie on
the wall surface.  The projection of such a rim onto any plane perpendicular
to the axis is therefore an approximate circle of the wall radius, which is
exactly the relaxed acceptance rule the hole modules apply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from contract.partmodel import PartModel


@dataclass(frozen=True)
class RimGroup:
    edge_ids: list[str]
    axis_position: float
    sample_points: list[list[float]] = field(default_factory=list)


@dataclass(frozen=True)
class HoleEnd:
    kind: str
    axis_position: float
    face_ids: list[str]


_SURFACE_TOL = 1e-3          # mm — endpoint distance to the wall surface
_BBOX_AXIAL_TOL = 1e-3       # mm — endpoint slack outside the wall axial range
_WINDING_THRESHOLD = 0.9     # |winding| >= this counts as a full turn around the axis
_SEAM_SPAN_TOL = 0.1         # rad — orphan edges with smaller angular span are wall seams
_DEGENERATE_LENGTH_TOL = 1e-9  # mm — zero-length apex edges are ignored when reading loops


def is_reversed_cylindrical_wall(face: object) -> bool:
    return bool(
        getattr(getattr(face, "surface_type", None), "value", None) == "CYLINDER"
        and getattr(face, "is_reversed", False)
    )


def cylindrical_wall_axial_extent(model: PartModel, wall_id: str) -> float | None:
    """Return the depth of a cylindrical wall: ``max(p·axis) - min(p·axis)``.

    The extrema are the ``axial_min`` / ``axial_max`` values persisted on the
    wall face by the STEP parser (computed from the wall's real boundary
    geometry before seam merging).  This is the single depth definition
    shared by ``ThroughHole`` and ``BlindHole`` — bottom geometry and rim
    positions are not involved.  Returns ``None`` when the wall has no
    persisted axial extent.
    """
    wall = model.faces.get(wall_id)
    if wall is None:
        return None
    minimum = wall.parameters.get("axial_min")
    maximum = wall.parameters.get("axial_max")
    if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
        return None
    minimum, maximum = float(minimum), float(maximum)
    if not (math.isfinite(minimum) and math.isfinite(maximum)) or maximum <= minimum:
        return None
    return maximum - minimum


def wall_rims(
    model: PartModel,
    wall_id: str,
    *,
    surface_tolerance: float = _SURFACE_TOL,
    radial_tolerance: float | None = None,
) -> list[RimGroup]:
    """Return the complete closed rims of a cylindrical wall face.

    A rim is a closed boundary curve of the wall patch that winds once
    around the wall axis.  Candidate edges are all model edges that lie on
    the wall surface (both endpoints within *surface_tolerance*), fall
    inside the wall's axial bounding extent, and are not orphaned seam
    connectors between the two wall halves (seams are same-meridian lines
    left over from seam-face merging).  Every simple cycle of the candidate
    graph is then validated by its winding number around the wall axis and
    by how closely its sampled points match the wall radius — this is the
    projection-onto-a-perpendicular-plane ≈ circle acceptance rule.

    The wall is a hole wall only when exactly two such rims exist.
    """
    wall = model.faces.get(wall_id)
    if wall is None:
        return []
    axis = unit_vector(wall.parameters.get("axis_direction"))
    location = wall.parameters.get("axis_location")
    radius = wall.parameters.get("radius")
    if axis is None or not valid_vector(location) or not isinstance(radius, (int, float)) or radius <= 0.0:
        return []
    radius = float(radius)
    radial_tol = radial_tolerance if radial_tolerance is not None else max(1e-2, 2e-3 * radius)
    axial_lo, axial_hi = _bbox_axial_range(wall, axis)
    if axial_lo is None:
        return []
    orphaned = _orphaned_edge_ids(model)
    basis = _perpendicular_basis(axis)

    candidates: dict[str, object] = {}
    for edge_id, edge in model.edges.items():
        points = _edge_endpoints(model, edge)
        if points is None or not _edge_in_axial_range(points, axis, axial_lo, axial_hi):
            continue
        if not all(abs(_distance_to_axis(point, location, axis) - radius) <= surface_tolerance for point in points):
            continue
        if edge_id in orphaned and _edge_angular_span(points, location, axis, basis) <= _SEAM_SPAN_TOL:
            continue
        candidates[edge_id] = edge

    deduped = _dedupe_edges(model, candidates)
    adjacency = _build_vertex_graph(model, deduped)
    cycles = _find_cycles(adjacency)
    if not cycles:
        return []

    rims: list[RimGroup] = []
    for cycle in cycles:
        polyline = _cycle_polyline(model, cycle)
        if len(polyline) < 3:
            continue
        if not all(abs(_distance_to_axis(point, location, axis) - radius) <= radial_tol for point in polyline):
            continue
        if abs(_winding_number(polyline, location, axis)) < _WINDING_THRESHOLD:
            continue
        positions = [dot(point, axis) for point in polyline]
        axis_position = float(sum(positions)) / len(positions)
        rims.append(RimGroup(edge_ids=cycle, axis_position=axis_position, sample_points=polyline))
    return sorted(rims, key=lambda rim: rim.axis_position)


def classify_rim_end(model: PartModel, wall_id: str, rim: RimGroup) -> HoleEnd:
    """Classify one rim as ``open``, ``capped``, or ``unknown``.

    Faces on the far side of the rim are gathered from the wall's own
    adjacency records and from every face loop that geometrically matches a
    rim edge.  A face caps the end only when **all** of its non-degenerate
    *outer* loops consist entirely of rim edges — the face's complete
    boundary then lies on the rim, i.e. the face truly fills the hole
    cross-section.  Zero-length apex edges (e.g. the degenerate circle a
    cone bottom stores at its tip) are ignored, and a loop only needs to be
    a subset of the rim so seam-split copies of the rim (a single half-arc
    inside the cap's loop) still count.

    The all-outer-loops condition matters for chamfer / counterbore cones:
    such a cone contributes only one of its two rim circles to the hole wall
    rim (the other circle borders the part surface), so it must not be
    treated as a cap.  Other far faces — planes with a hole, passage walls,
    mouth planes — make the end open.
    """
    rim_signatures = {_edge_signature(model.edges[edge_id]) for edge_id in rim.edge_ids if edge_id in model.edges}
    if not rim_signatures:
        return HoleEnd("unknown", rim.axis_position, [])

    far_faces: dict[str, set[str]] = {}
    wall = model.faces.get(wall_id)
    if wall is not None:
        for adjacency in wall.adjacencies:
            if adjacency.shared_edge_id in rim.edge_ids:
                far_faces.setdefault(adjacency.adjacent_face_id, set()).add(adjacency.shared_edge_id)
    for face_id, face in model.faces.items():
        if face_id == wall_id:
            continue
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None:
                continue
            for edge_id in loop.edge_ids:
                edge = model.edges.get(edge_id)
                if edge is not None and _edge_signature(edge) in rim_signatures:
                    far_faces.setdefault(face_id, set()).add(edge_id)

    open_faces: list[str] = []
    capped_faces: list[str] = []
    for face_id in sorted(far_faces):
        face = model.faces.get(face_id)
        if face is None:
            continue
        saw_outer = False
        all_outer_on_rim = True
        for loop_id in face.loop_ids:
            loop = model.loops.get(loop_id)
            if loop is None or not loop.is_outer:
                continue
            saw_outer = True
            loop_signatures = {
                _edge_signature(model.edges[edge_id])
                for edge_id in loop.edge_ids
                if edge_id in model.edges and model.edges[edge_id].length > _DEGENERATE_LENGTH_TOL
            }
            if loop_signatures and not (loop_signatures <= rim_signatures):
                all_outer_on_rim = False
                break
        capped = saw_outer and all_outer_on_rim
        if capped:
            capped_faces.append(face_id)
        else:
            open_faces.append(face_id)

    if open_faces and not capped_faces:
        return HoleEnd("open", rim.axis_position, sorted(set(open_faces)))
    if capped_faces and not open_faces:
        return HoleEnd("capped", rim.axis_position, sorted(set(capped_faces)))
    return HoleEnd("unknown", rim.axis_position, [])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def valid_vector(value: object) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)


def unit_vector(value: object) -> list[float] | None:
    if not valid_vector(value):
        return None
    magnitude = math.sqrt(sum(float(item) ** 2 for item in value))
    if magnitude <= 1.0e-12:
        return None
    return [float(item) / magnitude for item in value]


def dot(left: object, right: object) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _distance_to_axis(point: object, location: object, axis: object) -> float:
    delta = [float(point[index]) - float(location[index]) for index in range(3)]
    cross = [
        delta[1] * axis[2] - delta[2] * axis[1],
        delta[2] * axis[0] - delta[0] * axis[2],
        delta[0] * axis[1] - delta[1] * axis[0],
    ]
    return math.sqrt(sum(item * item for item in cross))


def _edge_endpoints(model: PartModel, edge) -> list[list[float]] | None:
    points: list[list[float]] = []
    for vertex_id in edge.vertex_ids:
        vertex = model.vertices.get(vertex_id)
        if vertex is None:
            return None
        points.append([float(value) for value in vertex.coordinates])
    return points if len(points) == 2 else None


def _bbox_axial_range(wall, axis: list[float]) -> tuple[float | None, float | None]:
    bbox = getattr(wall, "bbox", None)
    if not isinstance(bbox, dict) or not isinstance(bbox.get("min"), list) or not isinstance(bbox.get("max"), list):
        return None, None
    lo = 0.0
    hi = 0.0
    for index in range(3):
        low_value = bbox["min"][index]
        high_value = bbox["max"][index]
        if not isinstance(low_value, (int, float)) or not isinstance(high_value, (int, float)):
            return None, None
        component_lo = min(float(low_value) * axis[index], float(high_value) * axis[index])
        component_hi = max(float(low_value) * axis[index], float(high_value) * axis[index])
        lo += component_lo
        hi += component_hi
    return lo - _BBOX_AXIAL_TOL, hi + _BBOX_AXIAL_TOL


def _edge_in_axial_range(points: list[list[float]], axis: list[float], lo: float, hi: float) -> bool:
    return all(lo <= dot(point, axis) <= hi for point in points)


def _edge_signature(edge) -> tuple:
    samples = [list(point) for point in getattr(edge, "sample_points", None) or []]
    if samples:
        rounded = tuple(sorted(tuple(round(float(value), 6) for value in point) for point in samples))
    else:
        rounded = ()
    return (edge.curve_type.value if hasattr(edge.curve_type, "value") else str(edge.curve_type), round(edge.length, 6), rounded)


def _dedupe_edges(model: PartModel, candidates: dict[str, object]) -> dict[str, object]:
    """Drop geometric duplicates — seam-split copies of the same physical curve."""
    seen: set = set()
    result: dict[str, object] = {}
    for edge_id in sorted(candidates):
        signature = _edge_signature(model.edges[edge_id])
        if signature in seen:
            continue
        seen.add(signature)
        result[edge_id] = candidates[edge_id]
    return result


def _build_vertex_graph(model: PartModel, edges: dict[str, object]) -> dict[int, list[tuple[int, str]]]:
    """Build ``node -> [(neighbor_node, edge_id), ...]`` with coordinates merged."""
    node_points: list[list[float]] = []
    node_index: dict[tuple, int] = {}

    def _node_of(point: list[float]) -> int:
        key = (round(point[0], 4), round(point[1], 4), round(point[2], 4))
        node = node_index.get(key)
        if node is None:
            node = len(node_points)
            node_points.append(point)
            node_index[key] = node
        return node

    adjacency: dict[int, list[tuple[int, str]]] = {}
    for edge_id in sorted(edges):
        points = _edge_endpoints(model, model.edges[edge_id])
        if points is None:
            continue
        left = _node_of(points[0])
        right = _node_of(points[1])
        if left == right:
            adjacency.setdefault(left, []).extend([(left, edge_id), (left, edge_id)])
        else:
            adjacency.setdefault(left, []).append((right, edge_id))
            adjacency.setdefault(right, []).append((left, edge_id))
    return adjacency


def _find_cycles(adjacency: dict[int, list[tuple[int, str]]]) -> list[list[str]]:
    """Enumerate all simple cycles of the (small) multigraph.

    Returns ordered edge-id lists; duplicate cycles (same edge set reached
    from different starts) are reported once.
    """
    seen: set[frozenset] = set()
    cycles: list[list[str]] = []

    def _dfs(start: int, node: int, path_edges: list[str], visited: set[int]) -> None:
        for neighbor, edge_id in adjacency.get(node, []):
            if neighbor == start and path_edges and edge_id not in path_edges:
                signature = frozenset([*path_edges, edge_id])
                if signature not in seen:
                    seen.add(signature)
                    cycles.append([*path_edges, edge_id])
                continue
            if neighbor == node and not path_edges:
                signature = frozenset([edge_id])
                if signature not in seen:
                    seen.add(signature)
                    cycles.append([edge_id])
                continue
            if neighbor in visited:
                continue
            visited.add(neighbor)
            path_edges.append(edge_id)
            _dfs(start, neighbor, path_edges, visited)
            path_edges.pop()
            visited.remove(neighbor)

    for start in sorted(adjacency):
        _dfs(start, start, [], {start})
    return cycles


def _cycle_polyline(model: PartModel, edge_ids: list[str]) -> list[list[float]]:
    """Concatenate per-edge sample points into a continuous closed polyline."""
    polyline: list[list[float]] = []
    for index, edge_id in enumerate(edge_ids):
        edge = model.edges.get(edge_id)
        if edge is None:
            return []
        samples = [list(point) for point in getattr(edge, "sample_points", None) or []]
        if not samples:
            points = _edge_endpoints(model, edge)
            if points is None:
                return []
            samples = points
        if polyline:
            distance_forward = math.dist(samples[0], polyline[-1])
            distance_backward = math.dist(samples[-1], polyline[-1])
            if distance_backward < distance_forward:
                samples = list(reversed(samples))
        polyline.extend(samples)
    return polyline


def _winding_number(polyline: list[list[float]], location: object, axis: object) -> float:
    """Winding number of the closed polyline around the axis line (in turns)."""
    basis = _perpendicular_basis(axis)
    angles = []
    for point in polyline:
        angles.append(_point_angle(point, location, axis, basis))
    total = 0.0
    count = len(angles)
    for index in range(count):
        delta = angles[(index + 1) % count] - angles[index]
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        total += delta
    return total / (2.0 * math.pi)


def _perpendicular_basis(axis: object) -> tuple[list[float], list[float]]:
    """Return two orthonormal vectors spanning the plane perpendicular to *axis*."""
    reference = [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0]
    cross = [
        reference[1] * axis[2] - reference[2] * axis[1],
        reference[2] * axis[0] - reference[0] * axis[2],
        reference[0] * axis[1] - reference[1] * axis[0],
    ]
    magnitude = math.sqrt(sum(item * item for item in cross))
    if magnitude <= 1e-12:
        reference = [0.0, 1.0, 0.0]
        cross = [
            reference[1] * axis[2] - reference[2] * axis[1],
            reference[2] * axis[0] - reference[0] * axis[2],
            reference[0] * axis[1] - reference[1] * axis[0],
        ]
        magnitude = math.sqrt(sum(item * item for item in cross))
    u = [item / magnitude for item in cross]
    v = [
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
    ]
    return u, v


def _point_angle(point: object, location: object, axis: object, basis: tuple[list[float], list[float]]) -> float:
    """Angle of *point* around the axis line, in the perpendicular plane."""
    delta = [float(point[index]) - float(location[index]) for index in range(3)]
    axial = dot(delta, axis)
    perpendicular = [
        delta[0] - axial * axis[0],
        delta[1] - axial * axis[1],
        delta[2] - axial * axis[2],
    ]
    u, v = basis
    return math.atan2(dot(perpendicular, v), dot(perpendicular, u))


def _edge_angular_span(points: list[list[float]], location: object, axis: object, basis: tuple[list[float], list[float]]) -> float:
    """Short-way angular difference between the two endpoints around the axis."""
    if len(points) != 2:
        return math.inf
    first = _point_angle(points[0], location, axis, basis)
    last = _point_angle(points[1], location, axis, basis)
    delta = abs(last - first)
    while delta > 2.0 * math.pi:
        delta -= 2.0 * math.pi
    return min(delta, 2.0 * math.pi - delta)


def _orphaned_edge_ids(model: PartModel) -> set[str]:
    """Edges that appear in no face loop and no face adjacency record."""
    referenced: set[str] = set()
    for loop in model.loops.values():
        referenced.update(loop.edge_ids)
    for face in model.faces.values():
        for adjacency in face.adjacencies:
            referenced.add(adjacency.shared_edge_id)
    return set(model.edges.keys()) - referenced
