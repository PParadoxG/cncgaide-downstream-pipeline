"""The NX line's PartModel builder: NX topology and geometry, as plain data.

Fills the same table the OCC line fills (``contract.partmodel``), from a live
NX session.  The module itself never imports NXOpen — every NX handle (the
UF session, the work part, the body) is passed in, which keeps it importable
and testable anywhere; only the code that *calls* it needs NX.

Identity decisions (user-approved):

* NX's STEP translator sews seam halves on import, so the NX face list IS
  the active layer: face ids are ``face_<i>`` and ``merged_from`` stays empty.
  *What order* ``i`` runs in is decided by :mod:`nxline.identity` — NX's
  persistent object handle, never the tag (node 16; see that module for why
  the tag was not good enough).
* ``is_reversed`` is NX's own answer: ``AskFaceData``'s normDir is +1 when
  the face normal equals the natural (du x dv) surface normal and -1 when
  reversed — the same convention the OCC line's flag encodes.  Because that
  flag is stored, the plane normal beside it has to be the natural one too:
  ``AskFaceData`` returns the already-oriented normal, so it is negated back
  when normDir is -1 (node 21'; the contract now pins the pairing).

Where NX and OCC genuinely differ (edge concavity is classified here by a
local material probe rather than OCC's oriented-edge walk; a cone's apex is
reconstructed from the axis point and half angle), the divergence is a
measurable fact about the two kernels and shows up in the partmodel diff,
not hidden behind an adapter.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from contract.normalize import annotate_face_axial_extents, normalize_seam_faces
from nxline import identity
from contract.partmodel import (
    AdjacencyInfo,
    ConcavityType,
    CurveType,
    EdgeData,
    FaceData,
    LoopData,
    PartMetadata,
    PartModel,
    ShellData,
    SurfaceType,
    VertexData,
)

# NX surface type codes (AskFaceData) -> SurfaceType.
NX_SURFACE_TYPES = {
    22: SurfaceType.PLANE,
    16: SurfaceType.CYLINDER,
    17: SurfaceType.CONE,
    18: SurfaceType.SPHERE,
    19: SurfaceType.REVOLUTION,   # NX: revolved (toroidal)
    20: SurfaceType.EXTRUSION,
    43: SurfaceType.BSPLINE,
}
# A true torus arrives as code 19 with both radii set; 19 without a minor
# radius stays REVOLUTION.

UV_FRACTIONS = (0.2, 0.4, 0.5, 0.6, 0.8)
NUM_CURVE_SAMPLES = 9
SMOOTH_TOLERANCE_DEG = 0.1


def _ordinal(identifier: str) -> int:
    """The number in a ``<kind>_<n>`` id, for sorting by it.

    Finding E-17: this module numbers faces, edges and vertices and then sorts
    them - and a bare ``sorted()`` puts ``edge_10`` before ``edge_2``.  Every
    other place in the line already sorts by the number; the two that did not
    now read this.
    """
    return int(identifier.rsplit("_", 1)[-1])


def _unit(vector):
    magnitude = math.sqrt(sum(float(v) ** 2 for v in vector))
    if magnitude <= 1.0e-12:
        return None
    return [float(v) / magnitude for v in vector]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _handle_of(uf, tag: int) -> str:
    """NX's persistent object handle for *tag*, or ``""`` if it withholds one.

    A refusal is answered with the empty string rather than an exception, so
    the identity layer sees "no handle" as data and can name the degradation.
    """
    try:
        value = uf.Tag.AskHandleOfTag(tag)
    except Exception:
        return ""
    if isinstance(value, (list, tuple)):
        value = "|".join(str(item) for item in value)
    return str(value or "")


# ------------------------------------------------------------------- faces

def _face_parameters(uf_type: int, point, direction, radius, rad_data,
                     norm_dir: int = 1) -> tuple[SurfaceType, dict]:
    surface = NX_SURFACE_TYPES.get(uf_type, SurfaceType.OTHER)
    point = [float(v) for v in point]
    direction = [float(v) for v in direction]
    if surface == SurfaceType.PLANE:
        # ``AskFaceData`` hands back the *oriented* face normal; the contract
        # stores the *natural* surface normal and lets ``is_reversed`` say
        # which way the material is (``contract.partmodel.FaceData``).  So the
        # oriented answer is turned back into the natural one here rather than
        # stored as it came and read some other way downstream: until node 21'
        # the shared relation machinery, which reads the contract, got 181 of
        # this corpus's 268 NX planes pointing the wrong way.
        natural = direction if norm_dir >= 0 else [-v for v in direction]
        return surface, {"normal": natural, "location": point}
    if surface == SurfaceType.CYLINDER:
        return surface, {"radius": float(radius), "axis_direction": direction, "axis_location": point}
    if surface == SurfaceType.CONE:
        half_angle = float(rad_data)
        parameters = {
            "semi_angle": half_angle,
            "axis_direction": direction,
            "axis_location": point,
        }
        # The apex sits where the radius linearly reaches zero along the
        # axis.  AskFaceData gives the radius *at the returned axis point*;
        # the sign of the offset (which way the cone narrows) is not in the
        # data, so the apex is reconstructed on the -axis side by convention
        # and recorded as such - a named approximation, visible in the diff.
        if abs(math.tan(half_angle)) > 1.0e-12 and float(radius) > 0.0:
            offset = float(radius) / math.tan(half_angle)
            parameters["apex"] = [point[i] - direction[i] * offset for i in range(3)]
            parameters["apex_convention"] = "axis_point_minus_axis_times_radius_over_tan_half_angle"
        return surface, parameters
    if surface == SurfaceType.SPHERE:
        return surface, {"radius": float(radius), "center": point}
    if surface == SurfaceType.REVOLUTION:
        if rad_data and float(rad_data) > 0.0:
            return SurfaceType.TORUS, {
                "major_radius": float(radius),
                "minor_radius": float(rad_data),
                "axis": direction,
                "axis_location": point,
            }
        return surface, {"axis_direction": direction, "axis_location": point}
    return surface, {}


def _interior_samples(uf, face_tag: int, norm_dir: int) -> list[dict]:
    """uv-grid samples strictly inside the trim, with outward unit normals."""
    samples: list[dict] = []
    try:
        evaluator = uf.Evalsf.Initialize(face_tag)
    except Exception:
        return samples
    try:
        limits = uf.Evalsf.AskFaceUvMinmax(evaluator)
        u_min, u_max, v_min, v_max = (float(v) for v in limits)
        if not (u_max > u_min and v_max > v_min):
            return samples
        for fu in UV_FRACTIONS:
            for fv in UV_FRACTIONS:
                u = u_min + (u_max - u_min) * fu
                v = v_min + (v_max - v_min) * fv
                try:
                    inside = uf.Modeling.AskUvPointsContainment(1, u, v, face_tag)
                    if not inside or int(inside[0]) != 1:
                        continue
                    result = uf.Evalsf.Evaluate(evaluator, 1, [u, v])
                    position = [float(x) for x in result.SrfPos]
                    natural = _unit(_cross(list(result.SrfDu), list(result.SrfDv)))
                except Exception:
                    continue
                if natural is None:
                    continue
                outward = natural if norm_dir >= 0 else [-x for x in natural]
                samples.append({"point": position, "normal": outward})
    finally:
        try:
            uf.Evalsf.Free(evaluator)
        except Exception:
            pass
    return samples


def _facet_area_centroids(uf, body_tag: int, face_tags: list[int],
                          notes: list[str] | None = None) -> dict[int, tuple[float, list[float]]]:
    """Per face: (facet area, area centroid) from one default-tolerance facet model.

    The reader below takes the first three vertices of each facet, which is
    the whole facet only while NX returns triangles.  ``AskVerticesOfFacet``
    is documented to return four for a quadrangle, and there is no parameter
    that forbids one, so a quadrangle would silently cost this face half of
    every such facet's area and pull its centroid (finding E-08).  Measured on
    this install - NX 2506.4021, ``AskMaxFacetVerts`` = 3 over the
    084_818_012_solid_001 body, 654 facets across 55 faces, every one a
    triangle - so the guard below has never fired; it is here so that the day
    it does, the face says so instead of quietly measuring short.  *notes*
    collects those statements for the model's normalization log.
    """
    result: dict[int, tuple[float, list[float]]] = {}
    try:
        parameters = uf.Facet.AskDefaultParameters()
        model = uf.Facet.FacetSolid(body_tag, parameters)
    except Exception:
        return result
    for tag in face_tags:
        try:
            face_id = uf.Facet.AskFaceIdOfSolidFace(model, tag)
        except Exception:
            continue
        total = 0.0
        moment = [0.0, 0.0, 0.0]
        oversized = 0
        widest = 0
        facet_id = -1
        while True:
            try:
                facet_id = uf.Facet.CycleFacetsInFace(model, face_id, facet_id)
            except Exception:
                break
            if facet_id is None or facet_id < 0:
                break
            try:
                count, vertices = uf.Facet.AskVerticesOfFacet(model, facet_id)
            except Exception:
                continue
            if count < 3:
                continue
            if count > 3:
                # Not measured, not guessed: taking the first three would
                # report a fraction of this facet as the whole of it.
                oversized += 1
                widest = max(widest, int(count))
                continue
            a, b, c = vertices[0], vertices[1], vertices[2]
            ab = [b[i] - a[i] for i in range(3)]
            ac = [c[i] - a[i] for i in range(3)]
            normal = _cross(ab, ac)
            area = 0.5 * math.sqrt(sum(v * v for v in normal))
            if area <= 0.0:
                continue
            centre = [(a[i] + b[i] + c[i]) / 3.0 for i in range(3)]
            total += area
            for i in range(3):
                moment[i] += area * centre[i]
        if oversized and notes is not None:
            notes.append(
                "FACET GAP face tag %d: %d facet(s) carry more than three vertices "
                "(widest %d); this reader measures triangles only, so their area and "
                "moment are left out rather than counted short" % (tag, oversized, widest)
            )
        if total > 0.0:
            result[tag] = (total, [moment[i] / total for i in range(3)])
    return result


# ------------------------------------------------------------------- edges

def _edge_geometry(uf, edge) -> tuple[CurveType, dict, list[list[float]]]:
    """Curve type, ontology-shaped parameters and 9 uniform sample points."""
    curve_type = CurveType.OTHER
    parameters: dict = {}
    points: list[list[float]] = []
    try:
        evaluator = uf.Eval.Initialize2(edge.Tag)
    except Exception:
        return curve_type, parameters, points
    try:
        limits = uf.Eval.AskLimits(evaluator)
        t0, t1 = float(limits[0]), float(limits[1])
        try:
            if uf.Eval.IsLine(evaluator):
                curve_type = CurveType.LINE
                line = uf.Eval.AskLine(evaluator)
                unit = _unit(list(line.Unit))
                if unit is not None:
                    parameters["direction"] = unit
            elif uf.Eval.IsArc(evaluator):
                curve_type = CurveType.CIRCLE
                arc = uf.Eval.AskArc(evaluator)
                normal = _unit(_cross(list(arc.XAxis), list(arc.YAxis)))
                parameters["radius"] = float(arc.Radius)
                parameters["center"] = [float(v) for v in arc.Center]
                if normal is not None:
                    parameters["normal"] = normal
            elif uf.Eval.IsEllipse(evaluator):
                curve_type = CurveType.ELLIPSE
                ellipse = uf.Eval.AskEllipse(evaluator)
                for name, key in (("MajorRadius", "major_radius"), ("MinorRadius", "minor_radius"), ("Center", "center")):
                    value = getattr(ellipse, name, None)
                    if value is not None:
                        parameters[key] = [float(v) for v in value] if hasattr(value, "__len__") else float(value)
            elif uf.Eval.IsSpline(evaluator):
                curve_type = CurveType.BSPLINE
        except Exception:
            pass
        for index in range(NUM_CURVE_SAMPLES):
            t = t0 + (t1 - t0) * index / (NUM_CURVE_SAMPLES - 1)
            try:
                evaluated = uf.Eval.EvaluateUnitVectors(evaluator, t)
                point = evaluated[0]
                points.append([float(v) for v in point])
            except Exception:
                continue
    finally:
        try:
            uf.Eval.Free(evaluator)
        except Exception:
            pass
    return curve_type, parameters, points


def _edge_tangent(uf, edge, at_fraction: float = 0.5) -> list[float] | None:
    try:
        evaluator = uf.Eval.Initialize2(edge.Tag)
    except Exception:
        return None
    try:
        limits = uf.Eval.AskLimits(evaluator)
        t = float(limits[0]) + (float(limits[1]) - float(limits[0])) * at_fraction
        evaluated = uf.Eval.EvaluateUnitVectors(evaluator, t)
        return _unit([float(v) for v in evaluated[1]])
    except Exception:
        return None
    finally:
        try:
            uf.Eval.Free(evaluator)
        except Exception:
            pass


# ------------------------------------------------------------------- loops

def _reconstruct_loops(face_index: int, edge_ids: list[str], edges: dict[str, EdgeData]) -> list[LoopData]:
    """Group a face's edges into closed cycles by shared vertices.

    The binding exposes no loop query, so loops are rebuilt from vertex
    connectivity: repeatedly walk from an unused edge to whichever unused
    edge shares its open endpoint.  A periodic edge (both endpoints equal,
    or none) is its own loop.  The loop whose sample points span the largest
    box is called outer - the same answer OCC's OuterWire gives on every
    sane face, and a divergence shows up in the diff, not in a crash.
    """
    remaining = list(edge_ids)
    loops: list[LoopData] = []
    while remaining:
        current = remaining.pop(0)
        cycle = [current]
        edge = edges[current]
        if len(set(edge.vertex_ids)) < 2:
            loops.append(LoopData(id="", edge_ids=cycle))
            continue
        start, cursor = edge.vertex_ids[0], edge.vertex_ids[-1]
        guard = 0
        while cursor != start and guard < 10000:
            guard += 1
            follow = None
            for candidate in remaining:
                vertex_ids = edges[candidate].vertex_ids
                if cursor in vertex_ids:
                    follow = candidate
                    break
            if follow is None:
                break
            remaining.remove(follow)
            cycle.append(follow)
            vertex_ids = edges[follow].vertex_ids
            cursor = vertex_ids[-1] if vertex_ids[0] == cursor else vertex_ids[0]
        loops.append(LoopData(id="", edge_ids=cycle))

    def loop_extent(loop: LoopData) -> float:
        points = [p for eid in loop.edge_ids for p in (edges[eid].sample_points or [])]
        if not points:
            return 0.0
        lows = [min(p[i] for p in points) for i in range(3)]
        highs = [max(p[i] for p in points) for i in range(3)]
        return math.dist(lows, highs)

    outer = max(range(len(loops)), key=lambda i: loop_extent(loops[i])) if loops else 0
    for index, loop in enumerate(loops):
        loop.id = f"loop_{face_index}_{index}"
        loop.is_outer = index == outer
    return loops


# --------------------------------------------------------------- concavity

def _face_normal_near(uf, face_tag: int, norm_dir: int, point) -> list[float] | None:
    """Outward face normal near *point*, via the surface evaluator."""
    try:
        evaluator = uf.Evalsf.Initialize(face_tag)
    except Exception:
        return None
    try:
        uv = None
        try:
            closest = uf.Evalsf.FindClosestPoint(evaluator, [float(v) for v in point])
            for attr in ("Uv", "SrfUv", "uv"):
                value = getattr(closest, attr, None)
                if value is not None and len(value) == 2:
                    uv = [float(value[0]), float(value[1])]
                    break
        except Exception:
            uv = None
        if uv is None:
            limits = uf.Evalsf.AskFaceUvMinmax(evaluator)
            uv = [(float(limits[0]) + float(limits[1])) / 2.0,
                  (float(limits[2]) + float(limits[3])) / 2.0]
        result = uf.Evalsf.Evaluate(evaluator, 1, uv)
        natural = _unit(_cross(list(result.SrfDu), list(result.SrfDv)))
        if natural is None:
            return None
        return natural if norm_dir >= 0 else [-v for v in natural]
    except Exception:
        return None
    finally:
        try:
            uf.Evalsf.Free(evaluator)
        except Exception:
            pass


def _classify_concavity(uf, body_tag: int, edge, edge_data: EdgeData,
                        face_a: FaceData, tag_a: int, norm_a: int,
                        face_b: FaceData, tag_b: int, norm_b: int,
                        probe_step: float) -> ConcavityType:
    """Convex / concave / smooth for the edge between two faces.

    Smoothness is NX's own verdict (``AskEdgeSmoothness``).  The sharp cases
    are decided by a material probe: from the edge midpoint, step along the
    bisector of the two *in-face* directions (tangent to each face, away
    from the edge); that bisector leaves the material exactly on a concave
    edge and stays inside it on a convex one.
    """
    try:
        if uf.Modeling.AskEdgeSmoothness(edge.Tag, SMOOTH_TOLERANCE_DEG):
            return ConcavityType.SMOOTH
    except Exception:
        pass
    samples = edge_data.sample_points or []
    if not samples:
        return ConcavityType.UNKNOWN
    midpoint = samples[len(samples) // 2]
    tangent = _edge_tangent(uf, edge)
    normal_a = _face_normal_near(uf, tag_a, norm_a, midpoint)
    normal_b = _face_normal_near(uf, tag_b, norm_b, midpoint)
    if tangent is None or normal_a is None or normal_b is None:
        return ConcavityType.UNKNOWN

    def in_face_direction(normal: list[float], face: FaceData) -> list[float] | None:
        candidate = _unit(_cross(normal, tangent))
        if candidate is None:
            return None
        anchor = face.centroid or (
            [(face.bbox["min"][i] + face.bbox["max"][i]) / 2.0 for i in range(3)]
            if isinstance(face.bbox, dict) else None
        )
        if anchor is not None:
            toward = [anchor[i] - midpoint[i] for i in range(3)]
            if sum(candidate[i] * toward[i] for i in range(3)) < 0.0:
                candidate = [-v for v in candidate]
        return candidate

    direction_a = in_face_direction(normal_a, face_a)
    direction_b = in_face_direction(normal_b, face_b)
    if direction_a is None or direction_b is None:
        return ConcavityType.UNKNOWN
    bisector = _unit([direction_a[i] + direction_b[i] for i in range(3)])
    if bisector is None:
        return ConcavityType.UNKNOWN
    probe = [midpoint[i] + bisector[i] * probe_step for i in range(3)]
    try:
        state = uf.Modeling.AskPointContainment(probe, body_tag)
    except Exception:
        return ConcavityType.UNKNOWN
    if state == 1:
        return ConcavityType.CONVEX
    if state == 2:
        return ConcavityType.CONCAVE
    return ConcavityType.UNKNOWN


# ------------------------------------------------------------------ builder

def part_length_unit(work, units_enum=None) -> str:
    """NX's own name for the part's units - ``MILLIMETERS``, ``INCHES``, ``""``.

    ``BasePart.PartUnits`` is an enumeration member, and what the binding hands
    back is a property of the binding rather than of the part.  On NX 2506 it
    is an opaque ``BasePartUnitsMemberType`` whose ``str()`` is ``'1'`` and
    which carries no ``name`` (probed:
    ``NODE_REPORT/Node28_scripts/d/nx_probe_part_units.py``).  So the name is
    asked of **NX's own enumeration**: *units_enum* is ``NXOpen.BasePart.Units``,
    and the member that ``is`` this one names it.  Nothing is inferred from the
    ordinal - an integer whose meaning is a version of NXOpen is not a unit -
    so an unnamed member, a missing enumeration or a session that refuses all
    give ``""``.

    The enumeration is a parameter because this module never imports NXOpen:
    every NX handle is passed in, which is what keeps it importable and
    testable outside a session.

    NX's word, not the STEP file's - the two lines record the unit each of them
    was told, in the vocabulary each of them speaks, and neither is translated
    into the other's.  Nothing computes with it either way.
    """
    try:
        units = work.PartUnits
    except Exception:
        return ""
    if units is None:
        return ""
    text = (getattr(units, "name", None)
            or _enum_member_name(units, units_enum)
            or str(units))
    words = re.findall(r"[A-Za-z]+", str(text))
    if not words or words[-1].lower() == "units":
        return ""
    return words[-1].upper()


def _enum_member_name(member, units_enum) -> str:
    """Which attribute of *units_enum* is *member*, by identity then equality."""
    if units_enum is None:
        return ""
    names = [name for name in dir(units_enum) if not name.startswith("_")]
    for name in names:
        try:
            if getattr(units_enum, name) is member:
                return name
        except Exception:
            continue
    for name in names:
        try:
            if getattr(units_enum, name) == member:
                return name
        except Exception:
            continue
    return ""


def build_partmodel(uf, work, body, *, source_file: str, part_name: str,
                    part_units_enum=None) -> tuple[PartModel, list]:
    """One NX body -> a filled ``PartModel`` plus its face list in id order.

    The returned face list is the id currency: index ``i`` is cad face id
    ``i`` is ``face_<i>``, for the kernel services and the viewer alike.

    Three phases, and the split is the whole point of node 16: everything the
    kernel is asked is gathered first, the id order is then decided by a pure
    function of that gathering (:mod:`nxline.identity`), and only afterwards
    is the table filled.  No id may depend on the order NX happened to
    enumerate things in - that order is a property of the session, not of the
    solid.
    """
    identity_notes: list[str] = []

    # ---- phase 1a: every face, measured.  Order irrelevant here by design.
    face_objects = list(body.GetFaces())
    face_data_raw = {fc.Tag: uf.Modeling.AskFaceData(fc.Tag) for fc in face_objects}

    # -- measured area/perimeter units
    try:
        unit_mm = work.UnitCollection.FindObject("MilliMeter")
        unit_mm2 = work.UnitCollection.FindObject("SquareMilliMeter")
        measure = work.MeasureManager
    except Exception:
        unit_mm = unit_mm2 = measure = None

    facet_moments = _facet_area_centroids(uf, body.Tag, [fc.Tag for fc in face_objects],
                                          identity_notes)

    gathered_faces: list[dict] = []
    for fc in face_objects:
        raw = face_data_raw[fc.Tag]
        uf_type, point, direction, box, radius, rad_data, norm_dir = (
            raw[0], list(raw[1]), list(raw[2]), list(raw[3]), raw[4], raw[5], int(raw[6]))
        surface, parameters = _face_parameters(uf_type, point, direction, radius, rad_data, norm_dir)
        area = None
        if measure is not None:
            try:
                measured = measure.NewFaceProperties(unit_mm2, unit_mm, 0.99, [fc])
                area = float(measured.Area)
                measured.Dispose()
            except Exception:
                area = None
        moments = facet_moments.get(fc.Tag)
        if area is None:
            area = moments[0] if moments else 0.0
        centroid = list(moments[1]) if moments else None
        bbox = {"min": [float(v) for v in box[:3]], "max": [float(v) for v in box[3:]]}
        gathered_faces.append({
            "object": fc,
            "handle": _handle_of(uf, fc.Tag),
            "surface": surface,
            "parameters": parameters,
            "area": area,
            "centroid": centroid,
            "bbox": bbox,
            "norm_dir": norm_dir,
            "interior_samples": _interior_samples(uf, fc.Tag, norm_dir),
            "key": identity.face_key(surface, area, centroid, bbox["min"], bbox["max"], parameters),
        })

    # ---- phase 1b: every edge, deduplicated across the faces that share it
    edge_objects: dict[int, object] = {}
    for entry in gathered_faces:
        for edge in entry["object"].GetEdges():
            edge_objects.setdefault(edge.Tag, edge)

    gathered_edges: list[dict] = []
    for tag, edge in edge_objects.items():
        curve_type, parameters, samples = _edge_geometry(uf, edge)
        try:
            p1, p2, vertex_count = uf.Modeling.AskEdgeVerts(edge.Tag)
        except Exception:
            p1 = p2 = None
            vertex_count = 0
        endpoints: list[list[float]] = []
        if vertex_count >= 2 and p1 is not None and p2 is not None:
            endpoints = [[float(v) for v in p1], [float(v) for v in p2]]
        elif vertex_count == 1 and p1 is not None:
            # A closed periodic edge with one topological vertex: the OCC
            # line records it as [v, v] (start == end), and the rim-cycle
            # walk depends on that shape.
            point = [float(v) for v in p1]
            endpoints = [point, list(point)]
        elif samples:
            # NX reports zero vertices for a full circle; synthesize the
            # start/end vertex from the first curve sample, same shape.
            point = [float(v) for v in samples[0]]
            endpoints = [point, list(point)]
        try:
            length = float(edge.GetLength())
        except Exception:
            length = 0.0
        gathered_edges.append({
            "object": edge,
            "tag": tag,
            "handle": _handle_of(uf, tag),
            "curve_type": curve_type,
            "parameters": parameters,
            "samples": samples,
            "endpoints": endpoints,
            "length": length,
            "key": identity.edge_key(curve_type, length, endpoints, parameters),
        })

    # ---- phase 2: the ids.  A pure function of what phase 1 gathered.
    edge_order, _edge_mode, notes = identity.assign_identity(gathered_edges)
    identity_notes.extend(notes)
    face_order, _face_mode, notes = identity.assign_identity(gathered_faces)
    identity_notes.extend(notes)
    gathered_edges = [gathered_edges[index] for index in edge_order]
    gathered_faces = [gathered_faces[index] for index in face_order]
    cad_faces = [entry["object"] for entry in gathered_faces]
    identity_notes.insert(0, f"identity: faces numbered by {_face_mode}, edges by {_edge_mode}")

    # One representative per location, chosen from the coincident group rather
    # than from whichever edge reached it first: that choice used to move with
    # the edge order and left the stored coordinate a ULP apart between runs.
    vertex_id_by_key, vertex_coordinates = identity.vertex_table(
        [point for entry in gathered_edges for point in entry["endpoints"]])
    vertices: dict[str, VertexData] = {
        vertex_id: VertexData(id=vertex_id, coordinates=coordinates)
        for vertex_id, coordinates in vertex_coordinates.items()
    }

    # ---- phase 3: fill the table in id order
    edges: dict[str, EdgeData] = {}
    edge_id_by_tag: dict[int, str] = {}
    for index, entry in enumerate(gathered_edges):
        edge_id = f"edge_{index}"
        edge_id_by_tag[entry["tag"]] = edge_id
        edges[edge_id] = EdgeData(
            id=edge_id,
            curve_type=entry["curve_type"],
            parameters=entry["parameters"],
            length=entry["length"],
            vertex_ids=[vertex_id_by_key[identity.quantize_point(point)]
                        for point in entry["endpoints"]],
            is_seam=False,
            sample_points=entry["samples"],
        )

    # -- faces with loops
    faces: dict[str, FaceData] = {}
    face_id_by_tag: dict[int, str] = {}
    loops: dict[str, LoopData] = {}
    for index, entry in enumerate(gathered_faces):
        fc = entry["object"]
        face_id = f"face_{index}"
        face_id_by_tag[fc.Tag] = face_id
        face_edge_ids = sorted(
            {edge_id_by_tag[e.Tag] for e in fc.GetEdges() if e.Tag in edge_id_by_tag},
            key=_ordinal)
        face_loops = _reconstruct_loops(index, face_edge_ids, edges)
        for loop in face_loops:
            loops[loop.id] = loop
        faces[face_id] = FaceData(
            id=face_id,
            surface_type=entry["surface"],
            parameters=entry["parameters"],
            area=entry["area"],
            is_outer=True,
            is_reversed=entry["norm_dir"] < 0,
            loop_ids=[loop.id for loop in face_loops],
            bbox=entry["bbox"],
            adjacencies=[],
            centroid=entry["centroid"],
            interior_samples=entry["interior_samples"],
        )

    # -- adjacency with concavity, walked in edge-id order so the adjacency
    #    lists come out the same way twice
    lows = [min(faces[f].bbox["min"][i] for f in faces) for i in range(3)] if faces else [0.0] * 3
    highs = [max(faces[f].bbox["max"][i] for f in faces) for i in range(3)] if faces else [0.0] * 3
    probe_step = max(1.0e-3, 1.0e-3 * math.dist(lows, highs))
    for entry in gathered_edges:
        edge, tag = entry["object"], entry["tag"]
        try:
            attached = [fc for fc in edge.GetFaces() if fc.Tag in face_id_by_tag]
        except Exception:
            continue
        if len(attached) != 2:
            continue
        # Which of the two NX names first is not a property of the edge.
        attached.sort(key=lambda fc: _ordinal(face_id_by_tag[fc.Tag]))
        fa, fb = attached
        id_a, id_b = face_id_by_tag[fa.Tag], face_id_by_tag[fb.Tag]
        norm_a = int(face_data_raw[fa.Tag][6])
        norm_b = int(face_data_raw[fb.Tag][6])
        concavity = _classify_concavity(
            uf, body.Tag, edge, edges[edge_id_by_tag[tag]],
            faces[id_a], fa.Tag, norm_a, faces[id_b], fb.Tag, norm_b, probe_step)
        shared = edge_id_by_tag[tag]
        faces[id_a].adjacencies.append(AdjacencyInfo(adjacent_face_id=id_b, shared_edge_id=shared, connection_type=concavity))
        faces[id_b].adjacencies.append(AdjacencyInfo(adjacent_face_id=id_a, shared_edge_id=shared, connection_type=concavity))

    shell = ShellData(id="shell_0", face_ids=sorted(faces, key=_ordinal),
                      edge_ids=sorted(edges, key=_ordinal),
                      vertex_ids=sorted(vertices, key=_ordinal))

    model = PartModel(
        metadata=PartMetadata(
            source_file=source_file,
            ap_version="AP242 via NX step242",
            part_name=part_name,
            extraction_timestamp=datetime.now(timezone.utc).isoformat(),
            geometry_provider="nxline",
            length_unit=part_length_unit(work, part_units_enum),
        ),
        shells=[shell],
        faces=faces,
        edges=edges,
        loops=loops,
        vertices=vertices,
        normalization_log=list(identity_notes),
    )
    model = annotate_face_axial_extents(model)
    model = normalize_seam_faces(model)
    return model, cad_faces
