"""The NX line's implementation of :class:`contract.kernel.KernelServices`.

Alive only inside an NX session: the constructor takes the live handles (UF
session, work part, body) plus the tag-sorted face list that *is* the cad
face id currency (index i == cad id i == ``face_<i>``), exactly as the
PartModel builder assigned it.  The module imports no NX package - handles
come in from the composition root (``journal_fullnx``), which keeps this
importable everywhere and NX-bound nowhere but at runtime.

Method provenance (node 10 §5.3): each measurement names its NX route, so
stored evidence says which kernel produced it.

Refusal remains a first-class answer: anything the session cannot measure
returns the contract's named refusals rather than guessing.
"""

from __future__ import annotations

import math
import sys

from contract.kernel import (
    DISTANCE_MEASURED,
    DISTANCE_SOLVER_EXCEPTION,
    DISTANCE_SOLVER_FAILED,
    FaceMesh,
    FaceSetDistance,
    KernelServices,
    POINT_IN,
    POINT_ON,
    POINT_OUT,
    POINT_UNAVAILABLE,
    POINT_UNKNOWN,
)

_IDENTITY_16 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
_CONTAINMENT = {1: POINT_IN, 2: POINT_OUT, 3: POINT_ON}

# Ceiling on the points one boundary curve may be refined to.  A curve the
# tolerance cannot satisfy - a tiny wiggle in a spline, a degenerate seam -
# stops being refined rather than running away.
EDGE_MAX_POINTS = 512


def _chord_error(start, end, middle) -> float:
    """How far *middle* sits off the straight line from *start* to *end*."""
    ax, ay, az = start
    bx, by, bz = end
    mx, my, mz = middle
    ux, uy, uz = bx - ax, by - ay, bz - az
    length = math.sqrt(ux * ux + uy * uy + uz * uz)
    wx, wy, wz = mx - ax, my - ay, mz - az
    if length <= 1.0e-12:
        # A closed curve arrives as a chord of zero length: the far side of it
        # is the error, which is what makes a full circle subdivide at all.
        return math.sqrt(wx * wx + wy * wy + wz * wz)
    cx = uy * wz - uz * wy
    cy = uz * wx - ux * wz
    cz = ux * wy - uy * wx
    return math.sqrt(cx * cx + cy * cy + cz * cz) / length


def adaptive_curve_points(evaluate, start_t, end_t, deflection, max_points=EDGE_MAX_POINTS):
    """Walk a curve until no chord departs from it by more than *deflection*.

    ``evaluate(t)`` returns a point, or ``None`` where the curve cannot be
    read there.  Subdivision is driven by the sagitta of each chord measured
    at its own parameter midpoint, so a straight edge keeps its two points, a
    tight arc earns many, and nothing here has to know which kind of curve it
    is looking at - which is the point, since a boundary can be a line, an
    arc, an ellipse or a spline and they all have to come out looking right.
    """
    first, last = evaluate(start_t), evaluate(end_t)
    if first is None or last is None:
        return []
    tolerance = max(float(deflection), 1.0e-9)
    walked = [(float(start_t), first), (float(end_t), last)]
    index = 0
    while index < len(walked) - 1 and len(walked) < max_points:
        (left_t, left), (right_t, right) = walked[index], walked[index + 1]
        middle_t = (left_t + right_t) / 2.0
        middle = evaluate(middle_t)
        if middle is None or _chord_error(left, right, middle) <= tolerance:
            index += 1
            continue
        walked.insert(index + 1, (middle_t, middle))
    return [point for _, point in walked]


class NxKernelServices(KernelServices):
    """Geometry services answered by a live NX session against one body."""

    def __init__(self, uf, work, body, cad_faces, model) -> None:
        self._uf = uf
        self._work = work
        self._body = body
        self._cad_faces = list(cad_faces)          # index == cad face id
        self._tag_by_id = {index: face.Tag for index, face in enumerate(self._cad_faces)}
        self._id_by_tag = {face.Tag: index for index, face in enumerate(self._cad_faces)}
        self._model = model

    _PROVENANCE = {
        "face_set_distance": "nx_measure_manager_minimum_distance_pairwise_v1",
        "min_wall_thickness": "nx_trace_a_ray_from_stored_wall_samples_v1",
        "remaining_floor_thickness": "nx_trace_a_ray_from_stored_floor_samples_v1",
    }

    def provenance(self, key: str) -> str:
        return self._PROVENANCE.get(key) or super().provenance(key)

    # --------------------------------------------------------- solid queries

    def supports_point_classification(self) -> bool:
        return True

    def classify_point(self, point: list[float], tolerance: float) -> str:
        try:
            state = self._uf.Modeling.AskPointContainment(
                [float(v) for v in point], self._body.Tag)
        except Exception:
            return POINT_UNAVAILABLE
        return _CONTAINMENT.get(int(state), POINT_UNKNOWN)

    def face_set_distance(self, left_cad_face_ids, right_cad_face_ids) -> FaceSetDistance:
        try:
            unit = self._work.UnitCollection.FindObject("MilliMeter")
            measure = self._work.MeasureManager
            import NXOpen  # runtime-only: this method executes inside NX

            best = None
            witness = None
            for left_id in left_cad_face_ids:
                for right_id in right_cad_face_ids:
                    left_face = self._face(left_id)
                    right_face = self._face(right_id)
                    if left_face is None or right_face is None:
                        continue
                    measured = measure.NewDistance(
                        unit, NXOpen.MeasureManager.MeasureType.Minimum, left_face, right_face)
                    value = float(measured.Value)
                    measured.Dispose()
                    if best is None or value < best:
                        best = value
                        witness = [int(left_id), int(right_id)]
            if best is None or not math.isfinite(best):
                return FaceSetDistance(DISTANCE_SOLVER_FAILED,
                                       error="no measurable face pair in the given sets")
            # NewDistance exposes only the value; the witness pair is proven
            # (it is the argmin over every measured pair), the points are not.
            return FaceSetDistance(
                DISTANCE_MEASURED, distance=best,
                witness_cad_face_ids=witness,
                witness_reason="argmin pair of pairwise NewDistance(minimum) measurements",
            )
        except Exception as exc:
            return FaceSetDistance(DISTANCE_SOLVER_EXCEPTION, error=f"{type(exc).__name__}: {exc}")

    # --------------------------------------------------------------- ray casts

    def _trace(self, origin, direction):
        """Sorted (distance, cad_face_id) hits of one ray against the body."""
        count, hits = self._uf.Modeling.TraceARay(
            1, [self._body.Tag], [float(v) for v in origin],
            [float(v) for v in direction], _IDENTITY_16, 0)
        results = []
        for hit in hits or []:
            point = [float(v) for v in hit.HitPoint]
            distance = math.dist([float(v) for v in origin], point)
            face_id = self._id_by_tag.get(int(getattr(hit, "HitFace", -1)))
            results.append((distance, face_id, point))
        results.sort(key=lambda item: item[0])
        return results

    def remaining_floor_thickness(self, feature_cad_face_ids, bottom_cad_face_ids, direction):
        axis = self._unit(direction)
        if axis is None:
            return None
        feature_ids = {int(v) for v in feature_cad_face_ids}
        best = None
        for bottom_id in sorted(int(v) for v in bottom_cad_face_ids):
            for sample in self._stored_samples(bottom_id):
                origin = sample["point"]
                try:
                    hits = self._trace(origin, axis)
                except Exception:
                    continue
                for distance, face_id, _point in hits:
                    if distance <= 1.0e-6:
                        continue
                    if face_id is not None and face_id in feature_ids:
                        continue
                    if best is None or distance < best:
                        best = distance
                    break
        return best

    def min_wall_thickness(self, feature_cad_face_ids, wall_cad_face_ids, *,
                           material_side: bool = True, diagnostics: dict | None = None):
        feature_ids = {int(v) for v in feature_cad_face_ids}
        sampled = 0
        hit_count = 0
        best = None
        for wall_id in sorted(int(v) for v in wall_cad_face_ids):
            for sample in self._stored_samples(wall_id):
                normal = self._unit(sample.get("normal"))
                if normal is None:
                    continue
                direction = [-v for v in normal] if material_side else normal
                sampled += 1
                try:
                    hits = self._trace(sample["point"], direction)
                except Exception:
                    continue
                for distance, face_id, _point in hits:
                    if distance <= 1.0e-6:
                        continue
                    if face_id is not None and face_id in feature_ids:
                        continue
                    hit_count += 1
                    if best is None or distance < best:
                        best = distance
                    break
        if diagnostics is not None:
            diagnostics.update({
                "method": self.provenance("min_wall_thickness"),
                "sampledRays": sampled,
                "nonSelfHits": hit_count,
                "materialSide": material_side,
            })
        return best

    # ------------------------------------------------------------ tessellation

    def tessellation_bounds(self) -> list[float]:
        boxes = [face.bbox for face in self._model.faces.values() if isinstance(face.bbox, dict)]
        if not boxes:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lows = [min(float(b["min"][i]) for b in boxes) for i in range(3)]
        highs = [max(float(b["max"][i]) for b in boxes) for i in range(3)]
        return [*lows, *highs]

    def tessellate(self, linear_deflection, angular_deflection, edge_deflection) -> list[FaceMesh]:
        uf = self._uf
        parameters = uf.Facet.AskDefaultParameters()
        # The caller's display policy maps onto the UF tolerance knobs; the
        # angular tolerances are radians on both sides.
        try:
            parameters.SpecifySurfaceTolerance = True
            parameters.SurfaceDistTolerance = float(linear_deflection)
            parameters.SurfaceAngularTolerance = float(angular_deflection)
            parameters.SpecifyCurveTolerance = True
            parameters.CurveDistTolerance = float(edge_deflection)
        except Exception:
            pass
        facet_model = uf.Facet.FacetSolid(self._body.Tag, parameters)
        meshes: list[FaceMesh] = []
        oversized_facets = 0
        for cad_id, face in enumerate(self._cad_faces):
            flat_positions: list[float] = []
            flat_normals: list[float] = []
            indices: list[int] = []
            try:
                face_id = uf.Facet.AskFaceIdOfSolidFace(facet_model, face.Tag)
            except Exception:
                face_id = None
            facet_id = -1
            while face_id is not None:
                try:
                    facet_id = uf.Facet.CycleFacetsInFace(facet_model, face_id, facet_id)
                except Exception:
                    break
                if facet_id is None or facet_id < 0:
                    break
                try:
                    vertex_count, positions = uf.Facet.AskVerticesOfFacet(facet_model, facet_id)
                    _, normals = uf.Facet.AskNormalsOfFacet(facet_model, facet_id)
                except Exception:
                    continue
                if vertex_count < 3:
                    continue
                if vertex_count > 3:
                    # The same guard the area reader carries (finding E-08):
                    # emitting the first three corners of a quadrangle draws
                    # half of it and calls that the face.  NX 2506 returns
                    # only triangles here - AskMaxFacetVerts is 3 - so this
                    # counts nothing today; when it does, the count travels
                    # with the mesh rather than the half-facet.
                    oversized_facets += 1
                    continue
                base = len(flat_positions) // 3
                for corner in range(3):
                    flat_positions.extend(float(v) for v in positions[corner])
                    flat_normals.extend(float(v) for v in normals[corner])
                indices.extend((base, base + 1, base + 2))
            meshes.append(FaceMesh(
                cad_face_id=cad_id,
                positions=flat_positions,
                normals=flat_normals,
                indices=indices,
                edge_segments=self._edge_segments(cad_id, edge_deflection),
            ))
        if oversized_facets:
            # Said out loud on the stream the refresh already watches: a mesh
            # that is missing part of a face must not look like a face that
            # small.  Nothing on this install reaches it (E-08).
            print("nx tessellate: %d facet(s) with more than three vertices were not "
                  "drawn; this reader emits triangles only" % oversized_facets,
                  file=sys.stderr)
        return meshes

    def _edge_segments(self, cad_id: int, deflection: float) -> list[float]:
        """Boundary polylines of one face, at the caller's display resolution.

        The curve is walked directly, because the PartModel's nine points per
        edge are a *semantic* sampling - enough for the ontology's geometry
        questions, and deliberately the same nine on both lines - and nine
        points around a circle is an octagon.  The viewer draws these as the
        part's outline, so it needs the display resolution ``edge_deflection``
        was always carrying and this line was ignoring: measured on
        084_818_012_solid_001, every hole rim came out with 8 segments and a
        0.98mm sagitta against the 0.01mm the open-source line gives.

        If the session will not evaluate the curves, the stored samples are
        still published rather than nothing - a coarse outline beats none.
        """
        points_by_edge = self._display_edge_points(cad_id, deflection)
        if points_by_edge:
            segments: list[float] = []
            for points in points_by_edge:
                for start, end in zip(points, points[1:]):
                    segments.extend((*map(float, start), *map(float, end)))
            return segments
        return self._stored_edge_segments(cad_id)

    def _display_edge_points(self, cad_id: int, deflection: float) -> list[list[list[float]]]:
        """Every boundary curve of one face, refined until it looks curved."""
        face = self._face(cad_id)
        if face is None:
            return []
        try:
            edges = list(face.GetEdges())
        except Exception:
            return []
        loops: list[list[list[float]]] = []
        for edge in edges:
            points = self._curve_points(edge, deflection)
            if len(points) >= 2:
                loops.append(points)
        return loops

    def _curve_points(self, edge, deflection: float) -> list[list[float]]:
        """One NX edge, sampled by chord error.  Empty when it cannot be read."""
        uf = self._uf
        try:
            evaluator = uf.Eval.Initialize2(edge.Tag)
        except Exception:
            return []
        try:
            limits = uf.Eval.AskLimits(evaluator)
            start, end = float(limits[0]), float(limits[1])

            def evaluate(t):
                try:
                    return [float(v) for v in uf.Eval.EvaluateUnitVectors(evaluator, t)[0]]
                except Exception:
                    return None

            return adaptive_curve_points(evaluate, start, end, deflection)
        except Exception:
            return []
        finally:
            try:
                uf.Eval.Free(evaluator)
            except Exception:
                pass

    def _stored_edge_segments(self, cad_id: int) -> list[float]:
        """Boundary polylines of one face, from the stored edge samples."""
        face = self._model.faces.get(f"face_{cad_id}")
        if face is None:
            return []
        segments: list[float] = []
        seen: set[str] = set()
        for loop_id in face.loop_ids:
            loop = self._model.loops.get(loop_id)
            if loop is None:
                continue
            for edge_id in loop.edge_ids:
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                edge = self._model.edges.get(edge_id)
                points = (edge.sample_points or []) if edge is not None else []
                for start, end in zip(points, points[1:]):
                    segments.extend((*map(float, start), *map(float, end)))
        return segments

    # ----------------------------------------------------------------- helpers

    def _face(self, cad_id):
        try:
            return self._cad_faces[int(cad_id)]
        except (IndexError, ValueError, TypeError):
            return None

    def _stored_samples(self, cad_id: int) -> list[dict]:
        face = self._model.faces.get(f"face_{int(cad_id)}")
        return list(face.interior_samples) if face is not None else []

    @staticmethod
    def _unit(vector):
        if not isinstance(vector, (list, tuple)) or len(vector) != 3:
            return None
        magnitude = math.sqrt(sum(float(v) ** 2 for v in vector))
        if magnitude <= 1.0e-12:
            return None
        return [float(v) / magnitude for v in vector]
