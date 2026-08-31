"""OCC ray casting for remaining material below feature floors."""

from __future__ import annotations

import math
from typing import Any

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.BRepTools import breptools
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON
from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt, gp_Pnt2d

from occline.kernel.occ_context import OccGeometryContext
from occline.parse.geometry import get_outward_normal


_UV_FRACTIONS = (0.18, 0.34, 0.5, 0.66, 0.82)


def compute_remaining_floor_thickness(
    context: OccGeometryContext,
    feature_cad_face_ids: set[int],
    bottom_cad_face_ids: set[int],
    drilling_direction: object,
    *,
    tolerance: float = 1.0e-7,
) -> float | None:
    """Return the minimum sampled first-hit distance below a feature floor."""
    direction = _unit_vector(drilling_direction)
    if direction is None or not bottom_cad_face_ids:
        return None

    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(context.shape, tolerance)
    epsilon = max(tolerance * 10.0, context.max_ray_distance * 1.0e-9)
    distances: list[float] = []
    for cad_face_id in sorted(bottom_cad_face_ids):
        face = context.cad_face(cad_face_id)
        if face is None:
            continue
        for point in _interior_face_points(face, tolerance):
            origin = [point[index] + direction[index] * epsilon for index in range(3)]
            line = gp_Lin(gp_Pnt(*origin), gp_Dir(*direction))
            intersector.Perform(line, 0.0, context.max_ray_distance)
            if not intersector.IsDone():
                continue
            intersector.SortResult()
            for hit_index in range(1, intersector.NbPnt() + 1):
                distance = float(intersector.WParameter(hit_index)) + epsilon
                if distance <= epsilon:
                    continue
                hit_cad_face_id = context.cad_face_id(intersector.Face(hit_index))
                if hit_cad_face_id is None or hit_cad_face_id in feature_cad_face_ids:
                    continue
                distances.append(distance)
                break
    if not distances:
        return None
    return round(min(distances), 6)


def compute_min_wall_thickness(
    context: OccGeometryContext,
    feature_cad_face_ids: set[int],
    wall_cad_face_ids: set[int],
    *,
    material_side: bool = True,
    tolerance: float = 1.0e-7,
    diagnostics: dict[str, Any] | None = None,
) -> float | None:
    """Sample wall faces and return the nearest non-self material hit.

    ``material_side=True`` is the cutting-feature convention: a solid's
    outward normal points into the void, so the ray travels along its
    opposite.  Boss clearance uses ``False`` and casts into the air along the
    outward normal.  Every hit belonging to the feature is discarded before
    selecting the first other face, which prevents a narrow cavity from
    reporting its own opposite wall as a remaining wall thickness.
    """
    details: dict[str, Any] = {
        "method": "sampled_wall_normals_first_non_self_hit_v1",
        "materialSide": bool(material_side),
        "rayDirection": "opposite_outward_normal" if material_side else "outward_normal",
        "featureCadFaceIds": sorted(int(value) for value in feature_cad_face_ids),
        "wallCadFaceIds": sorted(int(value) for value in wall_cad_face_ids),
        "sampledWallCadFaceIds": [],
        "excludedSelfCadFaceIds": sorted(int(value) for value in feature_cad_face_ids),
        "sampleCount": 0,
        "testedRayCount": 0,
        "intersectorDoneCount": 0,
        "hitCount": 0,
        "selfHitDiscardCount": 0,
        "acceptedHitCount": 0,
        "hitCadFaceIds": [],
        "acceptedHitCadFaceIds": [],
        "minimumDistance": None,
        "status": "not_measured",
    }

    def finish(value: float | None) -> float | None:
        details["sampledWallCadFaceIds"] = sorted(set(details["sampledWallCadFaceIds"]))
        details["hitCadFaceIds"] = sorted(set(details["hitCadFaceIds"]))
        details["acceptedHitCadFaceIds"] = sorted(set(details["acceptedHitCadFaceIds"]))
        if value is not None:
            details["minimumDistance"] = value
            details["status"] = "measured"
        elif context is None:
            details["status"] = "no_occ_context"
        elif not wall_cad_face_ids:
            details["status"] = "no_wall_faces"
        elif details["sampleCount"] == 0:
            details["status"] = "no_interior_wall_samples"
        else:
            details["status"] = "no_other_surface_hit"
        if diagnostics is not None:
            diagnostics.update(details)
        return value

    if context is None or not wall_cad_face_ids:
        return finish(None)
    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(context.shape, tolerance)
    epsilon = max(tolerance * 10.0, context.max_ray_distance * 1.0e-9)
    distances: list[float] = []
    for cad_face_id in sorted(wall_cad_face_ids):
        face = context.cad_face(cad_face_id)
        if face is None:
            continue
        samples = _interior_face_points_with_normals(face, tolerance)
        details["sampledWallCadFaceIds"].append(cad_face_id)
        details["sampleCount"] += len(samples)
        for point, normal in samples:
            details["testedRayCount"] += 1
            sign = -1.0 if material_side else 1.0
            direction = [sign * float(normal[index]) for index in range(3)]
            origin = [point[index] + direction[index] * epsilon for index in range(3)]
            line = gp_Lin(gp_Pnt(*origin), gp_Dir(*direction))
            intersector.Perform(line, 0.0, context.max_ray_distance)
            if not intersector.IsDone():
                continue
            details["intersectorDoneCount"] += 1
            intersector.SortResult()
            for hit_index in range(1, intersector.NbPnt() + 1):
                distance = float(intersector.WParameter(hit_index)) + epsilon
                if distance <= epsilon:
                    continue
                hit_cad_face_id = context.cad_face_id(intersector.Face(hit_index))
                if hit_cad_face_id is None:
                    continue
                details["hitCount"] += 1
                details["hitCadFaceIds"].append(hit_cad_face_id)
                if hit_cad_face_id in feature_cad_face_ids:
                    details["selfHitDiscardCount"] += 1
                    continue
                distances.append(distance)
                details["acceptedHitCount"] += 1
                details["acceptedHitCadFaceIds"].append(hit_cad_face_id)
                break
    if not distances:
        return finish(None)
    return finish(round(min(distances), 6))


def _interior_face_points(face, tolerance: float) -> list[list[float]]:
    try:
        u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    except Exception:
        return []
    bounds = (u_min, u_max, v_min, v_max)
    if not all(math.isfinite(float(value)) for value in bounds):
        return []
    if float(u_max) <= float(u_min) or float(v_max) <= float(v_min):
        return []

    surface = BRepAdaptor_Surface(face)
    classifier = BRepClass_FaceClassifier()
    candidates: list[tuple[float, float, float]] = []
    for u_fraction in _UV_FRACTIONS:
        for v_fraction in _UV_FRACTIONS:
            u = float(u_min) + (float(u_max) - float(u_min)) * u_fraction
            v = float(v_min) + (float(v_max) - float(v_min)) * v_fraction
            classifier.Perform(face, gp_Pnt2d(u, v), tolerance)
            if classifier.State() not in {TopAbs_IN, TopAbs_ON}:
                continue
            point = surface.Value(u, v)
            center_distance = abs(u_fraction - 0.5) + abs(v_fraction - 0.5)
            candidates.append((center_distance, float(point.X()), float(point.Y()), float(point.Z())))
    candidates.sort(key=lambda item: item[0])
    return [[x, y, z] for _, x, y, z in candidates]


def _interior_face_points_with_normals(face, tolerance: float) -> list[tuple[list[float], list[float]]]:
    try:
        u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    except Exception:
        return []
    bounds = (u_min, u_max, v_min, v_max)
    if not all(math.isfinite(float(value)) for value in bounds):
        return []
    if float(u_max) <= float(u_min) or float(v_max) <= float(v_min):
        return []
    surface = BRepAdaptor_Surface(face)
    classifier = BRepClass_FaceClassifier()
    result: list[tuple[list[float], list[float]]] = []
    for u_fraction in _UV_FRACTIONS:
        for v_fraction in _UV_FRACTIONS:
            u = float(u_min) + (float(u_max) - float(u_min)) * u_fraction
            v = float(v_min) + (float(v_max) - float(v_min)) * v_fraction
            try:
                classifier.Perform(face, gp_Pnt2d(u, v), tolerance)
                if classifier.State() not in {TopAbs_IN, TopAbs_ON}:
                    continue
                point = surface.Value(u, v)
                normal = get_outward_normal(face, u, v)
            except Exception:
                continue
            if isinstance(normal, (list, tuple)) and len(normal) == 3:
                vector = [float(normal[index]) for index in range(3)]
            else:
                vector = [float(normal.X()), float(normal.Y()), float(normal.Z())]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm <= 1.0e-12:
                continue
            result.append((
                [float(point.X()), float(point.Y()), float(point.Z())],
                [value / norm for value in vector],
            ))
    return result


def _unit_vector(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if not all(isinstance(component, (int, float)) and math.isfinite(float(component)) for component in value):
        return None
    magnitude = math.sqrt(sum(float(component) ** 2 for component in value))
    if magnitude <= 1.0e-12:
        return None
    return [float(component) / magnitude for component in value]
