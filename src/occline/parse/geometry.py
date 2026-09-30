"""Geometric parameter extraction for faces, edges, and vertices.

Uses OCCT adaptor classes (``BRepAdaptor_Surface``, ``BRepAdaptor_Curve``)
and property computation (``BRepGProp``).  All vectors and points are
returned as ``list[float]`` — never OCCT geometry objects.
"""

import logging
import math

import numpy as np

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Geom import Geom_SurfaceOfLinearExtrusion, Geom_SurfaceOfRevolution
from OCC.Core.GeomAdaptor import GeomAdaptor_Curve
from OCC.Core.GeomAbs import (
    GeomAbs_BezierCurve,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Line,
    GeomAbs_OffsetCurve,
    GeomAbs_OtherCurve,
    GeomAbs_OtherSurface,
    GeomAbs_Parabola,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Vertex

from contract.partmodel import CurveType, SurfaceType

logger = logging.getLogger(__name__)

# Tolerance constants for surface classification
_PARALLEL_TOL = 1e-4   # radians — angle below which directions are considered parallel
_ON_AXIS_TOL = 1e-4    # mm — distance below which a point is considered on-axis

# ---------------------------------------------------------------------------
# GeomAbs → model enum mappings
# ---------------------------------------------------------------------------

_SURFACE_TYPE_MAP: dict[int, SurfaceType] = {
    GeomAbs_Plane:              SurfaceType.PLANE,
    GeomAbs_Cylinder:           SurfaceType.CYLINDER,
    GeomAbs_Cone:               SurfaceType.CONE,
    GeomAbs_Sphere:             SurfaceType.SPHERE,
    GeomAbs_Torus:              SurfaceType.TORUS,
    GeomAbs_BSplineSurface:     SurfaceType.BSPLINE,
    GeomAbs_BezierSurface:      SurfaceType.BSPLINE,
    GeomAbs_SurfaceOfExtrusion: SurfaceType.OTHER,
    GeomAbs_SurfaceOfRevolution: SurfaceType.OTHER,
    GeomAbs_OtherSurface:       SurfaceType.OTHER,
}

_CURVE_TYPE_MAP: dict[int, CurveType] = {
    GeomAbs_Line:         CurveType.LINE,
    GeomAbs_Circle:       CurveType.CIRCLE,
    GeomAbs_Ellipse:      CurveType.ELLIPSE,
    GeomAbs_BSplineCurve: CurveType.BSPLINE,
    GeomAbs_BezierCurve:  CurveType.BSPLINE,
    GeomAbs_Hyperbola:    CurveType.OTHER,
    GeomAbs_Parabola:     CurveType.OTHER,
    GeomAbs_OffsetCurve:  CurveType.OTHER,
    GeomAbs_OtherCurve:   CurveType.OTHER,
}

_GEOMABS_SURFACE_NAMES: dict[int, str] = {
    GeomAbs_Plane: "GeomAbs_Plane",
    GeomAbs_Cylinder: "GeomAbs_Cylinder",
    GeomAbs_Cone: "GeomAbs_Cone",
    GeomAbs_Sphere: "GeomAbs_Sphere",
    GeomAbs_Torus: "GeomAbs_Torus",
    GeomAbs_BSplineSurface: "GeomAbs_BSplineSurface",
    GeomAbs_BezierSurface: "GeomAbs_BezierSurface",
    GeomAbs_SurfaceOfExtrusion: "GeomAbs_SurfaceOfExtrusion",
    GeomAbs_SurfaceOfRevolution: "GeomAbs_SurfaceOfRevolution",
    GeomAbs_OtherSurface: "GeomAbs_OtherSurface",
}

_GEOMABS_CURVE_NAMES: dict[int, str] = {
    GeomAbs_Line: "GeomAbs_Line",
    GeomAbs_Circle: "GeomAbs_Circle",
    GeomAbs_Ellipse: "GeomAbs_Ellipse",
    GeomAbs_Hyperbola: "GeomAbs_Hyperbola",
    GeomAbs_Parabola: "GeomAbs_Parabola",
    GeomAbs_BSplineCurve: "GeomAbs_BSplineCurve",
    GeomAbs_BezierCurve: "GeomAbs_BezierCurve",
    GeomAbs_OffsetCurve: "GeomAbs_OffsetCurve",
    GeomAbs_OtherCurve: "GeomAbs_OtherCurve",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_list(gp_obj) -> list[float]:
    """Convert a *gp_Pnt* or *gp_Dir* to ``[x, y, z]``."""
    return [gp_obj.X(), gp_obj.Y(), gp_obj.Z()]


def _axis_direction(ax1) -> list[float]:
    """Extract direction vector from a *gp_Ax1*."""
    return _to_list(ax1.Direction())


def _axis_location(ax1) -> list[float]:
    """Extract location point from a *gp_Ax1*."""
    return _to_list(ax1.Location())


# ---------------------------------------------------------------------------
# Surface-of-Revolution / Surface-of-Extrusion classification helpers
# ---------------------------------------------------------------------------

def _classify_basis_curve(basis_curve, revolution_axis) -> tuple[str, dict]:
    """Classify the generatrix curve of a surface of revolution.

    Returns ``(label, params_dict)`` where *label* is one of
    ``'CYLINDER'``, ``'CONE'``, ``'SPHERE'``, ``'TORUS'`` (successful
    reduction to an elementary surface) or ``'REVOLUTION'`` (non-elementary
    generatrix — keep as revolution surface).
    """
    curve_adaptor = GeomAdaptor_Curve(basis_curve)
    curve_type = curve_adaptor.GetType()

    axis_loc = revolution_axis.Location()
    axis_dir = revolution_axis.Direction()
    axis_dir_vec = [axis_dir.X(), axis_dir.Y(), axis_dir.Z()]
    axis_loc_vec = [axis_loc.X(), axis_loc.Y(), axis_loc.Z()]

    # ── Linear generatrix ──────────────────────────────────────────────
    if curve_type == GeomAbs_Line:
        line = curve_adaptor.Line()
        line_dir = line.Direction()
        angle = axis_dir.Angle(line_dir)
        angle = min(angle, math.pi - angle)

        if angle < _PARALLEL_TOL:
            # Generatrix parallel to axis → CYLINDER
            line_loc = line.Location()
            pt = np.array([line_loc.X(), line_loc.Y(), line_loc.Z()])
            ax_pt = np.array(axis_loc_vec)
            ax_dir_np = np.array(axis_dir_vec)
            diff = pt - ax_pt
            radius = float(np.linalg.norm(diff - np.dot(diff, ax_dir_np) * ax_dir_np))
            return 'CYLINDER', {
                'radius': radius,
                'axis_direction': axis_dir_vec,
                'axis_location': axis_loc_vec,
            }
        else:
            # Generatrix at an angle to axis → CONE
            line_loc = line.Location()
            P0 = np.array(axis_loc_vec)
            d1 = np.array(axis_dir_vec)
            P1 = np.array([line_loc.X(), line_loc.Y(), line_loc.Z()])
            d2 = np.array([line_dir.X(), line_dir.Y(), line_dir.Z()])
            w = P0 - P1
            b = float(np.dot(d1, d2))
            d = float(np.dot(d1, d1))
            e = float(np.dot(d2, d2))
            f = float(np.dot(d1, w))
            g = float(np.dot(d2, w))
            denom = d * e - b * b
            if abs(denom) < 1e-12:
                return 'REVOLUTION', _revolution_fallback_params(
                    revolution_axis, basis_curve, curve_adaptor)
            t = (b * g - e * f) / denom
            apex_pt = P0 + t * d1
            return 'CONE', {
                'semi_angle': float(angle),
                'axis_direction': axis_dir_vec,
                'apex': [float(apex_pt[0]), float(apex_pt[1]), float(apex_pt[2])],
            }

    # ── Circular generatrix ────────────────────────────────────────────
    elif curve_type == GeomAbs_Circle:
        circle = curve_adaptor.Circle()
        center = circle.Location()
        radius = circle.Radius()

        cx = np.array([center.X(), center.Y(), center.Z()])
        ax_pt = np.array(axis_loc_vec)
        ax_dir_np = np.array(axis_dir_vec)
        diff = cx - ax_pt
        dist_to_axis = float(np.linalg.norm(
            diff - np.dot(diff, ax_dir_np) * ax_dir_np))

        if dist_to_axis < _ON_AXIS_TOL:
            # Center on revolution axis → SPHERE
            return 'SPHERE', {
                'radius': float(radius),
                'center': [float(center.X()), float(center.Y()), float(center.Z())],
            }
        else:
            # Center off axis → TORUS
            return 'TORUS', {
                'major_radius': float(dist_to_axis),
                'minor_radius': float(radius),
                'axis': axis_dir_vec,
            }

    # ── Non-elementary generatrix → REVOLUTION fallback ────────────────
    else:
        return 'REVOLUTION', _revolution_fallback_params(
            revolution_axis, basis_curve, curve_adaptor)


def _revolution_fallback_params(revolution_axis, basis_curve, curve_adaptor) -> dict:
    """Extract informative parameters from a non-elementary surface of revolution.

    Samples the generatrix uniformly to compute radial extent and records
    the curve type for diagnostics.
    """
    ax_pt = np.array([revolution_axis.Location().X(),
                       revolution_axis.Location().Y(),
                       revolution_axis.Location().Z()])
    ax_dir = np.array([revolution_axis.Direction().X(),
                        revolution_axis.Direction().Y(),
                        revolution_axis.Direction().Z()])

    NUM_SAMPLES = 20
    u_first = curve_adaptor.FirstParameter()
    u_last = curve_adaptor.LastParameter()
    radii = []
    for i in range(NUM_SAMPLES):
        u = u_first + (u_last - u_first) * i / (NUM_SAMPLES - 1)
        pt = curve_adaptor.Value(u)
        p = np.array([pt.X(), pt.Y(), pt.Z()])
        diff = p - ax_pt
        r = float(np.linalg.norm(diff - np.dot(diff, ax_dir) * ax_dir))
        radii.append(r)

    type_map = {
        GeomAbs_Line: 'LINEAR',
        GeomAbs_Circle: 'CIRCULAR',
        GeomAbs_BSplineCurve: 'BSPLINE',
        GeomAbs_BezierCurve: 'BEZIER',
    }
    basis_curve_type = type_map.get(curve_adaptor.GetType(), 'OTHER')

    return {
        'min_radius': float(min(radii)),
        'max_radius': float(max(radii)),
        'axis_direction': [float(ax_dir[0]), float(ax_dir[1]), float(ax_dir[2])],
        'axis_location': [float(ax_pt[0]), float(ax_pt[1]), float(ax_pt[2])],
        'basis_curve_type': basis_curve_type,
    }


def _classify_extrusion_surface(basis_curve, direction) -> tuple[str, dict]:
    """Classify the profile curve of a surface of linear extrusion.

    Returns ``(label, params_dict)`` where *label* is ``'PLANE'``,
    ``'CYLINDER'`` (successful reduction) or ``'EXTRUSION'`` (non-elementary
    profile — keep as extrusion surface).
    """
    curve_adaptor = GeomAdaptor_Curve(basis_curve)
    curve_type = curve_adaptor.GetType()

    ext_dir = np.array([direction.X(), direction.Y(), direction.Z()])
    ext_dir_list = [float(ext_dir[0]), float(ext_dir[1]), float(ext_dir[2])]

    # ── Linear profile → PLANE ─────────────────────────────────────────
    if curve_type == GeomAbs_Line:
        line = curve_adaptor.Line()
        line_dir = line.Direction()
        ld = np.array([line_dir.X(), line_dir.Y(), line_dir.Z()])
        normal = np.cross(ld, ext_dir)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-10:
            return 'EXTRUSION', {
                'direction': ext_dir_list,
                'basis_curve_type': 'LINEAR',
            }
        normal = normal / norm_mag
        loc = line.Location()
        return 'PLANE', {
            'normal': [float(normal[0]), float(normal[1]), float(normal[2])],
            'location': [float(loc.X()), float(loc.Y()), float(loc.Z())],
        }

    # ── Circular profile ∥ extrusion direction → CYLINDER ──────────────
    elif curve_type == GeomAbs_Circle:
        circle = curve_adaptor.Circle()
        circle_normal = circle.Axis().Direction()
        cn = np.array([circle_normal.X(), circle_normal.Y(), circle_normal.Z()])
        cos_angle = float(np.clip(np.dot(ext_dir, cn), -1.0, 1.0))
        angle = math.acos(abs(cos_angle))

        if angle < _PARALLEL_TOL:
            center = circle.Location()
            return 'CYLINDER', {
                'radius': float(circle.Radius()),
                'axis_direction': ext_dir_list,
                'axis_location': [float(center.X()), float(center.Y()), float(center.Z())],
            }
        else:
            return 'EXTRUSION', {
                'direction': ext_dir_list,
                'basis_curve_type': 'CIRCULAR',
            }

    # ── Other profile → EXTRUSION fallback ─────────────────────────────
    else:
        type_map = {
            GeomAbs_BSplineCurve: 'BSPLINE',
            GeomAbs_BezierCurve: 'BEZIER',
        }
        basis_type_str = type_map.get(curve_type, 'OTHER')
        return 'EXTRUSION', {
            'direction': ext_dir_list,
            'basis_curve_type': basis_type_str,
        }


# ---------------------------------------------------------------------------
# Public API — faces
# ---------------------------------------------------------------------------

def extract_face_geometry(face: TopoDS_Face) -> tuple[SurfaceType, dict]:
    """Return ``(SurfaceType, parameters_dict)`` for *face*.

    Uses *BRepAdaptor_Surface*.  If the surface type is not mapped to a
    known *SurfaceType*, returns ``(OTHER, {})`` and logs a warning with
    the OCCT type name.
    """
    try:
        adaptor = BRepAdaptor_Surface(face, True)
        geom_type = adaptor.GetType()
    except Exception:
        logger.warning("BRepAdaptor_Surface failed for face", exc_info=True)
        return SurfaceType.OTHER, {}

    surface_type = _SURFACE_TYPE_MAP.get(geom_type, SurfaceType.OTHER)

    # ── SurfaceOfRevolution: attempt reduction to elementary surface ────
    if geom_type == GeomAbs_SurfaceOfRevolution:
        try:
            surface_handle = BRep_Tool.Surface(face)
            rev_surf = Geom_SurfaceOfRevolution.DownCast(surface_handle)
            if rev_surf is None:
                raise ValueError("DownCast to Geom_SurfaceOfRevolution failed")

            basis_curve = rev_surf.BasisCurve()
            revolution_axis = rev_surf.Axis()

            classified_type, classified_params = _classify_basis_curve(
                basis_curve, revolution_axis)

            if classified_type in ('CYLINDER', 'CONE', 'SPHERE', 'TORUS'):
                return SurfaceType[classified_type], classified_params
            else:
                logger.info(
                    "SurfaceOfRevolution not reducible to elementary surface; "
                    "generatrix type: %s, radial range: [%.3f, %.3f] mm",
                    classified_params.get('basis_curve_type', 'UNKNOWN'),
                    classified_params.get('min_radius', 0),
                    classified_params.get('max_radius', 0),
                )
                return SurfaceType.REVOLUTION, classified_params

        except Exception:
            logger.warning(
                "SurfaceOfRevolution classification failed, falling back to "
                "REVOLUTION (no parameters)", exc_info=True)
            return SurfaceType.REVOLUTION, {}

    # ── SurfaceOfExtrusion: attempt reduction to elementary surface ─────
    if geom_type == GeomAbs_SurfaceOfExtrusion:
        try:
            surface_handle = BRep_Tool.Surface(face)
            ext_surf = Geom_SurfaceOfLinearExtrusion.DownCast(surface_handle)
            if ext_surf is None:
                raise ValueError("DownCast to Geom_SurfaceOfLinearExtrusion failed")

            basis_curve = ext_surf.BasisCurve()
            direction = ext_surf.Direction()

            classified_type, classified_params = _classify_extrusion_surface(
                basis_curve, direction)

            if classified_type in ('PLANE', 'CYLINDER'):
                return SurfaceType[classified_type], classified_params
            else:
                logger.info(
                    "SurfaceOfExtrusion not reducible to elementary surface; "
                    "profile curve type: %s",
                    classified_params.get('basis_curve_type', 'UNKNOWN'),
                )
                return SurfaceType.EXTRUSION, classified_params

        except Exception:
            logger.warning(
                "SurfaceOfExtrusion classification failed, falling back to "
                "EXTRUSION (no parameters)", exc_info=True)
            return SurfaceType.EXTRUSION, {}

    if surface_type == SurfaceType.OTHER:
        name = _GEOMABS_SURFACE_NAMES.get(geom_type, f"<unknown {geom_type}>")
        logger.warning("Unsupported surface type: %s", name)
        return SurfaceType.OTHER, {}

    try:
        params = _extract_surface_params(adaptor, geom_type)
    except Exception:
        logger.warning(
            "Parameter extraction failed for %s",
            _GEOMABS_SURFACE_NAMES.get(geom_type, str(geom_type)),
            exc_info=True,
        )
        return surface_type, {}

    return surface_type, params


def _extract_surface_params(adaptor, geom_type: int) -> dict:
    """Dispatch to the correct parameter extractor based on *geom_type*."""
    if geom_type == GeomAbs_Plane:
        pln = adaptor.Plane()
        return {
            "normal": _axis_direction(pln.Axis()),
            "location": _to_list(pln.Location()),
        }

    if geom_type == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        return {
            "radius": cyl.Radius(),
            "axis_direction": _axis_direction(cyl.Axis()),
            "axis_location": _axis_location(cyl.Axis()),
        }

    if geom_type == GeomAbs_Cone:
        cone = adaptor.Cone()
        return {
            "semi_angle": cone.SemiAngle(),
            "axis_direction": _axis_direction(cone.Position().Axis()),
            "apex": _to_list(cone.Apex()),
        }

    if geom_type == GeomAbs_Sphere:
        sph = adaptor.Sphere()
        return {
            "radius": sph.Radius(),
            "center": _to_list(sph.Location()),
        }

    if geom_type == GeomAbs_Torus:
        tor = adaptor.Torus()
        return {
            "major_radius": tor.MajorRadius(),
            "minor_radius": tor.MinorRadius(),
            "axis": _axis_direction(tor.Axis()),
        }

    if geom_type in (GeomAbs_BSplineSurface, GeomAbs_BezierSurface):
        bs = adaptor.BSpline()
        return {
            "degree_u": bs.UDegree(),
            "degree_v": bs.VDegree(),
            "num_poles_u": bs.NbUPoles(),
            "num_poles_v": bs.NbVPoles(),
        }

    # Should not reach here — caller checks OTHER before dispatching
    return {}


def extract_face_orientation(face: TopoDS_Face) -> bool:
    """Return *True* if the face orientation is *TopAbs_REVERSED*.

    When True, the geometric surface normal must be negated to obtain the
    true outward normal of the solid.
    """
    return face.Orientation() == TopAbs_REVERSED


def extract_face_area(face: TopoDS_Face) -> float:
    """Compute the area of *face* in mm² via *BRepGProp.SurfaceProperties*.

    Returns 0.0 and logs a warning on failure.
    """
    try:
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        return props.Mass()
    except Exception:
        logger.warning("Face area computation failed", exc_info=True)
        return 0.0


def extract_face_bbox(face: TopoDS_Face) -> dict | None:
    """Return the face's tight world-coordinate axis-aligned bounding box."""
    try:
        bounds = Bnd_Box()
        bounds.SetGap(0.0)
        brepbndlib.AddOptimal(face, bounds, False, False)
        if bounds.IsVoid():
            return None
        xmin, ymin, zmin, xmax, ymax, zmax = bounds.Get()
        values = [xmin, ymin, zmin, xmax, ymax, zmax]
        if not all(math.isfinite(float(value)) for value in values):
            return None
        return {
            "min": [float(xmin), float(ymin), float(zmin)],
            "max": [float(xmax), float(ymax), float(zmax)],
        }
    except Exception:
        logger.warning("Face bounding-box computation failed", exc_info=True)
        return None


def get_outward_normal(
    face: TopoDS_Face,
    u: float,
    v: float,
    adaptor: BRepAdaptor_Surface | None = None,
) -> list[float] | None:
    """Return the **outward** surface normal of *face* at parameters *(u, v)*.

    ``BRepLProp_SLProps`` computes the geometric normal from the underlying
    ``Geom_Surface`` — it does **not** consult ``TopoDS_Face.Orientation()``.
    This function corrects for that: if the face is reversed, the normal is
    negated so the caller always receives the true outward normal of the solid.

    Returns ``None`` if the normal is not defined at the given parameters
    (e.g. singular point on a cone apex).

    *adaptor* is an optional ``BRepAdaptor_Surface`` **of this same face**,
    supplied by a caller that walks a whole uv grid over one face and already
    holds one.  Building it here is the only per-call cost this function has
    and it does not depend on *(u, v)*, so a 5x5 station loop was paying for
    25 identical adaptors (4706 constructions for 4259 normals on a 119-face
    part).  Passing an adaptor built from a *different* face would silently
    return that other face's normal, so only a caller that owns both the face
    and the adaptor may use it; when it is omitted the adaptor is built here
    exactly as before.  Same technique as the per-face 2d classifier in
    ``face_sampling.interior_uv`` (B-P7(a)), and scoped the same way: to one
    face inside one pass, never a module-level cache keyed on ``hash(face)``.

    ``face_sampling.sample_face_interior`` passes one since node 28D.  It is
    worth knowing what that cost, because node 28S measured it and declined to
    spend it: a reused ``BRepAdaptor_Surface`` carries an evaluation cache, and
    on one b-spline face of the corpus (``Automotive Coolant Thermostat
    Housing``, face_103) that cache moves the stored normal by one ulp - seven
    of the twenty-five stations print a different last digit, and the part's
    ``artifactDigest`` moves with them.  No conclusion and no piece of evidence
    moves.  So a caller who wires this up outside a batch that is allowed to
    move a digest will fail the digest comparison, and correctly.
    """
    if adaptor is None:
        adaptor = BRepAdaptor_Surface(face)
    props = BRepLProp_SLProps(adaptor, u, v, 1, 1e-6)
    if not props.IsNormalDefined():
        return None
    n = props.Normal()
    if face.Orientation() == TopAbs_REVERSED:
        n.Reverse()
    return [n.X(), n.Y(), n.Z()]


# ---------------------------------------------------------------------------
# Public API — edges
# ---------------------------------------------------------------------------

def extract_edge_geometry(edge: TopoDS_Edge) -> tuple[CurveType, dict, float]:
    """Return ``(CurveType, parameters_dict, length)`` for *edge*.

    Uses *BRepAdaptor_Curve* for type and parameters, *BRepGProp* for length.
    *parameters_dict* additionally receives ``sample_points`` — a fixed
    number of points sampled uniformly along the curve parameter range —
    so downstream topology analysis can evaluate curve shape without
    depending on OCCT.
    """
    # --- length -----------------------------------------------------------
    length = 0.0
    try:
        props = GProp_GProps()
        brepgprop.LinearProperties(edge, props)
        length = props.Mass()
    except Exception:
        logger.warning("Edge length computation failed", exc_info=True)

    # --- curve type & parameters ------------------------------------------
    try:
        adaptor = BRepAdaptor_Curve(edge)
        geom_type = adaptor.GetType()
    except Exception:
        logger.warning("BRepAdaptor_Curve failed for edge", exc_info=True)
        return CurveType.OTHER, {}, length

    curve_type = _CURVE_TYPE_MAP.get(geom_type, CurveType.OTHER)

    if curve_type == CurveType.OTHER:
        name = _GEOMABS_CURVE_NAMES.get(geom_type, f"<unknown {geom_type}>")
        logger.warning("Unsupported curve type: %s", name)
        return CurveType.OTHER, {}, length

    try:
        params = _extract_curve_params(adaptor, geom_type)
    except Exception:
        logger.warning(
            "Parameter extraction failed for %s",
            _GEOMABS_CURVE_NAMES.get(geom_type, str(geom_type)),
            exc_info=True,
        )
        return curve_type, {}, length

    params["sample_points"] = _sample_curve_points(adaptor)
    return curve_type, params, length


_NUM_CURVE_SAMPLES = 9


def _sample_curve_points(adaptor) -> list[list[float]]:
    """Sample ``_NUM_CURVE_SAMPLES`` points uniformly over the curve range."""
    try:
        u_first = adaptor.FirstParameter()
        u_last = adaptor.LastParameter()
    except Exception:
        return []
    points: list[list[float]] = []
    try:
        for index in range(_NUM_CURVE_SAMPLES):
            u = u_first + (u_last - u_first) * index / (_NUM_CURVE_SAMPLES - 1)
            pnt = adaptor.Value(u)
            points.append([float(pnt.X()), float(pnt.Y()), float(pnt.Z())])
    except Exception:
        logger.warning("Curve sampling failed", exc_info=True)
        return []
    return points


def _extract_curve_params(adaptor, geom_type: int) -> dict:
    """Dispatch to the correct curve-parameter extractor."""
    if geom_type == GeomAbs_Line:
        lin = adaptor.Line()
        return {"direction": _axis_direction(lin.Position())}

    if geom_type == GeomAbs_Circle:
        circ = adaptor.Circle()
        ax3 = circ.Position()
        return {
            "radius": circ.Radius(),
            "center": _to_list(ax3.Location()),
            "normal": _axis_direction(ax3.Axis()),
        }

    if geom_type == GeomAbs_Ellipse:
        ell = adaptor.Ellipse()
        return {
            "major_radius": ell.MajorRadius(),
            "minor_radius": ell.MinorRadius(),
            "center": _to_list(ell.Location()),
        }

    if geom_type in (GeomAbs_BSplineCurve, GeomAbs_BezierCurve):
        bs = adaptor.BSpline()
        return {"degree": bs.Degree(), "num_poles": bs.NbPoles()}

    return {}


# ---------------------------------------------------------------------------
# Public API — vertices
# ---------------------------------------------------------------------------

def extract_vertex_coords(vertex: TopoDS_Vertex) -> tuple[float, float, float]:
    """Return ``(x, y, z)`` coordinates of *vertex* in model space (mm)."""
    pnt = BRep_Tool.Pnt(vertex)
    return (pnt.X(), pnt.Y(), pnt.Z())

