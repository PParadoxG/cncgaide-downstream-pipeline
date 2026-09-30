"""Parse-time face facts: interior samples and area centroids, on OCC.

These two measurements used to be :class:`contract.kernel.KernelServices`
queries (``sample_face_interior`` / ``face_area_centroid``).  They are static
properties of a face, so they are computed once while the shape is loaded and
stored in ``FaceData`` — which is also what lets a second kernel line supply
them by filling the same table instead of implementing two more services.

The measuring code is the code that used to sit in
``occline.kernel.occ_sampling``; only its address and its output shape
(plain dicts, ready for ``partmodel.json``) changed.
"""

from __future__ import annotations

import math

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepTopAdaptor import BRepTopAdaptor_FClass2d
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import breptools
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_IN
from OCC.Core.gp import gp_Pnt2d

from occline.face_ids import raw_cad_face_ids
from occline.parse.geometry import get_outward_normal

# The uv stations sampled inside a face.  Five per direction: enough for the
# draft-spread test to see a twisted wall vary, few enough to stay cheap on a
# part with hundreds of faces.
UV_FRACTIONS = (0.2, 0.4, 0.5, 0.6, 0.8)


def interior_uv(occ_face, tolerance: float = 1.0e-7) -> list[tuple[float, float]]:
    """uv stations that fall strictly inside the face's trim."""
    try:
        u_min, u_max, v_min, v_max = breptools.UVBounds(occ_face)
    except Exception:
        return []
    if not all(math.isfinite(float(value)) for value in (u_min, u_max, v_min, v_max)):
        return []
    if float(u_max) <= float(u_min) or float(v_max) <= float(v_min):
        return []
    # ``BRepClass_FaceClassifier.Perform(face, uv, tol)`` rebuilds the face's
    # 2d classification structure on every station; ``BRepTopAdaptor_FClass2d``
    # is that structure, built once for the face and then asked 25 times.  Same
    # answers - 35,650 stations across the corpus, zero disagreements - at a
    # fifth of the cost.
    try:
        classifier = BRepTopAdaptor_FClass2d(occ_face, tolerance)
    except Exception:
        return []
    result: list[tuple[float, float]] = []
    for fu in UV_FRACTIONS:
        for fv in UV_FRACTIONS:
            u = float(u_min) + (float(u_max) - float(u_min)) * fu
            v = float(v_min) + (float(v_max) - float(v_min)) * fv
            try:
                if classifier.Perform(gp_Pnt2d(u, v)) != TopAbs_IN:
                    continue
            except Exception:
                continue
            result.append((u, v))
    return result


def sample_face_interior(occ_face) -> list[dict]:
    """``{"point", "normal"}`` pairs sampled inside one raw CAD face.

    Normals are outward unit normals of the solid; points are strictly inside
    the trim.  An empty list means "could not sample".

    The one adaptor built here serves both the point and the normal.  It used
    to serve only the point, and ``get_outward_normal`` built its own for each
    of the 25 stations - 16,606 constructions over the nine-part corpus for
    answers that do not depend on the station.  Reusing it is not free of
    consequence and the consequence is measured: a ``BRepAdaptor_Surface``
    carries an evaluation cache, and on one b-spline face of the corpus
    (``Automotive Coolant Thermostat Housing``, face_103) seven of the stored
    normals move by one ulp in their last digit.  Node 28S found that, refused
    to spend it in a digest-neutral batch and left the wiring out (28S §3 #1);
    node 28D is the batch that may, and the seven values are attributed one by
    one in its report.  No conclusion, no piece of evidence and no verdict
    differs either side of it.
    """
    if occ_face is None:
        return []
    try:
        adaptor = BRepAdaptor_Surface(occ_face)
    except Exception:
        return []
    samples: list[dict] = []
    for u, v in interior_uv(occ_face):
        normal = _unit(get_outward_normal(occ_face, u, v, adaptor))
        if normal is None:
            continue
        try:
            point = adaptor.Value(u, v)
        except Exception:
            continue
        samples.append({"point": [point.X(), point.Y(), point.Z()], "normal": normal})
    return samples


def face_area_centroid(occ_face) -> tuple[float, list[float]] | None:
    """Exact surface area and area centroid of one raw CAD face."""
    if occ_face is None:
        return None
    props = GProp_GProps()
    brepgprop.SurfaceProperties(occ_face, props)
    mass = float(props.Mass())
    if mass <= 0.0:
        return None
    centre = props.CentreOfMass()
    return mass, [centre.X(), centre.Y(), centre.Z()]


def fill_face_samples(model, raw_faces: list) -> None:
    """Fill ``centroid`` and ``interior_samples`` on every active face.

    *raw_faces* is the shape's face list in traversal order, so index ``i`` is
    raw CAD face id ``i`` — the same numbering ``merged_from`` carries.  A
    merged active face pools its members by ascending raw id, which is the
    order the former kernel-query consumers already used.
    """
    for face in model.faces.values():
        raw_ids, unreadable = raw_cad_face_ids(face)
        if unreadable:
            # An id this line does not issue means the face facts below are
            # measured over fewer raw faces than the face is made of.  Say so
            # in the model rather than returning a short list that looks whole.
            model.normalization_log.append(
                f"FACE FACTS {face.id}: {len(unreadable)} member id(s) are not "
                f"raw CAD face ids and were not sampled: " + ", ".join(unreadable)
            )
        samples: list[dict] = []
        moments: list[tuple[float, list[float]]] = []
        out_of_range: list[int] = []
        unsampled: list[int] = []
        for raw_id in raw_ids:
            if not 0 <= raw_id < len(raw_faces):
                out_of_range.append(raw_id)
                continue
            occ_face = raw_faces[raw_id]
            member = sample_face_interior(occ_face)
            if not member:
                unsampled.append(raw_id)
            samples.extend(member)
            measured = face_area_centroid(occ_face)
            if measured is not None:
                moments.append(measured)
        if out_of_range:
            model.normalization_log.append(
                f"FACE FACTS {face.id}: {len(out_of_range)} member raw id(s) are outside "
                f"the shape's face list and were not sampled: "
                + ", ".join(str(raw_id) for raw_id in out_of_range)
            )
        if unsampled:
            # Finding B-13: an empty ``interior_samples`` used to be silent, and
            # its readers - min_wall_thickness, the outward-normal argument, the
            # band tests - cannot tell "this face has no interior" from "the
            # sampler could not find one".  Only the second is true here: every
            # one of the 25 stations landed outside the trim, or no outward
            # normal could be taken there.
            model.normalization_log.append(
                f"FACE FACTS {face.id}: no interior sample on {len(unsampled)} member "
                f"raw face(s) - every uv station fell outside the trim or had no "
                f"outward normal: " + ", ".join(str(raw_id) for raw_id in unsampled)
            )
        face.interior_samples = samples
        if moments:
            total = sum(item[0] for item in moments)
            face.centroid = [
                sum(mass * centre[index] for mass, centre in moments) / total
                for index in range(3)
            ]
        else:
            face.centroid = None


def _unit(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        components = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in components):
        return None
    norm = math.sqrt(sum(v * v for v in components))
    if norm <= 1.0e-12:
        return None
    return [v / norm for v in components]
