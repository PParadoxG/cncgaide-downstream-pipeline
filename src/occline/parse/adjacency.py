"""Face adjacency graph construction from B-Rep topology.

Uses ``TopTools_IndexedDataMapOfShapeListOfShape`` and
``TopExp.MapShapesAndAncestors`` to build edge-face incidence maps,
then derives face-face adjacency from shared edges with concavity
classification (CONCAVE / CONVEX / SMOOTH / UNKNOWN).
"""

import logging
import math
from collections import defaultdict
from typing import Any

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepLProp import BRepLProp_CLProps
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_FORWARD,
    TopAbs_SHAPE,
)
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
from OCC.Core.TopoDS import TopoDS_Edge, TopoDS_Face, TopoDS_Shape, topods
from OCC.Core.gp import gp_Dir

from occline.parse.geometry import extract_face_geometry, get_outward_normal
from contract.partmodel import AdjacencyInfo, ConcavityType, SurfaceType
from occline.parse.traversal import iter_faces

logger = logging.getLogger(__name__)

# Floor for the edge-midpoint projection residual check in ``detect_concavity``.
# Generous on purpose: it exists to catch a wrong coordinate frame (residuals
# there are on the order of the part size), not to police surface fitting.
_PROJECTION_TOL = 1e-3  # mm

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_oriented_edge_on_face(face: TopoDS_Face, target_edge: TopoDS_Edge) -> TopoDS_Edge | None:
    """Return the *oriented* edge from *face* whose TShape matches *target_edge*.

    The edge from ``TopExp.MapShapesAndAncestors`` is *neutral* (no
    face-relative orientation).  We need the oriented version from inside
    the face to query ``TopAbs_FORWARD`` / ``TopAbs_REVERSED``.
    """
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        e = topods.Edge(exp.Current())
        if hash(e) == hash(target_edge):
            return e
        exp.Next()
    return None


def _build_shape_lookups(shape: TopoDS_Shape) -> tuple[dict[int, TopoDS_Face], dict[int, TopoDS_Edge]]:
    """Return ``(face_hash→TopoDS_Face, edge_hash→TopoDS_Edge)`` for *shape*."""
    faces: dict[int, TopoDS_Face] = {}
    edges: dict[int, TopoDS_Edge] = {}
    for f in iter_faces(shape):
        faces[hash(f)] = f
    exp = TopExp_Explorer(shape, TopAbs_EDGE, TopAbs_SHAPE)
    while exp.More():
        e = topods.Edge(exp.Current())
        edges[hash(e)] = e
        exp.Next()
    return faces, edges


# ---------------------------------------------------------------------------
# Concavity detection
# ---------------------------------------------------------------------------

def detect_concavity(
    face1: TopoDS_Face,
    face2: TopoDS_Face,
    shared_edge: TopoDS_Edge,
) -> ConcavityType:
    """Classify the dihedral angle along *shared_edge* between *face1* and *face2*.

    Algorithm
    ---------
    The result is *material-side* concavity: with the observer outside the
    material at the edge, a local cross-section with material wedge < 180°
    is ``CONVEX`` (box corner) and > 180° is ``CONCAVE`` (pocket wall meeting
    its floor).

    Algorithm
    ---------
    1. Sample the edge at its parameter midpoint to get a 3D point **P**.
    2. Project **P** onto each face's *located* surface to obtain *(u, v)*.
    3. Call ``get_outward_normal`` at those parameters to get the true
       outward normals **n1**, **n2**.
    4. Get the edge tangent **t** at **P**.
    5. If ``angle(n1, n2) < 1°`` → ``SMOOTH``.
    6. Find the oriented edge on *face1* to determine whether the edge
       is ``FORWARD`` or ``REVERSED`` with respect to the face boundary.
    7. Compute ``(n1 × n2) · t``, negate if edge is ``FORWARD`` on face1.
       Positive → ``CONCAVE``, negative → ``CONVEX``.
    8. Any failure (projection, normal undefined, missing orientation)
       → ``UNKNOWN`` with a warning.

    Why the orientation in step 6 must come from ``TopExp_Explorer``
    ---------------------------------------------------------------
    ``TopExp_Explorer`` walks ``face → wire → edge`` through ``TopoDS_Iterator``
    with ``cumOri = True``, so the orientation it returns has the wire
    orientation **and the face's own orientation** composed in.
    ``BRepTools::OriEdgeInFace`` deliberately explores ``F.Oriented(FORWARD)``
    and therefore excludes the face orientation.  The two disagree on every
    edge of a REVERSED face — which is most faces of a typical solid.

    Step 6 is the required partner of step 3.  For a REVERSED face the
    outward normal is ``-n_geom`` *and* the composed edge orientation is
    flipped; both sign changes cancel, so ``interior = n_out × t`` holds
    uniformly.  Pairing the outward normal with ``OriEdgeInFace`` instead
    breaks that cancellation.  Measured over 1314 shared edges against an
    independent solid-classifier oracle: this pairing 100% correct, the
    ``OriEdgeInFace`` pairing 28%.  Do not "modernise" step 6 in isolation.

    Face-order invariance follows from orientation coherence of the shell:
    swapping the faces flips ``(n1 × n2) · t`` and flips the correcting
    orientation, and the two cancel.  See
    ``NODE_REPORT/MaterialSideConcavityAudit_T0.md`` for the full derivation,
    the corpus results and the counterexamples.
    """
    try:
        # --- 1. Edge midpoint P and tangent t -------------------------------
        curve = BRepAdaptor_Curve(shared_edge)
        t_min = curve.FirstParameter()
        t_max = curve.LastParameter()
        t_mid = (t_min + t_max) / 2.0

        cprops = BRepLProp_CLProps(curve, t_mid, 1, 1e-6)
        if not cprops.IsTangentDefined():
            logger.warning("Edge tangent undefined at midpoint")
            return ConcavityType.UNKNOWN

        P = cprops.Value()
        D = gp_Dir()
        cprops.Tangent(D)
        t = (D.X(), D.Y(), D.Z())

        # --- 2 & 3. Project onto each face, get outward normals -------------
        def _normal_at_point(face: TopoDS_Face, pnt) -> list[float] | None:
            # ``BRep_Tool.Surface(face)`` (one-argument form) returns the
            # surface with the face's TopLoc_Location already applied, i.e. in
            # world coordinates — the frame *pnt* lives in.
            #
            # ``BRepAdaptor_Surface(face).Surface().Surface()`` was used here
            # originally and is WRONG for any placed face: BRepAdaptor keeps
            # the location separately in ``myTrsf`` and hands back the
            # *unlocated* surface, so a world-space point was being projected
            # onto a local-space surface.  Planes survived (constant normal);
            # cylinders, cones, spheres and tori silently returned a normal
            # from the wrong (u, v), inverting the classification.
            surf = BRep_Tool.Surface(face)
            proj = GeomAPI_ProjectPointOnSurf(pnt, surf)
            if proj.NbPoints() == 0:
                return None
            # The point lies on the shared edge, hence on both surfaces.  A
            # non-trivial residual means the projection frame is wrong; fail
            # loudly rather than return a plausible-looking wrong normal.
            residual = proj.LowerDistance()
            budget = max(_PROJECTION_TOL, 1e3 * BRep_Tool.Tolerance(face))
            if residual > budget:
                logger.warning(
                    "Edge midpoint projects %.6g mm off the face surface "
                    "(budget %.6g) — normal rejected", residual, budget,
                )
                return None
            u, v = proj.LowerDistanceParameters()
            return get_outward_normal(face, u, v)

        n1 = _normal_at_point(face1, P)
        if n1 is None:
            logger.warning("Failed to project edge midpoint onto face1")
            return ConcavityType.UNKNOWN

        n2 = _normal_at_point(face2, P)
        if n2 is None:
            logger.warning("Failed to project edge midpoint onto face2")
            return ConcavityType.UNKNOWN

        # --- 4. Angle check -------------------------------------------------
        dot_n = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]
        dot_n = max(-1.0, min(1.0, dot_n))
        angle = math.degrees(math.acos(dot_n))

        if angle < 1.0:
            return ConcavityType.SMOOTH

        # --- 5. Edge orientation on face1 -----------------------------------
        oriented = _get_oriented_edge_on_face(face1, shared_edge)
        if oriented is None:
            logger.warning("Could not find oriented edge on face")
            return ConcavityType.UNKNOWN

        edge_ori = oriented.Orientation()

        # --- 6. Cross product test ------------------------------------------
        cross = (
            n1[1] * n2[2] - n1[2] * n2[1],
            n1[2] * n2[0] - n1[0] * n2[2],
            n1[0] * n2[1] - n1[1] * n2[0],
        )
        cross_dot_t = cross[0] * t[0] + cross[1] * t[1] + cross[2] * t[2]

        # Normalise by edge orientation on face1 so the result is
        # independent of which face we call "face1".
        # Under OCCT convention a FORWARD boundary edge keeps the face interior
        # on its left, so interior = n × t for FORWARD edges, interior = t × n
        # for REVERSED edges.  interior1 · n2 > 0 → CONCAVE.
        #   FORWARD:  (n1 × t)·n2 = -(n1×n2)·t  → CONCAVE iff cross_dot_t < 0 → negate for FORWARD
        #   REVERSED: (t × n1)·n2 = +(n1×n2)·t  → CONCAVE iff cross_dot_t > 0 → no negation
        # Negating for FORWARD makes the rule uniform: cross_dot_t > 0 → CONCAVE.
        if edge_ori == TopAbs_FORWARD:
            cross_dot_t = -cross_dot_t

        if cross_dot_t > 0:
            return ConcavityType.CONCAVE
        else:
            return ConcavityType.CONVEX

    except Exception:
        logger.warning("Concavity detection failed", exc_info=True)
        return ConcavityType.UNKNOWN


# ---------------------------------------------------------------------------
# Public API — edge-face map
# ---------------------------------------------------------------------------

def build_edge_face_map(shape: TopoDS_Shape) -> dict[int, list[int]]:
    """Return ``{edge_hash: [face_hash, ...]}`` for all edges in *shape*.

    Uses ``TopExp.MapShapesAndAncestors`` to build the OCCT incidence
    map, then converts it to a plain Python dict keyed by Python ``hash()``
    (equivalent to OCCT ``HashCode`` for topological identity).

    Edge cases:
    - Edge with >2 faces: logged as warning, edge is **skipped** (not in result)
    - Edge with 1 face: stored normally (boundary edge of the solid)
    - Edge with 0 faces: should not occur in a valid B-Rep; logged and skipped
    """
    occt_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, occt_map)

    result: dict[int, list[int]] = {}

    for i in range(1, occt_map.Size() + 1):
        edge = occt_map.FindKey(i)
        face_list = occt_map.FindFromIndex(i)
        n_faces = face_list.Size()

        if n_faces == 0:
            logger.warning("Edge %d has 0 adjacent faces — skipping", hash(edge))
            continue

        if n_faces > 2:
            logger.warning(
                "Edge %d has %d adjacent faces (expected <=2) — skipping",
                hash(edge), n_faces,
            )
            continue

        edge_hash = hash(edge)
        # Deduplicate: MapShapesAndAncestors can return the same face
        # twice when the edge is a seam on a closed face (e.g. full
        # cylinder).  When this happens the edge connects the face to
        # itself — it is not a shared edge between two distinct faces.
        face_hashes = list(dict.fromkeys(
            hash(topods.Face(f)) for f in face_list
        ))
        result[edge_hash] = face_hashes

    return result


# ---------------------------------------------------------------------------
# Public API — adjacency with concavity
# ---------------------------------------------------------------------------

def build_adjacency_infos(shape: TopoDS_Shape) -> dict[int, list[AdjacencyInfo]]:
    """Return ``{face_hash: [AdjacencyInfo, ...]}`` with concavity classified.

    Calls ``detect_concavity`` for every shared edge to determine
    ``CONCAVE``, ``CONVEX``, ``SMOOTH``, or ``UNKNOWN``.
    """
    edge_face_map = build_edge_face_map(shape)
    face_lookup, edge_lookup = _build_shape_lookups(shape)

    # Group by face for output
    result: dict[int, list[AdjacencyInfo]] = defaultdict(list)

    for edge_hash, face_hashes in edge_face_map.items():
        if len(face_hashes) != 2:
            continue

        fh0, fh1 = face_hashes[0], face_hashes[1]
        face0 = face_lookup.get(fh0)
        face1 = face_lookup.get(fh1)
        edge = edge_lookup.get(edge_hash)

        if face0 is None or face1 is None or edge is None:
            result[fh0].append(
                AdjacencyInfo(fh1, edge_hash, ConcavityType.UNKNOWN)
            )
            result[fh1].append(
                AdjacencyInfo(fh0, edge_hash, ConcavityType.UNKNOWN)
            )
            continue

        concavity = detect_concavity(face0, face1, edge)

        result[fh0].append(AdjacencyInfo(fh1, edge_hash, concavity))
        result[fh1].append(AdjacencyInfo(fh0, edge_hash, concavity))

    # Ensure every face has a key
    for face in iter_faces(shape):
        fh = hash(face)
        if fh not in result:
            result[fh] = []

    return dict(result)


# ---------------------------------------------------------------------------
# Public API — seam-edge detection
# ---------------------------------------------------------------------------

def detect_seam_edges(shape: TopoDS_Shape) -> set[int]:
    """Return the set of edge hashes that are seam edges.

    A seam edge separates two faces that belong to the **same underlying
    geometric surface** (e.g. a cylindrical face split in half at the
    parameterisation boundary).  We detect this by comparing the surface
    type and key geometric parameters of the two adjacent faces — if they
    match within the same tolerances used in ``normalizer.py``, the edge
    is a seam.

    ``BRep_Tool.IsClosed`` was evaluated but rejected: when a STEP
    exporter pre-splits a cylinder into half-faces, neither half-face is
    closed in UV space, so ``IsClosed`` returns ``False`` even though the
    edge is semantically a seam.
    """
    import math

    _RADIUS_TOL = 1e-6
    _AXIS_ANGLE_TOL = 1e-4
    _AXIS_DIST_TOL = 1e-4

    def _angle(v1: list[float], v2: list[float]) -> float:
        dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    def _same_surface(st_a: SurfaceType, st_b: SurfaceType,
                       p_a: dict, p_b: dict) -> bool:
        """Return True if the two faces lie on the same geometric surface."""
        if st_a != st_b:
            return False
        if st_a == SurfaceType.CYLINDER:
            if abs(p_a.get("radius", 0.0) - p_b.get("radius", 0.0)) >= _RADIUS_TOL:
                return False
            ax_a = p_a.get("axis_direction", [0.0, 0.0, 1.0])
            ax_b = p_b.get("axis_direction", [0.0, 0.0, 1.0])
            if _angle(ax_a, ax_b) >= _AXIS_ANGLE_TOL:
                return False
            # Coaxiality: reject laterally offset parallel axes
            loc_a = p_a.get("axis_location")
            loc_b = p_b.get("axis_location")
            if loc_a is not None and loc_b is not None:
                d = ax_a
                mag = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
                if mag > 0:
                    dx, dy, dz = d[0] / mag, d[1] / mag, d[2] / mag
                    v = [loc_b[0] - loc_a[0], loc_b[1] - loc_a[1], loc_b[2] - loc_a[2]]
                    v_dot_d = v[0] * dx + v[1] * dy + v[2] * dz
                    perp = math.sqrt(
                        (v[0] - v_dot_d * dx) ** 2
                        + (v[1] - v_dot_d * dy) ** 2
                        + (v[2] - v_dot_d * dz) ** 2
                    )
                    if perp >= _AXIS_DIST_TOL:
                        return False
            return True
        if st_a == SurfaceType.PLANE:
            n_a = p_a.get("normal", [0.0, 0.0, 1.0])
            n_b = p_b.get("normal", [0.0, 0.0, 1.0])
            if _angle(n_a, n_b) >= _AXIS_ANGLE_TOL:
                return False
            return True
        # CONE, SPHERE, TORUS, BSPLINE, OTHER — not recognised as seam sources
        return False

    edge_face_map = build_edge_face_map(shape)
    face_lookup, _ = _build_shape_lookups(shape)

    seam_edges: set[int] = set()

    for edge_hash, face_hashes in edge_face_map.items():
        if len(face_hashes) != 2:
            continue
        face_a = face_lookup.get(face_hashes[0])
        face_b = face_lookup.get(face_hashes[1])
        if face_a is None or face_b is None:
            continue

        try:
            st_a, p_a = extract_face_geometry(face_a)
            st_b, p_b = extract_face_geometry(face_b)
        except Exception:
            continue

        if _same_surface(st_a, st_b, p_a, p_b):
            seam_edges.add(edge_hash)

    return seam_edges


# ---------------------------------------------------------------------------
# Public API — shell connectivity
# ---------------------------------------------------------------------------
