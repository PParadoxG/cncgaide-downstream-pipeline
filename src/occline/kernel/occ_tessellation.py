"""Per-raw-CAD-face tessellation of the loaded shape, on OCC.

This is the geometry half of what the viewer used to do itself.  The split
follows the seam: the kernel turns a solid into triangles and boundary
polylines and hands over plain numbers; the viewer decides the tolerances,
packs the buffers and writes the files.  Which means the tessellation policy
(how fine, relative to the part) stays a display decision and does not have
to be reargued for a second kernel.

Keeping the raw face as the drawing unit is deliberate and is what lets a
click in 3D reach a feature: the face's index here is its ``cadFaceId``.
"""

from __future__ import annotations

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopLoc import TopLoc_Location

from contract.kernel import FaceMesh
from occline.parse.traversal import iter_edges


def bounding_box(shape) -> list[float]:
    """``[xmin, ymin, zmin, xmax, ymax, zmax]`` of the whole shape's geometry.

    ``useTriangulation`` is off on purpose.  With it on, ``Add`` measures the
    triangulation when the shape has one, so the same shape answered a
    different box before and after :func:`tessellate` meshed it in place - and
    the only reason the viewer got the right one was that it happens to ask
    for the bounds on the line above the one that meshes.  A contract that
    reads "the box of the whole solid" must not depend on when it is asked.
    Measured over the corpus: the exact-geometry box equals the triangulated
    one on all nine parts *before* meshing, so this changes no stored value;
    what it changes is that the answer stops moving afterwards.
    """
    box = Bnd_Box()
    box.SetGap(0.0)
    brepbndlib.Add(shape, box, False)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return [float(xmin), float(ymin), float(zmin), float(xmax), float(ymax), float(zmax)]


def tessellate(
    shape,
    faces,
    linear_deflection: float,
    angular_deflection: float,
    edge_deflection: float,
) -> list[FaceMesh]:
    """Mesh *shape* once, then read it back one raw CAD face at a time."""
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)
    meshes: list[FaceMesh] = []
    for cad_face_id, face in enumerate(faces):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)
        positions, normals, indices = _tessellate_face(face, triangulation, location)
        segments = _face_edge_segments(face, triangulation, location, edge_deflection)
        meshes.append(FaceMesh(
            cad_face_id=cad_face_id,
            positions=positions,
            normals=normals,
            indices=indices,
            edge_segments=segments,
        ))
    return meshes


def _tessellate_face(face, triangulation, location):
    """Return (positions, normals, indices) in world space for one raw CAD face.

    Winding is normalised so a counter-clockwise triangle faces outward, and
    vertex normals are averaged *within the face only* — cylinders shade
    smoothly, planes stay flat, and a seam face keeps its own shading instead
    of blending into its neighbour.
    """
    if triangulation is None:
        return [], [], []

    trsf = location.Transformation()
    positions: list[float] = []
    for i in range(1, triangulation.NbNodes() + 1):
        point = triangulation.Node(i).Transformed(trsf)
        positions.extend((point.X(), point.Y(), point.Z()))

    reversed_face = face.Orientation() == TopAbs_REVERSED
    indices: list[int] = []
    for i in range(1, triangulation.NbTriangles() + 1):
        n1, n2, n3 = triangulation.Triangle(i).Get()
        if reversed_face:
            n2, n3 = n3, n2
        indices.extend((n1 - 1, n2 - 1, n3 - 1))

    normals = [0.0] * len(positions)
    for i in range(0, len(indices), 3):
        a, b, c = indices[i], indices[i + 1], indices[i + 2]
        ax, ay, az = positions[a * 3:a * 3 + 3]
        bx, by, bz = positions[b * 3:b * 3 + 3]
        cx, cy, cz = positions[c * 3:c * 3 + 3]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        for node in (a, b, c):
            normals[node * 3] += nx
            normals[node * 3 + 1] += ny
            normals[node * 3 + 2] += nz

    for i in range(0, len(normals), 3):
        nx, ny, nz = normals[i], normals[i + 1], normals[i + 2]
        length = (nx * nx + ny * ny + nz * nz) ** 0.5
        if length > 1.0e-12:
            normals[i], normals[i + 1], normals[i + 2] = nx / length, ny / length, nz / length
        else:
            normals[i + 2] = 1.0

    return positions, normals, indices


def _face_edge_segments(face, triangulation, location, deflection) -> list[float]:
    """Discretise every edge of one face into world-space line segments.

    Preference goes to the edge's *polygon on triangulation*: those points are
    the very mesh vertices the browser draws, so the outline sits exactly on
    the rendered surface.  Sampling the exact curve instead makes the line dip
    in and out of the tessellated shell on anything curved, which reads as a
    blurry, stippled outline.
    """
    segments: list[float] = []
    trsf = location.Transformation()

    for edge in iter_edges(face):
        points = _edge_points_on_triangulation(edge, triangulation, location, trsf)
        if points is None:
            points = _edge_points_on_curve(edge, deflection)
        if not points:
            continue
        for i in range(len(points) - 1):
            p, q = points[i], points[i + 1]
            segments.extend((p.X(), p.Y(), p.Z(), q.X(), q.Y(), q.Z()))
    return segments


def _edge_points_on_triangulation(edge, triangulation, location, trsf):
    if triangulation is None:
        return None
    try:
        polygon = BRep_Tool.PolygonOnTriangulation(edge, triangulation, location)
        if polygon is None or polygon.NbNodes() < 2:
            return None
        nodes = polygon.Nodes()
        return [
            triangulation.Node(nodes.Value(i)).Transformed(trsf)
            for i in range(nodes.Lower(), nodes.Upper() + 1)
        ]
    except Exception:
        return None


def _edge_points_on_curve(edge, deflection):
    try:
        curve = BRepAdaptor_Curve(edge)
        discretizer = GCPnts_QuasiUniformDeflection(curve, deflection)
        if not discretizer.IsDone() or discretizer.NbPoints() < 2:
            return None
        return [curve.Value(discretizer.Parameter(i)) for i in range(1, discretizer.NbPoints() + 1)]
    except Exception:  # degenerate or unbounded edges are simply skipped
        return None
