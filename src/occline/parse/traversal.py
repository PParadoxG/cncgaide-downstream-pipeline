"""TopExp_Explorer wrappers for traversing the B-Rep topological hierarchy.

Each iterator yields exactly one shape type.  For shells, faces, and edges
the parent may be any *TopoDS_Shape*; wires and vertices require their
natural parent (face and edge respectively).
"""

from typing import Iterator

from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHAPE,
    TopAbs_SHELL,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import (
    TopoDS_Edge,
    TopoDS_Face,
    TopoDS_Shape,
    TopoDS_Shell,
    TopoDS_Vertex,
    TopoDS_Wire,
    topods,
)


def iter_shells(shape: TopoDS_Shape) -> Iterator[TopoDS_Shell]:
    """Yield every *TopoDS_Shell* found under *shape*.

    Traverses through intermediate levels (Compound, Solid) so the root
    shape from ``STEPControl_Reader.OneShape()`` can be passed directly.
    """
    exp = TopExp_Explorer(shape, TopAbs_SHELL, TopAbs_SHAPE)
    while exp.More():
        yield topods.Shell(exp.Current())
        exp.Next()


def iter_faces(parent: TopoDS_Shape) -> Iterator[TopoDS_Face]:
    """Yield every *TopoDS_Face* found under *parent*.

    *parent* may be a shell or the root shape.  Traverses through
    intermediate levels so both calling conventions produce the same
    result.
    """
    exp = TopExp_Explorer(parent, TopAbs_FACE, TopAbs_SHAPE)
    while exp.More():
        yield topods.Face(exp.Current())
        exp.Next()


def iter_wires(face: TopoDS_Face) -> Iterator[TopoDS_Wire]:
    """Yield every *TopoDS_Wire* directly contained in *face*."""
    exp = TopExp_Explorer(face, TopAbs_WIRE, TopAbs_SHAPE)
    while exp.More():
        yield topods.Wire(exp.Current())
        exp.Next()


def iter_edges(parent: TopoDS_Shape) -> Iterator[TopoDS_Edge]:
    """Yield every *TopoDS_Edge* found under *parent*.

    *parent* may be a wire, face, shell, or root shape.  Traverses through
    intermediate levels (faces, wires) so all calling conventions produce
    the full edge set.
    """
    exp = TopExp_Explorer(parent, TopAbs_EDGE, TopAbs_SHAPE)
    while exp.More():
        yield topods.Edge(exp.Current())
        exp.Next()


def iter_vertices(edge: TopoDS_Edge) -> Iterator[TopoDS_Vertex]:
    """Yield every *TopoDS_Vertex* directly contained in *edge*.

    Uses ``TopAbs_SHAPE`` as *ToAvoid* so the explorer descends into the
    edge.  ``TopAbs_EDGE`` was originally used but it blocks traversal
    because the starting shape *is* an edge (same issue as D15 for the
    generic-parent iterators).
    """
    exp = TopExp_Explorer(edge, TopAbs_VERTEX, TopAbs_SHAPE)
    while exp.More():
        yield topods.Vertex(exp.Current())
        exp.Next()


def _iter_vertices_from_shape(shape: TopoDS_Shape) -> Iterator[TopoDS_Vertex]:
    """Yield every vertex in *shape* (recursive from root)."""
    exp = TopExp_Explorer(shape, TopAbs_VERTEX, TopAbs_SHAPE)
    while exp.More():
        yield topods.Vertex(exp.Current())
        exp.Next()

