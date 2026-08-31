"""Runtime-only OCC geometry context for exact post-parse measurements."""

from __future__ import annotations

import math
from pathlib import Path

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Shape

from occline.parse.reader import load_step
from occline.parse.traversal import iter_faces


class OccGeometryContext:
    """Loaded STEP shape plus stable zero-based source CAD-face lookup."""

    def __init__(self, shape: TopoDS_Shape, source_step: str | Path | None = None) -> None:
        self.shape = shape
        self.source_step = str(source_step) if source_step is not None else None
        self.cad_faces = list(iter_faces(shape))
        self._hash_to_indices: dict[int, list[int]] = {}
        for index, face in enumerate(self.cad_faces):
            self._hash_to_indices.setdefault(hash(face), []).append(index)
        self.max_ray_distance = _shape_ray_distance(shape)

    @classmethod
    def from_step(cls, source_step: str | Path) -> "OccGeometryContext":
        path = Path(source_step).resolve()
        return cls(load_step(str(path)), path)

    @classmethod
    def from_shape(cls, shape: TopoDS_Shape) -> "OccGeometryContext":
        return cls(shape)

    def cad_face(self, cad_face_id: int) -> TopoDS_Face | None:
        if not isinstance(cad_face_id, int) or cad_face_id < 0 or cad_face_id >= len(self.cad_faces):
            return None
        return self.cad_faces[cad_face_id]

    def cad_face_id(self, face: TopoDS_Face) -> int | None:
        for index in self._hash_to_indices.get(hash(face), []):
            if self.cad_faces[index].IsSame(face):
                return index
        for index, candidate in enumerate(self.cad_faces):
            if candidate.IsSame(face):
                return index
        return None


def _shape_ray_distance(shape: TopoDS_Shape) -> float:
    box = Bnd_Box()
    box.SetGap(0.0)
    brepbndlib.AddOptimal(shape, box, False, False)
    if box.IsVoid():
        return 1.0e6
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    diagonal = math.sqrt(
        (float(xmax) - float(xmin)) ** 2
        + (float(ymax) - float(ymin)) ** 2
        + (float(zmax) - float(zmin)) ** 2
    )
    return max(1.0, diagonal * 2.0)
