"""Exact distance between two sets of raw CAD faces, on OCC.

The relation layer asks this to publish a pocket-to-pocket clearance, so a
wrong answer is worse than none: every failure mode gets its own name in
:class:`~contract.kernel.FaceSetDistance` and travels to the edge's evidence
intact.  The witness faces are reported only when the solver actually exposes
its support sub-shapes - naming the first input face as the nearest one would
be a guess dressed as a measurement.
"""

from __future__ import annotations

import math

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopAbs import (
    TopAbs_COMPOUND,
    TopAbs_COMPSOLID,
    TopAbs_EDGE,
    TopAbs_FACE,
    TopAbs_SHAPE,
    TopAbs_SHELL,
    TopAbs_SOLID,
    TopAbs_VERTEX,
    TopAbs_WIRE,
)
from OCC.Core.TopoDS import TopoDS_Compound

from contract.kernel import (
    DISTANCE_FACE_MAPPING_FAILED,
    DISTANCE_MEASURED,
    DISTANCE_NONFINITE,
    DISTANCE_SOLVER_EXCEPTION,
    DISTANCE_SOLVER_FAILED,
    FaceSetDistance,
)

# So a refusal can name what the solver actually handed back.  A distance
# solver's nearest-point support is usually a vertex, and "vertex" is a far
# more useful thing to read in the evidence than a bare enum number.
_SHAPE_TYPE_NAMES = {
    TopAbs_COMPOUND: "compound",
    TopAbs_COMPSOLID: "compsolid",
    TopAbs_SOLID: "solid",
    TopAbs_SHELL: "shell",
    TopAbs_FACE: "face",
    TopAbs_WIRE: "wire",
    TopAbs_EDGE: "edge",
    TopAbs_VERTEX: "vertex",
    TopAbs_SHAPE: "shape",
}


def _shape_type_name(shape) -> str:
    try:
        return _SHAPE_TYPE_NAMES.get(shape.ShapeType(), "unknown")
    except Exception:
        return "unknown"


def face_set_distance(context, left_cad_face_ids, right_cad_face_ids) -> FaceSetDistance:
    """Minimum distance between two raw-CAD-face sets of the loaded shape."""
    try:
        def compound(ids):
            result = TopoDS_Compound()
            builder = BRep_Builder()
            builder.MakeCompound(result)
            added = 0
            for cad_id in ids:
                face = context.cad_face(int(cad_id))
                if face is None:
                    continue
                builder.Add(result, face)
                added += 1
            return result if added else None

        left_shape, right_shape = compound(left_cad_face_ids), compound(right_cad_face_ids)
        if left_shape is None or right_shape is None:
            return FaceSetDistance(DISTANCE_FACE_MAPPING_FAILED)
        solver = BRepExtrema_DistShapeShape(left_shape, right_shape)
        if not solver.Perform() or not solver.IsDone():
            return FaceSetDistance(DISTANCE_SOLVER_FAILED)
        value = float(solver.Value())
        if not math.isfinite(value):
            return FaceSetDistance(DISTANCE_NONFINITE)
        point_a, point_b = solver.PointOnShape1(1), solver.PointOnShape2(1)
        # The solver returns nearest points but does not consistently expose
        # support sub-shapes across pythonOCC versions.  Do not claim that the
        # first input face is the nearest witness.  If support access is
        # available, map the actual support faces back to raw CAD IDs.
        #
        # No witness is a normal outcome, not a rare one: the nearest point of
        # two face compounds usually sits on a vertex or an edge, and only a
        # face can be a witness face.  Whichever way it ends without one, the
        # reason travels with the measurement - a reader who sees
        # ``witnessFaceStatus: unavailable`` and no reason cannot tell a
        # geometric fact from a broken lookup.
        witness_ids: list[int] | None = None
        witness_reason: str | None = None
        witness_cad: list[int | None] = []
        non_face_supports: list[str] = []
        try:
            support_left = solver.SupportOnShape1(1)
            support_right = solver.SupportOnShape2(1)
            for support in (support_left, support_right):
                if _shape_type_name(support) != "face":
                    non_face_supports.append(_shape_type_name(support))
                    witness_cad.append(None)
                    continue
                mapped = context.cad_face_id(support)
                witness_cad.append(int(mapped) if mapped is not None else None)
        except Exception as exc:
            witness_reason = f"{type(exc).__name__}: {exc}"
        else:
            if len(witness_cad) == 2 and all(item is not None for item in witness_cad):
                witness_ids = [int(item) for item in witness_cad]
            elif non_face_supports:
                witness_reason = (
                    "nearest point support is "
                    + " and ".join(non_face_supports)
                    + ", not a face"
                )
            else:
                witness_reason = (
                    "support face is not one of the loaded raw CAD faces"
                )
        return FaceSetDistance(
            DISTANCE_MEASURED,
            distance=value,
            point_a=[float(point_a.X()), float(point_a.Y()), float(point_a.Z())],
            point_b=[float(point_b.X()), float(point_b.Y()), float(point_b.Z())],
            witness_cad_face_ids=witness_ids,
            witness_reason=witness_reason,
        )
    except Exception as exc:  # OCC version differences are a proof failure, not a guessed metric.
        return FaceSetDistance(DISTANCE_SOLVER_EXCEPTION, error=f"{type(exc).__name__}: {exc}")
