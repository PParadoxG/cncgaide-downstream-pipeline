"""Point classification against the loaded solid, on OCC.

Interior sampling and face centroids used to live here as kernel queries;
they are static face facts and moved to parse time
(``occline.parse.face_sampling`` fills them into ``FaceData``).  What remains
is the one genuinely runtime question: whether a point the recognition layer
just constructed sits in the material.
"""

from __future__ import annotations

from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT
from OCC.Core.gp import gp_Pnt

from contract.kernel import (
    POINT_IN,
    POINT_ON,
    POINT_OUT,
    POINT_UNAVAILABLE,
    POINT_UNKNOWN,
)
from occline.parse.traversal import iter_solids


class SolidClassifier:
    """``inside / outside / on`` against the loaded solid, or unavailable.

    One classifier can only speak for one solid.  Building it from the first
    solid of a multi-body shape answers ``out`` for every point inside the
    others - a wrong answer wearing the face of a right one, which is the one
    thing this seam forbids.  So a shape with more than one solid makes the
    service *unavailable* and the caller records a named gap.  Multi-body STEP
    is refused a layer earlier anyway
    (:class:`occline.parse.reader.MultiBodyStepUnsupported`); this is the same
    ruling stated where the wrong answer would have been produced.
    """

    def __init__(self, shape) -> None:
        self._classifier = None
        if shape is None:
            return
        solids = list(iter_solids(shape))
        if len(solids) > 1:
            return
        target = solids[0] if solids else shape
        try:
            self._classifier = BRepClass3d_SolidClassifier(target)
        except Exception:
            self._classifier = None

    @property
    def available(self) -> bool:
        return self._classifier is not None

    def state(self, point: list[float], tolerance: float) -> str:
        if self._classifier is None:
            return POINT_UNAVAILABLE
        try:
            self._classifier.Perform(gp_Pnt(float(point[0]), float(point[1]), float(point[2])), float(tolerance))
            state = self._classifier.State()
        except Exception:
            return POINT_UNAVAILABLE
        if state == TopAbs_IN:
            return POINT_IN
        if state == TopAbs_OUT:
            return POINT_OUT
        if state == TopAbs_ON:
            return POINT_ON
        return POINT_UNKNOWN
