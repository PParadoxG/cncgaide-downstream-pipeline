"""Point classification against the loaded solid, on OCC.

Interior sampling and face centroids used to live here as kernel queries;
they are static face facts and moved to parse time
(``occline.parse.face_sampling`` fills them into ``FaceData``).  What remains
is the one genuinely runtime question: whether a point the recognition layer
just constructed sits in the material.
"""

from __future__ import annotations

from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_ON, TopAbs_OUT, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.gp import gp_Pnt

from contract.kernel import (
    POINT_IN,
    POINT_ON,
    POINT_OUT,
    POINT_UNAVAILABLE,
    POINT_UNKNOWN,
)


class SolidClassifier:
    """``inside / outside / on`` against the loaded solid, or unavailable."""

    def __init__(self, shape) -> None:
        self._classifier = None
        if shape is None:
            return
        target = shape
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            target = explorer.Current()
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
