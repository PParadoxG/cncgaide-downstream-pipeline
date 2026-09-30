"""Face identity, ordering and bounding-box facts.

A PartModel face id is an opaque string everywhere except here: this module is
the one place that reads the number out of it, for ordering and for the raw CAD
identity layer.  The parsing rule itself is ``contract.face_ids`` - both kernel
lines have to agree on it, so it sits at the seam.
"""

from __future__ import annotations

import math

from contract.face_ids import raw_cad_face_id
from contract.partmodel import PartModel, SurfaceType

from topology.hole_topology import wall_rims
from topology.signatures import SignatureView
from topology.vectors import dot, unit_vector, valid_vector

#: An axial span at or below this is not a span; a direction within this of
#: the stored axis is that axis.  Both are the compound layer's own numbers,
#: unchanged - they moved here with the function that reads them.
_AXIAL_TOL = 1.0e-3
_AXIS_TOL = 1.0e-3


def face_sort_key(face_id: object) -> tuple[int | float, str]:
    """Sort faces by their CAD face number, so ``face_9`` precedes ``face_10``.

    Three copies of this key existed and they disagreed on ids that are not
    ``face_<digits>``: one matched the whole string, one took the tail after the
    last underscore, one concatenated every digit anywhere in the string, and
    the three fallback buckets were ``10**9``, ``10**12`` and ``inf`` (Node 26
    finding C-12).  One rule now: the anchored one, because that is the only id
    shape either line issues, and one fallback, ``inf``, because a sentinel
    that is *definitely* last is easier to reason about than a large integer
    that a real face number could in principle reach.
    """
    text = str(face_id)
    number = raw_cad_face_id(text)
    return (number if number is not None else math.inf, text)


def valid_bbox(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    lower, upper = value.get("min"), value.get("max")
    return (
        valid_vector(lower)
        and valid_vector(upper)
        and all(float(lower[index]) <= float(upper[index]) for index in range(3))
    )


def cad_face_ids(model: PartModel, partmodel_face_ids: list[str]) -> set[int]:
    """The raw CAD face numbers behind a set of active PartModel faces."""
    found: set[int] = set()
    for partmodel_id in partmodel_face_ids:
        face = model.faces.get(partmodel_id)
        candidates = [partmodel_id, *(getattr(face, "merged_from", []) if face else [])]
        for candidate in candidates:
            number = raw_cad_face_id(candidate)
            if number is not None:
                found.add(number)
    return found


def serialize_cad_face_ids(model: PartModel, partmodel_face_ids: list[str]) -> str | None:
    values = sorted(cad_face_ids(model, partmodel_face_ids))
    return ",".join(str(value) for value in values) if values else None


def first_face_bbox_center(model: PartModel, partmodel_face_ids: list[str],
                           preferred_face_id: str | None) -> list[float] | None:
    ordered = ([preferred_face_id] if preferred_face_id else []) + [
        face_id for face_id in sorted(set(partmodel_face_ids)) if face_id != preferred_face_id
    ]
    for face_id in ordered:
        face = model.faces.get(face_id)
        bbox = getattr(face, "bbox", None)
        if not valid_bbox(bbox):
            continue
        return [
            round((float(bbox["min"][index]) + float(bbox["max"][index])) * 0.5, 6)
            for index in range(3)
        ]
    return None


def face_interval(model: PartModel, face, axis: list[float] | None,
                  signatures: SignatureView) -> tuple[float, float] | None:
    """The axial span of *face* projected onto *axis*, or ``None``.

    Three answers in order of trust: the extrema the parser persisted from
    the real B-Rep boundary, the projected positions of a cylinder's
    complete rims, and the coarse projection of the axis-aligned bounding
    box.  The first two are exact; the third over-estimates a tilted face.
    """
    if axis is None:
        return None
    # The normalizer persists true axial extrema from the B-Rep boundary.
    # They are authoritative, especially for a tilted axis where projecting an
    # axis-aligned bounding box can substantially overestimate the interval.
    minimum = face.parameters.get("axial_min")
    maximum = face.parameters.get("axial_max")
    if isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
        minimum, maximum = float(minimum), float(maximum)
        if math.isfinite(minimum) and math.isfinite(maximum) and maximum - minimum > _AXIAL_TOL:
            stored_axis = unit_vector(face.parameters.get("axis_direction"))
            orientation = dot(stored_axis, axis) if stored_axis is not None else 1.0
            if abs(abs(orientation) - 1.0) <= _AXIS_TOL:
                converted = (minimum * (1.0 if orientation >= 0.0 else -1.0), maximum * (1.0 if orientation >= 0.0 else -1.0))
                return min(converted), max(converted)
            # A persisted axial range with an unrelated axis is not reliable
            # for this query; continue to the rim/AABB fallbacks.

    # For a cylindrical wall with complete rims, use their actual projected
    # positions before falling back to the coarse AABB projection.  This keeps
    # seam/split geometry accurate without inventing a second rim detector.
    if getattr(face, "surface_type", None) == SurfaceType.CYLINDER:
        rims = wall_rims(model, face.id, signatures)
        positions = [float(rim.axis_position) for rim in rims if math.isfinite(float(rim.axis_position))]
        if len(positions) >= 2 and max(positions) - min(positions) > _AXIAL_TOL:
            return min(positions), max(positions)

    bbox = face.bbox
    if isinstance(bbox, dict) and valid_vector(bbox.get("min")) and valid_vector(bbox.get("max")):
        values = []
        for mask in range(8):
            point = [bbox["max"][index] if mask & (1 << index) else bbox["min"][index] for index in range(3)]
            values.append(dot(point, axis))
        return min(values), max(values)
    return None


class FaceIntervalTable:
    """``(face, axis) -> axial span``, computed once per part.

    A face's span along one axis is a constant of the part, but the compound
    layer asked for it 1126 times over the nine-part corpus (Node 27C leftover
    #1): ``_counterbore_matches`` and ``_countersink_matches`` walk every face
    of the model once per base face, and for a cylinder each of those walks
    re-ran the whole rim detection.  The table is the memo.

    The key includes the axis because the answer genuinely depends on it - the
    caller's axis is the *base* hole's, not the face's own - and it is the axis
    as given, not a rounded form: two directions that differ in the last bit
    miss each other and recompute, which is slower and never different.
    """

    __slots__ = ("_model", "_signatures", "_cache")

    def __init__(self, model: PartModel, signatures: SignatureView | None = None) -> None:
        self._model = model
        self._signatures = signatures if signatures is not None else SignatureView(model)
        self._cache: dict[tuple, tuple[float, float] | None] = {}

    @property
    def model(self) -> PartModel:
        return self._model

    @property
    def signatures(self) -> SignatureView:
        return self._signatures

    def interval(self, face, axis: list[float] | None) -> tuple[float, float] | None:
        key = (getattr(face, "id", None),
               None if axis is None else tuple(float(value) for value in axis))
        if key in self._cache:
            return self._cache[key]
        answer = face_interval(self._model, face, axis, self._signatures)
        self._cache[key] = answer
        return answer

