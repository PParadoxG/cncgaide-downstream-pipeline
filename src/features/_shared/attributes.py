"""Pure numerical and CAD-face helpers for feature-local ontology properties."""

from __future__ import annotations

import math
import re

from contract.partmodel import PartModel

from features._shared.hole_topology import dot, unit_vector, valid_vector


_PARTMODEL_FACE_ID = re.compile(r"^face_(\d+)$")


def rounded_scalar(value: object) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return round(float(value), 6)


def canonical_axis(value: object) -> list[float] | None:
    axis = unit_vector(value)
    if axis is None:
        return None
    dominant = max(range(3), key=lambda index: abs(axis[index]))
    if axis[dominant] < 0.0:
        axis = [-component for component in axis]
    return [round(0.0 if abs(component) < 0.5e-6 else component, 6) for component in axis]


def cad_face_ids(model: PartModel, partmodel_face_ids: list[str]) -> set[int]:
    found: set[int] = set()
    for partmodel_id in partmodel_face_ids:
        face = model.faces.get(partmodel_id)
        candidates = [partmodel_id, *(getattr(face, "merged_from", []) if face else [])]
        for candidate in candidates:
            match = _PARTMODEL_FACE_ID.fullmatch(str(candidate))
            if match:
                found.add(int(match.group(1)))
    return found


def serialize_cad_face_ids(model: PartModel, partmodel_face_ids: list[str]) -> str | None:
    values = sorted(cad_face_ids(model, partmodel_face_ids))
    return ",".join(str(value) for value in values) if values else None


def first_face_bbox_center(model: PartModel, partmodel_face_ids: list[str], preferred_face_id: str | None) -> list[float] | None:
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


def valid_bbox(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    lower, upper = value.get("min"), value.get("max")
    return (
        valid_vector(lower)
        and valid_vector(upper)
        and all(float(lower[index]) <= float(upper[index]) for index in range(3))
    )

