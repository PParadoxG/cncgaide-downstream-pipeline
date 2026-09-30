"""JSON serialization and deserialization of ``PartModel``.

Handles all custom types: enums (via ``.value``), tuples (↔ lists),
datetimes (ISO format), and nested dataclass collections.

This lives beside the dataclasses rather than in the parser on purpose: it is
how a stored ``partmodel.json`` is read back *without* a kernel in the
process, which is what the heuristics-only recompute and the downstream
replay test both depend on.
"""

from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

from contract.partmodel import (
    AdjacencyInfo,
    EdgeData,
    FaceData,
    LoopData,
    PartMetadata,
    PartModel,
    ShellData,
    VertexData,
)

# ---- type registry for reconstruction ---------------------------------------
_TYPE_REGISTRY: dict[str, type] = {
    "PartModel":       PartModel,
    "PartMetadata":    PartMetadata,
    "ShellData":       ShellData,
    "FaceData":        FaceData,
    "EdgeData":        EdgeData,
    "VertexData":      VertexData,
    "AdjacencyInfo":   AdjacencyInfo,
    "LoopData":        LoopData,
}

# ---------------------------------------------------------------------------
# Serialize
# ---------------------------------------------------------------------------

def _serialize_value(value: Any) -> Any:
    """Recursively convert a value to a JSON-serializable primitive."""
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return _dataclass_to_dict(value)
    # Fallback — try str
    return str(value)


def _dataclass_to_dict(obj: Any) -> dict:
    """Convert a dataclass instance to a plain dict, recording its type name."""
    result: dict = {"__type__": type(obj).__name__}
    for f in dataclasses.fields(obj):
        raw = getattr(obj, f.name)
        result[f.name] = _serialize_value(raw)
    return result


class NonFiniteNumber(ValueError):
    """The model carries a number JSON has no way to write.

    ``json.dumps`` defaults to ``allow_nan=True`` and emits bare ``NaN`` /
    ``Infinity``, which Python reads back and nothing else does - and a
    ``partmodel.json`` whose whole point is being readable without a kernel
    would then be readable only in Python.  Refusing is the seam's own rule:
    a quantity that could not be measured is ``None`` with a named reason, not
    a float that means "no answer".
    """


def _non_finite_paths(payload: Any, prefix: str = "") -> list[str]:
    """Where the values ``json.dumps`` refused actually live.

    Walked only on the error path, so the check costs nothing in the normal
    case - and the caller gets ``faces.face_12.area`` rather than a message
    that only says some float somewhere was out of range.
    """
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            found += _non_finite_paths(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            found += _non_finite_paths(value, f"{prefix}[{index}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        found.append(f"{prefix or '<root>'}={payload!r}")
    return found


# No indent and no spaces after the separators.  ``partmodel.json`` is the
# largest artifact by a wide margin (17.1 MiB over the 18 line folders of the
# nine-part corpus) and about 40% of it was indentation; the digest cannot see
# the difference, because ``store.stamps._content_hash`` parses a ``.json``
# artifact, strips the volatile keys and re-serializes it canonically before
# hashing - it asks whether the data moved, not how it was laid out.  Read it
# with ``python -m json.tool`` or the viewer; nothing reads it by eye.
_COMPACT = (",", ":")


def to_json(model: PartModel) -> str:
    """Serialize *model* to a JSON string.

    Raises :class:`NonFiniteNumber` (a ``ValueError``) rather than writing a
    file only Python can read; see that class for why.
    """
    payload = _dataclass_to_dict(model)
    try:
        return json.dumps(payload, ensure_ascii=False, allow_nan=False,
                          separators=_COMPACT)
    except ValueError as error:
        where = _non_finite_paths(payload)
        raise NonFiniteNumber(
            "the PartModel carries value(s) JSON cannot represent: "
            + ("; ".join(where[:8]) + ("; ..." if len(where) > 8 else "")
               if where else str(error))
        ) from error


# ---------------------------------------------------------------------------
# Deserialize
# ---------------------------------------------------------------------------

def _deserialize_value(value: Any, field_type: type | None = None) -> Any:
    """Recursively reconstruct typed values from JSON primitives."""
    if value is None:
        return None

    if (
        isinstance(field_type, type)
        and issubclass(field_type, Enum)
        and not isinstance(value, field_type)
    ):
        return field_type(value)

    # Dataclass dict with __type__ marker
    if isinstance(value, dict) and "__type__" in value:
        return _dict_to_dataclass(value)

    # list — check if it's a tuple field or list of dataclasses
    if isinstance(value, list):
        origin = get_origin(field_type)
        args = get_args(field_type) if field_type else ()
        elem_type = args[0] if args else None

        # Check if the original field type was tuple
        if origin is tuple:
            return tuple(
                _deserialize_value(v, elem_type if i < len(args) else None)
                for i, v in enumerate(value)
            )

        # Plain list — try to deserialize elements
        return [_deserialize_value(v, elem_type) for v in value]

    # dict without __type__ (e.g. parameters dict, faces dict)
    if isinstance(value, dict) and field_type and get_origin(field_type) is dict:
        args = get_args(field_type)
        val_type = args[1] if len(args) > 1 else None
        return {
            k: _deserialize_value(v, val_type) for k, v in value.items()
        }

    return value


def _dict_to_dataclass(d: dict) -> Any:
    """Reconstruct a dataclass instance from a ``__type__``-tagged dict."""
    type_name = d.pop("__type__")

    # No enum branch: ``_serialize_value`` turns every enum into its
    # ``.value`` string, so a ``__type__``-tagged enum dict is never
    # produced and the branch that read one was unreachable.  Pinned by
    # tests/test_node26_a.py::test_enums_are_serialized_as_plain_strings.
    cls = _TYPE_REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown type '{type_name}' in serialized data")

    field_types = get_type_hints(cls)
    kwargs: dict[str, Any] = {}

    for key, raw_value in d.items():
        ft = field_types.get(key)
        kwargs[key] = _deserialize_value(raw_value, ft)

    return cls(**kwargs)


def from_json(json_str: str) -> PartModel:
    """Deserialize a JSON string back into a ``PartModel``."""
    raw = json.loads(json_str)
    result = _deserialize_value(raw, PartModel)
    if not isinstance(result, PartModel):
        raise TypeError(
            f"Deserialized object is {type(result).__name__}, expected PartModel"
        )
    return result


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_json(model: PartModel, filepath: str) -> None:
    """Serialize *model* to a JSON file at *filepath*.

    ``newline=""`` because the default translates every ``\\n`` into
    ``\\r\\n`` on Windows while the store's own writer
    (``store.records.atomic_write_text``, which encodes to bytes first)
    does not - so one part's folder used to hold two line-ending
    conventions.  LF is the one that comes out the same on either platform.
    """
    json_str = to_json(model)
    Path(filepath).write_text(json_str, encoding="utf-8", newline="")


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------

def verify_roundtrip(original: PartModel) -> tuple[bool, str]:
    """Check that ``to_json`` → ``from_json`` preserves the model faithfully.

    Returns ``(True, "")`` on success or ``(False, error_message)``.
    """
    try:
        json_str = to_json(original)
        reconstructed = from_json(json_str)
    except Exception as exc:
        return False, f"Serialization error: {exc}"

    errors: list[str] = []

    # Compare metadata
    if original.metadata != reconstructed.metadata:
        errors.append("metadata differs")

    # Compare shells
    if len(original.shells) != len(reconstructed.shells):
        errors.append(f"shell count: {len(original.shells)} vs {len(reconstructed.shells)}")
    else:
        for i, (os, rs) in enumerate(zip(original.shells, reconstructed.shells)):
            if os != rs:
                errors.append(f"shell_{i} differs")

    # Compare faces
    if original.faces.keys() != reconstructed.faces.keys():
        errors.append("face ID sets differ")
    else:
        for fid in original.faces:
            if original.faces[fid] != reconstructed.faces[fid]:
                errors.append(f"face {fid} differs")

    # Compare edges
    if original.edges.keys() != reconstructed.edges.keys():
        errors.append("edge ID sets differ")
    else:
        for eid in original.edges:
            if original.edges[eid] != reconstructed.edges[eid]:
                errors.append(f"edge {eid} differs")

    # Compare vertices
    if original.vertices.keys() != reconstructed.vertices.keys():
        errors.append("vertex ID sets differ")
    else:
        for vid in original.vertices:
            if original.vertices[vid] != reconstructed.vertices[vid]:
                errors.append(f"vertex {vid} differs")

    # Compare loops
    if original.loops.keys() != reconstructed.loops.keys():
        errors.append("loop ID sets differ")
    else:
        for lid in original.loops:
            if original.loops[lid] != reconstructed.loops[lid]:
                errors.append(f"loop {lid} differs")

    # Compare PMI
    if original.pmi_annotations != reconstructed.pmi_annotations:
        errors.append("pmi_annotations differ")

    # Compare normalization log
    if original.normalization_log != reconstructed.normalization_log:
        errors.append("normalization_log differs")

    if errors:
        return False, "; ".join(errors)
    return True, ""

