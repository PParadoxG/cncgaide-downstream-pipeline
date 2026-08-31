"""Derived display mapping for canonical v8 result nodes.

The result JSON stays ontology-facing.  This artifact holds the PartModel and
display bookkeeping required by graph and OCC renderers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from contract.partmodel import PartModel


def build_id_map(part_id: str, model: PartModel, nodes: list[Any]) -> dict[str, Any]:
    partmodel_to_display: dict[str, str] = {}
    display_mapping: dict[str, dict] = {}
    cad_to_display: dict[str, str] = {}
    cad_to_partmodel: dict[str, str] = {}
    for index, face_id in enumerate(sorted(model.faces, key=_face_sort_key), start=1):
        display_id = f"F{index:03d}"
        face = model.faces[face_id]
        cad_ids = _cad_face_ids(face_id, getattr(face, "merged_from", []))
        partmodel_to_display[face_id] = display_id
        display_mapping[display_id] = {
            "partmodelFaceId": face_id,
            "cadFaceIds": cad_ids,
            "surfaceType": face.surface_type.value,
        }
        for cad_id in cad_ids:
            cad_to_display[str(cad_id)] = display_id
            cad_to_partmodel[str(cad_id)] = face_id

    feature_to_display: dict[str, list[str]] = {}
    display_to_features: dict[str, list[str]] = {}
    cad_to_features: dict[str, list[str]] = {}
    summaries: dict[str, dict] = {}
    for node in nodes:
        node_data = node.to_dict() if hasattr(node, "to_dict") else dict(node)
        node_id = str(node_data["nodeId"])
        face_ids = list(getattr(node, "partmodelFaceIds", []))
        if not face_ids:
            face_ids = [cad_to_partmodel[str(cad)] for cad in _parse_cad_ids(node_data.get("cadFaceIds")) if str(cad) in cad_to_partmodel]
        display_ids = list(dict.fromkeys(partmodel_to_display[face_id] for face_id in face_ids if face_id in partmodel_to_display))
        feature_to_display[node_id] = display_ids
        summaries[node_id] = {
            "nodeId": node_id,
            "featureType": node_data.get("featureType"),
            "cadFaceIds": node_data.get("cadFaceIds"),
        }
        for display_id in display_ids:
            display_to_features.setdefault(display_id, []).append(node_id)
            for cad_id in display_mapping[display_id]["cadFaceIds"]:
                cad_to_features.setdefault(str(cad_id), []).append(node_id)

    return {
        "schemaVersion": "v8.id_map.v1",
        "partId": part_id,
        "idLayers": {
            "cadFaceIds": "raw source CAD face IDs before PartModel seam/merge normalization",
            "partmodelFaceIds": "active merged PartModel face IDs used by feature computation",
            "displayFaceIds": "F### labels derived from active PartModel faces for display only",
        },
        "partmodelFaceToDisplayFace": partmodel_to_display,
        "displayFaceMapping": display_mapping,
        "cadFaceToDisplayFace": cad_to_display,
        "cadFaceToPartmodelFace": cad_to_partmodel,
        "cadFaceToFeatureIds": cad_to_features,
        "featureToDisplayFaces": feature_to_display,
        "displayFaceToFeatureIds": display_to_features,
        "featureSummaries": summaries,
    }


def save_id_map(id_map: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(id_map, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_cad_ids(value: object) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value if isinstance(item, (int, float, str)) and str(item).strip().lstrip("-").isdigit()]
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip().lstrip("-").isdigit()]
    return []


def _face_sort_key(face_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"face_(\d+)", face_id)
    return (int(match.group(1)), face_id) if match else (10**9, face_id)


def _cad_face_ids(face_id: str, merged_from: list[str]) -> list[int]:
    values: list[int] = []
    for candidate in [face_id, *merged_from]:
        match = re.fullmatch(r"face_(\d+)", str(candidate))
        if match:
            values.append(int(match.group(1)))
    return sorted(set(values))
