"""Derived graph JSON and standalone dynamic graph page for v8 results."""

from __future__ import annotations

import json
from pathlib import Path

from visualization.palette import edge_color, feature_color


def build_graph_view(
    part_id: str,
    result: dict,
    id_map: dict,
    palette: dict,
    source_step: str,
) -> dict:
    """Build display-only graph data from canonical nodes and the ID map.

    No ``outputTag``.  It named the run this file happens to sit in, and this
    file is hashed into the line's artifactDigest, so the digest answered
    "which tag was this built under" as well as "did the result change" - the
    two questions it exists to keep apart (finding F-14 / P-06).  The tag is
    still stated at run level, in the manifest and in viewer_index.json, which
    is where the page reads it.
    """
    display_mapping = id_map.get("displayFaceMapping", {})
    nodes = []
    for node in result.get("nodes", []):
        node_id = node["nodeId"]
        display_face_ids = id_map.get("featureToDisplayFaces", {}).get(node_id, [])
        display_face_details = [
            {
                "displayFaceId": display_face_id,
                "cadFaceIds": display_mapping.get(display_face_id, {}).get("cadFaceIds", []),
                "partmodelFaceId": display_mapping.get(display_face_id, {}).get("partmodelFaceId"),
                "surfaceType": display_mapping.get(display_face_id, {}).get("surfaceType"),
            }
            for display_face_id in display_face_ids
        ]
        properties = {
            key: value
            for key, value in node.items()
            if key not in {"nodeId", "featureType", "partmodelFaceIds", "missingAttributes", "diagnostics", "derived"}
        }
        entry = {
            "nodeId": node_id,
            "featureType": node["featureType"],
            "displayFaceIds": display_face_ids,
            "displayFaceDetails": display_face_details,
            "partmodelFaceIds": [
                item["partmodelFaceId"]
                for item in display_face_details
                if item.get("partmodelFaceId")
            ],
            "properties": properties,
            "missingAttributes": node.get("missingAttributes", {}),
            "diagnostics": node.get("diagnostics", {}),
            "color": feature_color(palette, node["featureType"]),
        }
        # The pipeline-measured block stays apart from the recogniser's own
        # attributes all the way to the page; a node without one keeps its
        # exact former shape.
        if node.get("derived"):
            entry["derived"] = node["derived"]
        nodes.append(entry)
    partmodel_to_display = id_map.get("partmodelFaceToDisplayFace", {})
    node_types = {node["nodeId"]: node["featureType"] for node in result.get("nodes", [])}
    edges = [
        {
            **edge,
            "color": edge_color(palette, edge.get("edgeType", "")),
            "sourceFeatureType": node_types.get(edge.get("source")),
            "targetFeatureType": node_types.get(edge.get("target")),
            # Display-only projection of the evidence face pairs.  F### never
            # takes part in the relation computation itself.
            "evidenceFacePairs": _display_face_pairs(edge, partmodel_to_display),
            # Relation-specific witnesses use several field names (terminal,
            # wall, host, support).  Keep the canonical evidence untouched and
            # attach one data-driven face-reference projection for the Viewer.
            "evidenceFaceRefs": _evidence_face_refs(
                {
                    "properties": edge.get("properties") or {},
                    "evidence": edge.get("evidence") or {},
                },
                partmodel_to_display,
                id_map.get("displayFaceMapping", {}),
            ),
        }
        for edge in result.get("edges", [])
    ]
    return {
        "schemaVersion": "v8.graph_view.v1",
        "partId": part_id,
        "sourceStep": source_step,
        "nodes": nodes,
        "edges": edges,
    }


def _display_face_pairs(edge: dict, partmodel_to_display: dict) -> list[dict]:
    """Attach F### labels to whatever face pairs an edge's evidence carries."""
    pairs = (edge.get("evidence") or {}).get("adjacentFacePairs")
    if not isinstance(pairs, list):
        return []
    return [
        {
            **pair,
            "sourceDisplayFaceId": partmodel_to_display.get(pair.get("sourcePartmodelFaceId")),
            "targetDisplayFaceId": partmodel_to_display.get(pair.get("targetPartmodelFaceId")),
        }
        for pair in pairs
        if isinstance(pair, dict)
    ]


def _evidence_face_refs(evidence: dict, partmodel_to_display: dict, display_mapping: dict) -> list[dict]:
    """Project relation evidence face IDs to display/CAD identities.

    The relation layer only uses ``partmodelFaceId`` values.  This projection
    is display-only and deliberately ignores raw CAD/display IDs already
    present in diagnostics, preventing identity layers from being mixed.
    """
    refs: list[dict] = []
    seen: set[str] = set()

    def add(key: str, value: object) -> None:
        if not isinstance(value, str) or not value:
            return
        lowered = key.lower()
        if "cadface" in lowered or "displayface" in lowered:
            return
        if "face" not in lowered:
            return
        if not ("partmodel" in lowered or lowered.endswith("faceid") or lowered.endswith("faceids")):
            return
        display_id = partmodel_to_display.get(value)
        if not display_id or value in seen:
            return
        seen.add(value)
        detail = display_mapping.get(display_id, {})
        refs.append({
            "partmodelFaceId": value,
            "displayFaceId": display_id,
            "cadFaceIds": list(detail.get("cadFaceIds") or []),
            "evidenceKeys": [key],
        })

    def walk(value: object, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, str(child_key))
        elif isinstance(value, list):
            for child_value in value:
                walk(child_value, key)
        else:
            add(key, value)

    walk(evidence)
    return refs


def save_graph_artifacts(graph_view: dict, part_paths: dict[str, Path]) -> None:
    """Write the graph view.  Data only - the viewer is a separate static page.

    Written compactly, and with the newlines the JSON itself carries (none).
    ``artifactDigest`` parses a JSON artifact before hashing it, so how it is
    laid out is not part of what the digest answers; what it is part of is
    40% of this run directory (node 26 FP-1).  A human who wants it laid out
    has ``python -m json.tool``.
    """
    part_paths["graph_view"].write_text(
        json.dumps(graph_view, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8", newline="")
