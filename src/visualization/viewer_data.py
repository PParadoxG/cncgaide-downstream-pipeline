"""Browser-readable artifacts for the web viewer.

Runs as part of ``refresh``: for every part it tessellates the B-Rep *per raw
CAD face* and writes the mesh plus a display-oriented view of the finished
graph.  Keeping the raw face as the drawing unit is what lets the page bridge
a click in 3D back to a feature, using the identities ``id_map.json`` already
records — no display-only identity is invented here.

Nothing in this module recognises features or computes relations.  Feature
nodes, attributes and relation edges are copied out of the finished
``graph_view``; face ownership comes from the finished ``id_map``.

Nor does it hold a geometry kernel.  The triangles arrive as plain numbers
from ``KernelServices.tessellate``; what stays here is the part that is
actually about display - how fine to mesh relative to the part's size, how to
pack the buffers, and what the header says.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

# Chord height and angular deviation of the tessellation, relative to the part
# bounding-box diagonal.  Tighter than "looks fine" on purpose: the edge
# overlay is built from these very mesh vertices, so a coarse mesh shows up as
# a ragged outline, not just a faceted surface.
LINEAR_DEFLECTION_RATIO = 0.0006
ANGULAR_DEFLECTION = 0.16
EDGE_DEFLECTION_RATIO = 0.0006

# Feature types the graph hides until asked.  PlanarFace dominates every part
# and drowns out the manufacturing features it sits between.
DEFAULT_HIDDEN_FEATURE_TYPES = ("PlanarFace",)

# No relation is drawn until it is asked for.  Adjacency alone accounts for
# well over half of all edges and turns the graph into a hairball, and every
# relation registered from here on would otherwise quietly bury it further —
# the reviewer switches on the ones the current question needs.
DEFAULT_VISIBLE_EDGE_TYPES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def _diagonal(bbox: list[float]) -> float:
    dx, dy, dz = bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]
    return max(1.0e-6, (dx * dx + dy * dy + dz * dz) ** 0.5)


def _face_moments(positions, indices):
    """Tessellated area, area-weighted centroid and mean outward normal.

    The mean normal is what lets the viewer swing the camera to a side where
    the feature is exposed.  It is near-zero for a closed bore wall, which the
    browser detects and falls back from.
    """
    total = 0.0
    cx = cy = cz = 0.0
    nx = ny = nz = 0.0
    for i in range(0, len(indices), 3):
        a, b, c = indices[i] * 3, indices[i + 1] * 3, indices[i + 2] * 3
        ax, ay, az = positions[a:a + 3]
        bx, by, bz = positions[b:b + 3]
        cx3, cy3, cz3 = positions[c:c + 3]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx3 - ax, cy3 - ay, cz3 - az
        tx = uy * vz - uz * vy
        ty = uz * vx - ux * vz
        tz = ux * vy - uy * vx
        area = 0.5 * (tx * tx + ty * ty + tz * tz) ** 0.5
        if area <= 0.0:
            continue
        total += area
        cx += area * (ax + bx + cx3) / 3.0
        cy += area * (ay + by + cy3) / 3.0
        cz += area * (az + bz + cz3) / 3.0
        nx += 0.5 * tx
        ny += 0.5 * ty
        nz += 0.5 * tz

    if total <= 0.0:
        return 0.0, None, None
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    normal = [nx / length, ny / length, nz / length] if length > 1.0e-12 else None
    return total, [cx / total, cy / total, cz / total], normal


# ---------------------------------------------------------------------------
# mesh artifact
# ---------------------------------------------------------------------------

def build_mesh(kernel, part_paths: dict[str, Path]) -> dict:
    """Pack the kernel's per-face tessellation into mesh.bin / mesh.json."""
    bbox = kernel.tessellation_bounds()
    diagonal = _diagonal(bbox)

    linear_deflection = diagonal * LINEAR_DEFLECTION_RATIO
    edge_deflection = diagonal * EDGE_DEFLECTION_RATIO
    face_meshes = kernel.tessellate(linear_deflection, ANGULAR_DEFLECTION, edge_deflection)

    all_positions: list[float] = []
    all_normals: list[float] = []
    all_indices: list[int] = []
    all_edges: list[float] = []
    face_records: list[dict] = []
    empty_faces: list[int] = []

    for face_mesh in face_meshes:
        cad_face_id = face_mesh.cad_face_id
        positions, normals, indices = face_mesh.positions, face_mesh.normals, face_mesh.indices
        if not indices:
            empty_faces.append(cad_face_id)

        vertex_offset = len(all_positions) // 3
        index_offset = len(all_indices)
        all_positions.extend(positions)
        all_normals.extend(normals)
        all_indices.extend(indices)

        segments = face_mesh.edge_segments
        edge_offset = len(all_edges) // 3
        all_edges.extend(segments)

        area, centroid, normal = _face_moments(positions, indices)
        face_records.append({
            "cadFaceId": cad_face_id,
            "vertexOffset": vertex_offset,
            "vertexCount": len(positions) // 3,
            "indexOffset": index_offset,
            "indexCount": len(indices),
            "edgeVertexOffset": edge_offset,
            "edgeVertexCount": len(segments) // 3,
            "area": area,
            "centroid": centroid,
            "normal": normal,
        })

    position_bytes = struct.pack("<%df" % len(all_positions), *all_positions)
    normal_bytes = struct.pack("<%df" % len(all_normals), *all_normals)
    index_bytes = struct.pack("<%dI" % len(all_indices), *all_indices)
    edge_bytes = struct.pack("<%df" % len(all_edges), *all_edges)
    part_paths["mesh_bin"].write_bytes(position_bytes + normal_bytes + index_bytes + edge_bytes)

    header = {
        "schemaVersion": "v8.viewer_mesh.v1",
        "cadFaceCount": len(face_meshes),
        "boundingBox": {"min": bbox[0:3], "max": bbox[3:6]},
        "tessellation": {
            "linearDeflection": linear_deflection,
            "angularDeflection": ANGULAR_DEFLECTION,
            "edgeDeflection": edge_deflection,
        },
        "buffers": {
            "positions": {"byteOffset": 0, "count": len(all_positions) // 3, "type": "Float32", "itemSize": 3},
            "normals": {
                "byteOffset": len(position_bytes),
                "count": len(all_normals) // 3, "type": "Float32", "itemSize": 3,
            },
            "indices": {
                "byteOffset": len(position_bytes) + len(normal_bytes),
                "count": len(all_indices), "type": "Uint32", "itemSize": 1,
            },
            "edgePositions": {
                "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes),
                "count": len(all_edges) // 3, "type": "Float32", "itemSize": 3,
            },
        },
        "faces": face_records,
        "emptyCadFaceIds": empty_faces,
    }
    part_paths["mesh_index"].write_text(json.dumps(header), encoding="utf-8")
    return header


# ---------------------------------------------------------------------------
# view artifact
# ---------------------------------------------------------------------------

def _diagnostics_digest(node: dict) -> dict:
    """Keep the audit-relevant part of the diagnostics, drop the bulk.

    The allow-list below is right for a recognizer whose diagnostics are
    mostly per-face classifier scores, and wrong for one whose diagnostics are
    a dozen scalars a reader actually wants to see.  So a node may also name
    its own keys, in ``diagnostics["display"]``, and those are copied through
    as they are.  Nothing here learns what any of them mean - which is what
    keeps one line's vocabulary out of the shared writer.
    """
    diagnostics = node.get("diagnostics") or {}
    digest: dict = {}
    for name in diagnostics.get("display") or []:
        if name in diagnostics:
            digest[name] = diagnostics[name]
    seed = diagnostics.get("seed")
    if isinstance(seed, dict):
        digest["seedPartmodelFaceId"] = seed.get("partmodelFaceId")
        digest["seedTopLabel"] = seed.get("topLabel")
        top_k = seed.get("topK") or []
        if top_k and isinstance(top_k[0], dict):
            digest["seedTopScore"] = top_k[0].get("softmax_score")
    topology = diagnostics.get("topology")
    if isinstance(topology, dict) and topology.get("classification"):
        digest["topologyClassification"] = topology.get("classification")
    for key in ("method", "surfaceType", "area"):
        if key in diagnostics:
            digest[key] = diagnostics[key]
    return digest


def _edge_type_color(palette: dict, edge_type: str, fallback_index: int) -> str:
    """Palette colour when the edge type has one, otherwise a stable hue.

    A relation registered after this file was written must still be legible;
    falling back to one shared grey would make two new relations
    indistinguishable.  The hue comes from the type name, so it is stable
    across runs without needing a palette entry.
    """
    declared = (palette.get("edges") or {}).get(edge_type)
    if declared:
        return str(declared)
    hue = (sum(ord(char) * (index + 1) for index, char in enumerate(edge_type)) * 47) % 360
    return f"hsl({hue}, 45%, 55%)"


def build_view(
    part_id: str,
    graph_view: dict,
    id_map: dict,
    mesh_header: dict,
    palette: dict,
    face_count: int,
    claimed_face_count: int,
    hidden_feature_types: tuple[str, ...] | list[str] = DEFAULT_HIDDEN_FEATURE_TYPES,
    feature_labels: dict[str, str] | None = None,
    not_evaluated: dict[str, str] | None = None,
    used_attributes: dict[str, list[str]] | None = None,
) -> dict:
    display_face_mapping = id_map.get("displayFaceMapping", {})
    cad_to_display = id_map.get("cadFaceToDisplayFace", {})
    cad_to_partmodel = id_map.get("cadFaceToPartmodelFace", {})
    cad_to_features = id_map.get("cadFaceToFeatureIds", {})

    nodes = []
    for node in graph_view.get("nodes", []):
        cad_ids: list[int] = []
        for detail in node.get("displayFaceDetails") or []:
            for cad_id in detail.get("cadFaceIds") or []:
                if int(cad_id) not in cad_ids:
                    cad_ids.append(int(cad_id))
        entry = {
            "nodeId": node.get("nodeId"),
            "featureType": node.get("featureType"),
            "color": node.get("color"),
            "displayFaceIds": list(node.get("displayFaceIds") or []),
            "partmodelFaceIds": list(node.get("partmodelFaceIds") or []),
            "cadFaceIds": sorted(cad_ids),
            "displayFaceDetails": node.get("displayFaceDetails") or [],
            "properties": node.get("properties") or {},
            "missingAttributes": node.get("missingAttributes") or {},
            "diagnostics": _diagnostics_digest(node),
        }
        # Written only when the line derived something, so a line that never
        # does keeps byte-identical artifacts.
        if node.get("derived"):
            entry["derived"] = node["derived"]
        nodes.append(entry)

    node_ids = {node["nodeId"] for node in nodes}
    edges = []
    edge_type_counts: dict[str, int] = {}
    for edge in graph_view.get("edges", []):
        if edge.get("source") not in node_ids or edge.get("target") not in node_ids:
            continue
        evidence = edge.get("evidence") or {}
        edge_type = edge.get("edgeType")
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
        edges.append({
            "edgeId": edge.get("edgeId"),
            "source": edge.get("source"),
            "target": edge.get("target"),
            "edgeType": edge_type,
            "directed": bool(edge.get("directed")),
            "sharedEdgeCount": evidence.get("sharedEdgeCount"),
            "connectionTypes": evidence.get("connectionTypes") or {},
            # Keep the relation's own properties and the complete evidence so
            # the inspector can show relation-specific metrics/witnesses
            # without display branches.
            "properties": edge.get("properties") or {},
            "evidence": evidence,
            "evidenceFacePairs": edge.get("evidenceFacePairs") or [],
            "evidenceFaceRefs": edge.get("evidenceFaceRefs") or [],
            "diagnostics": edge.get("diagnostics") or {},
        })

    relation_types = [
        {
            "edgeType": edge_type,
            "count": count,
            "color": _edge_type_color(palette, edge_type, index),
            "defaultVisible": edge_type in DEFAULT_VISIBLE_EDGE_TYPES,
        }
        for index, (edge_type, count) in enumerate(sorted(edge_type_counts.items()))
    ]

    # Raw CAD face table.  Every tessellated face gets a row, including the
    # ones no feature claimed — those still have to be reviewable.
    faces = {}
    for record in mesh_header["faces"]:
        cad_id = record["cadFaceId"]
        key = str(cad_id)
        display_id = cad_to_display.get(key)
        detail = display_face_mapping.get(display_id, {}) if display_id else {}
        faces[key] = {
            "cadFaceId": cad_id,
            "displayFaceId": display_id,
            "partmodelFaceId": cad_to_partmodel.get(key),
            "surfaceType": detail.get("surfaceType"),
            "featureIds": list(cad_to_features.get(key) or []),
            "mergedCadFaceIds": [int(value) for value in (detail.get("cadFaceIds") or [])],
            "hasMesh": record["indexCount"] > 0,
        }

    feature_counts: dict[str, int] = {}
    for node in nodes:
        feature_counts[node["featureType"]] = feature_counts.get(node["featureType"], 0) + 1

    view = {
        "schemaVersion": "v8.viewer_view.v1",
        "partId": part_id,
        "sourceStep": graph_view.get("sourceStep"),
        "outputTag": graph_view.get("outputTag"),
        "nodes": nodes,
        "edges": edges,
        "relationTypes": relation_types,
        "faces": faces,
        "featureCounts": feature_counts,
        "defaultHiddenFeatureTypes": list(hidden_feature_types),
        "stats": {
            "rawCadFaceCount": mesh_header["cadFaceCount"],
            "manifestPartmodelFaceCount": face_count,
            "manifestClaimedFaceCount": claimed_face_count,
            "displayFaceCount": len(display_face_mapping),
            "featureNodeCount": len(nodes),
            "unmappedRawCadFaceIds": sorted(int(k) for k, v in faces.items() if v["displayFaceId"] is None),
            "unclaimedRawCadFaceIds": sorted(int(k) for k, v in faces.items() if not v["featureIds"]),
            "emptyMeshCadFaceIds": mesh_header["emptyCadFaceIds"],
            "mergedDisplayFaces": {
                display_id: [int(value) for value in detail.get("cadFaceIds") or []]
                for display_id, detail in display_face_mapping.items()
                if len(detail.get("cadFaceIds") or []) > 1
            },
        },
    }
    # Both optional blocks are omitted when empty rather than written as an
    # empty object: a line that has neither must produce byte-for-byte the
    # same view.json it produced before they existed, which is what lets the
    # artifact digest prove nothing else moved.
    if feature_labels:
        # ``{class: the words a human reads}``.  The page prints the label and
        # keeps the class in the tooltip; nothing here knows what either
        # string means, and no display code branches on one.
        view["featureLabels"] = dict(sorted(feature_labels.items()))
    if not_evaluated:
        # ``{layer: why this line evaluated none of it}``.  The panels read
        # this instead of guessing from an absent file, so "no verdicts here"
        # and "this run predates verdicts" stay different statements.
        view["notEvaluated"] = dict(sorted(not_evaluated.items()))
    if used_attributes:
        # ``{class: the attribute names this pipeline reads}``.  A vendor can
        # publish a few dozen attributes per feature, of which the pipeline
        # consumes a handful; the page expands the handful and folds the rest
        # into one line.  It compares names and never learns what one means,
        # and a line that does not supply the table gets everything expanded -
        # which is what the open-source line, whose attributes are all its
        # own, actually wants.
        view["usedAttributes"] = {
            name: sorted(values) for name, values in sorted(used_attributes.items())
        }
    return view


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

def verify(view: dict, mesh_header: dict) -> list[dict]:
    """Cheap invariants over what was just written, recorded in the index."""
    stats = view["stats"]
    checks = []

    mapped = mesh_header["cadFaceCount"] - len(stats["unmappedRawCadFaceIds"])
    checks.append({
        "check": "rawCadFaceCoverage",
        "status": "OK" if not stats["unmappedRawCadFaceIds"] else "WARN",
        "detail": "%d/%d raw CAD faces resolve to a display face" % (mapped, mesh_header["cadFaceCount"]),
    })
    checks.append({
        "check": "meshCompleteness",
        "status": "OK" if not mesh_header["emptyCadFaceIds"] else "WARN",
        "detail": "%d raw CAD faces produced no triangles" % len(mesh_header["emptyCadFaceIds"]),
    })

    face_feature_pairs = {(cid, fid) for cid, row in view["faces"].items() for fid in row["featureIds"]}
    node_feature_pairs = {(str(cid), node["nodeId"]) for node in view["nodes"] for cid in node["cadFaceIds"]}
    checks.append({
        "check": "featureFaceRoundTrip",
        "status": "OK" if face_feature_pairs == node_feature_pairs else "FAIL",
        "detail": "face->feature pairs=%d, feature->face pairs=%d, symmetric_difference=%d"
                  % (len(face_feature_pairs), len(node_feature_pairs),
                     len(face_feature_pairs ^ node_feature_pairs)),
    })
    checks.append({
        "check": "mergedFaceFanOut",
        "status": "OK",
        "detail": "%d display faces are backed by more than one raw CAD face"
                  % len(stats["mergedDisplayFaces"]),
    })
    checks.append({
        "check": "unclaimedFacesBrowsable",
        "status": "OK",
        "detail": "%d raw CAD faces carry no manufacturing feature"
                  % len(stats["unclaimedRawCadFaceIds"]),
    })
    return checks


# ---------------------------------------------------------------------------
# entry point used by the orchestrator
# ---------------------------------------------------------------------------

def build_part_viewer_artifacts(
    part_id: str,
    kernel,
    graph_view: dict,
    id_map: dict,
    palette: dict,
    part_paths: dict[str, Path],
    face_count: int,
    claimed_face_count: int,
    hidden_feature_types: tuple[str, ...] | list[str] = DEFAULT_HIDDEN_FEATURE_TYPES,
    feature_labels: dict[str, str] | None = None,
    not_evaluated: dict[str, str] | None = None,
    used_attributes: dict[str, list[str]] | None = None,
) -> dict:
    """Write mesh.bin / mesh.json / view.json for one part.

    *kernel* is the one the recognition layers just used, holding the solid
    already in memory; the STEP is never re-read, so the raw CAD face order is
    by construction the same one the rest of the pipeline used.

    The four optional arguments are what a *line* knows and this module must
    not: which of its classes are a background layer, what a class is called in
    words, which layers it did not evaluate at all, and which of the attributes
    it publishes the pipeline actually reads.  Their defaults are the
    open-source line's, so nothing here has to name a vendor's vocabulary.
    """
    mesh_header = build_mesh(kernel, part_paths)
    view = build_view(part_id, graph_view, id_map, mesh_header, palette,
                      face_count, claimed_face_count,
                      hidden_feature_types=hidden_feature_types,
                      feature_labels=feature_labels, not_evaluated=not_evaluated,
                      used_attributes=used_attributes)
    part_paths["viewer_view"].write_text(json.dumps(view), encoding="utf-8")

    return {
        "partId": part_id,
        "rawCadFaceCount": mesh_header["cadFaceCount"],
        "partmodelFaceCount": face_count,
        "featureNodeCount": view["stats"]["featureNodeCount"],
        "featureCounts": view["featureCounts"],
        # The run-level index folds these across every line, so the home page
        # can hide both lines' background layers and label both lines' classes
        # without knowing either vocabulary.
        "defaultHiddenFeatureTypes": list(hidden_feature_types),
        "featureLabels": dict(sorted((feature_labels or {}).items())),
        "relationTypes": view["relationTypes"],
        "relationEdgeCount": len(view["edges"]),
        "unclaimedRawCadFaceCount": len(view["stats"]["unclaimedRawCadFaceIds"]),
        "meshBytes": part_paths["mesh_bin"].stat().st_size,
        "checks": verify(view, mesh_header),
    }
