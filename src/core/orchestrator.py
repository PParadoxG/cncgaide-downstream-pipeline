"""One part, end to end: parse it, get evidence, recognise it, write the files.

This module owns the *outside* of processing a part - loading the STEP,
running AAGNet, building the kernel, and turning what came back into
``partmodel.json`` / ``result.json`` / ``heuristics.json`` / the viewer
artifacts.  The order the recognition layers run in is not here; it is in
``core.recognition``, which knows nothing about files or kernels and
can therefore be replayed from stored data.

``refresh`` itself is a thin wrapper over the parts store
(``store.sync``): the store decides which parts are stale and calls
``_process_one`` only for those; this module owns how *one* part is processed.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from occline.parse import process_step_file
from contract.partmodel_json import save_json

from occline.aagnet.face_bridge import build_cad_face_mappings
from occline.aagnet.per_face import run_aagnet_per_face
from occline.kernel.occ_kernel import OccKernelServices
from contract.registry import LINE_OCC
from core.paths import ensure_part_paths, part_paths, project_relative, run_relative
from core.recognition import run_recognition
from visualization.graph_view import build_graph_view, save_graph_artifacts
from visualization.id_map import build_id_map, save_id_map
from visualization.palette import load_palette
from visualization.viewer_data import build_part_viewer_artifacts


def refresh(
    input_path: str,
    project_root: str | Path,
    *,
    output_tag: str = "latest",
    limit: int = 0,
    device: str = "auto",
    build_viewer: bool = True,
    mode: str = "stale",
    only: list[str] | None = None,
    keep_missing: bool = False,
) -> dict:
    """Synchronise ``runs/<output_tag>`` with *input_path* and return the manifest.

    Incremental by default (``mode="stale"``): new and changed parts are
    processed, parts whose only stale layer is the heuristics get their
    verdicts recomputed, everything else is left untouched.  ``mode="all"``
    reprocesses every part that still has a source.  The run directory is
    never cleared; the indexes are rebuilt from the per-part records.
    """
    from store import sync

    plan = sync.plan(
        project_root, input_path, output_tag,
        only=only, mode=mode, limit=limit, build_viewer=build_viewer,
    )
    summary = sync.execute(plan, device=device, build_viewer=build_viewer,
                           keep_missing=keep_missing)
    return summary["manifest"]


def _process_one(
    root: Path,
    output_tag: str,
    step_path: Path,
    part_id: str,
    model_bundle: dict,
    build_viewer: bool = True,
    *,
    paths: dict[str, Path] | None = None,
) -> dict:
    """Process one STEP into the artifact set of one part.

    *paths* lets the store point the writes at a staging directory and publish
    them atomically afterwards; by default the part's live directory is used.
    """
    paths = paths or ensure_part_paths(root, output_tag, part_id, LINE_OCC)
    model = process_step_file(str(step_path))
    save_json(model, str(paths["partmodel"]))
    # The evidence records where things are *logically* - relative to the
    # project and to the run - not where this build happened to write them.
    # ``paths`` may be a staging directory, so the reference is derived from
    # the part's live location rather than from the file being written.
    live = part_paths(root, output_tag, part_id, LINE_OCC)
    evidence = _produce_evidence(
        model, step_path, paths, model_bundle,
        source_ref=project_relative(root, step_path),
        native_groups_ref=run_relative(root, output_tag, live["native_groups"]),
    )
    # The recognition chain itself lives in ``core.recognition``:
    # this module owns parsing, evidence and the files, that one owns the
    # order the layers run in.  The split is what lets the replay guard drive
    # the same sequence from a stored PartModel and a stub kernel.
    kernel = OccKernelServices.from_step(step_path)
    recognition = run_recognition(model, evidence, kernel)
    nodes, edges = recognition.nodes, recognition.edges
    result = {
        "schemaVersion": "v8.result.v1",
        "partId": part_id,
        "sourceStep": project_relative(root, step_path),
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges],
        "diagnostics": recognition.diagnostics,
    }
    paths["result"].write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["heuristics"].write_text(
        json.dumps({"partId": part_id, **recognition.heuristics}, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    id_map = build_id_map(part_id, model, nodes)
    save_id_map(id_map, paths["id_map"])
    palette = load_palette(root)
    graph_view = build_graph_view(part_id, result, id_map, palette, result["sourceStep"], output_tag)
    save_graph_artifacts(graph_view, paths)
    counts = Counter(node.featureType for node in nodes)
    claimed_ids = {face_id for node in nodes for face_id in node.partmodelFaceIds}
    # The viewer is fed from the finished graph and the kernel already in
    # memory, so it observes exactly what the pipeline decided and never
    # re-reads the STEP.
    viewer_entry = (
        build_part_viewer_artifacts(
            part_id, kernel, graph_view, id_map, palette, paths,
            face_count=len(model.faces), claimed_face_count=len(claimed_ids),
        )
        if build_viewer
        else None
    )
    return {
        "_viewer": viewer_entry,
        "partId": part_id,
        "sourceStep": result["sourceStep"],
        "sourceStepAbs": str(step_path.resolve()),
        "faceCount": len(model.faces),
        "claimedFaceCount": len(claimed_ids),
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "featureCounts": dict(sorted(counts.items())),
        "relationCounts": recognition.relation_diagnostics["relationCounts"],
        "heuristicCounts": recognition.heuristic_counts(),
        "heuristicSummary": recognition.heuristic_summary,
    }


def _produce_evidence(model, step_path: Path, paths: dict[str, Path], bundle: dict,
                      *, source_ref: str, native_groups_ref: str) -> dict:
    evidence = run_aagnet_per_face(
        str(step_path),
        device=bundle["device"],
        model=bundle["model"],
        stat=bundle["stat"],
        attr_schema=bundle["attr_schema"],
        face_mappings=build_cad_face_mappings(model),
        save_native_groups_path=str(paths["native_groups"]),
        save_tensor_shapes_path=str(paths["tensor_shapes"]),
        source_ref=source_ref,
        native_groups_ref=native_groups_ref,
    )
    if evidence.get("status") != "success":
        error = evidence.get("error") or {}
        warning = f"AAGNet evidence failed: {error.get('type', 'unknown')}: {error.get('message', '')}"
        evidence.update({"status": "skipped", "skipReason": warning, "per_face_evidence": []})
        if not paths["native_groups"].exists():
            paths["native_groups"].write_text(json.dumps({"schemaVersion": "aagnet_native_groups.audit.v1", "nativeFeatureGroups": [], "skipReason": warning}, indent=2), encoding="utf-8")
    paths["evidence"].write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    return evidence
