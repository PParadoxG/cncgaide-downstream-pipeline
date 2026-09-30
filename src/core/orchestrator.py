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

from occline.parse import process_step_shape
from contract.partmodel_json import save_json

from contract.evidence import FEATURE_NAMES
from occline.aagnet.face_bridge import build_cad_face_mappings
from occline.kernel.occ_kernel import OccKernelServices
from occwl_provenance import occwl_provenance
from contract.registry import LINE_OCC
from core.paths import (
    ensure_part_paths,
    part_paths,
    project_relative,
    project_relative_gap,
    run_relative,
)
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
    # One read of the STEP for both the parse and the kernel services: the
    # file used to be opened twice for the same part (node 26 B-P3 / P PERF-04).
    model, shape = process_step_shape(str(step_path))
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
    kernel = OccKernelServices.from_loaded_shape(shape, step_path)
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
    # Written only when ``sourceStep`` above had to fall back to a file name,
    # so a run whose inputs are inside the project produces the same bytes it
    # always did.
    source_gap = project_relative_gap(root, step_path)
    if source_gap:
        result["diagnostics"]["sourceStep"] = source_gap
    paths["result"].write_text(json.dumps(result, ensure_ascii=False,
                                          separators=(",", ":")),
                               encoding="utf-8", newline="")
    paths["heuristics"].write_text(
        json.dumps({"partId": part_id, **recognition.heuristics},
                   ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8", newline="",
    )
    id_map = build_id_map(part_id, model, nodes)
    save_id_map(id_map, paths["id_map"])
    palette = load_palette(root)
    graph_view = build_graph_view(part_id, result, id_map, palette, result["sourceStep"])
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
    # ``--no-aagnet`` is an ablation, not a failure: the store hands over a
    # bundle that says the model was switched off instead of one that carries
    # it, and this is the only place that has to know.  The recognition chain
    # then takes the path it already takes when AAGNet cannot answer - an
    # evidence document with an empty ``per_face_evidence`` list - so the
    # ablation exercises production code rather than a second code path.
    if bundle is not None and bundle.get("aagnetDisabled"):
        evidence = _disabled_evidence(step_path, source_ref=source_ref)
        _write_empty_native_groups(paths, evidence["skipReason"])
        paths["evidence"].write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                                     encoding="utf-8", newline="")
        return evidence
    # Imported here rather than at the top of the module because importing it
    # imports torch, dgl and the vendored AAGNet tree: with the model switched
    # off none of that may be loaded, and a top-level import would load it
    # before this function ever ran.
    from occline.aagnet.per_face import run_aagnet_per_face

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
            _write_empty_native_groups(paths, warning)
    paths["evidence"].write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                                 encoding="utf-8", newline="")
    return evidence


def _write_empty_native_groups(paths: dict[str, Path], reason: str) -> None:
    """The audit document a part gets when AAGNet produced no groups at all."""
    paths["native_groups"].write_text(
        json.dumps({"schemaVersion": "aagnet_native_groups.audit.v1",
                    "nativeFeatureGroups": [], "skipReason": reason}, indent=2),
        encoding="utf-8", newline="")


def _disabled_evidence(step_path: Path, *, source_ref: str) -> dict:
    """The evidence document of a part built with the model switched off.

    Every key the adapter writes is written here too, so a stored artifact is
    self-describing: nothing downstream has to be told that this run was an
    ablation, it can read it.  ``status`` and ``device`` say ``disabled``
    rather than naming an inference that did not happen, and the empty
    ``per_face_evidence`` list is what the recognition chain consumes.

    ``_elapsed_sec`` is 0.0 rather than absent: it is a volatile key, so it is
    not in the artifact digest, and keeping it makes the two documents the same
    shape - which ``tests/test_node30_aagnet_switch.py`` checks against the
    real adapter so the two cannot drift apart.
    """
    return {
        "schema_version": "aagnet_per_face_evidence.v1",
        "recognizer": "AAGNet",
        "recognizer_version": "vendor_snapshot",
        # The environment is still named: which occwl is installed is a fact
        # about this machine, and the ablation does not make it untrue.
        **occwl_provenance(),
        "part_id": Path(step_path).stem,
        "source_step": source_ref,
        "face_id_space": "cad_face_id + partmodel_face_id",
        "feature_classes": FEATURE_NAMES,
        "score_semantics": {
            "logits": "raw_model_logits",
            "softmax": "relative_softmax_over_25_classes",
            "top_softmax_score": "not_calibrated_probability",
        },
        "status": "disabled",
        "skipReason": "disabled by --no-aagnet",
        "device": "disabled",
        "per_face_evidence": [],
        "mismatch_diagnostics": {},
        "audit": {},
        "error": None,
        "_elapsed_sec": 0.0,
    }
