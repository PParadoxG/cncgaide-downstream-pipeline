"""Rebuild ``manifest.json`` and ``viewer_index.json`` from the part records.

The two indexes are *derived*: nothing in them is authoritative, everything is
folded from the records on disk, so a sync that touched two parts rebuilds the
same indexes a full run would have written.

Both speak for the **open-source line** at their top level - the manifest is
what the GED corpus reads, and the viewer's default view is that line - and
carry the other line beside it under ``lines``.  That keeps one reader (the
corpus) pointed at one recogniser without a second index, and lets the page
switch lines by reaching one key deeper.

Each viewer-index entry also carries the part's ``light`` (source STEP only)
and ``rulesChanged`` (the layers some built line is behind on), which are
different questions and are answered in different places on the page.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from contract.registry import LINE_OCC, LINE_ORDER
from core.paths import collect_step_files, make_part_id, resolve_input_path, run_paths
from store.records import atomic_write_json, get_line, iter_records, line_passed
from store.stamps import Light, layer_stamps, light, rules_changed


def new_run_id() -> str:
    """Unique per rebuild: the page reloads when it changes, so two rebuilds in
    one second must not collide."""
    return "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _previous_manifest(run: dict) -> dict:
    try:
        return json.loads(run["manifest"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _common_input_root(records: list[dict]) -> str:
    paths = [record.get("sourceStepAbs") for record in records if record.get("sourceStepAbs")]
    if not paths:
        return ""
    try:
        return os.path.commonpath([str(Path(p).parent) for p in paths])
    except ValueError:
        return ""


def step_index(project_root: str | Path, input_path: str | None) -> dict[str, Path]:
    """partId -> STEP path for everything currently under the input path."""
    if not input_path:
        return {}
    input_abs = resolve_input_path(input_path, project_root)
    input_root = input_abs if input_abs.is_dir() else input_abs.parent
    return {make_part_id(path, input_root): path for path in collect_step_files(input_abs)}


def lights_of(records: list[dict], steps: dict[str, Path]) -> dict[str, Light]:
    return {record["partId"]: light(record, steps.get(record["partId"])) for record in records}


def _line_summary(record: dict, line: str) -> dict:
    """What the page needs about one line of one part, without its artifacts."""
    block = get_line(record, line)
    if block is None:
        return {"status": None, "reason": "never built"}
    entry = block.get("entry") or {}
    summary = {
        "status": block.get("status"),
        "producedAt": block.get("producedAt"),
        "nodeCount": entry.get("nodeCount"),
        "edgeCount": entry.get("edgeCount"),
        "featureCounts": entry.get("featureCounts"),
        "heuristicCounts": block.get("heuristicCounts") or {},
        # Per-heuristic four-value counts, so the statistics view can fold the
        # shown line's own verdicts.  Each line's are its dialect's: node 19a
        # is explicit that a v5.1 rate and an nx_v1 rate are not one metric,
        # which is why they are kept apart rather than summed anywhere.
        "heuristicSummary": block.get("heuristicSummary") or {},
    }
    for key in ("reason", "error"):
        if block.get(key):
            summary[key] = block[key]
    return summary


def _vocabulary_fold(records: list[dict]) -> dict:
    """What every line calls its classes, folded across the whole run.

    The home page shows one line at a time but has to be able to show either,
    and only the line that produced a part knows its own vocabulary: which
    classes are a background layer, and what a class is called in words.  Both
    come out of the per-line viewer entries rather than out of a table in the
    viewer, so a line's vocabulary never has to be written down twice.
    """
    background: set[str] = set()
    labels: dict[str, str] = {}
    for record in records:
        for line in LINE_ORDER:
            entry = (get_line(record, line) or {}).get("viewerEntry") or {}
            background.update(entry.get("defaultHiddenFeatureTypes") or [])
            for feature_type, label in (entry.get("featureLabels") or {}).items():
                labels.setdefault(feature_type, label)
    return {"background_types": sorted(background),
            "feature_labels": dict(sorted(labels.items()))}


def rebuild_index(
    project_root: str | Path,
    output_tag: str,
    *,
    input_path: str | None = None,
    build_viewer: bool | None = None,
    stamps: dict[str, str] | None = None,
    lights: dict[str, Light] | None = None,
) -> tuple[dict, dict | None]:
    """Fold every record into the manifest (and viewer index) and write both.

    Returns ``(manifest, viewer_index)``; the index is ``None`` when it was not
    written.  ``build_viewer`` None means "write the viewer index if any record
    carries a viewer entry" - a run produced with ``--no-viewer`` keeps having
    none.
    """
    root = Path(project_root).resolve()
    run = run_paths(root, output_tag)
    run["parts"].mkdir(parents=True, exist_ok=True)
    records = iter_records(run["run_root"])
    previous = _previous_manifest(run)

    input_abs = input_path or previous.get("inputPath") or _common_input_root(records)
    if input_abs:
        input_abs = str(resolve_input_path(input_abs, root))

    passed = [record for record in records if line_passed(record, LINE_OCC)]
    parts = [dict(get_line(record, LINE_OCC)["entry"]) for record in passed]
    statuses = []
    for record in records:
        block = get_line(record, LINE_OCC) or {}
        if line_passed(record, LINE_OCC):
            entry = block["entry"]
            statuses.append({
                "partId": record["partId"],
                "status": "PASS",
                "nodeCount": entry.get("nodeCount", 0),
                "edgeCount": entry.get("edgeCount", 0),
                "relationCounts": entry.get("relationCounts", {}),
                "heuristicCounts": entry.get("heuristicCounts", block.get("heuristicCounts", {})),
            })
        else:
            statuses.append({
                "partId": record["partId"],
                "status": "FAIL",
                "error": block.get("error") or block.get("reason") or "",
                "traceback": block.get("traceback") or "",
            })

    from heuristics import corpus_summary
    from heuristics.v51.registry import HeuristicRegistry

    relation_totals: Counter = Counter()
    for entry in parts:
        relation_totals.update(entry.get("relationCounts", {}))
    devices = sorted({str(get_line(record, LINE_OCC).get("device")) for record in passed
                      if get_line(record, LINE_OCC).get("device")})
    device = devices[0] if len(devices) == 1 else ("mixed" if devices else "unknown")

    manifest = {
        "schemaVersion": "v8.run_manifest.v1",
        "runId": new_run_id(),
        "outputTag": output_tag,
        "inputPath": input_abs,
        "device": device,
        "edgeCount": sum(entry.get("edgeCount", 0) for entry in parts),
        "relationCounts": dict(sorted(relation_totals.items())),
        "heuristics": {
            "registry": HeuristicRegistry.version(),
            "corpus": corpus_summary(
                [get_line(record, LINE_OCC).get("heuristicSummary") or {} for record in passed]),
        },
        "parts": parts,
        "statuses": statuses,
    }
    atomic_write_json(run["manifest"], manifest, indent=2)

    viewer_parts = [record for record in passed
                    if get_line(record, LINE_OCC).get("viewerEntry")]
    if build_viewer is None:
        build_viewer = bool(viewer_parts)
    index = None
    if build_viewer:
        from visualization.palette import load_palette
        from visualization.viewer_index import build_viewer_index

        stamps = stamps or layer_stamps(root)
        if lights is None:
            lights = lights_of(records, step_index(root, input_abs))
        entries = []
        for record in viewer_parts:
            occ = get_line(record, LINE_OCC)
            viewer_entry = dict(occ["viewerEntry"])
            viewer_entry.setdefault("heuristicCounts", occ.get("heuristicCounts", {}))
            viewer_entry.setdefault("heuristicSummary", occ.get("heuristicSummary", {}))
            state = lights.get(record["partId"])
            viewer_entry["light"] = (state.to_dict() if state is not None
                                     else {"state": "fresh", "reasons": []})
            viewer_entry["rulesChanged"] = rules_changed(record, stamps)
            viewer_entry["lines"] = {line: _line_summary(record, line) for line in LINE_ORDER}
            entries.append(viewer_entry)
        index = build_viewer_index(manifest, entries, load_palette(root), run["run_root"],
                                   **_vocabulary_fold(records))
        atomic_write_json(run["viewer_index"], index, indent=1)
    return manifest, index
