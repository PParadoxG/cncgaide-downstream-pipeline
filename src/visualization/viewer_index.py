"""Run-level index the viewer's part list and statistics view read.

Also records every relation rule and every heuristic the registries know
about — including the ones that produced nothing in this run.  A rule that
registers but never matches is exactly the thing worth seeing while a new
relation or heuristic is being written, and it is invisible if the index only
reports what happened to exist.

The heuristic section speaks for the open-source line at its top level, like
the rest of the index, and carries every line beside it under ``lines``: one
dialect's frozen catalogue, its registry provenance and its corpus fold each.
The two are **not one metric** — node 19a is explicit that a pass rate under
v5.1 and one under nx_v1 are not comparable — so they are folded separately
and the page shows one line at a time.
"""

from __future__ import annotations

from pathlib import Path

from contract.registry import LINE_NX, LINE_OCC, LINE_ORDER
from visualization.viewer_data import DEFAULT_HIDDEN_FEATURE_TYPES, DEFAULT_VISIBLE_EDGE_TYPES, _edge_type_color


def _registered_relations(palette: dict, produced: dict[str, int]) -> list[dict]:
    try:
        from relations.v51.registry import RelationRegistry

        rules = RelationRegistry.all_rules()
    except Exception:  # registry problems must not take the whole run down
        rules = []

    seen: dict[str, dict] = {}
    for index, rule in enumerate(rules):
        edge_type = getattr(rule, "edge_type", None)
        if not edge_type or edge_type in seen:
            continue
        seen[edge_type] = {
            "edgeType": edge_type,
            "ruleId": getattr(rule, "rule_id", None),
            "directed": bool(getattr(rule, "directed", False)),
            "description": (getattr(rule, "description", "") or "").strip(),
            "count": produced.get(edge_type, 0),
            "color": _edge_type_color(palette, edge_type, index),
            "defaultVisible": edge_type in DEFAULT_VISIBLE_EDGE_TYPES,
        }

    # An edge type present in the data but not in the registry would mean the
    # graph and the registry disagree; surface it rather than dropping it.
    for offset, (edge_type, count) in enumerate(sorted(produced.items())):
        if edge_type in seen:
            continue
        seen[edge_type] = {
            "edgeType": edge_type,
            "ruleId": None,
            "directed": False,
            "description": "produced in this run but not found in the relation registry",
            "count": count,
            "color": _edge_type_color(palette, edge_type, len(rules) + offset),
            "defaultVisible": edge_type in DEFAULT_VISIBLE_EDGE_TYPES,
        }

    return [seen[key] for key in sorted(seen)]


def _dialect_registry(line: str):
    """The heuristic dialect whose vocabulary that line's verdicts are in.

    Named here rather than asked of the store so this module keeps working on
    a run written by an older layout; the mapping itself is the store's
    (``store.sync.heuristic_dialect``) and this is the read-only half of it.
    """
    if line == LINE_NX:
        from heuristics.nx.registry import NxHeuristicRegistry

        return NxHeuristicRegistry
    from heuristics.v51.registry import HeuristicRegistry

    return HeuristicRegistry


def _dialect_section(line: str, summaries: list[dict]) -> dict:
    """Registry provenance, the frozen catalogue and the corpus fold of one line.

    The statistics view needs every heuristic that is *registered*, not only
    the ones that produced a verdict in this run: a rule whose domain is empty
    in the whole corpus is exactly the coverage gap worth seeing, and it is
    invisible if the page is built from the verdicts alone.
    """
    try:
        from heuristics import corpus_summary
        from heuristics.executor import _catalogue

        registry = _dialect_registry(line)
        return {
            "registry": registry.version(),
            "coverage": registry.coverage(),
            "catalogue": _catalogue(registry),
            "corpus": corpus_summary([summary for summary in summaries if summary]),
        }
    except Exception:  # a registry problem must not take the whole run down
        return {"registry": {}, "coverage": {}, "catalogue": {}, "corpus": {}}


def _heuristics_section(manifest: dict, viewer_parts: list[dict]) -> dict:
    """The open-source fold at the top level, every line's beside it."""
    lines = {
        line: _dialect_section(
            line,
            [((part.get("lines") or {}).get(line) or {}).get("heuristicSummary") or {}
             for part in viewer_parts],
        )
        for line in LINE_ORDER
    }
    section = manifest.get("heuristics") or {}
    open_source = lines.get(LINE_OCC) or {}
    return {
        "registry": section.get("registry") or open_source.get("registry") or {},
        "coverage": open_source.get("coverage") or {},
        "catalogue": open_source.get("catalogue") or {},
        "corpus": section.get("corpus") or open_source.get("corpus") or {},
        "lines": lines,
    }


def build_viewer_index(manifest: dict, viewer_parts: list[dict], palette: dict, run_root: Path,
                       *, background_types: list[str] | None = None,
                       feature_labels: dict[str, str] | None = None) -> dict:
    produced: dict[str, int] = {}
    for part in viewer_parts:
        for entry in part.get("relationTypes", []):
            produced[entry["edgeType"]] = produced.get(entry["edgeType"], 0) + entry["count"]

    parts = [{key: value for key, value in part.items() if key != "checks"} for part in viewer_parts]
    heuristics = _heuristics_section(manifest, viewer_parts)
    failures = [
        {"partId": part["partId"], **check}
        for part in viewer_parts
        for check in part.get("checks", [])
        if check["status"] != "OK"
    ]

    return {
        "schemaVersion": "v8.viewer_index.v1",
        "runId": manifest.get("runId"),
        "outputTag": manifest.get("outputTag"),
        "generatedFrom": str(run_root),
        "palette": palette,
        "relationTypes": _registered_relations(palette, produced),
        # The background layer (what the part page hides by default); the
        # launcher leaves these out of its type bars unless asked.  Folded
        # across every line that built something, because the page shows one
        # line at a time and each names its own background classes.
        "defaultHiddenFeatureTypes": sorted(background_types
                                            or DEFAULT_HIDDEN_FEATURE_TYPES),
        # ``{class: the words a human reads}``, likewise folded across lines
        # and empty for a line whose class names are already the words.
        "featureLabels": dict(sorted((feature_labels or {}).items())),
        "heuristics": heuristics,
        "parts": parts,
        "checkFailures": failures,
    }
