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


def _run_label(run_root: Path) -> str:
    """``runs/<tag>`` - what the page's tooltip actually wants to say."""
    root = Path(run_root)
    parent = root.parent.name
    return f"{parent}/{root.name}" if parent else root.name


def _registered_relations(palette: dict, produced: dict[str, int],
                          problems: list[str] | None = None) -> list[dict]:
    try:
        from relations.v51.registry import RelationRegistry

        rules = RelationRegistry.all_rules()
    except Exception as error:  # registry problems must not take the whole run down
        # ... but a run that could not read the registry has to say so.  An
        # empty registry block looks exactly like "this corpus produced no
        # relations", which is a conclusion, and this is not one (F-12).
        rules = []
        if problems is not None:
            problems.append(f"relations.v51.registry: {type(error).__name__}: {error}")

    seen: dict[str, dict] = {}
    for rule in rules:
        edge_type = getattr(rule, "edge_type", None)
        if not edge_type or edge_type in seen:
            continue
        seen[edge_type] = {
            "edgeType": edge_type,
            "ruleId": getattr(rule, "rule_id", None),
            "directed": bool(getattr(rule, "directed", False)),
            "description": (getattr(rule, "description", "") or "").strip(),
            "count": produced.get(edge_type, 0),
            "color": _edge_type_color(palette, edge_type),
            "defaultVisible": edge_type in DEFAULT_VISIBLE_EDGE_TYPES,
        }

    # An edge type present in the data but not in the registry would mean the
    # graph and the registry disagree; surface it rather than dropping it.
    for edge_type, count in sorted(produced.items()):
        if edge_type in seen:
            continue
        seen[edge_type] = {
            "edgeType": edge_type,
            "ruleId": None,
            "directed": False,
            "description": "produced in this run but not found in the relation registry",
            "count": count,
            "color": _edge_type_color(palette, edge_type),
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


def _dialect_section(line: str, summaries: list[dict], problems: list[str] | None = None) -> dict:
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
    except Exception as error:  # a registry problem must not take the whole run down
        # An empty catalogue reads as "no heuristic in this dialect exists",
        # which is a statement about the corpus.  Say what really happened
        # instead of publishing that (F-12); the page shows the key when it
        # is there, and it is only there when something went wrong.
        if problems is not None:
            problems.append(f"heuristics registry for line {line}: {type(error).__name__}: {error}")
        return {"registry": {}, "coverage": {}, "catalogue": {}, "corpus": {},
                "unavailable": f"{type(error).__name__}: {error}"}


def _heuristics_section(manifest: dict, viewer_parts: list[dict],
                        problems: list[str] | None = None) -> dict:
    """The open-source fold at the top level, every line's beside it."""
    lines = {
        line: _dialect_section(
            line,
            [((part.get("lines") or {}).get(line) or {}).get("heuristicSummary") or {}
             for part in viewer_parts],
            problems,
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
    problems: list[str] = []
    heuristics = _heuristics_section(manifest, viewer_parts, problems)
    failures = [
        {"partId": part["partId"], **check}
        for part in viewer_parts
        for check in part.get("checks", [])
        if check["status"] != "OK"
    ]

    relation_types = _registered_relations(palette, produced, problems)

    index = {
        "schemaVersion": "v8.viewer_index.v1",
        "runId": manifest.get("runId"),
        "outputTag": manifest.get("outputTag"),
        # The run as the project names it, never this machine's absolute path
        # (node 26 F-04 / P-03): this file is served over HTTP and travels
        # with a packed run, and "where is the checkout" is not a fact about
        # what was built.
        "generatedFrom": _run_label(run_root),
        "palette": palette,
        "relationTypes": relation_types,
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
    # Only when something actually failed: a run that read every registry
    # writes exactly the bytes it used to (F-12 must not move a digest or a
    # page that reads a key it never saw before).
    if problems:
        index["errors"] = problems
    return index
