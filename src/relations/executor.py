"""Run every enabled relation rule over the final feature node set."""

from __future__ import annotations

from collections import Counter

from contract.partmodel import PartModel

from core.models import FeatureEdge, FeatureNode
from relations.base import RelationContext
from relations.registry import BaseRelationRegistry, default_registry


def run_relations(
    model: PartModel,
    nodes: list[FeatureNode],
    *,
    kernel=None,
    terminal_overrides: dict[str, list[dict]] | None = None,
    registry: type[BaseRelationRegistry] | None = None,
) -> tuple[list[FeatureEdge], dict]:
    """Return ``(edges, diagnostics)`` for the final node set.

    The executor owns three invariants that no individual rule has to repeat:
    an edge may only reference a surviving node id, an edge may never be a
    self loop, and one (source, target, edgeType) pair yields at most one
    edge.  A rule that returns several partial edges for the same pair has
    their evidence merged rather than one of them silently dropped.

    ``registry`` selects the dialect - which ontology's relation rules run.
    The invariants above are the same either way; only the rules differ.
    """

    registry = registry or default_registry()
    context = RelationContext.build(
        model, nodes, kernel=kernel, terminal_overrides=terminal_overrides
    )
    edges: list[FeatureEdge] = []
    by_key: dict[tuple, FeatureEdge] = {}
    rule_reports: list[dict] = []

    for rule in registry.enabled():
        produced = 0
        rejected: Counter = Counter()
        for edge in rule.build_edges(context):
            edge.edgeType = edge.edgeType or rule.edge_type
            edge.directed = bool(edge.directed or rule.directed)
            edge.sourceRule = edge.sourceRule or rule.rule_id
            if edge.source not in context.node_by_id or edge.target not in context.node_by_id:
                rejected["unknownNodeId"] += 1
                continue
            if edge.source == edge.target:
                rejected["selfEdge"] += 1
                continue
            if not edge.directed and edge.source > edge.target:
                edge = _flip(edge)
            key = (
                (edge.edgeType, edge.source, edge.target)
                if edge.directed
                else (edge.edgeType, *sorted((edge.source, edge.target)))
            )
            existing = by_key.get(key)
            if existing is not None:
                rejected["duplicatePair"] += 1
                _merge_duplicate(existing, edge)
                continue
            by_key[key] = edge
            edges.append(edge)
            produced += 1
        report = {
            "ruleId": rule.rule_id,
            "edgeType": rule.edge_type,
            "directed": rule.directed,
            "edgeCount": produced,
            "rejected": dict(sorted(rejected.items())),
        }
        extra = context.rule_diagnostics.get(rule.rule_id)
        if extra:
            report["diagnostics"] = extra
        rule_reports.append(
            report
        )

    edges.sort(key=lambda item: (item.edgeType, item.source, item.target))
    # Numbered within the relation type rather than across all of them.  A
    # global counter made ``hole_spacing_0126`` appear in a part holding
    # seventeen of them, and adding a single COAXIAL edge renumbered every
    # DIRECT_ADJACENT id after it.
    sequence: Counter = Counter()
    for edge in edges:
        sequence[edge.edgeType] += 1
        edge.edgeId = f"{edge.edgeType.lower()}_{sequence[edge.edgeType]:04d}"

    diagnostics = {
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "relationCounts": relation_counts(edges),
        "rules": rule_reports,
    }
    return edges, diagnostics


def relation_counts(edges: list[FeatureEdge]) -> dict[str, int]:
    return dict(sorted(Counter(edge.edgeType for edge in edges).items()))


def _flip(edge: FeatureEdge) -> FeatureEdge:
    """Put an undirected edge in canonical (sorted) node order."""

    edge.source, edge.target = edge.target, edge.source
    pairs = edge.evidence.get("adjacentFacePairs")
    if isinstance(pairs, list):
        edge.evidence["adjacentFacePairs"] = [
            _swap_keys(pair, "sourcePartmodelFaceId", "targetPartmodelFaceId") for pair in pairs
        ]
        edge.evidence["adjacentFacePairs"] = [
            _swap_keys(pair, "sourceSurfaceType", "targetSurfaceType")
            for pair in edge.evidence["adjacentFacePairs"]
        ]
    # Only the legacy DIRECT_ADJACENT fields encode graph orientation.  New
    # relations keep semantic direction in explicit witness records (for
    # example ``semanticSourceFeatureId`` -> ``hostOwnerNodeId``); flipping a
    # graph edge must never rewrite those records.
    if edge.edgeType == "DIRECT_ADJACENT":
        edge.evidence = _swap_keys(edge.evidence, "sourceFeatureType", "targetFeatureType")
        edge.evidence = _swap_keys(edge.evidence, "sourcePartmodelFaceIds", "targetPartmodelFaceIds")
    elif edge.edgeType != "LOCATED_ON":
        # The new undirected rules publish A/B or left/right evidence for
        # their two feature endpoints.  Keep those analytic witnesses aligned
        # with the canonical graph pair, while deliberately leaving
        # LOCATED_ON's semantic source -> host records untouched.
        edge.evidence = _swap_relation_orientation(edge.evidence)
        # Every relation property is a symmetric scalar today, but routing
        # them through the same canonicalisation means the next oriented one
        # only has to be named in _RELATION_ORIENTATION_KEYS.
        edge.properties = _swap_relation_orientation(edge.properties)
    return edge


def _swap_keys(data: dict, first: str, second: str) -> dict:
    if first not in data and second not in data:
        return data
    return {**data, first: data.get(second), second: data.get(first)}


_RELATION_ORIENTATION_KEYS = (
    ("featureA", "featureB"),
    ("axisA", "axisB"),
    ("axisPointA", "axisPointB"),
    ("leftPartmodelFaceId", "rightPartmodelFaceId"),
    ("leftSurfaceType", "rightSurfaceType"),
    ("leftTerminalWitnesses", "rightTerminalWitnesses"),
    ("leftPartmodelFaceIds", "rightPartmodelFaceIds"),
    ("leftCadFaceIds", "rightCadFaceIds"),
    ("leftBoundaryEdgeId", "rightBoundaryEdgeId"),
    # The plural pair is what the nx SLOT_CLEARANCE witness actually carries
    # (node 21' changed that rule to measure opening boundaries and named the
    # keys in the plural); without it a flipped edge said "left" of A on one
    # key and "left" of B on the next.
    ("leftBoundaryEdgeIds", "rightBoundaryEdgeIds"),
    ("nearestPointA", "nearestPointB"),
    ("courseDirectionA", "courseDirectionB"),
)


def _swap_relation_orientation(value):
    """Swap only known endpoint-oriented relation evidence recursively."""
    if isinstance(value, list):
        return [_swap_relation_orientation(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _swap_relation_orientation(item)
        for key, item in value.items()
    }
    for first, second in _RELATION_ORIENTATION_KEYS:
        if first in result or second in result:
            result[first], result[second] = result.get(second), result.get(first)
    return result


def _merge_duplicate(existing: FeatureEdge, extra: FeatureEdge) -> None:
    """Fold a duplicate pair's witnesses without relation-specific loss.

    DIRECT_ADJACENT historically had one list named ``adjacentFacePairs``;
    the other relations carry lists such as ``locatedOnWitnesses`` and
    ``principalWallFacePairs``, and LOCATED_ON carries one aligned
    ``terminals`` list on the properties.  Merge every list by a
    deterministic JSON key, preserve scalar metrics from the first proof,
    and then refresh the common count fields.  Semantic source/host fields
    are never swapped here.
    """

    _merge_section(existing.properties, extra.properties)
    _merge_section(existing.evidence, extra.evidence)


_COUNT_KEYS = {
    "adjacentFacePairs": "facePairCount",
    "locatedOnWitnesses": "locatedOnWitnessCount",
    "principalWallFacePairs": "principalWallPairCount",
    "clearanceWitnesses": "clearanceWitnessCount",
    "terminals": "terminalCount",
}


def _merge_section(current: dict, extra: dict) -> None:
    for key, more in extra.items():
        if not isinstance(more, list):
            if key not in current or current.get(key) in (None, "", {}):
                current[key] = more
            continue
        bucket = current.setdefault(key, [])
        if not isinstance(bucket, list):
            continue
        seen = {_stable_key(item) for item in bucket}
        for item in more:
            marker = _stable_key(item)
            if marker not in seen:
                seen.add(marker)
                bucket.append(item)

    for key, value in list(current.items()):
        count_key = _COUNT_KEYS.get(key)
        if count_key and isinstance(value, list):
            current[count_key] = len(value)


def _stable_key(value) -> str:
    """Return a deterministic, hashable marker for arbitrary JSON evidence."""
    import json

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError:
        return repr(value)
