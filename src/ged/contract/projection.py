"""Project one finished part (``result.json`` + ``partmodel.json``) to a ``GedGraph``.

This module *is* the representation layer.  Every rule here corresponds to a
decision in ``NODE_REPORT/GED_Design_Outline_v0.md`` §2:

- semantic leaf nodes enter the assignment; Surface leaves (``PlanarFace``,
  ``FreeformFace``) form the host background layer and are aggregated into
  mandatory fixed nodes (§2.2 C);
- only whitelisted own-geometry attributes are kept (``attribute_policy``),
  lengths are divided by ``L_ref`` and areas by ``L_ref²`` (§2.4);
- ``DIRECT_ADJACENT`` survives only between semantic nodes and carries the
  dominant connection type as its label; ``LOCATED_ON`` to a Surface host
  becomes ``SHARES_HOST`` derived edges between the host's members, while
  ``LOCATED_ON`` to a semantic host stays a real edge whose semantic
  direction is recovered from the evidence, not from the (sorted) endpoint
  order; the metric relations keep their normalised value (§2.5);
- unclaimed faces are aggregated into one low-weight fixed node (§2.6);
- ``L_ref`` (geometric diameter of the vertex cloud) is derived here and is
  not an ontology attribute (§2.3 / R-5);
- the finished graph is in **canonical order** (``canonical_node_key``), so a
  renumbering of the artifacts cannot move a distance (node 28 D-1).

Nothing here assigns a cost.  The audit collects every place where the
artifacts deviate from the contract so that a representation problem is
visible without re-running the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from contract.face_ids import raw_cad_face_id

from .representation import Representation, V5_1, for_line
from .schema import GedAttribute, GedEdge, GedFixedNode, GedGraph, GedNode


# The v5.1 relation tables, kept under their historical names because the
# semantics stamp and the tests read them here.  Which set a projection uses
# is now the ``Representation``'s answer (node 21'); these are ``V5_1``'s.
#
# GROUPED_RELATIONS: types that form cliques / groups and are normalised by
# group size in the cost layer - the projection only records ``group_size``.
# HOST_IMPLYING_RELATIONS: a pair connected by one of these does not
# additionally receive a SHARES_HOST edge.
# EDGE_METRICS: relation-valued properties that enter the edge attribute term,
# with the normalisation to apply (``courseAngle`` / ``transverseAngle`` of
# SLOT_CLEARANCE are deliberately left out in v0).
GROUPED_RELATIONS: frozenset[str] = V5_1.grouped_relations
HOST_IMPLYING_RELATIONS: frozenset[str] = V5_1.host_implying_relations
EDGE_METRICS: dict[str, dict[str, str]] = dict(V5_1.edge_metrics)
KNOWN_RELATIONS: frozenset[str] = V5_1.known_relations

NORMAL_PARALLEL_TOL_DEG = 1.0
FIXED_UNCLAIMED = "__unclaimed__"


@dataclass
class ProjectionAudit:
    """Everything the projection had to decide beyond the happy path."""

    graph_id: str = ""
    unknown_leaf_types: list[str] = field(default_factory=list)
    attributes_absent: list[dict[str, Any]] = field(default_factory=list)
    attributes_null: list[dict[str, Any]] = field(default_factory=list)
    unknown_relation_types: list[str] = field(default_factory=list)
    edges_dropped: dict[str, int] = field(default_factory=dict)
    located_on_unresolved: list[dict[str, Any]] = field(default_factory=list)
    area_source: dict[str, str] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)

    def note_drop(self, reason: str) -> None:
        self.edges_dropped[reason] = self.edges_dropped.get(reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "graphId": self.graph_id,
            "unknownLeafTypes": list(self.unknown_leaf_types),
            "attributesAbsent": list(self.attributes_absent),
            "attributesNull": list(self.attributes_null),
            "unknownRelationTypes": list(self.unknown_relation_types),
            "edgesDropped": dict(self.edges_dropped),
            "locatedOnUnresolved": list(self.located_on_unresolved),
            "areaSource": dict(self.area_source),
            "counts": dict(self.counts),
        }


# --------------------------------------------------------------------------
# helpers


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _unit(vector: Iterable[float]) -> tuple[float, float, float] | None:
    values = [float(v) for v in vector]
    if len(values) != 3:
        return None
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 0:
        return None
    return (values[0] / norm, values[1] / norm, values[2] / norm)


def geometric_diameter(partmodel: dict[str, Any]) -> float:
    """Max pairwise vertex distance: rigid-motion invariant, unlike a bbox diagonal."""
    vertices = partmodel.get("vertices")
    if not isinstance(vertices, dict) or not vertices:
        raise ValueError("PartModel has no vertices for L_ref")
    coords = np.asarray(
        [entry.get("coordinates") for entry in vertices.values()], dtype=float
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("PartModel vertex coordinate schema mismatch")
    # Chunked O(n²); parts have hundreds to a few thousand vertices.
    best = 0.0
    chunk = 2048
    for start in range(0, len(coords), chunk):
        block = coords[start : start + chunk]
        diff = block[:, None, :] - coords[None, :, :]
        best = max(best, float(np.sqrt((diff * diff).sum(axis=2)).max()))
    if best <= 0:
        raise ValueError("PartModel geometric diameter is not positive")
    return best


def _normalise_value(kind: str, raw: float, name: str, l_ref: float) -> float:
    if kind == "length":
        return raw / l_ref
    if kind == "area":
        return raw / (l_ref * l_ref)
    if kind == "angle":
        return raw
    if kind == "count":
        return raw
    raise ValueError(f"numeric normalisation undefined for {kind} ({name})")


def _node_attributes(
    raw_node: dict[str, Any],
    leaf: str,
    l_ref: float,
    audit: ProjectionAudit,
    representation: Representation = V5_1,
) -> dict[str, GedAttribute]:
    missing = raw_node.get("missingAttributes") or {}
    if not isinstance(missing, dict):
        missing = {}
    node_id = str(raw_node.get("nodeId"))
    out: dict[str, GedAttribute] = {}
    for name in representation.leaf_attributes[leaf]:
        kind = representation.kind_of(name)
        present = name in raw_node
        raw = raw_node.get(name)
        if kind == "category":
            if isinstance(raw, str) and raw:
                out[name] = GedAttribute(name, "category", raw, raw, "valid")
                continue
        elif _is_number(raw):
            out[name] = GedAttribute(
                name, kind, _normalise_value(kind, float(raw), name, l_ref), raw, "valid"  # type: ignore[arg-type]
            )
            continue
        reason = missing.get(name)
        if reason is None:
            reason = "absent_in_node" if not present else "null_without_reason"
            audit.attributes_absent.append({"nodeId": node_id, "leaf": leaf, "attribute": name})
        else:
            audit.attributes_null.append({"nodeId": node_id, "leaf": leaf, "attribute": name})
        out[name] = GedAttribute(name, kind, None, raw, "null", str(reason))  # type: ignore[arg-type]
    return out


def _normal_classes(normals: list[tuple[float, float, float]]) -> int:
    """Number of mutually non-parallel normal directions (pose invariant)."""
    cos_tol = math.cos(math.radians(NORMAL_PARALLEL_TOL_DEG))
    reps: list[tuple[float, float, float]] = []
    for normal in normals:
        if any(abs(sum(a * b for a, b in zip(normal, rep))) >= cos_tol for rep in reps):
            continue
        reps.append(normal)
    return len(reps)


def _face_area(face: dict[str, Any] | None) -> float | None:
    if isinstance(face, dict) and _is_number(face.get("area")):
        return float(face["area"])
    return None


def _surface_aggregate(
    leaf: str,
    raw_nodes: list[dict[str, Any]],
    faces: dict[str, Any],
    l_ref: float,
    audit: ProjectionAudit,
    representation: Representation = V5_1,
) -> GedFixedNode:
    count = 0
    area = 0.0
    normals: list[tuple[float, float, float]] = []
    sources: set[str] = set()
    for raw_node in raw_nodes:
        if raw_node.get("featureType") != leaf:
            continue
        count += 1
        # R-1: the node should publish ``area`` itself.  Until the pipeline
        # does, fall back to the PartModel face areas and say so.
        if _is_number(raw_node.get("area")):
            area += float(raw_node["area"])
            sources.add("node.area")
        else:
            for face_id in raw_node.get("partmodelFaceIds") or []:
                face_area = _face_area(faces.get(str(face_id)))
                if face_area is None:
                    raise ValueError(f"{raw_node.get('nodeId')}: face {face_id} has no area")
                area += face_area
            sources.add("partmodel.face.area")
        normal = raw_node.get("surface_normal")
        if isinstance(normal, list):
            unit = _unit(normal)
            if unit is not None:
                normals.append(unit)
    if sources:
        audit.area_source[leaf] = "+".join(sorted(sources))
    attributes: dict[str, float] = {
        "face_count": float(count),
        "area": area / (l_ref * l_ref),
    }
    if leaf in representation.surface_normal_classes:
        attributes["normal_classes"] = float(_normal_classes(normals))
    return GedFixedNode(
        id=f"__{leaf}__",
        type_key=f"{leaf}Aggregate",
        attributes=attributes,
        raw={"face_count": count, "area_mm2": area, "normal_classes": attributes.get("normal_classes")},
    )


def _unclaimed_aggregate(
    raw_nodes: list[dict[str, Any]],
    faces: dict[str, Any],
    l_ref: float,
) -> GedFixedNode:
    claimed: set[str] = set()
    for raw_node in raw_nodes:
        claimed.update(str(face_id) for face_id in raw_node.get("partmodelFaceIds") or [])
    count = 0
    area = 0.0
    surface_mix: dict[str, int] = {}
    for face_id, face in faces.items():
        if face_id in claimed:
            continue
        count += 1
        face_area = _face_area(face)
        area += face_area or 0.0
        surface_type = str(face.get("surface_type", "UNKNOWN")) if isinstance(face, dict) else "UNKNOWN"
        surface_mix[surface_type] = surface_mix.get(surface_type, 0) + 1
    return GedFixedNode(
        id=FIXED_UNCLAIMED,
        type_key="UnclaimedAggregate",
        attributes={"face_count": float(count), "area": area / (l_ref * l_ref)},
        raw={"face_count": count, "area_mm2": area, "surface_mix": dict(sorted(surface_mix.items()))},
    )


def _dominant_label(evidence: dict[str, Any]) -> str | None:
    histogram = evidence.get("connectionTypes")
    if not isinstance(histogram, dict) or not histogram:
        pairs = evidence.get("adjacentFacePairs")
        if isinstance(pairs, list):
            histogram = {}
            for pair in pairs:
                kind = pair.get("connectionType") if isinstance(pair, dict) else None
                if kind:
                    histogram[kind] = histogram.get(kind, 0) + 1
    if not histogram:
        return None
    # Deterministic: highest count, then lexical order.
    return sorted(histogram.items(), key=lambda item: (-int(item[1]), str(item[0])))[0][0]


def _located_on_endpoints(
    raw_edge: dict[str, Any],
    semantic_ids: set[str],
    surface_ids: set[str],
) -> tuple[str | None, list[str], str]:
    """Return (source feature id, host owner ids, direction) from the evidence.

    ``direction`` is ``"u_on_v"`` when the evidence says which end is located
    on which, and ``"unstated"`` when it does not.  The third case - the two
    ends witnessing each other - used to have no answer at all and cost the
    edge (finding F-09): nx's relation layer records a CORNER_NOTCH_STRAIGHT
    and the STEP1HOLE through it as meeting at these terminals, without
    saying which is on which, so neither the one-witness path nor the
    one-semantic-end fall-back applied and the edge was dropped.  Twelve of
    nx's 73 projected LOCATED_ON edges, on the heaviest relation channel that
    line has.  Leaving the direction blank is not an approximation: no cost
    function reads it - the graph's edge key is the sorted pair plus the type
    - so what was thrown away was a real relation for the sake of a field
    nobody uses.
    """
    evidence = raw_edge.get("evidence") or {}
    properties = raw_edge.get("properties") or {}
    sources: list[str] = []
    witnesses = evidence.get("locatedOnWitnesses") if isinstance(evidence, dict) else None
    if isinstance(witnesses, list):
        for witness in witnesses:
            if isinstance(witness, dict) and witness.get("semanticSourceFeatureId"):
                source = str(witness["semanticSourceFeatureId"])
                if source not in sources:
                    sources.append(source)
    hosts: list[str] = []
    terminals = properties.get("terminals") if isinstance(properties, dict) else None
    if isinstance(terminals, list):
        for terminal in terminals:
            if isinstance(terminal, dict) and terminal.get("hostOwnerNodeId"):
                host = str(terminal["hostOwnerNodeId"])
                if host not in hosts:
                    hosts.append(host)
    endpoints = [str(raw_edge.get("source", "")), str(raw_edge.get("target", ""))]
    if len(sources) == 1:
        return _with_hosts(sources[0], hosts, endpoints, "u_on_v")
    # Fallback: the non-Surface endpoint is the source.
    candidates = [e for e in endpoints if e in semantic_ids]
    others = [e for e in endpoints if e in surface_ids]
    if len(candidates) == 1 and len(others) == 1:
        return _with_hosts(candidates[0], hosts, endpoints, "u_on_v")
    # Mutual witness: exactly two witnesses, and they are exactly this edge's
    # two ends, both of them semantic.  The relation holds; its direction is
    # not stated, and is not invented here.
    if (len(sources) == 2 and endpoints[0] != endpoints[1]
            and set(sources) == set(endpoints)
            and all(end in semantic_ids for end in endpoints)):
        first, second = sorted(endpoints)
        return first, [second], "unstated"
    return None, [], "u_on_v"


def _with_hosts(source: str, hosts: list[str], endpoints: list[str],
                direction: str) -> tuple[str, list[str], str]:
    return source, hosts or [e for e in endpoints if e != source], direction


def _component_sizes(pairs: list[tuple[str, str]]) -> dict[str, int]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in pairs:
        parent.setdefault(u, u)
        parent.setdefault(v, v)
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    sizes: dict[str, int] = {}
    for node in parent:
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    return {node: sizes[find(node)] for node in parent}


# --------------------------------------------------------------------------
# canonical order


def _face_order(face_id: object) -> tuple[float, str]:
    """One active face id, ordered by its CAD face number.

    Same rule as ``topology.faces.face_sort_key`` and for the same reason -
    ``face_9`` before ``face_10`` - but spelled here because the projection
    reads artifacts, not a PartModel, and may not import a kernel line's
    topology helpers.  Both read the number through ``contract.face_ids``, so
    there is one parsing rule, not two.
    """
    number = raw_cad_face_id(face_id)
    return (float(number) if number is not None else math.inf, str(face_id))


def canonical_node_key(node: GedNode) -> tuple[str, tuple[tuple[float, str], ...], str]:
    """The order a graph's nodes enter the solver in.

    ``(class, active face ids in numeric order, node id)``.  The first two
    components are what the *part* is - a node is the class it recognised over
    the faces it claimed - and they do not move when the recognisers walk the
    model in a different order and hand out different ordinals.  The node id
    is only the tie-break, for the case two nodes of one class claim the very
    same face set; ``tools/conclusion_diff.py`` calls that case ambiguous and
    refuses to re-pair it, and here it simply means the ordinal decides.

    Why this exists (node 27 phase 2 §6): the BP solver is an approximation,
    and among equal-cost assignments Hungarian picks the one the enumeration
    order leads to.  Renumbering five parts moved 22 of 72 pairwise distances
    by up to 6.3e-3 while every node, edge and attribute stayed identical.
    Ordering the projection canonically makes the distance a function of the
    graph rather than of the ordinals the pipeline happened to issue.
    """
    return (node.type_key, tuple(sorted(_face_order(f) for f in node.face_ids)), node.id)


def _canonical_edge_key(edge: GedEdge, rank: dict[str, int]) -> tuple[int, int, str]:
    """An edge follows its two ends: the pair of canonical node positions.

    Unordered, because the cost layer's edge key is the sorted pair plus the
    type; ``u`` and ``v`` themselves are left alone, since LOCATED_ON's ``u``
    is the semantic source and swapping it would rewrite what the edge says.
    """
    left, right = rank[edge.u], rank[edge.v]
    return (min(left, right), max(left, right), edge.type_key)


# --------------------------------------------------------------------------
# projection


def project_part(
    result: dict[str, Any],
    partmodel: dict[str, Any],
    *,
    graph_id: str | None = None,
    audit: ProjectionAudit | None = None,
    representation: Representation = V5_1,
) -> GedGraph:
    surface_leaves = representation.surface_leaves
    semantic_leaves = representation.semantic_leaves
    family_of = representation.family_of
    known_relations = representation.known_relations
    grouped_relations = representation.grouped_relations
    host_implying = representation.host_implying_relations
    edge_metrics = representation.edge_metrics
    audit = audit if audit is not None else ProjectionAudit()
    graph_id = graph_id or str(result.get("partId") or "")
    audit.graph_id = graph_id
    if not graph_id:
        raise ValueError("result has no partId and no graph_id was given")

    raw_nodes = result.get("nodes")
    raw_edges = result.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError(f"{graph_id}: nodes/edges schema mismatch")
    faces = partmodel.get("faces")
    if not isinstance(faces, dict):
        raise ValueError(f"{graph_id}: PartModel faces schema mismatch")

    l_ref = geometric_diameter(partmodel)

    # --- nodes --------------------------------------------------------------
    nodes: list[GedNode] = []
    semantic_ids: set[str] = set()
    surface_ids: set[str] = set()
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict):
            raise ValueError(f"{graph_id}: invalid node record")
        node_id = str(raw_node.get("nodeId") or "")
        leaf = str(raw_node.get("featureType") or "")
        if not node_id:
            raise ValueError(f"{graph_id}: node without nodeId")
        if leaf in surface_leaves:
            surface_ids.add(node_id)
            continue
        if leaf not in semantic_leaves:
            if leaf not in audit.unknown_leaf_types:
                audit.unknown_leaf_types.append(leaf)
            # Unknown leaves still enter as nodes (type cost handles them)
            # but carry no comparable attributes.
            nodes.append(GedNode(node_id, leaf, family_of.get(leaf, "Unknown"), {}))
            semantic_ids.add(node_id)
            continue
        axis = raw_node.get("feature_axis")
        nodes.append(
            GedNode(
                id=node_id,
                type_key=leaf,
                family=family_of[leaf],
                attributes=_node_attributes(raw_node, leaf, l_ref, audit, representation),
                feature_axis=_unit(axis) if isinstance(axis, list) else None,
                face_ids=tuple(str(f) for f in raw_node.get("partmodelFaceIds") or []),
            )
        )
        semantic_ids.add(node_id)
    if len(semantic_ids) != len(nodes):
        raise ValueError(f"{graph_id}: duplicate node ids")

    # --- fixed nodes ------------------------------------------------------
    fixed: list[GedFixedNode] = [
        _surface_aggregate(leaf, raw_nodes, faces, l_ref, audit, representation)
        for leaf in surface_leaves
    ]
    fixed.append(_unclaimed_aggregate(raw_nodes, faces, l_ref))

    # --- edges ------------------------------------------------------------
    edges: list[GedEdge] = []
    slots: set[tuple[str, str, str]] = set()
    pair_relations: dict[tuple[str, str], set[str]] = {}
    host_members: dict[str, list[str]] = {}

    def add_edge(edge: GedEdge) -> None:
        key = (*sorted((edge.u, edge.v)), edge.type_key)
        if key in slots:
            raise ValueError(f"{graph_id}: duplicate typed edge slot {key}")
        slots.add(key)
        edges.append(edge)
        pair_relations.setdefault(key[:2], set()).add(edge.type_key)

    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict):
            raise ValueError(f"{graph_id}: invalid edge record")
        edge_type = str(raw_edge.get("edgeType") or "")
        source = str(raw_edge.get("source") or "")
        target = str(raw_edge.get("target") or "")
        evidence = raw_edge.get("evidence") or {}
        properties = raw_edge.get("properties") or {}
        if edge_type not in known_relations:
            if edge_type not in audit.unknown_relation_types:
                audit.unknown_relation_types.append(edge_type)
            audit.note_drop(f"unknown:{edge_type}")
            continue
        if edge_type == "DIRECT_ADJACENT":
            if source in semantic_ids and target in semantic_ids:
                add_edge(GedEdge(source, target, edge_type, label=_dominant_label(evidence)))
            else:
                audit.note_drop("DIRECT_ADJACENT:surface_endpoint")
            continue
        if edge_type == "LOCATED_ON":
            src, hosts, direction = _located_on_endpoints(raw_edge, semantic_ids, surface_ids)
            if src is None or src not in semantic_ids:
                audit.located_on_unresolved.append({"edgeId": raw_edge.get("edgeId"), "source": source, "target": target})
                audit.note_drop("LOCATED_ON:unresolved")
                continue
            for host in hosts:
                if host in surface_ids:
                    members = host_members.setdefault(host, [])
                    if src not in members:
                        members.append(src)
                elif host in semantic_ids and host != src:
                    key = (*sorted((src, host)), "LOCATED_ON")
                    if key in slots:
                        continue
                    roles = [t.get("terminalRole") for t in properties.get("terminals", []) if isinstance(t, dict)]
                    metadata = {"direction": direction, "terminalRoles": roles}
                    if direction == "unstated":
                        metadata["reason"] = "mutual_witness"
                    add_edge(GedEdge(src, host, "LOCATED_ON", metadata=metadata))
                else:
                    audit.located_on_unresolved.append({"edgeId": raw_edge.get("edgeId"), "host": host})
                    audit.note_drop("LOCATED_ON:host_unknown")
            continue
        # metric / axis relations between semantic nodes
        if source not in semantic_ids or target not in semantic_ids:
            audit.note_drop(f"{edge_type}:non_semantic_endpoint")
            continue
        attributes: dict[str, float] = {}
        raw_attributes: dict[str, object] = {}
        for name, norm in edge_metrics.get(edge_type, {}).items():
            value = properties.get(name) if isinstance(properties, dict) else None
            if _is_number(value):
                raw_attributes[name] = value
                attributes[name] = float(value) / l_ref if norm == "length" else float(value) / 90.0
        add_edge(GedEdge(source, target, edge_type, attributes=attributes, raw_attributes=raw_attributes))

    # SHARES_HOST derived edges (design outline §2.2 C / §2.5).
    shares: dict[tuple[str, str], list[str]] = {}
    for host, members in host_members.items():
        ordered = sorted(members)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                pair = (a, b)
                if pair_relations.get(pair, set()) & host_implying:
                    continue
                shares.setdefault(pair, []).append(host)
    for (a, b), hosts in sorted(shares.items()):
        add_edge(GedEdge(a, b, "SHARES_HOST", metadata={"hosts": sorted(hosts)}))

    # Group sizes for clique-normalised relation types.
    grouped: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        if edge.type_key in grouped_relations:
            grouped.setdefault(edge.type_key, []).append((edge.u, edge.v))
    sizes_by_type = {t: _component_sizes(pairs) for t, pairs in grouped.items()}
    finished: list[GedEdge] = []
    for edge in edges:
        if edge.type_key in grouped_relations:
            size = sizes_by_type[edge.type_key][edge.u]
            finished.append(
                GedEdge(
                    edge.u,
                    edge.v,
                    edge.type_key,
                    edge.label,
                    edge.attributes,
                    edge.raw_attributes,
                    {**edge.metadata, "group_size": size},
                )
            )
        else:
            finished.append(edge)

    # Canonical order, last: everything above may append in artifact order,
    # and nothing above reads a position.  The nodes decide, the edges follow.
    nodes.sort(key=canonical_node_key)
    rank = {node.id: index for index, node in enumerate(nodes)}
    finished.sort(key=lambda edge: _canonical_edge_key(edge, rank))

    histogram: dict[str, int] = {}
    for node in nodes:
        histogram[node.type_key] = histogram.get(node.type_key, 0) + 1
    relation_histogram: dict[str, int] = {}
    for edge in finished:
        relation_histogram[edge.type_key] = relation_histogram.get(edge.type_key, 0) + 1
    audit.counts = {
        "rawNodes": len(raw_nodes),
        "semanticNodes": len(nodes),
        "surfaceNodes": len(surface_ids),
        "rawEdges": len(raw_edges),
        "gedEdges": len(finished),
        "hostsWithMembers": len(host_members),
        "L_ref": l_ref,
    }
    meta = {
        "partId": graph_id,
        "sourceStep": result.get("sourceStep"),
        "typeHistogram": dict(sorted(histogram.items())),
        "relationHistogram": dict(sorted(relation_histogram.items())),
        "surfaceNodeCount": len(surface_ids),
        "aagnetStatus": (result.get("diagnostics") or {}).get("aagnetStatus"),
    }
    return GedGraph(
        graph_id=graph_id,
        nodes=tuple(nodes),
        edges=tuple(finished),
        fixed_nodes=tuple(fixed),
        l_ref=l_ref,
        meta=meta,
    )


def load_part(line_dir: str | Path, *, graph_id: str | None = None,
              audit: ProjectionAudit | None = None,
              representation: Representation | None = None) -> GedGraph:
    """Load ``result.json`` + ``partmodel.json`` from one line folder.

    *line_dir* is ``runs/<tag>/parts/<id>/<line>``; the graph is named after
    the *part*, not the line folder, so ``graph_id`` defaults to the parent's
    name rather than to "occ".  The representation likewise defaults to the
    one that line's artifacts are written in, so a caller who names the folder
    never has to name the ontology as well.

    A folder whose name is not a registered line raises: the two ontologies do
    not translate into one another, and reading nx artifacts with v5.1's leaf
    whitelist produces a graph in which every class is unknown and every
    relation is filtered out - a distance, computed without complaint, from
    almost nothing (node 26 F-13).  A caller who really means a particular
    ontology says so by passing ``representation``.
    """
    root = Path(line_dir)
    if representation is None:
        representation = for_line(root.name)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    partmodel = json.loads((root / "partmodel.json").read_text(encoding="utf-8"))
    return project_part(result, partmodel, graph_id=graph_id or root.parent.name,
                        audit=audit, representation=representation)
