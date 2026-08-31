"""Read-only evaluation context and the operators several heuristics share.

Everything here reads the finished artifacts only: final ``FeatureNode``
objects, materialized ``FeatureEdge`` objects and the merged ``PartModel``.
Nothing is recomputed from the STEP file, and nothing is written back.

This is machinery: it names no ontology's classes and no ontology's
attributes.  A dialect's own vocabulary — which leaf classes are holes, how
many terminals a hole is expected to have — lives beside that dialect's rules
(``heuristics.v51._shared``, ``heuristics.nx._shared``).

Three groups live here:

* the context and its indexes (nodes by id, edges by type, the LOCATED_ON
  terminal table), built once per part instead of once per heuristic;
* the query-time derived predicates of catalogue 02 chapter 7
  (``SHARES_HOST`` / ``PARALLEL_AXIS`` / ``ORTHOGONAL_AXIS``), which are facts
  a heuristic may use but are not relations and are never materialized;
* the tolerance-aware ``distinct`` operator and the aggregate missing-value
  rule of catalogue 03 chapters 0.6 and 0.7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from heuristics.base import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    Judgement,
    failed,
    not_evaluable,
    passed,
)
from heuristics.registry import BaseHeuristicRegistry, default_registry

# States of one attribute on one instance, per catalogue 03 chapter 7.2:
#   key absent     -> the attribute is not in this leaf class's attribute set
#   key with null  -> B-class gap, the reason is in missingAttributes
#   key with value -> evaluable
ATTR_OK = "ok"
ATTR_ABSENT = "absent"
ATTR_GAP = "gap"


@dataclass
class HeuristicContext:
    """Everything a heuristic rule is allowed to see."""

    model: Any
    nodes: tuple = ()
    edges: tuple = ()
    node_by_id: dict = field(default_factory=dict)
    edges_by_type: dict = field(default_factory=dict)
    # Named external operands (tool magazine, design intent).  The pipeline
    # never invents these; an absent key makes the dependent heuristic
    # NOT_EVALUABLE with an explicit "external input missing" reason.
    external_inputs: dict = field(default_factory=dict)
    # The relation layer's own report.  Read only to explain why a terminal
    # is missing; a heuristic never re-derives relation decisions from it.
    relation_diagnostics: dict = field(default_factory=dict)
    # Which dialect's definitions, domains and thresholds are in play.  A rule
    # asks the context, never a module-level registry, so the same machinery
    # serves both ontologies without a branch.
    registry: type | None = None
    cache: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = default_registry()

    @classmethod
    def build(
        cls,
        model,
        nodes,
        edges,
        *,
        external_inputs: dict | None = None,
        relation_diagnostics: dict | None = None,
        registry: type[BaseHeuristicRegistry] | None = None,
    ) -> "HeuristicContext":
        by_type: dict[str, list] = {}
        for edge in edges:
            by_type.setdefault(edge.edgeType, []).append(edge)
        return cls(
            model=model,
            nodes=tuple(nodes),
            edges=tuple(edges),
            node_by_id={node.nodeId: node for node in nodes},
            edges_by_type=by_type,
            external_inputs=dict(external_inputs or {}),
            relation_diagnostics=dict(relation_diagnostics or {}),
            registry=registry,
        )

    # -- nodes and attributes ---------------------------------------------
    def nodes_of(self, types: Iterable[str]) -> list:
        wanted = frozenset(types)
        return [node for node in self.nodes if node.featureType in wanted]

    def domain_nodes(self, h_id: str) -> list:
        """Instances of the leaf classes this heuristic applies to.

        Taken from ``appliesToFeatureType`` in the YAML, never from a class
        list written into a rule module, so a domain change is a source edit.
        """
        return [node for node in self.nodes
                if self.registry.in_domain(h_id, node.featureType)]

    def attributes(self, node) -> dict:
        return getattr(node, "attributes", {}) or {}

    def attribute(self, node, name: str):
        return self.attributes(node).get(name)

    def attribute_state(self, node, name: str) -> tuple[str, str]:
        """``(state, reason)`` for one attribute on one instance."""
        attributes = self.attributes(node)
        if name not in attributes:
            return ATTR_ABSENT, f"{name} is not in the attribute set of {node.featureType}"
        if attributes[name] is None:
            reason = (getattr(node, "missingAttributes", {}) or {}).get(name)
            return ATTR_GAP, reason or f"{name} is null on {node.nodeId} without a recorded reason"
        return ATTR_OK, ""

    def gate_attributes(self, node, names: Iterable[str]) -> tuple[str, str, str]:
        """First blocking attribute state over *names*: ``(state, name, reason)``."""
        for name in names:
            state, reason = self.attribute_state(node, name)
            if state != ATTR_OK:
                return state, name, reason
        return ATTR_OK, "", ""

    # -- thresholds --------------------------------------------------------
    def threshold(self, symbol: str):
        return self.registry.value(symbol)

    # -- relations ---------------------------------------------------------
    def edges_of(self, edge_type: str) -> list:
        return self.edges_by_type.get(edge_type, [])

    def _memo(self, key: str, factory):
        if key not in self.cache:
            self.cache[key] = factory()
        return self.cache[key]

    def terminals(self) -> list[dict]:
        """Flat LOCATED_ON terminal table.

        One entry per proven terminal: which feature it belongs to, which role
        it plays, which face hosts it and which node owns that face.  The
        ``hostSurfaceNormal`` is the physical outward normal the relation layer
        froze on the edge - H27 needs it even when the host owner is not a
        ``PlanarFace`` node.
        """

        def build() -> list[dict]:
            table: list[dict] = []
            for edge in self.edges_of("LOCATED_ON"):
                witnesses = (edge.evidence or {}).get("locatedOnWitnesses") or []
                terminals = (edge.properties or {}).get("terminals") or []
                for index, terminal in enumerate(terminals):
                    witness = witnesses[index] if index < len(witnesses) else {}
                    source_id = witness.get("semanticSourceFeatureId")
                    if source_id is None:
                        # Fall back to the graph endpoints: an edge always has
                        # one endpoint that is not the resolved host owner.
                        host = terminal.get("hostOwnerNodeId")
                        source_id = edge.target if edge.source == host else edge.source
                    table.append({
                        "edgeId": edge.edgeId,
                        "sourceNodeId": source_id,
                        "sourceFeatureType": (
                            witness.get("semanticSourceFeatureType")
                            or getattr(self.node_by_id.get(source_id), "featureType", None)
                        ),
                        "terminalRole": terminal.get("terminalRole") or witness.get("terminalRole"),
                        "hostFaceId": terminal.get("hostFaceId"),
                        "hostSurfaceType": terminal.get("hostSurfaceType"),
                        "hostSurfaceNormal": terminal.get("hostSurfaceNormal"),
                        "hostOwnerNodeId": terminal.get("hostOwnerNodeId"),
                        "hostOwnerFeatureType": getattr(
                            self.node_by_id.get(terminal.get("hostOwnerNodeId")), "featureType", None
                        ),
                    })
            return table

        return self._memo("terminals", build)

    def terminals_of(self, node_id: str) -> list[dict]:
        def build() -> dict[str, list[dict]]:
            index: dict[str, list[dict]] = {}
            for entry in self.terminals():
                index.setdefault(entry["sourceNodeId"], []).append(entry)
            return index

        return self._memo("terminalsBySource", build).get(node_id, [])

    def terminals_on(self, host_node_id: str) -> list[dict]:
        def build() -> dict[str, list[dict]]:
            index: dict[str, list[dict]] = {}
            for entry in self.terminals():
                index.setdefault(entry["hostOwnerNodeId"], []).append(entry)
            return index

        return self._memo("terminalsByHost", build).get(host_node_id, [])

    def _located_on_reports(self) -> list[dict]:
        """The relation layer's own reports for whichever rule proved LOCATED_ON.

        Keyed on the edge type rather than on a rule id: each dialect names its
        rule differently and the layer is entitled to, but the edge type is the
        thing both of them publish.
        """
        return [rule for rule in ((self.relation_diagnostics or {}).get("rules") or [])
                if rule.get("edgeType") == "LOCATED_ON"]

    def rejected_terminals(self, node_id: str) -> list[dict]:
        """Terminals the relation layer refused to prove, for this feature."""
        records = []
        for rule in self._located_on_reports():
            for record in (rule.get("diagnostics") or {}).get("rejectedTerminals") or []:
                if record.get("semanticSourceFeatureId") == node_id:
                    records.append(record)
        return records

    def rejected_terminals_for_host(self, host_node_id: str) -> list[dict]:
        """Terminals that named this node as a candidate host but were rejected.

        A boss whose holes lost their terminal proof cannot be asked whether a
        hole sits on it; H13 needs that to answer NOT_EVALUABLE instead of
        silently reading the ambiguity as "no hole here".
        """
        records = []
        for rule in self._located_on_reports():
            for record in (rule.get("diagnostics") or {}).get("rejectedTerminals") or []:
                if host_node_id in (record.get("candidateHostOwnerNodeIds") or []):
                    records.append(record)
        return records

    def missing_terminal_reason(self, node) -> str:
        rejected = self.rejected_terminals(node.nodeId)
        if rejected:
            roles = ", ".join(sorted({str(item.get("terminalRole")) for item in rejected}))
            reasons = ", ".join(sorted({str(item.get("reason")) for item in rejected}))
            return f"LOCATED_ON terminal not proven ({roles}): {reasons}"
        return "LOCATED_ON terminal not proven for this feature"

    def related(self, node_id: str, edge_type: str) -> list[tuple[str, Any]]:
        """``(other node id, edge)`` for every edge of *edge_type* at a node."""

        def build() -> dict[tuple[str, str], list[tuple[str, Any]]]:
            index: dict[tuple[str, str], list[tuple[str, Any]]] = {}
            for edge in self.edges:
                index.setdefault((edge.source, edge.edgeType), []).append((edge.target, edge))
                index.setdefault((edge.target, edge.edgeType), []).append((edge.source, edge))
            return index

        return self._memo("incident", build).get((node_id, edge_type), [])

    def has_edge(self, a: str, b: str, edge_type: str) -> bool:
        return any(other == b for other, _ in self.related(a, edge_type))

    # -- derived predicates (catalogue 02 chapter 7) -----------------------
    def hosts_of(self, node_id: str) -> set[str]:
        return {entry["hostOwnerNodeId"] for entry in self.terminals_of(node_id) if entry["hostOwnerNodeId"]}

    def shares_host(self, a: str, b: str) -> bool:
        """SHARES_HOST: two features resolve to intersecting effective hosts."""
        return bool(self.hosts_of(a) & self.hosts_of(b))

    def parallel_axis(self, a, b, tolerance_deg: float = 1.0) -> bool | None:
        """PARALLEL_AXIS over undirected feature axes; ``None`` when unknown."""
        left, right = unit(self.attribute(a, "feature_axis")), unit(self.attribute(b, "feature_axis"))
        if left is None or right is None:
            return None
        return acute_angle_deg(left, right) <= tolerance_deg

    def orthogonal_axis(self, a, b, tolerance_deg: float = 1.0) -> bool | None:
        """ORTHOGONAL_AXIS over undirected feature axes; ``None`` when unknown."""
        left, right = unit(self.attribute(a, "feature_axis")), unit(self.attribute(b, "feature_axis"))
        if left is None or right is None:
            return None
        return abs(acute_angle_deg(left, right) - 90.0) <= tolerance_deg


# ---------------------------------------------------------------- geometry

def unit(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        components = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in components):
        return None
    length = math.sqrt(sum(item * item for item in components))
    if length <= 1.0e-12:
        return None
    return [item / length for item in components]


def dot(left, right) -> float:
    return sum(left[index] * right[index] for index in range(3))


def acute_angle_deg(left, right) -> float:
    """Angle between two *undirected* directions, in [0, 90]."""
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(dot(left, right))))))


# ------------------------------------------------- distinct with tolerance

def distinct(values: list, tolerance: float) -> list[list]:
    """Cluster scalars by *tolerance* and return the clusters.

    Catalogue 03 chapter 0.6: exact float equality is not a usable criterion
    on millimetre measurements, so every "how many different values" rule is
    written as ``distinct(expr, tol)``.  Single-link clustering on the sorted
    values, which is what "one tool produced all of these" means.
    """
    if not values:
        return []
    ordered = sorted(values)
    clusters = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return clusters


def distinct_tuples(values: list[tuple], tolerances: tuple[float, ...]) -> list[list[tuple]]:
    """Cluster tuples: two agree when every component is within its tolerance."""
    clusters: list[list[tuple]] = []
    for value in values:
        for cluster in clusters:
            head = cluster[0]
            if all(abs(value[i] - head[i]) <= tolerances[i] for i in range(len(tolerances))):
                cluster.append(value)
                break
        else:
            clusters.append([value])
    return clusters


def distinct_axes(axes: list[list[float]], tolerance_deg: float) -> list[list[list[float]]]:
    """Cluster undirected axis directions; opposite vectors are one orientation."""
    clusters: list[list[list[float]]] = []
    for axis in axes:
        for cluster in clusters:
            if acute_angle_deg(cluster[0], axis) <= tolerance_deg:
                cluster.append(axis)
                break
        else:
            clusters.append([axis])
    return clusters


def aggregate_count_limit(
    cluster_count: int,
    limit: float,
    *,
    gap_count: int,
    values: dict,
    evidence: dict | None = None,
    quantity: str = "distinct values",
) -> Judgement:
    """The aggregate missing-value rule (catalogue 03 chapter 0.7).

    Counting rules all read ``count <= t`` and are monotone in the count, so
    the clusters of the *known* values are a lower bound on the true count:
    once that bound exceeds the limit no unknown value can turn the verdict
    around, and the aggregate is a proven FAIL.  Below the limit with an
    unknown present the count is genuinely not obtainable - NOT_EVALUABLE,
    not an optimistic PASS.
    """
    detail = dict(values)
    detail["distinctCount"] = cluster_count
    detail["nullCount"] = gap_count
    if cluster_count > limit:
        return failed(
            f"{quantity} {cluster_count} > {limit}"
            + (f" (lower bound; {gap_count} value(s) unknown)" if gap_count else ""),
            detail,
            evidence or {},
        )
    if gap_count:
        return not_evaluable(
            f"{quantity} lower bound {cluster_count} <= {limit}, but {gap_count} value(s) are null",
            detail,
            evidence or {},
        )
    return passed(f"{quantity} {cluster_count} <= {limit}", detail, evidence or {})


# ---------------------------------------------------------- simple compare

def compare_at_least(value: float, limit: float, name: str, symbol: str, extra: dict | None = None) -> Judgement:
    values = {name: value, **(extra or {})}
    operator = ">=" if value >= limit else "<"
    reason = f"{name} {round(value, 6)} {operator} {symbol} {limit}"
    return passed(reason, values) if value >= limit else failed(reason, values)


def compare_at_most(value: float, limit: float, name: str, symbol: str, extra: dict | None = None) -> Judgement:
    values = {name: value, **(extra or {})}
    operator = "<=" if value <= limit else ">"
    reason = f"{name} {round(value, 6)} {operator} {symbol} {limit}"
    return passed(reason, values) if value <= limit else failed(reason, values)


def external_missing(operand: str) -> Judgement:
    """NOT_EVALUABLE because an operand outside the geometry pipeline is absent.

    Catalogue 03 chapter 6.2: a tool magazine list or a design-intent flag is
    not a geometric quantity, so no amount of geometry work produces it.  The
    reason is tagged so the report never reads "the tool magazine is not
    connected" as a gap in what this pipeline owes.
    """
    return not_evaluable(
        f"external input missing: EXT.{operand}",
        {},
        {"blocked": "externalInputMissing", "externalOperand": f"EXT.{operand}"},
    )


def relation_missing(reason: str, evidence: dict | None = None) -> Judgement:
    """NOT_EVALUABLE because a relation the condition needs was not proven."""
    detail = {"blocked": "relationMissing", **(evidence or {})}
    return not_evaluable(reason, {}, detail)


def member_of(value: float, allowed: list, tolerance: float, name: str, symbol: str) -> Judgement:
    """Membership in a controlled numeric set, judged within *tolerance*."""
    hit = next((item for item in allowed if abs(value - item) <= tolerance), None)
    values = {name: value, "matched": hit}
    if hit is None:
        return failed(f"{name} {round(value, 6)} matches no member of {symbol} {allowed}", values)
    return passed(f"{name} {round(value, 6)} matches {hit} in {symbol}", values)


__all__ = [
    "ATTR_ABSENT",
    "ATTR_GAP",
    "ATTR_OK",
    "FAIL",
    "HeuristicContext",
    "NOT_EVALUABLE",
    "PASS",
    "acute_angle_deg",
    "aggregate_count_limit",
    "compare_at_least",
    "compare_at_most",
    "distinct",
    "distinct_axes",
    "distinct_tuples",
    "dot",
    "external_missing",
    "failed",
    "member_of",
    "not_evaluable",
    "passed",
    "relation_missing",
    "unit",
]
