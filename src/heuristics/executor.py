"""Run every registered heuristic over one finished part.

The executor owns the central gate of catalogue 03 chapter 0.1, so that no
rule module repeats it:

1. leaf class not in ``appliesToFeatureType``      -> NOT_APPLICABLE (counted
   only, never one record per instance: this is the largest bucket by far and
   listing it would bury the informative verdicts);
2. in domain, but the attribute the condition reads is not in that leaf
   class's attribute set (key absent)              -> NOT_APPLICABLE, structural;
3. in domain and in the attribute set, but the value is null (B-class gap)
   -> NOT_EVALUABLE, carrying the geometric reason from ``missingAttributes``;
4. otherwise the rule decides, and a conditional heuristic whose premise is
   false returns NOT_APPLICABLE - vacuous truth is never PASS.

The three-line rule of chapter 7.2 ("key absent / key null / key present") is
implemented once, in :meth:`HeuristicContext.gate_attributes`, and needs no
per-heuristic or per-attribute branch.
"""

from __future__ import annotations

import traceback
from collections import Counter

from heuristics._shared import ATTR_ABSENT, ATTR_GAP, HeuristicContext
from heuristics.base import (
    NA_NO_UNIT,
    NA_STRUCTURAL,
    NOT_APPLICABLE,
    NOT_EVALUABLE,
    OUTCOMES,
    PASS,
    FAIL,
    FeatureRule,
    Judgement,
    PairRule,
    PartRule,
    Verdict,
)
from heuristics.registry import BaseHeuristicRegistry, default_registry

SCHEMA_VERSION = "v8.heuristics.v1"


def run_heuristics(
    model,
    nodes: list,
    edges: list,
    *,
    external_inputs: dict | None = None,
    relation_diagnostics: dict | None = None,
    registry: type[BaseHeuristicRegistry] | None = None,
    context_factory=HeuristicContext,
) -> tuple[dict, dict]:
    """Return ``(payload, summary)`` for one part.

    ``payload`` is the content of ``heuristics.json`` minus the part id, which
    the orchestrator adds; ``summary`` is the per-heuristic four-value count
    the manifest aggregates over the corpus.

    ``registry`` selects the dialect - which ontology's heuristic set, domains
    and thresholds are in play - and ``context_factory`` the reading of a node
    that goes with it.  The gate below is the same one either way; what
    differs is the vocabulary it gates against, which is why neither is a
    branch inside this function.
    """

    registry = registry or default_registry()
    context = context_factory.build(
        model,
        nodes,
        edges,
        external_inputs=external_inputs,
        relation_diagnostics=relation_diagnostics,
        registry=registry,
    )

    records: list[dict] = []
    summary: dict[str, dict] = {}
    errors: list[dict] = []

    for rule in registry.rules():
        definition = registry.definition(rule.h_id)
        thresholds = registry.threshold_records(rule.USES_THRESHOLDS)
        domain_nodes = [node for node in context.nodes
                        if registry.in_domain(rule.h_id, node.featureType)]
        entry = _new_summary(rule, definition, len(domain_nodes), len(context.nodes) - len(domain_nodes))

        try:
            if isinstance(rule, FeatureRule):
                verdicts = _run_feature_rule(rule, context, domain_nodes, entry)
            elif isinstance(rule, PairRule):
                verdicts = _run_pair_rule(rule, context, entry)
            elif isinstance(rule, PartRule):
                verdicts = _run_part_rule(rule, context)
            else:
                verdicts = list(rule.evaluate(context))
        except Exception as exc:  # one broken rule must not lose the part
            errors.append({
                "hId": rule.h_id,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            entry["error"] = f"{type(exc).__name__}: {exc}"
            summary[rule.h_id] = entry
            continue

        for verdict in verdicts:
            verdict.hId = verdict.hId or rule.h_id
            verdict.scope = verdict.scope or rule.scope
            if not verdict.thresholds:
                verdict.thresholds = thresholds
            _count(entry, verdict)
            records.append(verdict.to_dict())

        summary[rule.h_id] = entry

    records.sort(key=lambda item: (item["hId"], item["targets"].get("nodeIds", [""])[:1], item["outcome"]))

    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "registry": registry.version(),
        "coverage": registry.coverage(),
        "heuristics": _catalogue(registry),
        "summary": summary,
        "records": records,
    }
    if errors:
        payload["errors"] = errors
    return payload, summary


def not_evaluated(reason: str, *, node_count: int = 0) -> tuple[dict, dict]:
    """The document a line publishes when it evaluates no rule at all.

    A missing file and a file saying "no rule was evaluated here, and here is
    why" are different statements, and only the second one still answers a
    question six months later.  The shape is exactly ``run_heuristics``'s, so
    the panel, the store's counts and the statistics view read it without a
    special case; ``notEvaluated`` is the one extra key, and it is the reason.

    This is the layer's own business rather than any line's: the file's shape
    belongs to whoever defines it, and a second line that evaluates nothing
    must not have to copy a vendor's spelling of an empty document.
    """
    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "notEvaluated": reason,
        "registry": {},
        "coverage": {},
        "heuristics": {},
        "summary": {},
        "records": [],
        "diagnostics": {"reason": reason, "featureNodeCount": node_count},
    }
    return payload, {}


# --------------------------------------------------------------- internals

def _new_summary(rule, definition: dict, domain_nodes: int, out_of_domain: int) -> dict:
    return {
        "hId": rule.h_id,
        "scope": rule.scope,
        "grade": definition.get("grade"),
        "externalDependency": definition.get("external_dependency_en"),
        "domainNodes": domain_nodes,
        "outOfDomainNodes": out_of_domain,
        "records": 0,
        "counts": {outcome: 0 for outcome in OUTCOMES},
        "naKinds": {},
        "blocked": {},
    }


def _count(entry: dict, verdict: Verdict) -> None:
    entry["records"] += 1
    entry["counts"][verdict.outcome] = entry["counts"].get(verdict.outcome, 0) + 1
    if verdict.outcome == NOT_APPLICABLE and verdict.naKind:
        entry["naKinds"][verdict.naKind] = entry["naKinds"].get(verdict.naKind, 0) + 1
    if verdict.outcome == NOT_EVALUABLE:
        # "The tool magazine has not been connected" and "the geometry could
        # not be measured here" are different findings for whoever reads the
        # report; catalogue 03 chapter 5.3 requires them to stay apart.
        blocked = (verdict.evidence or {}).get("blocked") or "attributeGap"
        entry["blocked"][blocked] = entry["blocked"].get(blocked, 0) + 1


def _run_feature_rule(rule, context: HeuristicContext, domain_nodes: list, entry: dict) -> list[Verdict]:
    """Central gate for single-feature heuristics."""
    verdicts: list[Verdict] = []
    structural = 0
    for node in domain_nodes:
        state, name, reason = context.gate_attributes(node, rule.required_attributes(node.featureType))
        if state == ATTR_ABSENT:
            # A-class: the attribute is not on this class at all.  Counted,
            # not recorded - it is a property of the class, not of the part.
            structural += 1
            continue
        if state == ATTR_GAP:
            verdicts.append(_verdict(
                rule,
                node,
                Judgement(
                    NOT_EVALUABLE,
                    {},
                    {"blocked": "attributeGap", "attribute": name},
                    f"{name} is null: {reason}",
                ),
            ))
            continue
        verdicts.append(_verdict(rule, node, rule.evaluate_feature(node, context)))
    if structural:
        entry["naKinds"][NA_STRUCTURAL] = entry["naKinds"].get(NA_STRUCTURAL, 0) + structural
        entry["counts"][NOT_APPLICABLE] += structural
    return verdicts


def _run_pair_rule(rule, context: HeuristicContext, entry: dict) -> list[Verdict]:
    """Central gate for feature-pair heuristics.

    The unit is one edge of the rule's relation type, never an arbitrary pair
    of features.  A pair with no edge is *not* a violation (catalogue 02
    forbids reading a missing edge as one) and is not a unit either, so it
    produces no record - the relation layer already recorded why it withheld
    the edge.
    """
    verdicts: list[Verdict] = []
    structural = 0
    for edge in context.edges_of(rule.edge_type):
        source = context.node_by_id.get(edge.source)
        target = context.node_by_id.get(edge.target)
        if source is None or target is None:
            continue
        registry = context.registry
        if not (registry.in_domain(rule.h_id, source.featureType)
                and registry.in_domain(rule.h_id, target.featureType)):
            continue
        targets = {"nodeIds": [source.nodeId, target.nodeId], "edgeIds": [edge.edgeId]}
        blocked = None
        for node in (source, target):
            state, name, reason = context.gate_attributes(node, rule.required_attributes(node.featureType))
            if state == ATTR_ABSENT:
                blocked = (NOT_APPLICABLE, name, reason, node)
                break
            if state == ATTR_GAP:
                blocked = (NOT_EVALUABLE, name, reason, node)
                break
        if blocked and blocked[0] == NOT_APPLICABLE:
            structural += 1
            continue
        if blocked:
            _, name, reason, node = blocked
            verdicts.append(Verdict(
                hId=rule.h_id,
                scope=rule.scope,
                targets=targets,
                outcome=NOT_EVALUABLE,
                evidence={"blocked": "attributeGap", "attribute": name, "nodeId": node.nodeId},
                reason=f"{name} is null on {node.nodeId}: {reason}",
            ))
            continue
        judgement = rule.evaluate_pair(edge, source, target, context)
        verdicts.append(Verdict(
            hId=rule.h_id,
            scope=rule.scope,
            targets=targets,
            outcome=judgement.outcome,
            values=judgement.values,
            evidence=judgement.evidence,
            reason=judgement.reason,
            naKind=judgement.na_kind,
        ))
    if structural:
        entry["naKinds"][NA_STRUCTURAL] = entry["naKinds"].get(NA_STRUCTURAL, 0) + structural
        entry["counts"][NOT_APPLICABLE] += structural
    return verdicts


def _run_part_rule(rule, context: HeuristicContext) -> list[Verdict]:
    """Part-level aggregate: exactly one unit per part, or none at all.

    ``None`` means the part carries no instance the aggregate counts over -
    an empty set is NOT_APPLICABLE, never a vacuous PASS.
    """
    judgement = rule.evaluate_part(context)
    if judgement is None:
        return [Verdict(
            hId=rule.h_id,
            scope=rule.scope,
            targets={},
            outcome=NOT_APPLICABLE,
            reason="the part carries no instance this aggregate counts over",
            naKind=NA_NO_UNIT,
        )]
    members = (judgement.evidence or {}).get("memberNodeIds") or []
    return [Verdict(
        hId=rule.h_id,
        scope=rule.scope,
        targets={"nodeIds": list(members)},
        outcome=judgement.outcome,
        values=judgement.values,
        evidence=judgement.evidence,
        reason=judgement.reason,
        naKind=judgement.na_kind,
    )]


def _verdict(rule, node, judgement: Judgement) -> Verdict:
    return Verdict(
        hId=rule.h_id,
        scope=rule.scope,
        targets={"nodeIds": [node.nodeId]},
        outcome=judgement.outcome,
        values=judgement.values,
        evidence=judgement.evidence,
        reason=judgement.reason,
        naKind=judgement.na_kind,
    )


def _catalogue(registry: type[BaseHeuristicRegistry]) -> dict:
    """The frozen definition of every registered heuristic, for the readers.

    ``heuristics.json`` is meant to be readable without the ontology at hand:
    the viewer, the statistics page and a later parts store all need the
    statement, grade and thresholds next to the verdicts.
    """
    catalogue = {}
    for rule in registry.rules():
        definition = registry.definition(rule.h_id)
        catalogue[rule.h_id] = {
            "labelEn": definition.get("label_en"),
            "statementEn": definition.get("statement_en"),
            "condition": definition.get("condition"),
            "grade": definition.get("grade"),
            "scope": rule.scope,
            "scopeText": definition.get("scope_en"),
            "appliesTo": list(definition.get("applies_to") or []),
            "usesRelations": list(definition.get("uses_relations") or []),
            "externalDependency": definition.get("external_dependency_en"),
            # The ontology names the dependency; the rule module names the
            # operand itself, which says more anyway: not "tool magazine" but
            # which list out of it is missing.
            "externalOperand": f"EXT.{rule.EXTERNAL_INPUT}" if rule.EXTERNAL_INPUT else None,
            "thresholds": registry.threshold_records(rule.USES_THRESHOLDS),
            "affectsKpi": list(definition.get("affects_kpi") or []),
        }
    return catalogue


def corpus_summary(part_summaries: list[dict]) -> dict:
    """Fold per-part summaries into one corpus-level table."""
    folded: dict[str, dict] = {}
    for summary in part_summaries:
        for h_id, entry in summary.items():
            target = folded.setdefault(h_id, {
                "hId": h_id,
                "scope": entry.get("scope"),
                "grade": entry.get("grade"),
                "externalDependency": entry.get("externalDependency"),
                "domainNodes": 0,
                "records": 0,
                "counts": {outcome: 0 for outcome in OUTCOMES},
                "naKinds": {},
                "blocked": {},
                "parts": 0,
            })
            target["parts"] += 1
            target["domainNodes"] += entry.get("domainNodes", 0)
            target["records"] += entry.get("records", 0)
            for key, value in entry.get("counts", {}).items():
                target["counts"][key] = target["counts"].get(key, 0) + value
            for bucket in ("naKinds", "blocked"):
                for key, value in (entry.get(bucket) or {}).items():
                    target[bucket][key] = target[bucket].get(key, 0) + value
    return dict(sorted(folded.items()))


def outcome_counts(summary: dict) -> dict[str, int]:
    """Flat four-value total over one part's summary, for the manifest."""
    totals: Counter = Counter()
    for entry in summary.values():
        for outcome, count in entry.get("counts", {}).items():
            totals[outcome] += count
    return {outcome: int(totals.get(outcome, 0)) for outcome in (PASS, FAIL, NOT_EVALUABLE, NOT_APPLICABLE)}
