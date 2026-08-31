"""Verdict records and the rule interface of the heuristic evaluation layer.

The layer is a strictly downstream reader: it runs after the relation layer,
sees the final ``FeatureNode`` set, the materialized ``FeatureEdge`` set and
the merged ``PartModel``, and writes nothing back.  A heuristic never edits a
node, an edge or a diagnostic; it only produces :class:`Verdict` records.

One heuristic is one module under its dialect's rule package
(``heuristics.v51.rules``, ``heuristics.nx.rules``).  A module declares which
attributes, relations and named thresholds it consumes, and implements the
condition itself — the condition expressions in that dialect's
``heuristics.yaml`` are documentation, not a DSL that is interpreted here.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

# The four values of catalogue 03 chapter 0.1.  NOT_APPLICABLE is a *gate*
# result ("this rule does not talk about this instance"), the other three are
# evaluation results.  Vacuous truth is NOT_APPLICABLE, never PASS.
PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUABLE = "NOT_EVALUABLE"
NOT_APPLICABLE = "NOT_APPLICABLE"
OUTCOMES = (PASS, FAIL, NOT_EVALUABLE, NOT_APPLICABLE)

# Why a unit came out NOT_APPLICABLE.  The three are different findings and
# catalogue 03 chapter 7 forbids collapsing them: "out of domain" is the leaf
# class not being in appliesToFeatureType, "structural" is the attribute not
# being in that leaf class's attribute set (the A-class null), and
# "false antecedent" is a conditional heuristic whose premise does not hold.
NA_OUT_OF_DOMAIN = "outOfDomain"
NA_STRUCTURAL = "structural"
NA_FALSE_ANTECEDENT = "falseAntecedent"
NA_NO_UNIT = "noUnit"


@dataclass
class Judgement:
    """What a rule decides about one evaluation unit."""

    outcome: str
    values: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    na_kind: str | None = None


def passed(reason: str, values: dict | None = None, evidence: dict | None = None) -> Judgement:
    return Judgement(PASS, values or {}, evidence or {}, reason)


def failed(reason: str, values: dict | None = None, evidence: dict | None = None) -> Judgement:
    return Judgement(FAIL, values or {}, evidence or {}, reason)


def not_evaluable(reason: str, values: dict | None = None, evidence: dict | None = None) -> Judgement:
    return Judgement(NOT_EVALUABLE, values or {}, evidence or {}, reason)


def not_applicable(
    reason: str,
    kind: str = NA_FALSE_ANTECEDENT,
    values: dict | None = None,
    evidence: dict | None = None,
) -> Judgement:
    return Judgement(NOT_APPLICABLE, values or {}, evidence or {}, reason, na_kind=kind)


@dataclass
class Verdict:
    """One (heuristic, evaluation unit) result, as written to heuristics.json.

    ``targets`` names the unit in the identities the rest of the pipeline
    already uses — node ids and edge ids — so a reader can drill from a
    verdict straight back to the geometry without re-running anything.
    """

    hId: str
    scope: str
    targets: dict[str, list[str]]
    outcome: str
    values: dict[str, Any] = field(default_factory=dict)
    thresholds: list[dict[str, Any]] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    naKind: str | None = None

    def to_dict(self) -> dict[str, Any]:
        record = {
            "hId": self.hId,
            "scope": self.scope,
            "targets": {key: list(value) for key, value in self.targets.items() if value},
            "outcome": self.outcome,
            "values": self.values,
            "thresholds": self.thresholds,
            "evidence": self.evidence,
            "reason": self.reason,
        }
        if self.naKind:
            record["naKind"] = self.naKind
        return record


class HeuristicRule(ABC):
    """One heuristic.  Pure function of the read-only evaluation context."""

    # Frozen id from the dialect's heuristics.yaml.  The module file name is
    # the same id, one heuristic per file.
    h_id: str = ""
    # feature | pair | terminal | group | chain | part
    scope: str = "feature"
    # Contract declarations, cross-checked against the YAML by the tests:
    # every name here must appear in that heuristic's condition expression.
    USES_ATTRIBUTES: tuple[str, ...] = ()
    USES_RELATIONS: tuple[str, ...] = ()
    USES_THRESHOLDS: tuple[str, ...] = ()
    # Named external operand (catalogue 03 chapter 6.2) this rule needs, if
    # any.  Absent input yields NOT_EVALUABLE with an explicit reason, which
    # must stay distinguishable from a B-class geometric gap.
    EXTERNAL_INPUT: str | None = None

    def required_attributes(self, feature_type: str) -> tuple[str, ...]:
        """Attributes the central gate must check for one leaf class.

        Defaults to :attr:`USES_ATTRIBUTES`; the few heuristics whose carrier
        attribute differs per leaf class (H21, H50) override this.
        """
        return self.USES_ATTRIBUTES

    def evaluate_feature(self, node, context) -> Judgement:  # pragma: no cover - interface
        raise NotImplementedError

    def evaluate(self, context) -> list[Verdict]:  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} {self.h_id} {self.scope}>"


class FeatureRule(HeuristicRule):
    """Single-feature heuristic; the executor owns domain and attribute gating."""

    scope = "feature"


class PairRule(HeuristicRule):
    """Feature-pair heuristic evaluated once per edge of one relation type."""

    scope = "pair"
    edge_type: str = ""

    def evaluate_pair(self, edge, source, target, context) -> Judgement:  # pragma: no cover
        raise NotImplementedError


class PartRule(HeuristicRule):
    """Part-level aggregate; evaluated once per part."""

    scope = "part"

    def evaluate_part(self, context) -> Judgement:  # pragma: no cover - interface
        raise NotImplementedError
