"""H26 - uniform compound-hole operation geometry (advisory, same-host group)."""

from __future__ import annotations

from heuristics.v51._shared import ATTR_OK, aggregate_count_limit, distinct_tuples
from heuristics.base import NOT_EVALUABLE, HeuristicRule, Verdict
from heuristics.v51.registry import register

# Which measurements make up "the operation geometry" of one compound-hole
# leaf class.  The two branches compare different quantities, which is why
# the group key carries the leaf class as well as the host.
GROUP_FIELDS = {
    "Counterbore": ("counterbore_diameter", "counterbore_depth"),
    "Countersink": ("countersink_diameter", "derived_countersink_angle"),
}


@register
class H26(HeuristicRule):
    """Grouping is (effective host, leaf class), never a pattern relation.

    A pattern cannot be proven from B-Rep, and grouping by "same host, same
    type, same size" would use the very sizes this rule checks - the rule
    would prove itself.  Grouping by the LOCATED_ON host owner avoids that
    and works even when the host is another feature rather than a plane.
    """

    h_id = "H26"
    scope = "group"
    USES_ATTRIBUTES = (
        "counterbore_diameter",
        "counterbore_depth",
        "countersink_diameter",
        "derived_countersink_angle",
    )
    USES_RELATIONS = ("LOCATED_ON",)
    USES_THRESHOLDS = ("tol_dim", "tol_angle")

    def required_attributes(self, feature_type):
        # The two branches read different measurements, so the gate is per
        # leaf class rather than the union of both branches.
        return GROUP_FIELDS.get(feature_type, ())

    def evaluate(self, context):
        verdicts = []
        groups: dict[tuple[str, str], list] = {}
        for node in context.domain_nodes(self.h_id):
            hosts = sorted(context.hosts_of(node.nodeId))
            if not hosts:
                verdicts.append(Verdict(
                    hId=self.h_id,
                    scope=self.scope,
                    targets={"nodeIds": [node.nodeId]},
                    outcome=NOT_EVALUABLE,
                    evidence={"blocked": "relationMissing"},
                    reason=context.missing_terminal_reason(node) + "; the hole cannot be grouped",
                ))
                continue
            for host in hosts:
                groups.setdefault((host, node.featureType), []).append(node)

        tol_dim = context.threshold("tol_dim")
        tol_angle = context.threshold("tol_angle")
        for (host, leaf), members in sorted(groups.items()):
            if len(members) < 2:
                # A single compound hole has nothing to be consistent with.
                continue
            fields = self.required_attributes(leaf)
            tolerances = (tol_dim, tol_dim if leaf == "Counterbore" else tol_angle)
            tuples, gaps = [], 0
            for member in members:
                state, _, _ = context.gate_attributes(member, fields)
                if state != ATTR_OK:
                    gaps += 1
                    continue
                tuples.append(tuple(context.attribute(member, field) for field in fields))
            clusters = distinct_tuples(tuples, tolerances)
            judgement = aggregate_count_limit(
                len(clusters),
                1,
                gap_count=gaps,
                values={"groupSize": len(members), "fields": list(fields)},
                evidence={
                    "hostOwnerNodeId": host,
                    "leafClass": leaf,
                    "memberNodeIds": [member.nodeId for member in members],
                    "operationGeometries": [list(cluster[0]) for cluster in clusters],
                },
                quantity=f"distinct {leaf} operation geometries on host {host}",
            )
            verdicts.append(Verdict(
                hId=self.h_id,
                scope=self.scope,
                targets={"nodeIds": [member.nodeId for member in members]},
                outcome=judgement.outcome,
                values=judgement.values,
                evidence=judgement.evidence,
                reason=judgement.reason,
                naKind=judgement.na_kind,
            ))
        return verdicts
