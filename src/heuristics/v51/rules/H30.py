"""H30 - monotonic diameter along a coaxial hole chain (advisory, chain)."""

from __future__ import annotations

from heuristics.v51._shared import ATTR_OK, HOLE_TYPES, dot, unit
from heuristics.base import (
    FAIL,
    NA_FALSE_ANTECEDENT,
    NOT_APPLICABLE,
    NOT_EVALUABLE,
    PASS,
    HeuristicRule,
    Verdict,
)
from heuristics.v51.registry import register


@register
class H30(HeuristicRule):
    """Order comes from the chain topology, never from a position coordinate.

    Two holes that are coaxial *and* share a real B-Rep edge are consecutive
    segments of one bore, so walking the chain is walking along the axis.  The
    entry end is the member whose LOCATED_ON host is not another hole, i.e.
    the one that opens on the outside of the part.  Two such ends means the
    bore was machined from both sides and a single monotone sequence is not
    what the design intends - not applicable rather than a violation.
    """

    h_id = "H30"
    scope = "chain"
    USES_ATTRIBUTES = ("diameter", "machining_direction")
    USES_RELATIONS = ("COAXIAL", "DIRECT_ADJACENT", "LOCATED_ON")

    def required_attributes(self, feature_type):
        return ()

    def evaluate(self, context):
        holes = {node.nodeId: node for node in context.domain_nodes(self.h_id)}
        links: dict[str, set[str]] = {node_id: set() for node_id in holes}
        for edge in context.edges_of("COAXIAL"):
            if edge.source not in holes or edge.target not in holes:
                continue
            if not context.has_edge(edge.source, edge.target, "DIRECT_ADJACENT"):
                continue
            links[edge.source].add(edge.target)
            links[edge.target].add(edge.source)

        verdicts = []
        for component in _components(links):
            if len(component) < 2:
                continue
            verdicts.append(self._judge(component, links, holes, context))
        return verdicts

    def _judge(self, component, links, holes, context):
        members = sorted(component)
        targets = {"nodeIds": members}
        evidence = {"chainMemberNodeIds": members}

        unresolved = [node_id for node_id in members if not context.terminals_of(node_id)]
        if unresolved:
            return Verdict(
                hId=self.h_id, scope=self.scope, targets=targets, outcome=NOT_EVALUABLE,
                evidence={**evidence, "blocked": "relationMissing", "unresolvedNodeIds": unresolved},
                reason="a chain member has no proven LOCATED_ON terminal, so the entry end "
                       "cannot be identified",
            )

        entries = [
            node_id for node_id in members
            if any(
                terminal["hostOwnerFeatureType"] not in HOLE_TYPES
                for terminal in context.terminals_of(node_id)
            )
        ]
        evidence["entryNodeIds"] = entries
        if len(entries) != 1:
            return Verdict(
                hId=self.h_id, scope=self.scope, targets=targets, outcome=NOT_APPLICABLE,
                values={"entryCount": len(entries)}, evidence=evidence,
                reason=f"the chain has {len(entries)} entries into the part; a single feed "
                       "sequence is not what this chain is",
                naKind=NA_FALSE_ANTECEDENT,
            )

        directions = []
        for node_id in members:
            state, _ = context.attribute_state(holes[node_id], "machining_direction")
            direction = (
                unit(context.attribute(holes[node_id], "machining_direction"))
                if state == ATTR_OK else None
            )
            if direction is not None:
                directions.append((node_id, direction))
        for index, (left_id, left) in enumerate(directions):
            for right_id, right in directions[index + 1:]:
                if dot(left, right) <= 0.0:
                    return Verdict(
                        hId=self.h_id, scope=self.scope, targets=targets, outcome=NOT_APPLICABLE,
                        evidence={**evidence, "opposedNodeIds": [left_id, right_id]},
                        reason="two chain members are fed from opposite directions, so they are "
                               "not one single-sided sequence",
                        naKind=NA_FALSE_ANTECEDENT,
                    )

        order = _walk(entries[0], links, component)
        if order is None:
            return Verdict(
                hId=self.h_id, scope=self.scope, targets=targets, outcome=NOT_EVALUABLE,
                evidence={**evidence, "blocked": "relationMissing"},
                reason="the coaxial-adjacent component branches, so it is not a single chain",
            )

        diameters = []
        for node_id in order:
            state, _ = context.attribute_state(holes[node_id], "diameter")
            if state != ATTR_OK:
                return Verdict(
                    hId=self.h_id, scope=self.scope, targets=targets, outcome=NOT_EVALUABLE,
                    evidence={**evidence, "blocked": "attributeGap", "nodeId": node_id},
                    reason=f"diameter is not available on chain member {node_id}",
                )
            diameters.append(context.attribute(holes[node_id], "diameter"))

        evidence["traversal"] = order
        values = {"diameters": diameters, "entryNodeId": entries[0]}
        rising = [
            f"{order[index]}->{order[index + 1]}"
            for index in range(len(diameters) - 1)
            if diameters[index + 1] > diameters[index]
        ]
        if rising:
            return Verdict(
                hId=self.h_id, scope=self.scope, targets=targets, outcome=FAIL,
                values={**values, "increasingSteps": rising}, evidence=evidence,
                reason="the diameter increases away from the entry at " + ", ".join(rising),
            )
        return Verdict(
            hId=self.h_id, scope=self.scope, targets=targets, outcome=PASS,
            values=values, evidence=evidence,
            reason="the diameter is non-increasing along the chain from its single entry",
        )


def _components(links: dict[str, set[str]]) -> list[set[str]]:
    seen: set[str] = set()
    components = []
    for node_id in links:
        if node_id in seen:
            continue
        stack, component = [node_id], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(links[current] - component)
        seen |= component
        components.append(component)
    return components


def _walk(entry: str, links: dict[str, set[str]], component: set[str]) -> list[str] | None:
    """Linear order of a simple chain from *entry*, or ``None`` if it branches."""
    if any(len(links[node_id] & component) > 2 for node_id in component):
        return None
    order, previous, current = [entry], None, entry
    while True:
        following = [node for node in links[current] & component if node != previous]
        if not following:
            break
        if len(following) > 1:
            return None
        previous, current = current, following[0]
        order.append(current)
    return order if len(order) == len(component) else None
