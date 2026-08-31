"""LOCATED_ON (nx_v1) — a feature's opening, resolved outward to one host."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.nx._shared import (
    _report,
    is_located_source,
    located_terminals,
    nx_edge,
)
from relations.nx.registry import register_rule


@register_rule
class NxLocatedOn(RelationRule):
    """The two fields the ontology declares travel as properties; the rest is evidence.

    ``ontology/nx_v1/source/relations.yaml`` declares a ``LocatedOnTerminal``
    with exactly ``hostSurfaceType`` and ``hostSurfaceNormal`` — the two a
    rule can ask about today.  Which feature the terminal belongs to and which
    node owns the host face are identity rather than quantities, so they
    travel in the evidence beside the terminal, aligned by index, and the
    graph's own endpoints say the same thing.
    """

    rule_id = "located_on_nx_v1"
    edge_type = "LOCATED_ON"
    priority = 20
    enabled_by_default = True
    directed = False
    description = (
        "Undirected graph edge whose evidence records one opening of an NX "
        "feature resolving to a single effective host face and its owner."
    )

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        for node in context.nodes:
            if not is_located_source(node):
                continue
            witnesses = located_terminals(context, node)
            if not witnesses:
                _report(context, self.rule_id, "terminalProofUnavailable")
                continue
            for witness in witnesses:
                host_id = witness.get("hostOwnerNodeId")
                if not host_id or host_id == node.nodeId or host_id not in context.node_by_id:
                    _report(context, self.rule_id, "invalidHostOwner")
                    continue
                edges.append(nx_edge(
                    node, context.node_by_id[host_id], self.edge_type, self.rule_id,
                    "nx_topology",
                    {
                        "relationBasis": "openWallRimToEffectiveHost",
                        "nxBasis": "PartModel rims and face ownership; NX states no relation",
                        "semanticDirection": "sourceFeatureTerminalToEffectiveHost",
                        "locatedOnWitnesses": [witness],
                        "locatedOnWitnessCount": 1,
                    },
                    properties={
                        "terminals": [{
                            "hostSurfaceType": witness.get("hostSurfaceType"),
                            "hostSurfaceNormal": witness.get("hostSurfaceNormal"),
                        }],
                        "terminalCount": 1,
                    },
                ))
        return edges
