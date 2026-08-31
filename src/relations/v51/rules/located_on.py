"""LOCATED_ON — semantic terminal to effective host evidence."""

from __future__ import annotations

from core.models import FeatureEdge
from relations.base import RelationContext, RelationRule
from relations.v51.registry import register_rule
from relations.v51._shared import (
    LOCATED_SOURCE_TYPES,
    _feature_edge,
    _report,
    located_terminals,
)


@register_rule
class LocatedOn(RelationRule):
    rule_id = "located_on_v1"
    edge_type = "LOCATED_ON"
    priority = 20
    enabled_by_default = True
    directed = False
    description = (
        "Undirected graph edge whose evidence records a semantic source "
        "terminal resolving to one effective host face and owner."
    )

    def build_edges(self, context: RelationContext) -> list[FeatureEdge]:
        edges: list[FeatureEdge] = []
        for node in context.nodes:
            if node.featureType not in LOCATED_SOURCE_TYPES:
                continue
            terminals = located_terminals(context, node)
            if not terminals:
                _report(context, self.rule_id, "terminalProofUnavailable")
                continue
            for witness in terminals:
                host_id = witness.get("hostOwnerNodeId")
                if not host_id or host_id == node.nodeId or host_id not in context.node_by_id:
                    _report(context, self.rule_id, "invalidHostOwner")
                    continue
                edges.append(
                    _feature_edge(
                        node,
                        context.node_by_id[host_id],
                        self.edge_type,
                        self.rule_id,
                        {
                            "relationBasis": "semanticTerminalEffectiveHost",
                            "semanticDirection": "sourceFeatureTerminalToEffectiveHost",
                            "locatedOnWitnesses": [witness],
                            "locatedOnWitnessCount": 1,
                        },
                        {
                            "terminalRole": witness.get("terminalRole"),
                            "hostResolution": "single_effective_host",
                        },
                        # The frozen contract names terminalRole / hostFaceId /
                        # hostSurfaceType / hostSurfaceNormal / hostOwner as
                        # relation-valued properties.  One graph edge can carry
                        # more than one terminal of the same feature pair, so
                        # they are published as one list of aligned records
                        # rather than as five parallel arrays that a duplicate
                        # merge could knock out of step.  An ABox projection
                        # that wants one relation instance per terminal reads
                        # this list directly.
                        properties={
                            "terminals": [{
                                "terminalRole": witness.get("terminalRole"),
                                "hostFaceId": witness.get("hostFaceId"),
                                "hostFaceIds": list(witness.get("hostFaceIds") or []),
                                "hostSurfaceType": witness.get("hostSurfaceType"),
                                "hostSurfaceNormal": witness.get("hostSurfaceNormal"),
                                "hostOwnerNodeId": witness.get("hostOwnerNodeId"),
                            }],
                            "terminalCount": 1,
                        },
                    )
                )
        return edges
