"""H49 (nx_v1) - how many distinct hole diameters a part needs."""

from __future__ import annotations

from heuristics.base import PartRule
from heuristics.nx._shared import (
    aggregate_count_limit,
    distinct,
    section_diameter_names,
    sections,
)
from heuristics.nx.registry import register


@register
class H49(PartRule):
    """Every section needs its own cutter, so the count runs over all of them.

    A six-section ladder therefore contributes six diameters by itself - which
    is what it costs.  Node 19a left the threshold at five and recorded the
    consequence rather than moving a calibration to make a number look better.
    """

    h_id = "H49"
    USES_ATTRIBUTES = ("DIAMETER_1", "DIAMETER_n")
    USES_THRESHOLDS = ("t_diameter_variety", "tol_diameter")

    def required_attributes(self, feature_type):
        return section_diameter_names(feature_type)

    def evaluate_part(self, context):
        nodes = context.domain_nodes(self.h_id)
        if not nodes:
            return None
        values, gaps, members = [], 0, []
        for node in nodes:
            members.append(node.nodeId)
            table, reason = sections(context, node)
            if reason:
                gaps += 1
                continue
            values.extend(row["diameter"] for row in table)
        clusters = distinct(values, context.threshold("tol_diameter"))
        return aggregate_count_limit(
            len(clusters), context.threshold("t_diameter_variety"),
            gap_count=gaps,
            values={"instanceCount": len(nodes), "sectionCount": len(values)},
            evidence={"memberNodeIds": members,
                      "clusters": [[round(value, 6) for value in cluster]
                                   for cluster in clusters]},
            quantity="distinct section diameters in the part",
        )
