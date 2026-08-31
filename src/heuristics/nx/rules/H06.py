"""H06 (nx_v1) - minimum round-boss diameter."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_least
from heuristics.nx.registry import register


@register
class H06(FeatureRule):
    """NX publishes one DIAMETER on a round boss.

    The English statement, copied verbatim from v5.1, names a base/top pair;
    NX has no such pair, so statement and condition read differently here.
    Node 19a keeps the statement frozen and lets ``condition`` carry the NX
    reading, which is what this implements.
    """

    h_id = "H06"
    USES_ATTRIBUTES = ("DIAMETER",)
    USES_THRESHOLDS = ("t_boss_diameter",)

    def evaluate_feature(self, node, context):
        return compare_at_least(
            context.attribute(node, "DIAMETER"), context.threshold("t_boss_diameter"),
            "DIAMETER", "t_boss_diameter",
        )
