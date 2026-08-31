"""H50 (nx_v1) - feature depth within the usable tool length."""

from __future__ import annotations

from heuristics.base import FeatureRule
from heuristics.nx._shared import compare_at_most, external_missing
from heuristics.nx.registry import register


@register
class H50(FeatureRule):
    """Only the depth half of the frozen rule.

    v5.1 reads a step's maximum face extent as well, because a Step has no
    depth; NX has no Step class, so that half of the condition has no domain
    here and is left out rather than approximated.
    """

    h_id = "H50"
    USES_ATTRIBUTES = ("DEPTH",)
    EXTERNAL_INPUT = "max_usable_length"

    def evaluate_feature(self, node, context):
        limit = context.external_inputs.get(self.EXTERNAL_INPUT)
        if limit is None:
            return external_missing(self.EXTERNAL_INPUT)
        return compare_at_most(
            context.attribute(node, "DEPTH"), limit, "DEPTH",
            f"EXT.{self.EXTERNAL_INPUT}",
        )
