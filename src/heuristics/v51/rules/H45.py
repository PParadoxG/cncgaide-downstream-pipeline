"""H45 - tool length-to-diameter capability (hard, needs the magazine)."""

from __future__ import annotations

from heuristics.v51._shared import compare_at_most, external_missing
from heuristics.base import FeatureRule
from heuristics.v51.registry import register


@register
class H45(FeatureRule):
    """Same attribute as H01, different question.

    H01 is a design guideline with a generic limit; this one checks the ratio
    against what the tools of a specific magazine can actually reach.  The two
    are not duplicates and must not be merged.
    """

    h_id = "H45"
    USES_ATTRIBUTES = ("derived_ratio_depth_to_diameter",)
    # One spelling for both dialects.  nx H45 has always used this name, and
    # v5.1's own condition line in heuristics.yaml already reads
    # ``tool_magazine(machine).max_length_to_diameter`` - only the page's
    # rendering string and this constant said ``max_LD``.
    EXTERNAL_INPUT = "max_length_to_diameter"

    def evaluate_feature(self, node, context):
        limit = context.external_inputs.get(self.EXTERNAL_INPUT)
        if limit is None:
            return external_missing(self.EXTERNAL_INPUT)
        return compare_at_most(
            context.attribute(node, "derived_ratio_depth_to_diameter"),
            limit,
            "derived_ratio_depth_to_diameter",
            f"EXT.{self.EXTERNAL_INPUT}",
        )
