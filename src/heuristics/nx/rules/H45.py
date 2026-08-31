"""H45 (nx_v1) - section L/D against what the magazine can reach."""

from __future__ import annotations

from heuristics.base import FeatureRule, failed, not_evaluable, passed
from heuristics.nx._shared import (
    external_missing,
    section_depth_names,
    section_diameter_names,
    sections,
)
from heuristics.nx.registry import register


@register
class H45(FeatureRule):
    """Same quantity as H01, different question.

    H01 is a design guideline with a generic limit; this one asks whether the
    tools of a specific magazine can actually reach.  Per section, as in H01,
    and on the blind half of the STEP family - a through ladder is drilled
    from both sides in the general case, which is a different reach argument.
    """

    h_id = "H45"
    # Only the name this rule's own prose carries.  Its note says "per section,
    # as in H01", and it is H01's note that spells out how a section depth is
    # made out of the cumulative ``DEPTH_k``; the concrete names this gate
    # checks are below, where they come from the class in hand.
    USES_ATTRIBUTES = ("DIAMETER_k",)
    EXTERNAL_INPUT = "max_length_to_diameter"

    def required_attributes(self, feature_type):
        return section_diameter_names(feature_type) + section_depth_names(feature_type)

    def evaluate_feature(self, node, context):
        limit = context.external_inputs.get(self.EXTERNAL_INPUT)
        if limit is None:
            return external_missing(self.EXTERNAL_INPUT)
        table, reason = sections(context, node)
        if reason:
            return not_evaluable(f"the ladder could not be walked: {reason}",
                                 evidence={"blocked": "attributeGap"})
        rows = [{
            "index": row["index"],
            "sectionDepth": round(row["depth"], 6),
            "diameter": row["diameter"],
            "ratio": round(row["depth"] / row["diameter"], 6) if row["diameter"] else None,
        } for row in table]
        usable = [row for row in rows if row["ratio"] is not None]
        if not usable:
            return not_evaluable("every section diameter is zero, so no ratio exists",
                                 {"sections": rows}, {"blocked": "attributeGap"})
        worst = max(usable, key=lambda row: row["ratio"])
        values = {"sections": rows, "sectionCount": len(rows),
                  "maxRatio": worst["ratio"], "worstSection": worst["index"]}
        if worst["ratio"] > limit:
            return failed(
                f"section {worst['index']} needs L/D {worst['ratio']} > "
                f"EXT.{self.EXTERNAL_INPUT} {limit}", values)
        return passed(
            f"every section is within EXT.{self.EXTERNAL_INPUT} {limit} "
            f"(worst {worst['ratio']})", values)
