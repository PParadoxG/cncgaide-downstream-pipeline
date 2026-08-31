"""H01 (nx_v1) - hole depth-to-diameter ratio, per ladder section."""

from __future__ import annotations

from heuristics.base import FeatureRule, failed, not_evaluable, passed
from heuristics.nx._shared import section_depth_names, section_diameter_names, sections
from heuristics.nx.registry import register


@register
class H01(FeatureRule):
    """Every section is its own drilling job, so every section is checked.

    NX publishes ``DIAMETER_1..n`` and a cumulative ``DEPTH_1..``, so the depth
    of section k is arithmetic on what NX said and needs no re-cut of the
    solid (node 19a fact 1).  The condition is a ``forall``, so one feature is
    one unit: the verdict is the worst section and every section's numbers are
    in ``values``, which is what a reader - or a survey that only wants the
    quantities - needs to see which one it was.
    """

    h_id = "H01"
    # As the ontology writes them: ``DIAMETER_k`` in the condition, ``DEPTH_k``
    # and ``DEPTH`` in the note that defines ``sectionDepth``.  What the gate
    # checks are the concrete names of the class in hand, below.
    USES_ATTRIBUTES = ("DIAMETER_k", "DEPTH_k", "DEPTH")
    USES_THRESHOLDS = ("t_hole_ld",)

    def required_attributes(self, feature_type):
        return section_diameter_names(feature_type) + section_depth_names(feature_type)

    def evaluate_feature(self, node, context):
        limit = context.threshold("t_hole_ld")
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
                f"section {worst['index']} depth-to-diameter {worst['ratio']} > t_hole_ld {limit}",
                values)
        return passed(
            f"every section is at or below t_hole_ld {limit} (worst {worst['ratio']})", values)
