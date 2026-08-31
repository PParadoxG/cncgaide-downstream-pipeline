"""H12 (nx_v1) - the material web between two parallel holes."""

from __future__ import annotations

from heuristics.base import PairRule
from heuristics.nx._shared import compare_at_least, entry_diameter_name, relation_missing
from heuristics.nx.registry import register


@register
class H12(PairRule):
    """Remaining material, read at the mouths.

    ``DIAMETER_1`` is the entry section and, because a ladder runs
    monotonically inward, the widest one - so subtracting the two entry radii
    from the axis spacing is the thinnest reading of the pair, which is the
    conservative one for a web check.
    """

    h_id = "H12"
    edge_type = "HOLE_SPACING"
    USES_ATTRIBUTES = ("DIAMETER_1",)
    USES_RELATIONS = ("HOLE_SPACING",)
    USES_THRESHOLDS = ("t_hole_web",)

    def required_attributes(self, feature_type):
        name = entry_diameter_name(feature_type)
        return (name,) if name else ()

    def evaluate_pair(self, edge, source, target, context):
        spacing = (edge.properties or {}).get("axisSpacing")
        if spacing is None:
            return relation_missing("HOLE_SPACING edge carries no axisSpacing",
                                    {"edgeId": edge.edgeId})
        left = context.attribute(source, entry_diameter_name(source.featureType))
        right = context.attribute(target, entry_diameter_name(target.featureType))
        web = spacing - (left + right) / 2.0
        return compare_at_least(
            round(web, 6), context.threshold("t_hole_web"), "web", "t_hole_web",
            {"axisSpacing": spacing, "entryDiameterA": left, "entryDiameterB": right},
        )
