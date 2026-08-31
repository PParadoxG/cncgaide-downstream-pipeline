"""H36 - edge-treatment tool availability (advisory, needs the magazine)."""

from __future__ import annotations

from heuristics.v51._shared import external_missing
from heuristics.base import FeatureRule, failed, passed
from heuristics.v51.registry import register

# Which measurements identify the tool, per leaf class.  The two branches are
# different criteria and catalogue 03 chapter 0.10 recommends splitting them
# into two heuristics; that renumbering is the collaborator's decision, so
# both branches stay in this one rule for now.
TOOL_FIELDS = {
    "Chamfer": ("min_chamfer_angle", "chamfer_width"),
    "Fillet": ("fillet_radius",),
}


@register
class H36(FeatureRule):
    """Angle is matched, width is only bounded by the tool's usable length."""

    h_id = "H36"
    USES_ATTRIBUTES = ("min_chamfer_angle", "chamfer_width", "fillet_radius")
    USES_THRESHOLDS = ("tol_chamfer_angle", "tol_fillet_radius")
    EXTERNAL_INPUT = "edge_tools"

    def required_attributes(self, feature_type):
        return TOOL_FIELDS.get(feature_type, ())

    def evaluate_feature(self, node, context):
        tools = context.external_inputs.get(self.EXTERNAL_INPUT)
        if not tools:
            return external_missing(self.EXTERNAL_INPUT)
        if node.featureType == "Chamfer":
            return self._chamfer(node, context, tools)
        return self._fillet(node, context, tools)

    def _chamfer(self, node, context, tools):
        angle = context.attribute(node, "min_chamfer_angle")
        width = context.attribute(node, "chamfer_width")
        tolerance = context.threshold("tol_chamfer_angle")
        values = {"min_chamfer_angle": angle, "chamfer_width": width}
        for tool in tools:
            if tool.get("angle") is None:
                continue
            if abs(angle - tool["angle"]) > tolerance:
                continue
            if tool.get("max_width") is not None and width > tool["max_width"]:
                continue
            return passed(
                f"chamfer matches magazine tool {tool.get('id', tool)}",
                values, {"tool": tool},
            )
        return failed("no magazine edge tool matches this chamfer", values)

    def _fillet(self, node, context, tools):
        radius = context.attribute(node, "fillet_radius")
        tolerance = context.threshold("tol_fillet_radius")
        values = {"fillet_radius": radius}
        for tool in tools:
            if tool.get("radius") is None:
                continue
            if abs(radius - tool["radius"]) <= tolerance:
                return passed(
                    f"fillet matches magazine tool {tool.get('id', tool)}",
                    values, {"tool": tool},
                )
        return failed("no magazine edge tool matches this fillet radius", values)
