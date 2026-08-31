"""H42 - freeform faces only where the function needs them (informational)."""

from __future__ import annotations

from heuristics.v51._shared import external_missing
from heuristics.base import FeatureRule, failed, passed
from heuristics.v51.registry import register


@register
class H42(FeatureRule):
    """Kept rather than retired, because what is missing is not a geometry.

    Whether a surface is functionally required is design intent; no amount of
    geometric work produces it, and the pipeline's input is STEP plus per-face
    evidence.  The geometry side owes exactly one thing here - enumerating the
    FreeformFace instances - and that already exists.
    """

    h_id = "H42"
    USES_RELATIONS = ()
    EXTERNAL_INPUT = "required_by_function"

    def required_attributes(self, feature_type):
        return ()

    def evaluate_feature(self, node, context):
        intent = context.external_inputs.get(self.EXTERNAL_INPUT)
        if not intent:
            return external_missing(self.EXTERNAL_INPUT)
        required = intent.get(node.nodeId) if isinstance(intent, dict) else None
        if required is None:
            return external_missing(self.EXTERNAL_INPUT)
        values = {"requiredByFunction": bool(required)}
        if required:
            return passed("the freeform face is recorded as functionally required", values)
        return failed("the freeform face is not recorded as functionally required", values)
