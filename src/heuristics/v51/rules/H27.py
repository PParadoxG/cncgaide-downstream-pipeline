"""H27 - a hole enters through a plane whose normal is its axis (hard, terminal)."""

from __future__ import annotations

from heuristics.v51._shared import ATTR_OK, EXPECTED_TERMINALS, acute_angle_deg, unit
from heuristics.base import (
    FAIL,
    NOT_EVALUABLE,
    PASS,
    HeuristicRule,
    Verdict,
)
from heuristics.v51.registry import register


@register
class H27(HeuristicRule):
    """One unit per terminal, including the terminals that were never proven.

    The host normal is read from the LOCATED_ON evidence rather than from a
    PlanarFace node: a hole may open on a boss or a pocket floor, and those
    hosts are feature nodes without a surface_normal attribute.

    A non-planar host is a FAIL, not an unmeasurable case - the statement
    asks for a plane, and a cylindrical entry violates it outright.  A
    terminal the relation layer refused to prove is the unmeasurable case.
    """

    h_id = "H27"
    scope = "terminal"
    USES_ATTRIBUTES = ("feature_axis",)
    USES_RELATIONS = ("LOCATED_ON",)
    USES_THRESHOLDS = ("t_normal_entry",)

    def evaluate(self, context):
        limit = context.threshold("t_normal_entry")
        verdicts = []
        for node in context.domain_nodes(self.h_id):
            state, name, reason = context.gate_attributes(node, self.USES_ATTRIBUTES)
            axis = unit(context.attribute(node, "feature_axis")) if state == ATTR_OK else None
            terminals = context.terminals_of(node.nodeId)
            for terminal in terminals:
                targets = {"nodeIds": [node.nodeId], "edgeIds": [terminal["edgeId"]]}
                evidence = {
                    "terminalRole": terminal["terminalRole"],
                    "hostOwnerNodeId": terminal["hostOwnerNodeId"],
                    "hostFaceId": terminal["hostFaceId"],
                    "hostSurfaceType": terminal["hostSurfaceType"],
                }
                if axis is None:
                    verdicts.append(Verdict(
                        hId=self.h_id, scope=self.scope, targets=targets, outcome=NOT_EVALUABLE,
                        evidence={**evidence, "blocked": "attributeGap", "attribute": name},
                        reason=f"feature_axis is not usable: {reason or 'not a finite direction'}",
                    ))
                    continue
                normal = unit(terminal["hostSurfaceNormal"])
                if terminal["hostSurfaceType"] != "PLANE":
                    verdicts.append(Verdict(
                        hId=self.h_id, scope=self.scope, targets=targets, outcome=FAIL,
                        values={"hostSurfaceType": terminal["hostSurfaceType"]},
                        evidence=evidence,
                        reason=f"the hole enters on a {terminal['hostSurfaceType']} host, not a plane",
                    ))
                    continue
                if normal is None:
                    verdicts.append(Verdict(
                        hId=self.h_id, scope=self.scope, targets=targets, outcome=NOT_EVALUABLE,
                        evidence={**evidence, "blocked": "relationMissing"},
                        reason="the planar host terminal carries no usable surface normal",
                    ))
                    continue
                angle = round(acute_angle_deg(axis, normal), 6)
                ok = angle <= limit
                verdicts.append(Verdict(
                    hId=self.h_id, scope=self.scope, targets=targets,
                    outcome=PASS if ok else FAIL,
                    values={"axisToHostNormalAngle": angle},
                    evidence=evidence,
                    reason=(
                        f"axis to host normal {angle} deg "
                        f"{'<=' if ok else '>'} t_normal_entry {limit}"
                    ),
                ))
            missing = EXPECTED_TERMINALS.get(node.featureType, 1) - len(terminals)
            for _ in range(max(0, missing)):
                verdicts.append(Verdict(
                    hId=self.h_id, scope=self.scope, targets={"nodeIds": [node.nodeId]},
                    outcome=NOT_EVALUABLE,
                    evidence={"blocked": "relationMissing"},
                    reason=context.missing_terminal_reason(node),
                ))
        return verdicts
