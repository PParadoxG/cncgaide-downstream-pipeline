"""H27 (nx_v1) - a hole enters through a plane normal to its axis."""

from __future__ import annotations

from heuristics.base import FAIL, NOT_EVALUABLE, PASS, HeuristicRule, Verdict
from heuristics.nx._shared import (
    ATTR_OK,
    ORIENTATION_D,
    acute_angle_deg,
    expected_terminals,
    unit,
)
from heuristics.nx.registry import register


@register
class H27(HeuristicRule):
    """One unit per terminal, the terminals that were never proven included.

    A non-planar host is a FAIL and not an unmeasurable case - the statement
    asks for a plane, and a cylindrical entry violates it outright.  The
    unmeasurable case is a terminal the relation layer refused to prove, and
    how many a feature owes is what NX's own name for its class says: a
    ``STEP2HOLE`` goes through and opens twice, a ``STEP2POCKET`` is blind and
    opens once (node 19a D2).
    """

    h_id = "H27"
    scope = "terminal"
    USES_ATTRIBUTES = ORIENTATION_D
    USES_RELATIONS = ("LOCATED_ON",)
    USES_THRESHOLDS = ("t_normal_entry",)

    def evaluate(self, context):
        limit = context.threshold("t_normal_entry")
        verdicts = []
        for node in context.domain_nodes(self.h_id):
            state, name, reason = context.gate_attributes(node, ORIENTATION_D)
            axis = context.axis(node) if state == ATTR_OK else None
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
                        hId=self.h_id, scope=self.scope, targets=targets,
                        outcome=NOT_EVALUABLE,
                        evidence={**evidence, "blocked": "attributeGap", "attribute": name},
                        reason=f"the feature axis is not usable: "
                               f"{reason or 'not a finite direction'}",
                    ))
                    continue
                if terminal["hostSurfaceType"] != "PLANE":
                    verdicts.append(Verdict(
                        hId=self.h_id, scope=self.scope, targets=targets, outcome=FAIL,
                        values={"hostSurfaceType": terminal["hostSurfaceType"]},
                        evidence=evidence,
                        reason=f"the hole enters on a {terminal['hostSurfaceType']} host, "
                               "not a plane",
                    ))
                    continue
                normal = unit(terminal["hostSurfaceNormal"])
                if normal is None:
                    verdicts.append(Verdict(
                        hId=self.h_id, scope=self.scope, targets=targets,
                        outcome=NOT_EVALUABLE,
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
                    reason=f"axis to host normal {angle} deg "
                           f"{'<=' if ok else '>'} t_normal_entry {limit}",
                ))
            missing = expected_terminals(node.featureType) - len(terminals)
            for _ in range(max(0, missing)):
                verdicts.append(Verdict(
                    hId=self.h_id, scope=self.scope, targets={"nodeIds": [node.nodeId]},
                    outcome=NOT_EVALUABLE,
                    evidence={"blocked": "relationMissing"},
                    reason=context.missing_terminal_reason(node),
                ))
        return verdicts
