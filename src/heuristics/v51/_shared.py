"""The v5.1 dialect's own vocabulary, plus the machinery its rules import.

Everything here names v5.1 leaf classes.  The operators, the evaluation
context and the four-value gate are machinery and live in
``heuristics._shared``; a v5.1 rule imports both through this module, so the
import line in a rule file says which ontology it is written against.
"""

from __future__ import annotations

from heuristics._shared import (  # noqa: F401
    ATTR_ABSENT,
    ATTR_GAP,
    ATTR_OK,
    FAIL,
    NOT_EVALUABLE,
    PASS,
    HeuristicContext,
    acute_angle_deg,
    aggregate_count_limit,
    compare_at_least,
    compare_at_most,
    distinct,
    distinct_axes,
    distinct_tuples,
    dot,
    external_missing,
    failed,
    member_of,
    not_evaluable,
    passed,
    relation_missing,
    unit,
)

HOLE_TYPES = frozenset({"ThroughHole", "BlindHole", "Counterbore", "Countersink", "StepBore"})
SLOT_TYPES = frozenset({"RectangularSlot", "TriangularSlot", "CircularSlot", "GeneralSlot"})
POCKET_TYPES = frozenset({"RectangularPocket", "GeneralPocket"})
BOSS_TYPES = frozenset({"CircularBoss", "RectangularBoss", "GeneralBoss"})

# How many LOCATED_ON terminals a hole must have for its openings to be fully
# resolved.  A through hole enters and leaves the material, every other hole
# leaf has one opening.  Used by H27 to tell "the terminal says the entry is
# not planar" (FAIL) from "the terminal was never proven" (NOT_EVALUABLE).
# A StepBore has one: its major end, which opens either to the outside or onto
# the shoulder of the preceding section; its minor end is its own shoulder.
EXPECTED_TERMINALS = {"ThroughHole": 2, "BlindHole": 1, "Counterbore": 1, "Countersink": 1,
                      "StepBore": 1}

__all__ = [
    "ATTR_ABSENT",
    "ATTR_GAP",
    "ATTR_OK",
    "BOSS_TYPES",
    "EXPECTED_TERMINALS",
    "FAIL",
    "HOLE_TYPES",
    "HeuristicContext",
    "NOT_EVALUABLE",
    "PASS",
    "POCKET_TYPES",
    "SLOT_TYPES",
    "acute_angle_deg",
    "aggregate_count_limit",
    "compare_at_least",
    "compare_at_most",
    "distinct",
    "distinct_axes",
    "distinct_tuples",
    "dot",
    "external_missing",
    "failed",
    "member_of",
    "not_evaluable",
    "passed",
    "relation_missing",
    "unit",
]
