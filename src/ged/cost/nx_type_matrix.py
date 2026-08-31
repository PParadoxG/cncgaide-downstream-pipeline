"""Leaf-class substitution cost for nx_v1: Wu-Palmer over NX's own class tree.

v5.1 carries an explicit matrix because its twenty leaves were arranged by
hand and the hierarchy above them is shallow.  NX arranges its own: a
``STEP2HOLE`` **is a** ``STEP1HOLE`` with one more section, ``SLOT_V_SHAPED``
sits under ``SLOTS``, and everything hangs off ``MachiningFeature``.  Node 19'
§1.4 measured what that tree gives:

    cost(a, b) = 1 - 2 * depth(LCA) / (depth(a) + depth(b))

over the 84 reachable classes that is 3486 unordered pairs taking **56
distinct values**, none of which anybody chose - a counterbore next to a
through hole lands at 0.091, six sections away at 0.333, a V-slot at 0.400,
a wire-cut pocket at 0.750.  Raw hop count gets that ordering wrong (it calls
``STEP1HOLE`` ↔ ``SLOT_V_SHAPED`` *nearer* than ``STEP1HOLE`` ↔
``STEP6HOLE``, because the STEP ladder is deep); the normalisation fixes it
without introducing a constant of ours.

``OVERRIDES`` stays, and stays **empty**.  Node 19' D8 put one candidate in
front of it - through vs blind costs more the longer the ladder, because the
LCA ``STEP`` is relatively shallower - and the user's decision was to accept
what the tree says rather than develop a formula with parameters we would
then own.  §4.4 of that report is the archived analysis of the case that was
declined; this is where an exception would go if one is ever proven.

The price of the whole scheme, also from 19' §3.1: Wu-Palmer reads the
*depth* of NX's tree, and depth is an artifact of how Siemens modelled it.
A different NX install can move every number.  That is why the two installed
files' sha256 travel in the nx_v1 semantics stamp.
"""

from __future__ import annotations

from functools import lru_cache

from tbox import nx_v1 as tbox


# Empty by decision (19' D8 = b), not by omission.
OVERRIDES: dict[frozenset[str], float] = {}

SAME_LEAF = 0.0
# What an unknown class costs against anything else.  A class the ontology
# does not know has no depth, so Wu-Palmer has nothing to say; the answer is
# "as different as this scheme can express", stated rather than crashed.
UNKNOWN = 1.0


@lru_cache(maxsize=None)
def _depth(name: str) -> int:
    """Number of classes from the root down to *name*, inclusive.  0 = unknown."""
    return len(tbox.ancestry(name))


@lru_cache(maxsize=None)
def _lca_depth(left: str, right: str) -> int:
    right_chain = tbox.ancestry(right)
    if not right_chain:
        return 0
    seen = set(right_chain)
    for ancestor in tbox.ancestry(left):
        if ancestor in seen:
            return _depth(ancestor)
    return 0


def type_cost(left: str, right: str) -> float:
    if left == right:
        return SAME_LEAF
    key = frozenset((left, right))
    if key in OVERRIDES:
        return OVERRIDES[key]
    left_depth, right_depth = _depth(left), _depth(right)
    if not left_depth or not right_depth:
        return UNKNOWN
    shared = _lca_depth(left, right)
    if not shared:
        return UNKNOWN
    return 1.0 - (2.0 * shared) / (left_depth + right_depth)


def type_matrix() -> dict[str, dict[str, float]]:
    """Full class x class matrix over the reachable classes, for reports and tests."""
    classes = sorted(tbox.reachable_classes())
    return {a: {b: type_cost(a, b) for b in classes} for a in classes}


def distinct_costs() -> tuple[float, ...]:
    """The sorted set of off-diagonal values the tree produces."""
    classes = sorted(tbox.reachable_classes())
    values = {round(type_cost(a, b), 12)
              for index, a in enumerate(classes) for b in classes[index + 1:]}
    return tuple(sorted(values))
