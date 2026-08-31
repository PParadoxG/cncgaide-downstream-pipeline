"""The nx_v1 representation tables, derived from the NX ontology.

Nothing here is a list somebody wrote down.  ``ontology/nx_v1/source`` holds
NX's own class tree and its own attribute declarations (transcribed from the
installed ``customization_base.xml`` and checked against it by the ontology's
S8), and every table below is a function of that file plus the four rules
node 19' §3.4 froze:

* an attribute enters the geometry term when NX gives it a ``unit`` of
  ``Length`` / ``Angle`` / ``Area``;
* ``*_UPPER`` / ``*_LOWER`` are the two halves of a tolerance band, not a
  second and third dimension, and are excluded;
* ``STOCK_*`` is machining stock, a process allowance rather than a shape;
* positions are excluded, the same decision that keeps ``posX/Y/Z`` out of the
  v5.1 table - two identical pockets in different corners are the same pocket.

Two attributes are boolean in meaning while NX types them as a length or a
string (``TOP_OBSTRUCTED``, ``IS_THROUGH``).  Node 19' §3.4's third dimension
decision makes them categories: the difference between obstructed and not is
not a distance in millimetres.

The one number that is ours and not NX's is the angle span: NX declares no
value range for an angle, so every angle difference is normalised by 180
degrees (node 19' §3.2).  It is named here rather than hidden in a formula.

The **family** of a class is its nearest abstract ancestor.  That is a
reading of NX's tree, not a grouping we invented, and it reproduces node 19'
§3.2's twelve families on the reachable classes - with one honest deviation:
``TURNING_GROOVE`` is a *concrete* class in NX's file, so its three children
group under ``TurningFeature``.  The label is metadata; no cost reads it (the
type cost is Wu-Palmer over the whole tree, which needs no family at all).
"""

from __future__ import annotations

from tbox import nx_v1 as tbox


# The unit vocabulary NX itself uses for the quantities that can enter a
# geometric distance, mapped to the kinds ``GedAttribute`` knows.
UNIT_KIND: dict[str, str] = {"Length": "length", "Angle": "angle", "Area": "area"}

# Excluded by name, with the reason each exclusion encodes.
TOLERANCE_BAND_SUFFIXES: tuple[str, ...] = ("_UPPER", "_LOWER")
STOCK_PREFIX = "STOCK_"
POSITION_NAMES: frozenset[str] = frozenset((
    "X_POSITION", "Y_POSITION", "Z_POSITION",
    "START_HOLE_POS_X", "START_HOLE_POS_Y",
))

# Boolean in meaning, numeric or textual in NX's type system (19' §3.4).
BOOLEAN_AS_CATEGORY: tuple[str, ...] = ("TOP_OBSTRUCTED", "IS_THROUGH")

# NX declares no value range for an angle; this one is ours (19' §3.2).
NX_ANGLE_RANGE_DEG = 180.0

# The background layer: NX's own ``SURFACES`` family, by the family and not
# by a list of class names (19' D6).
SURFACE_FAMILY = "SURFACES"

# How the ``W_RES`` budget splits across the aggregates.  v5.1 weights its two
# surface aggregates unequally because a planar background and a freeform one
# are different amounts of part; NX's surface leaves are all planar, so the
# split is even and derived from how many there are.  The residue keeps
# v5.1's share so the two representations spend the same budget on "what the
# assignment did not look at".
SURFACE_BUDGET = 0.7
UNCLAIMED_BUDGET = 0.3


def _is_geometric(name: str) -> bool:
    record = tbox.attribute_record(name)
    if record is None or record.get("unit") not in UNIT_KIND:
        return False
    if name in POSITION_NAMES or name.startswith(STOCK_PREFIX):
        return False
    return not name.endswith(TOLERANCE_BAND_SUFFIXES)


def attribute_kind(name: str) -> str | None:
    """``length`` / ``angle`` / ``area`` / ``category``, or None if excluded."""
    if name in BOOLEAN_AS_CATEGORY:
        return "category"
    if not _is_geometric(name):
        return None
    record = tbox.attribute_record(name)
    return UNIT_KIND[record["unit"]]


def build() -> dict:
    """Every nx_v1 representation table, derived in one pass over the ontology."""
    classes = tuple(sorted(tbox.reachable_classes()))
    surface_leaves = tuple(c for c in classes if tbox.in_family(c, SURFACE_FAMILY))
    leaf_attributes: dict[str, tuple[str, ...]] = {}
    kinds: dict[str, str] = {}
    for leaf in classes:
        kept = []
        for name in tbox.effective_attributes(leaf):
            kind = attribute_kind(name)
            if kind is None:
                continue
            kept.append(name)
            kinds[name] = kind
        # A surface leaf is aggregated, so its own attributes are never
        # compared one by one - the same treatment v5.1 gives PlanarFace.
        leaf_attributes[leaf] = () if leaf in surface_leaves else tuple(sorted(kept))
    return {
        "classes": classes,
        "surface_leaves": surface_leaves,
        "leaf_attributes": leaf_attributes,
        "attribute_kind": kinds,
        "angle_range_deg": {name: NX_ANGLE_RANGE_DEG
                            for name, kind in kinds.items() if kind == "angle"},
        "family_of": {leaf: family_of(leaf) for leaf in classes},
    }


def family_of(name: str) -> str:
    """The class's nearest abstract ancestor, NX's own name for its family."""
    for ancestor in tbox.ancestry(name)[1:]:
        record = tbox.class_record(ancestor)
        if record is not None and record.get("abstract"):
            return ancestor
    return "Unknown"


def surface_budget_shares(surface_leaves) -> dict[str, float]:
    """Fixed-node budget shares, keyed by aggregate type; they sum to 1."""
    leaves = tuple(surface_leaves)
    shares = {f"{leaf}Aggregate": SURFACE_BUDGET / len(leaves) for leaf in leaves} if leaves else {}
    shares["UnclaimedAggregate"] = UNCLAIMED_BUDGET if leaves else 1.0
    return shares


def surface_attribute_weights(surface_leaves) -> dict[str, dict[str, float]]:
    """What an aggregate is compared on.  ``normal_classes`` is not among them:
    an NX node publishes NX's attributes, and a face normal is not one."""
    weights = {f"{leaf}Aggregate": {"face_count": 0.5, "area": 0.5}
               for leaf in surface_leaves}
    weights["UnclaimedAggregate"] = {"face_count": 0.5, "area": 0.5}
    return weights
