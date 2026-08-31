"""Which ontology attributes enter the GED geometry term, per leaf class.

Source of truth: the running ontology, ``ontology/v5.1/source/classes.yaml``
(leaf classes and their attribute sets; ``NODE_REPORT/catalogue/01`` is the
frozen v5.0 origin of the same table and no longer lists the classes v5.1
added), filtered by the decisions recorded in
``NODE_REPORT/GED_Design_Outline_v0.md`` §2.4:

- only attributes that describe the feature's *own* geometry are compared;
- ``posX/Y/Z``, ``feature_axis``, ``machining_direction``, ``surface_normal``,
  ``cadFaceIds`` / ``partmodelFaceIds`` and every ``derived_*`` attribute are
  excluded;
- controlled vocabularies (``hole_bottom_type``, ``pocket_bottom_type``)
  are compared as categories.

Changing this table is a representation-layer change: it must track the
ontology, never the other way round.  ``tests/test_ged_stepbore.py`` derives
both tables from ``classes.yaml`` and fails when they drift apart, so a leaf
class added to the ontology cannot stay unknown here unnoticed.
"""

from __future__ import annotations


# --- class hierarchy (catalogue 1.2 / 1.9) --------------------------------

FAMILY_OF: dict[str, str] = {
    "ThroughHole": "Hole",
    "BlindHole": "Hole",
    "Counterbore": "Hole",
    "Countersink": "Hole",
    "StepBore": "Hole",
    "RectangularSlot": "Recess",
    "TriangularSlot": "Recess",
    "CircularSlot": "Recess",
    "GeneralSlot": "Recess",
    "RectangularPocket": "Recess",
    "GeneralPocket": "Recess",
    "ThroughCut": "Recess",
    "CircularBoss": "Boss",
    "RectangularBoss": "Boss",
    "GeneralBoss": "Boss",
    "Chamfer": "EdgeTreatment",
    "Fillet": "EdgeTreatment",
    "PlanarFace": "Surface",
    "FreeformFace": "Surface",
    "Step": "Step",
}

# Recess is the only family whose leaves split into sub-families with
# clearly different manufacturing meaning; the type matrix uses this.
SUBFAMILY_OF: dict[str, str] = {
    "RectangularSlot": "Slot",
    "TriangularSlot": "Slot",
    "CircularSlot": "Slot",
    "GeneralSlot": "Slot",
    "RectangularPocket": "Pocket",
    "GeneralPocket": "Pocket",
    "ThroughCut": "ThroughCut",
}

LEAF_TYPES: frozenset[str] = frozenset(FAMILY_OF)

# Surface leaves form the host background layer (design outline §2.2 C):
# they never enter the assignment and are aggregated into fixed nodes.
SURFACE_LEAVES: tuple[str, ...] = ("PlanarFace", "FreeformFace")

SEMANTIC_LEAVES: frozenset[str] = frozenset(
    leaf for leaf in LEAF_TYPES if leaf not in SURFACE_LEAVES
)


# --- attribute kinds --------------------------------------------------------

ATTRIBUTE_KIND: dict[str, str] = {
    # lengths (mm → / L_ref)
    "diameter": "length",
    "depth": "length",
    "counterbore_diameter": "length",
    "counterbore_depth": "length",
    "countersink_diameter": "length",
    "countersink_depth": "length",
    "course_length": "length",
    "min_opening_width": "length",
    "min_bottom_width": "length",
    "profile_radius": "length",
    "min_corner_radius": "length",
    "height": "length",
    "base_diameter": "length",
    "top_diameter": "length",
    "base_thickness": "length",
    "top_thickness": "length",
    "min_face_extent": "length",
    "max_face_extent": "length",
    "chamfer_width": "length",
    "fillet_radius": "length",
    # angles (degrees, compared against the attribute's range)
    "v_included_angle": "angle",
    "max_wall_draft_angle": "angle",
    "min_corner_angle": "angle",
    "step_included_angle": "angle",
    "min_chamfer_angle": "angle",
    # controlled vocabularies
    "hole_bottom_type": "category",
    "pocket_bottom_type": "category",
}

# Width of the value space an angle difference is normalised by.
ANGLE_RANGE_DEG: dict[str, float] = {
    "v_included_angle": 180.0,
    "max_wall_draft_angle": 90.0,
    "min_corner_angle": 180.0,
    "step_included_angle": 180.0,
    "min_chamfer_angle": 90.0,
}


# --- per-leaf attribute sets (catalogue chapter 2, filtered) ---------------

_HOLE_COMMON = ("diameter", "depth")
_SLOT_COMMON = ("depth", "course_length")

LEAF_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "ThroughHole": _HOLE_COMMON,
    "BlindHole": _HOLE_COMMON + ("hole_bottom_type",),
    "Counterbore": _HOLE_COMMON + ("counterbore_diameter", "counterbore_depth"),
    "Countersink": _HOLE_COMMON + ("countersink_diameter", "countersink_depth"),
    # A ladder's non-terminal section (D-7): no bottom, so no
    # ``hole_bottom_type``; what is left of the BlindHole set after the
    # derived attributes are filtered out is the plain cylinder pair.
    "StepBore": _HOLE_COMMON,
    "RectangularSlot": _SLOT_COMMON
    + ("min_opening_width", "min_bottom_width", "max_wall_draft_angle"),
    "TriangularSlot": _SLOT_COMMON + ("min_opening_width", "v_included_angle"),
    "CircularSlot": _SLOT_COMMON + ("min_opening_width", "profile_radius"),
    "GeneralSlot": _SLOT_COMMON,
    "RectangularPocket": (
        "depth",
        "min_opening_width",
        "min_bottom_width",
        "min_corner_radius",
        "min_corner_angle",
        "max_wall_draft_angle",
        "pocket_bottom_type",
    ),
    "GeneralPocket": (
        "depth",
        "min_corner_radius",
        "min_corner_angle",
        "max_wall_draft_angle",
        "pocket_bottom_type",
    ),
    "ThroughCut": ("min_corner_radius", "min_corner_angle", "max_wall_draft_angle"),
    "CircularBoss": ("height", "base_diameter", "top_diameter"),
    "RectangularBoss": ("height", "base_thickness", "top_thickness", "max_wall_draft_angle"),
    "GeneralBoss": ("height", "max_wall_draft_angle"),
    "Chamfer": ("chamfer_width", "min_chamfer_angle"),
    "Fillet": ("fillet_radius",),
    "Step": ("min_face_extent", "max_face_extent", "step_included_angle"),
    # Surface leaves are aggregated; their per-node attributes are not
    # compared individually.
    "PlanarFace": (),
    "FreeformFace": (),
}


def attribute_kind(name: str) -> str:
    try:
        return ATTRIBUTE_KIND[name]
    except KeyError as exc:  # pragma: no cover - guarded by tests on the table
        raise KeyError(f"attribute {name!r} has no registered kind") from exc


def _check_table() -> None:
    for leaf, names in LEAF_ATTRIBUTES.items():
        if leaf not in FAMILY_OF:
            raise AssertionError(f"{leaf} is not a frozen leaf class")
        for name in names:
            if name not in ATTRIBUTE_KIND:
                raise AssertionError(f"{leaf}.{name} has no attribute kind")
            if ATTRIBUTE_KIND[name] == "angle" and name not in ANGLE_RANGE_DEG:
                raise AssertionError(f"{name} is an angle without a range")
    missing = LEAF_TYPES - set(LEAF_ATTRIBUTES)
    if missing:
        raise AssertionError(f"leaf classes without an attribute policy: {sorted(missing)}")


_check_table()
