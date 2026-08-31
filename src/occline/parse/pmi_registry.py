"""PMI entity registry for STEP AP242 text parsing.

PMI entities vary by authoring system and AP242 implementation level.
This registry maps entity type names to annotation categories.  Entities
not listed here that match the ``_TOLERANCE`` suffix pattern are dispatched
as tolerances automatically.  Entities that look like PMI but match neither
the registry nor the suffix pattern are logged as "unknown_pmi" so they can
be added to the registry incrementally — no information is silently dropped.
"""

# ---------------------------------------------------------------------------
# Exact-match registry — entity name → annotation metadata
# ---------------------------------------------------------------------------
PMI_ENTITY_REGISTRY: dict[str, dict] = {
    # ---- Dimensions ------------------------------------------------
    "DIMENSIONAL_SIZE":     {"type": "dimension"},
    "DIMENSIONAL_LOCATION": {"type": "dimension"},
    # ---- Surface condition / texture --------------------------------
    "SURFACE_CONDITION":                  {"type": "surface_finish"},
    "SURFACE_TEXTURE_REPRESENTATION":     {"type": "surface_finish"},
}

# ---------------------------------------------------------------------------
# Suffix-based dispatch
# ---------------------------------------------------------------------------
# Any entity whose name ends with this suffix is classified as a tolerance
# without being individually enumerated.  Validated against:
#   CYLINDRICITY_TOLERANCE  (SolidWorks 2022)
#   PLUS_MINUS_TOLERANCE    (SolidWorks 2022)
PMI_TOLERANCE_SUFFIX = "_TOLERANCE"

# ---------------------------------------------------------------------------
# Blocklist — entity names that contain PMI-like substrings but are NOT
# annotation entities (geometry, topology, reference data, etc.)
# ---------------------------------------------------------------------------
_PMI_BLOCKLIST = frozenset({
    # Geometric — never PMI
    "CARTESIAN_POINT", "DIRECTION", "VECTOR", "AXIS2_PLACEMENT_3D",
    "CIRCLE", "ELLIPSE", "LINE", "B_SPLINE_CURVE", "B_SPLINE_CURVE_WITH_KNOTS",
    "PLANE", "CYLINDRICAL_SURFACE", "SPHERICAL_SURFACE", "TOROIDAL_SURFACE",
    "COMPLEX_TRIANGULATED_SURFACE_SET",
    "CONICAL_SURFACE", "B_SPLINE_SURFACE", "B_SPLINE_SURFACE_WITH_KNOTS",
    "SEAM_CURVE", "SURFACE_CURVE", "INTERSECTION_CURVE",
    "EDGE_CURVE", "FACE_OUTER_BOUND", "FACE_BOUND", "ORIENTED_EDGE",
    "EDGE_LOOP", "FACE_SURFACE", "ADVANCED_FACE", "MANIFOLD_SOLID_BREP",
    "CLOSED_SHELL", "OPEN_SHELL", "VERTEX_POINT",
    "SHAPE_REPRESENTATION", "SHAPE_REPRESENTATION_RELATIONSHIP",
    "SHAPE_DEFINITION_REPRESENTATION", "PRODUCT_DEFINITION_SHAPE",
    # Supporting / reference — never direct PMI
    "TOLERANCE_VALUE", "LENGTH_MEASURE", "LENGTH_MEASURE_WITH_UNIT",
    "MEASURE_REPRESENTATION_ITEM", "MEASURE_WITH_UNIT",
    "DIMENSIONAL_EXPONENTS", "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION",
    "SHAPE_ASPECT", "SHAPE_ASPECT_RELATIONSHIP",
    "PROPERTY_DEFINITION", "PROPERTY_DEFINITION_REPRESENTATION",
    "UNCERTAINTY_MEASURE_WITH_UNIT", "QUALIFIED_REPRESENTATION_ITEM",
    "REPRESENTATION", "REPRESENTATION_RELATIONSHIP",
    "REPRESENTATION_ITEM", "REPRESENTATION_CONTEXT",
    "APPLICATION_PROTOCOL_DEFINITION", "APPLICATION_CONTEXT",
    "PRODUCT", "PRODUCT_DEFINITION", "PRODUCT_DEFINITION_FORMATION",
    "PRODUCT_RELATED_PRODUCT_CATEGORY", "PRODUCT_CONTEXT",
    "MECHANICAL_CONTEXT", "PRODUCT_DEFINITION_CONTEXT",
    "DESIGN_CONTEXT", "SECURITY_CLASSIFICATION",
    "APPROVAL", "DATE_TIME_ROLE", "PERSON_AND_ORGANIZATION",
    "CC_DESIGN_PERSON_AND_ORGANIZATION_ASSIGNMENT",
    # Generic / utility
    "STYLED_ITEM", "PRESENTATION_STYLE_ASSIGNMENT",
    "SURFACE_STYLE_USAGE", "CURVE_STYLE", "COLOUR_RGB",
    "DRAUGHTING_PRE_DEFINED_COLOUR", "FILL_AREA_STYLE_COLOUR",
    "CONTEXT_DEPENDENT_SHAPE_REPRESENTATION",
    "SHAPE_DIMENSION_REPRESENTATION",
    # Not PMI per se
    "APPLIED_ORGANIZATION_ASSIGNMENT", "ORGANIZATION", "ORGANIZATION_ROLE",
    "PERSON", "PERSON_AND_ORGANIZATION_ROLE",
    "APPROVAL_PERSON_ORGANIZATION", "APPROVAL_ROLE",
    "APPROVAL_STATUS", "APPROVAL_DATE_TIME",
    "COORDINATED_UNIVERSAL_TIME_OFFSET", "CALENDAR_DATE",
    "LOCAL_TIME", "DATE_AND_TIME",
})


def is_registered(entity_name: str) -> bool:
    """True if *entity_name* is an exact match in the registry."""
    return entity_name in PMI_ENTITY_REGISTRY


def matches_tolerance_suffix(entity_name: str) -> bool:
    """True if *entity_name* ends with ``PMI_TOLERANCE_SUFFIX``."""
    return entity_name.endswith(PMI_TOLERANCE_SUFFIX)


def looks_like_pmi(entity_name: str) -> bool:
    """Heuristic: could *entity_name* be a PMI annotation entity?

    Returns True if the name is NOT in the blocklist and contains at least
    one PMI-signalling substring.  This is intentionally broad — false
    positives are logged and can be added to the blocklist.
    """
    if entity_name in _PMI_BLOCKLIST:
        return False

    # PMI-signalling substrings (case-insensitive)
    signals = [
        "TOLERANCE", "DIMENSIONAL", "SURFACE", "DATUM",
        "FLATNESS", "CYLINDRICITY", "CIRCULARITY", "STRAIGHTNESS",
        "PERPENDICULARITY", "PARALLELISM", "ANGULARITY",
        "POSITION", "CONCENTRICITY", "SYMMETRY", "PROFILE",
        "RUNOUT", "TOTAL_RUNOUT", "ROUNDNESS",
    ]
    upper = entity_name.upper()
    return any(s in upper for s in signals)

