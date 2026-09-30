"""Which ontology a GED graph is written in, as one object the layers pass around.

The GED had exactly one representation until node 21': the v5.1 tables in
``attribute_policy``, reached by module-level import from the projection and
the cost layer.  A second ontology now publishes parts too, and its classes,
attributes and relations are not the first one's - so the tables become a
parameter instead of an import, and the two live side by side.

What a ``Representation`` fixes is *what the graph is*: which leaf classes
enter the assignment, which are the background layer, which attributes are
compared and in what unit, which relations survive and how their values are
normalised.  What it deliberately does **not** fix is the eight adjustable
weights - node 21' reuses one weight vocabulary and one preset table across
both, because a weight says how much a kind of difference matters and that
question is the same question on either side.

**Distances still do not cross.**  Two graphs in different representations
are never compared: the corpus is per line, the pair cache is per line, and
each representation carries its own semantics stamp (``ged.semantics``) so a
cached entry cannot be served to the other one even by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from contract.registry import LINE_NX, LINE_OCC

from . import attribute_policy, nx_policy


@dataclass(frozen=True)
class Representation:
    """The tables that say what a graph in one ontology looks like."""

    name: str
    family_of: Mapping[str, str]
    surface_leaves: tuple[str, ...]
    semantic_leaves: frozenset[str]
    leaf_attributes: Mapping[str, tuple[str, ...]]
    attribute_kind: Mapping[str, str]
    angle_range_deg: Mapping[str, float]
    # Which relation types the projection keeps, and how it reads them.
    known_relations: frozenset[str]
    grouped_relations: frozenset[str]
    host_implying_relations: frozenset[str]
    edge_metrics: Mapping[str, Mapping[str, str]]
    # Insert / delete base weight of the relations this ontology publishes and
    # does *not* clique-normalise.  A representation decision like the ones
    # above: which relations exist is the ontology's answer, and so is what
    # detaching one of them is worth.  A type absent here falls back to
    # ``cost.edge_cost.DEFAULT_BASE_WEIGHT``.
    static_edge_base_weight: Mapping[str, float] = field(default_factory=dict)
    # Whether a surface aggregate carries the pose-invariant normal count.
    surface_normal_classes: frozenset[str] = frozenset()
    # Fixed-node budget shares and attribute weights, by aggregate type key.
    fixed_budget_shares: Mapping[str, float] = field(default_factory=dict)
    fixed_attribute_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # Extra facts the semantics stamp must cover: for nx_v1 the sha256 of the
    # two installed NX files every table above is a function of (19' D14).
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def kind_of(self, name: str) -> str:
        try:
            return self.attribute_kind[name]
        except KeyError as exc:  # pragma: no cover - guarded by the table checks
            raise KeyError(f"attribute {name!r} has no registered kind") from exc

    def angle_span(self, name: str) -> float:
        return float(self.angle_range_deg.get(name, 180.0))


# ------------------------------------------------------------------- v5.1

V5_1 = Representation(
    name="v5_1",
    family_of=attribute_policy.FAMILY_OF,
    surface_leaves=attribute_policy.SURFACE_LEAVES,
    semantic_leaves=attribute_policy.SEMANTIC_LEAVES,
    leaf_attributes=attribute_policy.LEAF_ATTRIBUTES,
    attribute_kind=attribute_policy.ATTRIBUTE_KIND,
    angle_range_deg=attribute_policy.ANGLE_RANGE_DEG,
    known_relations=frozenset(
        ("DIRECT_ADJACENT", "LOCATED_ON", "COAXIAL", "INTERSECTS",
         "HOLE_SPACING", "POCKET_CLEARANCE", "SLOT_CLEARANCE")
    ),
    grouped_relations=frozenset(
        ("COAXIAL", "HOLE_SPACING", "SHARES_HOST", "POCKET_CLEARANCE", "SLOT_CLEARANCE")
    ),
    host_implying_relations=frozenset(("HOLE_SPACING", "POCKET_CLEARANCE", "SLOT_CLEARANCE")),
    edge_metrics={
        "INTERSECTS": {"crossingAngle": "angle90"},
        "HOLE_SPACING": {"axisSpacing": "length"},
        "POCKET_CLEARANCE": {"pocketClearance": "length"},
        "SLOT_CLEARANCE": {"slotClearance": "length"},
    },
    static_edge_base_weight={
        "DIRECT_ADJACENT": 0.6,
        "LOCATED_ON": 0.6,
        "INTERSECTS": 1.0,
    },
    surface_normal_classes=frozenset(("PlanarFace",)),
    fixed_budget_shares={
        "PlanarFaceAggregate": 0.5,
        "FreeformFaceAggregate": 0.2,
        "UnclaimedAggregate": 0.3,
    },
    fixed_attribute_weights={
        "PlanarFaceAggregate": {"face_count": 0.4, "area": 0.4, "normal_classes": 0.2},
        "FreeformFaceAggregate": {"face_count": 0.5, "area": 0.5},
        "UnclaimedAggregate": {"face_count": 0.5, "area": 0.5},
    },
)


# ------------------------------------------------------------------ nx_v1

def _nx_representation() -> Representation:
    from tbox import nx_v1 as tbox

    tables = nx_policy.build()
    surfaces = tables["surface_leaves"]
    return Representation(
        name="nx_v1",
        family_of=tables["family_of"],
        surface_leaves=surfaces,
        semantic_leaves=frozenset(c for c in tables["classes"] if c not in surfaces),
        leaf_attributes=tables["leaf_attributes"],
        attribute_kind=tables["attribute_kind"],
        angle_range_deg=tables["angle_range_deg"],
        # The six ``relations.yaml`` declares.  DIRECT_ADJACENT is not among
        # them: node 19a withdrew its only client (H30), and nx_v1 does not
        # publish the edge at all, so keeping it in the whitelist would only
        # describe a relation nobody produces.
        known_relations=frozenset(
            ("LOCATED_ON", "COAXIAL", "INTERSECTS",
             "HOLE_SPACING", "POCKET_CLEARANCE", "SLOT_CLEARANCE")
        ),
        grouped_relations=frozenset(
            ("COAXIAL", "HOLE_SPACING", "SHARES_HOST", "POCKET_CLEARANCE", "SLOT_CLEARANCE")
        ),
        host_implying_relations=frozenset(
            ("HOLE_SPACING", "POCKET_CLEARANCE", "SLOT_CLEARANCE")
        ),
        edge_metrics={
            "INTERSECTS": {"crossingAngle": "angle90"},
            "HOLE_SPACING": {"axisSpacing": "length"},
            "POCKET_CLEARANCE": {"pocketClearance": "length"},
            "SLOT_CLEARANCE": {"slotClearance": "length"},
        },
        # The two relations nx_v1 publishes and does not clique-normalise, at
        # the same prices v5.1 gives them.  DIRECT_ADJACENT is not here for
        # the reason it is not in ``known_relations``: nx_v1 does not publish
        # the edge, so a price for it would be a price nothing can ever pay.
        static_edge_base_weight={
            "LOCATED_ON": 0.6,
            "INTERSECTS": 1.0,
        },
        surface_normal_classes=frozenset(),
        fixed_budget_shares=nx_policy.surface_budget_shares(surfaces),
        fixed_attribute_weights=nx_policy.surface_attribute_weights(surfaces),
        provenance={
            # Every table above is a function of these files; 19' D14 asks
            # for their fingerprints to travel with anything derived from
            # them, so a new NX install invalidates the cache by itself.
            "installedSources": [
                {"path": item.get("path"), "sha256": item.get("sha256")}
                for item in _class_sources()
            ],
            "ontologySourceVersion": dict(tbox.source_version()),
        },
    )


def _class_sources() -> list[dict]:
    """The installed NX files ``classes.yaml`` says it was transcribed from."""
    import yaml

    from tbox import nx_v1 as tbox

    document = yaml.safe_load(tbox.CLASSES_YAML.read_text(encoding="utf-8"))
    return list((document.get("head") or {}).get("source") or [])


NX_V1 = _nx_representation()


BY_NAME: dict[str, Representation] = {rep.name: rep for rep in (V5_1, NX_V1)}

# Which representation a kernel line's artifacts are written in.  The line is
# who filled the PartModel; the representation is whose vocabulary the result
# speaks.  They are 1:1 today and the mapping says so once, here.
BY_LINE: dict[str, Representation] = {LINE_OCC: V5_1, LINE_NX: NX_V1}


def for_line(line: str) -> Representation:
    try:
        return BY_LINE[line]
    except KeyError:
        raise KeyError(f"no GED representation registered for line {line!r}; "
                       f"known: {', '.join(sorted(BY_LINE))}") from None


def check(rep: Representation) -> None:
    """Every attribute a leaf publishes has a kind, and every angle a span."""
    for leaf, names in rep.leaf_attributes.items():
        if leaf not in rep.family_of:
            raise AssertionError(f"{rep.name}: {leaf} has attributes but no family")
        for name in names:
            kind = rep.attribute_kind.get(name)
            if kind is None:
                raise AssertionError(f"{rep.name}: {leaf}.{name} has no attribute kind")
            if kind == "angle" and name not in rep.angle_range_deg:
                raise AssertionError(f"{rep.name}: {name} is an angle without a range")
    priced = set(rep.static_edge_base_weight) - set(rep.known_relations)
    if priced:
        raise AssertionError(
            f"{rep.name}: base weights for relations it does not publish: {sorted(priced)}")
    missing = set(rep.semantic_leaves) - set(rep.leaf_attributes)
    if missing:
        raise AssertionError(f"{rep.name}: leaves without an attribute policy: {sorted(missing)}")
    total = sum(rep.fixed_budget_shares.values())
    if rep.fixed_budget_shares and abs(total - 1.0) > 1.0e-9:
        raise AssertionError(f"{rep.name}: fixed-node budget shares sum to {total}, not 1")
    for type_key in rep.fixed_budget_shares:
        if type_key not in rep.fixed_attribute_weights:
            raise AssertionError(f"{rep.name}: {type_key} has a budget but no weights")


for _rep in BY_NAME.values():
    check(_rep)
