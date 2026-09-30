"""NX's native feature-based recognition, projected onto feature nodes.

Node 18' swapped this line's core.  Until now the NX side answered in *our*
vocabulary: the FBM output was looked up in a provider vocabulary, split into
trust piles, and then re-segmented by the D-7 hole grammar into ontology leaf
classes.  Two translations stood between NX and the stored result, and node
17' measured what they cost - 447 NX features became 514 of ours, 36 of 98
hole assertions changed shape, and 13 ``Chamfer`` nodes appeared that NX never
said anything about.

This module does none of that.  NX's answer *is* the answer:

* the class is NX's ``Type`` (``STEP1HOLE``) - stable, unique, and unlike the
  display name it does not move when the NX interface language does;
* the label is NX's ``TypeDisplayName`` (``Through Hole``) - the exact string
  the Machining Feature Navigator shows, carried for every place a human
  reads;
* the attributes are NX's attributes, under NX's own names, with NX's own
  values.  Nothing is renamed, converted, re-derived or re-measured.

What we add is *reading*, never data: the catalogue (node 17' stage 3) showed
the 141 enumerated attributes fall into four piles, and three of them are not
measurements - vendor rule defaults out of ISO 2768-1 class m, annotation
sub-fields, identity flags, and slots that read ``0`` because they are unused
rather than because something was measured as zero.  Every value is still
published verbatim; which pile it belongs to is recorded beside it so a reader
never mistakes a table lookup for a measurement.

Nothing here imports NX.  The journal hands over plain records and the
geometry provider's ``tag -> face id`` table; everything below is a pure
function of those, which is what lets it be tested without a licence.
"""

from __future__ import annotations

import json
from fnmatch import fnmatch, fnmatchcase
from typing import Any

from tbox import nx_v1 as tbox
from core.models import FeatureNode

# How this line publishes NX's answer.  The *provider* is still ``nx_fbm``
# (the registry's name for NX's feature-based recognition, which is what
# actually finds the features); this names the second half - that its output
# is published as NX states it rather than translated into our ontology.
PROJECTION_ID = "nx_native"

# NX's own words for "this face is just a face of the part".  The catalogue
# counted 259 of these against 447 features - 58% of the corpus - which is the
# same judgement the v5.1 ontology makes about ``PlanarFace``: a background
# layer to be looked past, not a manufacturing feature.
BACKGROUND_TYPES: tuple[str, ...] = (
    "SURFACE_PLANAR",
    "SURFACE_PLANAR_RECTANGULAR",
    "SURFACE_PLANAR_ROUND",
)

# Classes NX *does* claim as features, which this corpus shows are dominated
# by noise, and which the page therefore starts with switched off.  A pattern
# rather than a list, because the rule library registers seven notch classes
# (CORNER_NOTCHES(4) + SIDE_NOTCHES(3)) and only two have been observed so
# far; the fifth to appear should not need a code edit to behave like the
# other four.
#
# This is deliberately *not* BACKGROUND_TYPES.  A SURFACE_PLANAR is NX saying
# "this is just a face of the part"; a CORNER_NOTCH_STRAIGHT is NX asserting a
# machining feature, and disagreeing with that assertion is our judgement, not
# a fact about the part.  So the node is kept whole - faces, attributes, id,
# everything - and only the display default moves.  One click in the type
# filter brings it back.
NOISY_TYPE_PATTERNS: tuple[str, ...] = ("*NOTCH*",)
NOISY_TYPE_REASON = (
    "hidden by default (user decision 2026-08-29): on this corpus the notch classes are "
    "dominated by the right angles between large planar faces rather than by features worth "
    "reviewing.  NX asserts this feature and the node keeps everything it asserted; only the "
    "page's starting state treats it as noise, and the type filter switches it back on"
)


def is_noisy(native_type: str) -> bool:
    return any(fnmatch(native_type, pattern) for pattern in NOISY_TYPE_PATTERNS)


def default_hidden_types(native_types) -> list[str]:
    """What the page starts with switched off, for a part holding *native_types*.

    Two different reasons in one list, because the page only needs to know
    *whether* to draw something: NX's own background classes, plus the classes
    we judge to be noise.  Which reason applied to which class is on the node,
    where a reader can see it.
    """
    present = set(native_types)
    return sorted(present & set(BACKGROUND_TYPES)) + sorted(t for t in present if is_noisy(t))

# Attribute names whose value comes from a vendor rule table rather than from
# the geometry.  ``MACH/resource/feature/*/feature_tolerances.xml`` fills the
# tolerance pair from the ISO 2768-1 class m band for the nominal size, and
# every roughness field from a fixed default.  Publishing these as if they
# were measured would put a few thousand invented tolerances into the store.
RULE_DEFAULT_PATTERNS: tuple[str, ...] = ("*_UPPER", "*_LOWER", "*ROUGHNESS*")
RULE_DEFAULT_SOURCE = (
    "vendor rule default, not a measurement: NX fills tolerance pairs from the "
    "ISO 2768-1 class m size bands and roughness from a fixed default "
    "(MACH/resource/feature/*/feature_tolerances.xml)"
)

# Flags about the feature's provenance, not about its shape.
IDENTITY_ATTRIBUTES: tuple[str, ...] = ("SUBTYPE", "ISRECOGNIZEDFEATURE")

# Attribute names that are *slots*: an optional treatment NX carries a field
# for whether or not the feature has one.  Only on these can a zero mean "the
# slot is unused" rather than "this was measured as zero", which is what node
# 18b B1 fixed: the blanket ``value == 0`` test flagged four direction cosines
# of ``POCKET_RECTANGULAR_STRAIGHT_001`` - a real ``-0.0`` component of a real
# measured direction - as an unused slot.  A position or an orientation is
# never a slot, and neither is a radius, a depth or an obstruction flag: a
# zero there is the measurement.
SLOT_ATTRIBUTE_PATTERNS: tuple[str, ...] = ("*CHAMFER*", "TILTED_*", "*OFFSET*")

ZERO_NOTE = (
    "reads 0 on this instance; this is an optional-treatment slot, so on this class "
    "the slot is frequently unused rather than measured as zero (catalogue pile B) "
    "and 0 must not be read as a measurement"
)

# The attribute slots no API route can return.  ``CAMAttribute.ValueType`` has
# six members - Integer, Double, String, Bool, Point3, Vector3 - and neither
# Face nor Class is among them, so the per-face slots NX keeps on a hole
# (which faces are the top chamfer, which the bottom) and the ``CHAMFER``
# sub-object are declared by NX and unreachable through this binding.  They are
# stated as gaps rather than filled with a plausible value.
STRUCTURAL_GAPS: dict[str, str] = {
    "FACES_*": (
        "NX declares per-face attribute slots (DataType=Face) on this class; "
        "CAMAttribute.ValueType has no Face member, so no API route returns them"
    ),
    "CHAMFER": (
        "NX declares this slot as DataType=Class; the six scalar getters "
        "CAMAttribute offers cannot return an object, so it is unreadable"
    ),
}


# --------------------------------------------------------------- attributes

def _pile(name: str) -> str:
    """Which of the catalogue's piles this attribute name belongs to."""
    if name in IDENTITY_ATTRIBUTES:
        return "identity"
    if any(fnmatch(name, pattern) for pattern in RULE_DEFAULT_PATTERNS):
        return "ruleDefault"
    return "measured"


def _canonical(value: Any) -> str:
    """A comparable spelling of an attribute value, for tie-breaking only."""
    return json.dumps(value, sort_keys=True, default=str)


def _read_attributes(raw: dict) -> tuple[dict[str, Any], dict[str, list[str]], list[str]]:
    """Split one feature's enumerated attributes into values and their reading.

    Returns ``(values, piles, zero_valued)``.  Every enumerated value ends up
    in *values* under NX's own name - the piles say how to read them, they
    never withhold anything.
    """
    values: dict[str, Any] = {}
    piles: dict[str, list[str]] = {"measured": [], "ruleDefault": [], "identity": []}
    zero_valued: list[str] = []
    for name in sorted(raw.get("attributes") or {}):
        record = (raw["attributes"] or {})[name]
        value = record.get("value") if isinstance(record, dict) else record
        values[name] = value
        piles[_pile(name)].append(name)
        if (_is_slot(name)
                and isinstance(value, (int, float)) and not isinstance(value, bool)
                and value == 0):
            zero_valued.append(name)
    return values, piles, zero_valued


def _is_slot(name: str) -> bool:
    """Is a zero here plausibly "no such treatment" rather than a measurement?"""
    return any(fnmatchcase(name, pattern) for pattern in SLOT_ATTRIBUTE_PATTERNS)


def _gaps(raw: dict, native_type: str) -> dict[str, str]:
    """What NX declares on this feature and will not hand over.

    Two kinds, kept apart: the slots that are structurally unreachable through
    the binding, and the ones the enumeration met and could not read - almost
    always an annotation sub-field NX declares as String and never sets.

    The first kind is a constant table, because ``CAMFeature.Attributes`` never
    enumerates a ``DataType=Face`` or ``DataType=Class`` field at all - there
    is no run-time channel through which those names could arrive.  Which of
    them a class actually declares is therefore read from the transcription of
    NX's own schema; applying the table unconditionally, as node 18' shipped
    it, made the artifacts say NX had declared a ``CHAMFER`` on a
    ``SURFACE_PLANAR`` (node 19' 1.2a, 546 such claims in this corpus).
    """
    # The table's keys are patterns, but a gap has to be filed under the name
    # its readers ask for.  ``attribute_state`` looks a name up with
    # ``if name in missing``, so a rule asking about ``FACES_CYLINDER_1`` never
    # matched the literal key ``FACES_*`` and fell through to a generic "NX
    # declares X on Y and did not publish it" - losing the reason NX itself
    # gives (finding E-05).  Expanded against the same class table
    # ``declares`` reads, so nothing is claimed that NX did not declare.
    gaps = {}
    for pattern, reason in STRUCTURAL_GAPS.items():
        for name in tbox.effective_attributes(native_type):
            if fnmatchcase(name, pattern):
                gaps[name] = reason
    for item in raw.get("unreadable") or []:
        name = str(item.get("name") or "")
        if not name:
            continue
        gaps[name] = (
            f"NX declares this attribute (type {item.get('type')}) on this feature and "
            f"the value could not be read: {item.get('reason')}"
        )
    return gaps


# -------------------------------------------------------------------- faces

def _face_ids(raw: dict, face_id_by_tag: dict[int, str]) -> tuple[list[str], list[dict]]:
    """The faces one NX feature names, as a set in face-id order.

    The order NX lists a feature's faces in is a property of the session, not
    of the feature (node 16), so it is sorted by the geometry provider's own
    numbering and never preserved.
    """
    matched: set[str] = set()
    unmatched: list[dict] = []
    for tag in raw.get("faceTags") or []:
        face_id = face_id_by_tag.get(tag)
        if face_id is None:
            unmatched.append({"tag": tag, "reason": "face tag not in the geometry provider's table"})
        else:
            matched.add(face_id)
    return (sorted(matched, key=lambda fid: int(fid.rsplit("_", 1)[-1])),
            sorted(unmatched, key=lambda item: item["tag"]))


# ---------------------------------------------------------------- projection

def _sort_key(record: dict) -> tuple:
    """The order that decides the ordinals in the node ids.

    Deliberately not NX's own listing order: ``CAMFeatures`` enumerates in
    recognition order, which is a property of the run.  The class, then the
    faces the feature owns (already on the handle-based numbering that node 16
    made session-independent), then the attribute block, decide it instead - so
    the same solid produces the same ids whether it was rebuilt alone or in a
    batch of twenty-one.
    """
    faces = record["partmodelFaceIds"]
    return (
        record["type"],
        [int(fid.rsplit("_", 1)[-1]) for fid in faces],
        _canonical(record["values"]),
    )


def project(part_id: str, raw: dict, face_id_by_tag: dict[int, str],
            *, model=None) -> tuple[list[FeatureNode], dict]:
    """One part's native NX extraction -> feature nodes plus diagnostics.

    *raw* is what the journal read out of NX (see ``journal_fullnx``);
    *face_id_by_tag* is the geometry provider's exact ``NX tag -> face id``
    table, which on this line is an identity by construction - the PartModel's
    faces *are* the NX faces.  *model* is optional and only used to report how
    many active faces no NX feature ever mentioned.
    """
    prepared: list[dict] = []
    unmatched_total: list[dict] = []
    for feature in raw.get("features") or []:
        face_ids, unmatched = _face_ids(feature, face_id_by_tag)
        values, piles, zero_valued = _read_attributes(feature)
        prepared.append({
            "type": str(feature.get("type") or ""),
            "displayName": str(feature.get("typeDisplayName") or ""),
            "raw": feature,
            "partmodelFaceIds": face_ids,
            "values": values,
            "piles": piles,
            "zeroValued": zero_valued,
            "unmatched": unmatched,
        })
        unmatched_total.extend(unmatched)

    prepared.sort(key=_sort_key)

    nodes: list[FeatureNode] = []
    counters: dict[str, int] = {}
    claimed: set[str] = set()
    labels: dict[str, str] = {}
    counts: dict[str, int] = {}
    for record in prepared:
        native_type = record["type"]
        index = counters.get(native_type, 0) + 1
        counters[native_type] = index
        counts[native_type] = index
        if record["displayName"]:
            labels.setdefault(native_type, record["displayName"])
        feature = record["raw"]
        diagnostics: dict[str, Any] = {
            "nx": {
                "type": native_type,
                "typeDisplayName": record["displayName"],
                # NX's own name for this feature, as the Machining Feature
                # Navigator shows it.  Its number is NX's within-part find
                # order, which is why it names the feature but never decides
                # this node's id.
                "name": feature.get("name"),
                "sourceType": feature.get("sourceType"),
                "status": feature.get("status"),
                "csys": feature.get("csys") or {},
                "attributeRoute": feature.get("attributeRoute"),
            },
            "attributeReading": {
                "measured": record["piles"]["measured"],
                "ruleDefault": record["piles"]["ruleDefault"],
                "ruleDefaultSource": RULE_DEFAULT_SOURCE,
                "identity": record["piles"]["identity"],
                "zeroValued": record["zeroValued"],
            },
        }
        if record["zeroValued"]:
            diagnostics["attributeReading"]["zeroValuedNote"] = ZERO_NOTE
        if native_type in BACKGROUND_TYPES:
            diagnostics["backgroundLayer"] = (
                "NX's own name for a plain face of the part; a background layer to look "
                "past, not a manufacturing feature")
        elif is_noisy(native_type):
            diagnostics["hiddenByDefault"] = NOISY_TYPE_REASON
        if record["unmatched"]:
            diagnostics["unmatchedNativeFaces"] = record["unmatched"]
        if feature.get("attributeErrors"):
            diagnostics["attributeErrors"] = list(feature["attributeErrors"])
        # Which of these the viewer should carry through to the reader.  Every
        # one of them is a fact about *this* feature that no attribute states:
        # NX's own name for it, how to read the numbers above, and - when it
        # applies - why the page started with it switched off.
        diagnostics["display"] = [key for key in
                                  ("nx", "attributeReading", "backgroundLayer",
                                   "hiddenByDefault", "unmatchedNativeFaces",
                                   "attributeErrors")
                                  if key in diagnostics]
        nodes.append(FeatureNode(
            nodeId=f"{native_type}_{index:03d}",
            featureType=native_type,
            partmodelFaceIds=list(record["partmodelFaceIds"]),
            attributes=dict(record["values"]),
            missingAttributes=_gaps(feature, native_type),
            diagnostics=diagnostics,
        ))
        claimed.update(record["partmodelFaceIds"])

    active = set(getattr(model, "faces", {}) or {})
    diagnostics = {
        # Who recognised is unchanged and is still the registry's name for it:
        # NX's feature-based machining recognition.  What changed in node 18'
        # is how its answer is published - which is the projection, below.
        "recognitionProvider": "nx_fbm",
        "projection": PROJECTION_ID,
        "projectionNote": "NX's Type is the class, TypeDisplayName is the label, NX's attributes "
                          "are the attributes; nothing is re-segmented, renamed or re-measured",
        "recipe": dict(raw.get("recipe") or {}),
        "environment": dict(raw.get("environment") or {}),
        "registeredTypeCount": raw.get("registeredTypeCount"),
        "featureCount": len(nodes),
        "featureCountsByType": dict(sorted(counts.items())),
        "typeDisplayNames": dict(sorted(labels.items())),
        "backgroundTypes": [t for t in BACKGROUND_TYPES if t in counts],
        # Two lists, because they answer different questions: what NX itself
        # calls a plain face, and what the page starts with switched off.  The
        # second is a display default and nothing else - every node is still
        # here, with everything NX said about it.
        "noisyTypes": sorted(t for t in counts if is_noisy(t)),
        "defaultHiddenTypes": default_hidden_types(counts),
        "claimedFaceCount": len(claimed),
        "unclaimedFaceIds": sorted(active - claimed,
                                   key=lambda fid: int(fid.rsplit("_", 1)[-1]))
        if active else [],
        "unmatchedNativeFaceCount": len(unmatched_total),
        "modulesNotRun": {
            "features/*": "no leaf-class recognizer runs on this line; the class is NX's Type",
        },
    }
    return nodes, diagnostics


def used_attributes(nodes: list[FeatureNode]) -> dict[str, list[str]]:
    """``{class: the attribute names this pipeline actually reads}``.

    NX publishes a few dozen attributes on every feature and the pipeline
    consumes a handful of them; node 19' D11c/D12 asks the part page to expand
    that handful and fold the rest into one line.  The list is *derived*, not
    written down: every rule already declares which attributes it reads for a
    given class, so this asks the rules.  Nothing here is a display decision -
    the page receives names, compares them to the keys it is about to render,
    and never learns what any of them mean.
    """
    from tbox import nx_v1
    from heuristics.nx.registry import NxHeuristicRegistry
    from relations.nx._shared import ORIENTATION_D, POSITION, is_hole_like

    # Tier-E rules read nxd: quantities, which are never node top-level keys -
    # they live in the separate ``derived`` block the page renders on its own.
    # This list stays NX vocabulary only.
    derived_names = set(nx_v1.derived_attributes())
    used: dict[str, set[str]] = {}
    for node in nodes:
        names = used.setdefault(node.featureType, set())
        for rule in NxHeuristicRegistry.rules():
            if NxHeuristicRegistry.in_domain(rule.h_id, node.featureType):
                names.update(set(rule.required_attributes(node.featureType)) - derived_names)
        # The relation layer reads the two frames NX publishes on every
        # feature it can pair up; those readings are as real as a rule's.
        if is_hole_like(node) or node.featureType.startswith("BOSS_ROUND"):
            names.update(ORIENTATION_D)
            names.update(POSITION)
    # A class with an empty list is kept: "the pipeline reads nothing on this
    # class" is an answer, and dropping the row would make the page fall back
    # to expanding everything - the opposite of what it says.
    return {name: sorted(values) for name, values in sorted(used.items())}


def feature_labels(nodes: list[FeatureNode]) -> dict[str, str]:
    """``{Type: TypeDisplayName}`` for the classes these nodes use.

    The label is an instance property in NX, so it can only be known for a
    class something was actually recognised as - which is exactly the set the
    page has to render.
    """
    labels: dict[str, str] = {}
    for node in nodes:
        display = ((node.diagnostics or {}).get("nx") or {}).get("typeDisplayName")
        if display:
            labels.setdefault(node.featureType, str(display))
    return labels
