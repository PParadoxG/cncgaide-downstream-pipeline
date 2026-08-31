"""The nx_v1 dialect's own vocabulary, plus the machinery its rules import.

Three things live here that the v5.1 dialect has no use for:

* **a four-branch attribute gate.**  An NX node carries the values NX handed
  over; what NX declared and would not hand over is in ``missingAttributes``;
  what the class declares and this instance never carried is in neither.  The
  three-line rule of catalogue 03 chapter 7.2 still holds, but "the key is
  absent" splits: absent *from the class* is structural (NOT_APPLICABLE),
  absent *from this instance* is a gap (NOT_EVALUABLE), and the two are told
  apart by the class's inherited attribute set.
* **the ladder.**  NX publishes ``DIAMETER_1..n`` and a *cumulative*
  ``DEPTH_1..`` per stepped hole or pocket, so a per-section depth is
  arithmetic on what NX said (node 19a fact 1) and needs no re-cut of the
  solid.  Every rule that reads a section reads it through here.
* **the axis and the locating point**, which NX publishes as nine separate
  scalars rather than as vectors.

Everything else - the evaluation context, the operators, the four values - is
machinery and comes from ``heuristics._shared``.
"""

from __future__ import annotations

import re

from tbox import nx_v1 as source

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

# NX's own names for a ladder's k-th section.  Anchored, because
# ``DEPTH_BOTTOM_CHAMFER_1`` is a chamfer slot and not section 1.
_DIAMETER_K = re.compile(r"^DIAMETER_(\d+)$")
_DEPTH_K = re.compile(r"^DEPTH_(\d+)$")

# NX publishes the feature's access direction and locating point as nine
# scalars.  These are the names, not a translation of them.
ORIENTATION_D = ("X_ORIENTATION_D", "Y_ORIENTATION_D", "Z_ORIENTATION_D")
POSITION = ("X_POSITION", "Y_POSITION", "Z_POSITION")

# NX's own two hole families: STEP is the drilled/stepped one, HOLES the
# milled through-cavities.  Named rather than enumerated - which leaves belong
# to each is a fact in the installed schema, transcribed into classes.yaml and
# checked against that file by the ontology's own S8.
HOLE_FAMILIES = ("STEP", "HOLES")


def is_hole_class(name) -> bool:
    """Is this NX class a member of one of NX's two hole families?"""
    return bool(name) and any(source.in_family(str(name), family)
                              for family in HOLE_FAMILIES)


class NxHeuristicContext(HeuristicContext):
    """What an NX-dialect rule is allowed to see."""

    # -- the gate ----------------------------------------------------------
    def attribute_state(self, node, name: str) -> tuple[str, str]:
        """``(state, reason)`` for one NX attribute on one NX feature.

        Four cases rather than three, because NX has two ways of not giving a
        value and they are different findings:

        1. the value is there                          -> ``ATTR_OK``;
        2. NX declared it and would not hand it over
           (``missingAttributes``)                     -> ``ATTR_GAP``,
           carrying NX's own reason;
        3. the class declares it and this instance did
           not carry it                                -> ``ATTR_GAP``;
        4. the class does not declare it at all        -> ``ATTR_ABSENT``.

        A name from the ``nxd:`` derived vocabulary is answered from the
        node's ``derived`` block instead — see :meth:`derived_state`.  It can
        never be ``ATTR_ABSENT``: derived quantities are attached to rules,
        not classes, so "this class does not carry it" is already the domain
        gate's answer, and everything else is a measurement present or a
        measurement refused.
        """
        if name in source.derived_attributes():
            return self.derived_state(node, name)
        attributes = self.attributes(node)
        if name in attributes and attributes[name] is not None:
            return ATTR_OK, ""
        missing = getattr(node, "missingAttributes", {}) or {}
        if name in missing:
            return ATTR_GAP, str(missing[name])
        if name in attributes:
            return ATTR_GAP, f"{name} is null on {node.nodeId} without a recorded reason"
        if source.declares(node.featureType, name):
            return ATTR_GAP, (
                f"NX declares {name} on {node.featureType} and did not publish it on "
                f"{node.nodeId}"
            )
        return ATTR_ABSENT, f"{name} is not declared on {node.featureType}"

    # -- the derived block -------------------------------------------------
    def derived_entry(self, node, name: str) -> dict | None:
        return (getattr(node, "derived", None) or {}).get(name)

    def derived_state(self, node, name: str) -> tuple[str, str]:
        """``(state, reason)`` for one ``nxd:`` quantity on one node.

        The measurement happened (or was refused) in the session that built
        the part; here only the recorded answer is read, which is what keeps
        the seconds-tier recompute kernel-free.  No entry at all means the
        build predates the measurement or the domain was widened since - a
        named gap either way, never a guess.
        """
        entry = self.derived_entry(node, name)
        if entry is None:
            return ATTR_GAP, (f"no {entry_tier(name)} measurement of {name} is recorded "
                              f"on {node.nodeId}")
        if entry.get("value") is not None:
            return ATTR_OK, ""
        return ATTR_GAP, str(entry.get("reason")
                             or f"{name} was refused on {node.nodeId} without a recorded reason")

    def attribute(self, node, name: str):
        if name in source.derived_attributes():
            entry = self.derived_entry(node, name) or {}
            return entry.get("value")
        return super().attribute(node, name)

    # -- geometry NX published --------------------------------------------
    def axis(self, node) -> list[float] | None:
        """The feature's access direction, as the unit vector NX's nine give."""
        return unit([self.attribute(node, name) for name in ORIENTATION_D])

    def position(self, node) -> list[float] | None:
        values = [self.attribute(node, name) for name in POSITION]
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               for value in values):
            return None
        return [float(value) for value in values]

    # -- terminals ---------------------------------------------------------
    def terminals(self) -> list[dict]:
        """Flat LOCATED_ON terminal table, joined from properties and evidence.

        The ontology declares exactly two fields on a terminal
        (``hostSurfaceType`` / ``hostSurfaceNormal``), so those are all the
        relation carries as *properties*; which feature the terminal belongs
        to and which node owns the host face are identity, and travel in the
        edge's evidence beside them, aligned by index.
        """

        def build() -> list[dict]:
            table: list[dict] = []
            for edge in self.edges_of("LOCATED_ON"):
                terminals = (edge.properties or {}).get("terminals") or []
                witnesses = (edge.evidence or {}).get("locatedOnWitnesses") or []
                for index, terminal in enumerate(terminals):
                    witness = witnesses[index] if index < len(witnesses) else {}
                    source_id = witness.get("semanticSourceFeatureId")
                    host_id = witness.get("hostOwnerNodeId")
                    if source_id is None:
                        source_id = edge.target if edge.source == host_id else edge.source
                    if host_id is None:
                        host_id = edge.target if edge.source == source_id else edge.source
                    table.append({
                        "edgeId": edge.edgeId,
                        "sourceNodeId": source_id,
                        "sourceFeatureType": getattr(
                            self.node_by_id.get(source_id), "featureType", None),
                        "terminalRole": witness.get("terminalRole"),
                        "hostFaceId": witness.get("hostFaceId"),
                        "hostSurfaceType": terminal.get("hostSurfaceType"),
                        "hostSurfaceNormal": terminal.get("hostSurfaceNormal"),
                        "hostOwnerNodeId": host_id,
                        "hostOwnerFeatureType": getattr(
                            self.node_by_id.get(host_id), "featureType", None),
                    })
            return table

        return self._memo("terminals", build)


def entry_tier(name: str) -> str:
    """The declared ``derivedBy`` tier of one ``nxd:`` quantity (``nx_kernel``)."""
    record = source.derived_attributes().get(name) or {}
    return str(record.get("derivedBy") or "derived")


# --------------------------------------------------------------- the ladder

def section_count(feature_type: str) -> int:
    """How many sections NX declares on this class.

    Counted from the class's own ``DIAMETER_k`` declarations rather than from
    the digit in its name: both say the same thing, and only one of them is a
    field NX publishes.
    """
    indexes = [int(match.group(1))
               for match in (_DIAMETER_K.match(name)
                             for name in source.effective_attributes(feature_type))
               if match]
    return max(indexes) if indexes else 0


def section_diameter_names(feature_type: str) -> tuple[str, ...]:
    return tuple(f"DIAMETER_{index}" for index in range(1, section_count(feature_type) + 1))


def entry_diameter_name(feature_type: str) -> str | None:
    """``DIAMETER_1`` — the entry and, on a monotone ladder, the widest."""
    return "DIAMETER_1" if section_count(feature_type) else None


def base_diameter_name(feature_type: str) -> str | None:
    """``DIAMETER_n`` — the last section, which is the base hole."""
    count = section_count(feature_type)
    return f"DIAMETER_{count}" if count else None


def section_depth_names(feature_type: str) -> tuple[str, ...]:
    """The cumulative depths a section walk needs, in order, ``DEPTH`` last.

    ``DEPTH_k`` is measured from the entry, so the boundaries of the n
    sections are ``0, DEPTH_1 … DEPTH_(n-1), DEPTH``.  A through ladder
    declares exactly n-1 of them and a blind one declares n, whose last is
    ``DEPTH`` again; taking ``DEPTH`` as the final boundary is right either
    way.
    """
    count = section_count(feature_type)
    if not count:
        return ()
    declared = {name for name in source.effective_attributes(feature_type)
                if _DEPTH_K.match(name)}
    interior = tuple(name for index in range(1, count)
                     for name in (f"DEPTH_{index}",) if name in declared)
    return interior + ("DEPTH",)


def sections(context, node) -> tuple[list[dict], str | None]:
    """``([{index, diameter, depth, cumulative}], reason)`` for one ladder.

    The reason is non-empty exactly when the walk could not be completed, and
    names the attribute that stopped it, so a rule can hand it straight to the
    NOT_EVALUABLE it has to return.
    """
    count = section_count(node.featureType)
    if not count:
        return [], f"{node.featureType} declares no DIAMETER_k"

    boundaries = [0.0]
    for name in section_depth_names(node.featureType):
        state, reason = context.attribute_state(node, name)
        if state != ATTR_OK:
            return [], f"{name}: {reason}"
        boundaries.append(float(context.attribute(node, name)))
    if len(boundaries) != count + 1:
        return [], (
            f"{node.featureType} declares {count} section(s) but "
            f"{len(boundaries) - 1} cumulative depth(s)"
        )

    records = []
    for index in range(1, count + 1):
        name = f"DIAMETER_{index}"
        state, reason = context.attribute_state(node, name)
        if state != ATTR_OK:
            return [], f"{name}: {reason}"
        records.append({
            "index": index,
            "diameter": float(context.attribute(node, name)),
            "depth": boundaries[index] - boundaries[index - 1],
            "cumulativeDepth": boundaries[index],
        })
    return records, None


def expected_terminals(feature_type: str) -> int:
    """How many openings NX's own name for the class implies.

    Node 19a D2: the through / blind split on this line is NX's, not ours -
    a ``STEP2HOLE`` goes through and opens twice, a ``STEP2POCKET`` is blind
    and opens once.  The class name is the statement; nothing is measured.
    """
    return 1 if "POCKET" in feature_type else 2


__all__ = [
    "ATTR_ABSENT",
    "ATTR_GAP",
    "ATTR_OK",
    "FAIL",
    "HOLE_FAMILIES",
    "NOT_EVALUABLE",
    "ORIENTATION_D",
    "PASS",
    "POSITION",
    "NxHeuristicContext",
    "acute_angle_deg",
    "aggregate_count_limit",
    "base_diameter_name",
    "compare_at_least",
    "compare_at_most",
    "distinct",
    "distinct_axes",
    "distinct_tuples",
    "dot",
    "entry_diameter_name",
    "expected_terminals",
    "external_missing",
    "failed",
    "is_hole_class",
    "member_of",
    "not_evaluable",
    "passed",
    "relation_missing",
    "section_count",
    "section_depth_names",
    "section_diameter_names",
    "sections",
    "unit",
]
