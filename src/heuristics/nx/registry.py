"""The nx_v1 dialect's registry: ``ontology/nx_v1/source`` + ``heuristics.nx.rules``.

Three differences from the v5.1 dialect, all of them because NX's vocabulary
is shaped differently, and none of them a translation:

* **status** is ``nxStatus``.  Node 19a's audit splits the frozen H set four
  ways — ``evaluable`` / ``noDomain`` / ``withdrawn`` / ``retired`` — and only
  the first is a rule this line can carry.
* **domains are patterns**, not lists of leaves.  NX names families
  (``STEP*``, ``POCKET_*``), because its 84 reachable classes are a tree whose
  names carry the family; writing the leaves out would be us restating a fact
  NX already states.
* **the attribute set of a class is inherited**.  ``STEP2HOLE`` carries
  ``DIAMETER_2`` of its own and ``DEPTH`` from an ancestor, so the central
  gate's "is this attribute on this class at all" question is answered from
  the whole chain (``tbox.nx_v1.effective_attributes``).

``coverage()`` is empty since node 20'b: the six tier-E rules read kernel
measurements the journal stores on the node (the ``derived`` block) and on
the two clearance edges, so every one of the 28 evaluable heuristics is
implemented.  The tier split still matters for *when* a value can exist -
tier E is measured only inside an NX session - not for whether the rule runs.
"""

from __future__ import annotations

from fnmatch import fnmatchcase

from tbox import nx_v1 as source

from heuristics.base import HeuristicRule
from heuristics.registry import BaseHeuristicRegistry

# The tiers node 19' 4.2 cut 20'a from 20'b on: A/B/C/D need only what is
# already on disk (result.json, partmodel.json), E needs a live NX session.
TIERS_WITHOUT_A_SESSION = ("A", "B", "C", "D")


class NxHeuristicRegistry(BaseHeuristicRegistry):
    """Frozen nx_v1 definitions + the rule modules that implement them."""

    package = "heuristics.nx.rules"
    _loaded = False
    _rules: dict[str, HeuristicRule] = {}

    @classmethod
    def source(cls) -> dict:
        return {
            "definitions": source.definitions(),
            "symbols": source.symbols(),
            # Keyed by NX class; the machinery asks for it by the name it uses
            # for the v5.1 leaves, and the answer is the inherited set.
            "classAttributes": {},
            "version": source.source_version(),
        }

    @classmethod
    def leaf_attributes(cls, leaf: str) -> tuple[str, ...]:
        return source.effective_attributes(leaf)

    @classmethod
    def leaf_classes(cls) -> tuple[str, ...]:
        return source.reachable_classes()

    @classmethod
    def active_ids(cls) -> list[str]:
        """Every H this ontology publishes — the 28 node 19a kept.

        ``withdrawn`` is deliberately not among them.  A withdrawn rule is one
        that NX can supply numbers for and cannot state clearly in its own
        words, and publishing a verdict for it would be exactly the claim the
        audit refused to make.
        """
        return sorted(key for key, entry in cls.definitions().items()
                      if entry.get("nxStatus") == "evaluable")

    @classmethod
    def retired_ids(cls) -> list[str]:
        return sorted(key for key, entry in cls.definitions().items()
                      if entry.get("nxStatus") == "retired")

    @classmethod
    def tier(cls, h_id: str) -> str | None:
        return cls.definitions()[h_id].get("nxTier")

    @classmethod
    def ids_without_a_session(cls) -> list[str]:
        """The 23 an evaluation can reach with no NX licence in the room.

        22 with node 20'a and one more with node 27 (finding D-08): H16 reads
        SLOT_CLEARANCE, which node 21' had already moved off the kernel, and
        its tier had stayed behind at E.
        """
        return sorted(key for key in cls.active_ids()
                      if cls.tier(key) in TIERS_WITHOUT_A_SESSION)

    @classmethod
    def in_domain(cls, h_id: str, feature_type: str) -> bool:
        return any(fnmatchcase(feature_type, pattern)
                   for pattern in cls.definitions()[h_id]["applies_to"])


def register(rule):
    """Class decorator: instantiate and register one NX heuristic rule."""

    NxHeuristicRegistry.register(rule() if isinstance(rule, type) else rule)
    return rule
