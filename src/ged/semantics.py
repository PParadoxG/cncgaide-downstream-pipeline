"""Version stamp for everything a cached pair result depends on besides the weights.

Two things already version a cached ``GedResult``: the configuration hash
covers the eight adjustable scalars and *nothing else* (``cost/presets.py``
says so deliberately — a renamed or re-derived weight set must share the
cache), and each part's ``producedAt`` covers the artifacts it was projected
from.  Neither covers the code in between: the representation tables that say
which leaf classes exist and which attributes are compared, and the
cost-layer tables that are not weights.  Editing one of them changes every
distance while the hash and both ``producedAt`` stamps stay the same, so a
cached pair would keep answering with what the old tables produced.

This module closes that hole the same way ``producedAt`` closes its own: the
stamp joins the cache key, an entry whose stamp differs simply never matches
again and is overwritten by the next computation (lazy invalidation, no
migration, no orphaned files).

The stamp is **derived from the live tables**, so adding a leaf class or
changing an attribute set invalidates the affected cache by itself — nobody
has to remember to bump anything.  What derivation cannot see is a change to
the *algorithms* around those tables: a new projection rule, a different
geometry distance, another normalisation.  ``EPOCH`` covers those — raise it
when you change how a distance is computed without changing any table below.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ged.contract import attribute_policy, nx_policy, projection
from ged.contract.representation import NX_V1, Representation, V5_1
from ged.cost import edge_cost, node_cost, nx_type_matrix, type_matrix


# Raise when representation or cost *code* changes without any table below
# changing.  A table change needs no bump: it moves the hash on its own.
#
# 2 - node 27 phase 2, finding F-09: a LOCATED_ON edge whose two ends witness
#     each other is now projected as one undirected edge with its direction
#     left unstated, instead of being dropped.  Twelve nx edges that used to
#     be absent are present, which moves the distance of every pair that
#     touches them; no table changed, so nothing else would have said so.
#     The occ representation's numbers are unaffected - it has no such edge -
#     but EPOCH is shared, so its cache is rebuilt once and lands on the same
#     distances.
# 3 - node 28 D-1: the projection now emits its nodes in canonical order
#     (class + active face ids + id) and its edges in the order of their ends,
#     so an approximate solver's choice among equal-cost assignments no longer
#     follows the ordinals the pipeline happened to issue.  Distances move on
#     both sides without any table moving, which is again what EPOCH is for.
EPOCH = 3


def _canonical(value: Any) -> Any:
    """A deterministic, JSON-serialisable image of a table.

    Sets and frozensets become sorted lists; a mapping keyed by frozensets
    (the unordered pair tables) becomes a sorted list of ``[members, value]``
    so that neither insertion order nor set iteration order can move the
    stamp.
    """
    if isinstance(value, (frozenset, set)):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, dict):
        if any(isinstance(key, (frozenset, set)) for key in value):
            return sorted([_canonical(key), _canonical(item)] for key, item in value.items())
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


# Every table outside the eight weights whose value changes a distance.
# This is the **v5.1** set, kept spelled out because these tables are written
# by hand and each name here is a promise that somebody looked at it.
# The promise is no longer only a promise: ``tests/test_node27_f.py`` walks
# ``ged/cost`` and ``ged/contract`` for module-level constants and fails on
# any that is neither reachable from a stamp below nor exempted by name with
# a reason (node 26 F-01 was exactly a table nobody remembered to add).
TABLES: dict[str, Any] = {
    # representation: what becomes a node, what is compared, how it is scaled
    "FAMILY_OF": attribute_policy.FAMILY_OF,
    "SUBFAMILY_OF": attribute_policy.SUBFAMILY_OF,
    "SURFACE_LEAVES": attribute_policy.SURFACE_LEAVES,
    "LEAF_ATTRIBUTES": attribute_policy.LEAF_ATTRIBUTES,
    "ATTRIBUTE_KIND": attribute_policy.ATTRIBUTE_KIND,
    "ANGLE_RANGE_DEG": attribute_policy.ANGLE_RANGE_DEG,
    # representation: which relations survive and how they are normalised
    "KNOWN_RELATIONS": projection.KNOWN_RELATIONS,
    "GROUPED_RELATIONS": projection.GROUPED_RELATIONS,
    "HOST_IMPLYING_RELATIONS": projection.HOST_IMPLYING_RELATIONS,
    "EDGE_METRICS": projection.EDGE_METRICS,
    "NORMAL_PARALLEL_TOL_DEG": projection.NORMAL_PARALLEL_TOL_DEG,
    # representation: what an aggregate of the background layer is worth
    "FIXED_BUDGET_SHARES": V5_1.fixed_budget_shares,
    "FIXED_ATTRIBUTE_WEIGHTS": V5_1.fixed_attribute_weights,
    # cost layer: what an edge costs to delete, insert or change
    "EDGE_STATIC_BASE_WEIGHT": V5_1.static_edge_base_weight,
    "EDGE_DEFAULT_BASE_WEIGHT": edge_cost.DEFAULT_BASE_WEIGHT,
    "LABEL_MISMATCH_BASE": edge_cost.LABEL_MISMATCH_BASE,
    "ATTRIBUTE_MISMATCH_BASE": edge_cost.ATTRIBUTE_MISMATCH_BASE,
    # cost layer, the parts the eight weights do not reach
    "SAME_LEAF": type_matrix.SAME_LEAF,
    "SAME_FAMILY": type_matrix.SAME_FAMILY,
    "CROSS_FAMILY": type_matrix.CROSS_FAMILY,
    "RECESS_CROSS": type_matrix.RECESS_CROSS,
    "OVERRIDES": type_matrix.OVERRIDES,
    "CATEGORY_DISTANCE": node_cost.CATEGORY_DISTANCE,
}


# The nx_v1 set.  Every table here is *derived* from the ontology rather than
# written down, so the entries are the ontology's own fingerprints plus the
# four rules the derivation applies (19' §3.4) and the one constant that is
# ours (the 180-degree angle span).  Node 19' D14 asks for the two installed
# NX files to be in here: the class tree, every attribute's unit and the whole
# Wu-Palmer type matrix are functions of them, so a different NX install has
# to invalidate every cached distance - and now does, by itself.
NX_TABLES: dict[str, Any] = {
    "installedSources": NX_V1.provenance.get("installedSources"),
    "ontologySourceVersion": NX_V1.provenance.get("ontologySourceVersion"),
    # the derivation rules, so editing one of them moves the stamp too
    "UNIT_KIND": nx_policy.UNIT_KIND,
    "TOLERANCE_BAND_SUFFIXES": nx_policy.TOLERANCE_BAND_SUFFIXES,
    "STOCK_PREFIX": nx_policy.STOCK_PREFIX,
    "POSITION_NAMES": nx_policy.POSITION_NAMES,
    "BOOLEAN_AS_CATEGORY": nx_policy.BOOLEAN_AS_CATEGORY,
    "NX_ANGLE_RANGE_DEG": nx_policy.NX_ANGLE_RANGE_DEG,
    "SURFACE_FAMILY": nx_policy.SURFACE_FAMILY,
    "SURFACE_BUDGET": nx_policy.SURFACE_BUDGET,
    "UNCLAIMED_BUDGET": nx_policy.UNCLAIMED_BUDGET,
    # what the derivation produced, so a change upstream shows even if the
    # rules above are untouched
    "SURFACE_LEAVES": NX_V1.surface_leaves,
    "LEAF_ATTRIBUTES": NX_V1.leaf_attributes,
    "ATTRIBUTE_KIND": NX_V1.attribute_kind,
    "KNOWN_RELATIONS": NX_V1.known_relations,
    "GROUPED_RELATIONS": NX_V1.grouped_relations,
    "HOST_IMPLYING_RELATIONS": NX_V1.host_implying_relations,
    "EDGE_METRICS": NX_V1.edge_metrics,
    "FIXED_BUDGET_SHARES": NX_V1.fixed_budget_shares,
    "FIXED_ATTRIBUTE_WEIGHTS": NX_V1.fixed_attribute_weights,
    # what an edge costs on this side: the priced relations are nx_v1's own
    # (DIRECT_ADJACENT is not among them), the three constants are shared.
    "EDGE_STATIC_BASE_WEIGHT": NX_V1.static_edge_base_weight,
    "EDGE_DEFAULT_BASE_WEIGHT": edge_cost.DEFAULT_BASE_WEIGHT,
    "LABEL_MISMATCH_BASE": edge_cost.LABEL_MISMATCH_BASE,
    "ATTRIBUTE_MISMATCH_BASE": edge_cost.ATTRIBUTE_MISMATCH_BASE,
    # cost layer: OVERRIDES is empty by decision (D8 = b), and its emptiness
    # is part of what a cached distance was computed under.
    "NX_OVERRIDES": nx_type_matrix.OVERRIDES,
    "NX_SAME_LEAF": nx_type_matrix.SAME_LEAF,
    "NX_UNKNOWN": nx_type_matrix.UNKNOWN,
    "CATEGORY_DISTANCE": node_cost.CATEGORY_DISTANCE,
    "SKIP_BOTH_ZERO": True,
    # the projection rules both representations share
    "NORMAL_PARALLEL_TOL_DEG": projection.NORMAL_PARALLEL_TOL_DEG,
}


TABLES_BY_REPRESENTATION: dict[str, dict[str, Any]] = {
    V5_1.name: TABLES,
    NX_V1.name: NX_TABLES,
}


def tables_hash(representation: Representation | str | None = None) -> str:
    """First 8 hex digits of sha256 over the canonical image of the tables."""
    name = representation.name if isinstance(representation, Representation) else (
        representation or V5_1.name)
    tables = TABLES_BY_REPRESENTATION[name]
    payload = json.dumps({key: _canonical(table) for key, table in sorted(tables.items())},
                         sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def semantics_stamp(representation: Representation | str | None = None) -> str:
    """``r<EPOCH>-<tables hash>`` — the stamp that versions a cached pair.

    One per representation, because the tables are.  Two graphs written in
    different ontologies therefore cannot collide in the cache even if the
    weights, the part ids and the ``producedAt`` stamps all matched.
    """
    return f"r{EPOCH}-{tables_hash(representation)}"


#: Evaluated once at import; the tables are module-level constants.
SEMANTICS_STAMP = semantics_stamp()
STAMP_BY_REPRESENTATION: dict[str, str] = {
    name: semantics_stamp(name) for name in TABLES_BY_REPRESENTATION
}
