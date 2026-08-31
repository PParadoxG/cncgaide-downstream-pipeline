"""The nx_v1 relation dialect's registry: ``relations.nx.rules``.

One module per relation type under that package; the decorator below is what
an NX rule uses to register itself.  The discovery and ordering machinery is
in ``relations.registry``.
"""

from __future__ import annotations

from relations.base import RelationRule
from relations.registry import BaseRelationRegistry


class NxRelationRegistry(BaseRelationRegistry):
    """The nx_v1 dialect: ``relations.nx.rules``."""

    package = "relations.nx.rules"
    _loaded = False
    _rules: list[RelationRule] = []


def register_rule(rule):
    """Class decorator: instantiate and register the rule, return the class."""

    NxRelationRegistry.register(rule() if isinstance(rule, type) else rule)
    return rule
