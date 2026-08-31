"""The v5.1 relation dialect's registry: ``relations.v51.rules``.

One module per relation type under that package; the decorator below is what
a v5.1 rule uses to register itself.  The discovery and ordering machinery is
in ``relations.registry``.
"""

from __future__ import annotations

from relations.base import RelationRule
from relations.registry import BaseRelationRegistry


class RelationRegistry(BaseRelationRegistry):
    """The v5.1 dialect: ``relations.v51.rules``."""

    package = "relations.v51.rules"
    _loaded = False
    _rules: list[RelationRule] = []


def register_rule(rule):
    """Class decorator: instantiate and register the rule, return the class."""

    RelationRegistry.register(rule() if isinstance(rule, type) else rule)
    return rule
