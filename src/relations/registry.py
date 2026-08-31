"""Auto-discovery machinery for relation rules.

Every module under a registry's rule package is imported once; classes
decorated with that package's ``register_rule`` register a single instance.
A new relation therefore needs no orchestrator edit.

One subclass per **dialect**, where a dialect is one ontology's relation set:
``relations.v51.registry`` and ``relations.nx.registry``.  They share this
machinery and nothing else — node 19' D16-r asserts no equivalence between
``nx:COAXIAL`` and ``cnc:COAXIAL`` even though the names match, and two
separate rule sets is what that looks like in code.

Split by ontology and not by kernel line on purpose: which vocabulary a rule
is written in is a different question from who filled the ``PartModel``.
"""

from __future__ import annotations

import importlib
import pkgutil

from relations.base import RelationRule


class BaseRelationRegistry:
    """Discovery and ordering for one dialect's relation rules.

    A subclass restates ``package``, ``_loaded`` and ``_rules`` so that no
    dialect ever discovers another's modules.
    """

    package = ""
    _loaded = False
    _rules: list[RelationRule] = []

    @classmethod
    def discover(cls) -> None:
        if cls._loaded:
            return
        cls._loaded = True
        module = importlib.import_module(cls.package)
        for info in pkgutil.iter_modules(module.__path__, cls.package + "."):
            if info.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            importlib.import_module(info.name)

    @classmethod
    def register(cls, rule: RelationRule) -> None:
        if any(existing.rule_id == rule.rule_id for existing in cls._rules):
            return
        cls._rules.append(rule)

    @classmethod
    def all_rules(cls) -> list[RelationRule]:
        cls.discover()
        return sorted(cls._rules, key=lambda rule: (rule.priority, rule.rule_id))

    @classmethod
    def enabled(cls) -> list[RelationRule]:
        return [rule for rule in cls.all_rules() if rule.enabled_by_default]


def default_registry() -> type[BaseRelationRegistry]:
    """The dialect a caller gets when it does not name one.

    v5.1, because that is what the open-source line, the replay guard and
    every rule written before there was a second ontology mean.  Imported
    inside the function so the machinery carries no module-level dependency
    on any dialect's rule package.
    """
    from relations.v51.registry import RelationRegistry

    return RelationRegistry
