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


class RuleDiscoveryError(RuntimeError):
    """One rule module of a dialect's package did not import.

    Named rather than left as whatever the module happened to raise, because
    the interesting fact is *which* module refused and the original error is
    only half of that.  The cause travels on ``__cause__``.
    """


class DuplicateRuleError(RuntimeError):
    """Two rule modules of one dialect claim the same relation rule id.

    One of them would never run and which one is decided by the walk order of
    the package.  The heuristic side has a standing test pinning "one H, one
    file"; the relation side has none, which is exactly why the silent drop
    had to become a refusal.
    """


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
        """Import every rule module of this dialect, or refuse to be loaded.

        ``_loaded`` is set *after* the walk, never before it.  Set first, a
        module that raised on import would leave the flag standing and every
        later ``all_rules()`` would answer with the rules that happened to
        register before the failure - and a missing relation rule is a whole
        edge type silently absent from every part of the run that follows.
        """
        if cls._loaded:
            return
        module = importlib.import_module(cls.package)
        for info in pkgutil.iter_modules(module.__path__, cls.package + "."):
            if info.name.rsplit(".", 1)[-1].startswith("_"):
                continue
            try:
                importlib.import_module(info.name)
            except Exception as exc:
                raise RuleDiscoveryError(
                    f"relation rule module {info.name!r} did not import: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        cls._loaded = True

    @classmethod
    def register(cls, rule: RelationRule) -> None:
        existing = next((item for item in cls._rules if item.rule_id == rule.rule_id), None)
        if existing is not None:
            raise DuplicateRuleError(
                f"{rule.rule_id} is registered twice in {cls.__name__}: "
                f"{type(existing).__module__} and {type(rule).__module__}"
            )
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
