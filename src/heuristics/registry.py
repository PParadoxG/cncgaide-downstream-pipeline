"""Registry machinery: rule discovery, thresholds, provenance.

This module is the *machine*, not the content.  It knows how a dialect is
shaped — a frozen definition table, a threshold table, a class-to-attribute
table, and a package of hand-written rule modules — and nothing about which
ontology is in play.

One subclass per **dialect**, where a dialect is one ontology's heuristic set:
``heuristics.v51.registry`` for ``ontology/v5.1`` and ``heuristics.nx.registry``
for ``ontology/nx_v1``.  They are separated by ontology and not by kernel line
on purpose: which vocabulary a rule is written in is a different question from
who filled the ``PartModel``, and the provider registry already allows those
two axes to come apart.

Nothing in this layer hard-codes a threshold number.  A rule declares the
symbols it consumes and its dialect's registry resolves them, so changing a
threshold is a YAML edit followed by a re-run of the heuristic layer only.
The source files are hashed and the digests travel in ``heuristics.json`` —
that is what makes "this verdict was produced under that threshold set"
checkable later, and what a parts store needs to decide whether the H layer
alone has to be recomputed.
"""

from __future__ import annotations

import hashlib
import importlib
import pkgutil
from pathlib import Path
from typing import Iterable

from heuristics.base import HeuristicRule


def digest(path: Path) -> str:
    """Short content digest of one source file, or ``""`` when it is absent."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else ""


class BaseHeuristicRegistry:
    """Frozen definitions + the rule modules that implement them.

    A subclass restates ``package``, ``_loaded`` and ``_rules`` so that no
    dialect ever discovers another's modules — ``H01`` means one thing under
    ``ontology/v5.1`` and another under ``ontology/nx_v1``, and node 19a is
    explicit that the two verdicts are not one metric.
    """

    package = ""
    _loaded = False
    _rules: dict[str, HeuristicRule] = {}

    @classmethod
    def source(cls) -> dict:  # pragma: no cover - interface
        """``{definitions, symbols, classAttributes, version}`` for this dialect."""
        raise NotImplementedError

    # -- rule discovery ---------------------------------------------------
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
    def register(cls, rule: HeuristicRule) -> None:
        cls._rules.setdefault(rule.h_id, rule)

    @classmethod
    def rules(cls) -> list[HeuristicRule]:
        cls.discover()
        return [cls._rules[key] for key in sorted(cls._rules)]

    @classmethod
    def rule(cls, h_id: str) -> HeuristicRule | None:
        cls.discover()
        return cls._rules.get(h_id)

    @classmethod
    def implemented_ids(cls) -> list[str]:
        return [rule.h_id for rule in cls.rules()]

    # -- frozen definitions ----------------------------------------------
    @classmethod
    def definitions(cls) -> dict[str, dict]:
        return cls.source()["definitions"]

    @classmethod
    def active_ids(cls) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def retired_ids(cls) -> list[str]:  # pragma: no cover - interface
        raise NotImplementedError

    @classmethod
    def definition(cls, h_id: str) -> dict:
        return cls.source()["definitions"][h_id]

    @classmethod
    def applies_to(cls, h_id: str) -> frozenset[str]:
        return frozenset(cls.source()["definitions"][h_id]["applies_to"])

    @classmethod
    def in_domain(cls, h_id: str, feature_type: str) -> bool:
        """Is this class in that heuristic's domain?

        Plain membership by default: ``appliesTo`` names leaf classes one by
        one.  A dialect whose ontology names families instead overrides it.
        """
        return feature_type in cls.applies_to(h_id)

    # -- thresholds --------------------------------------------------------
    @classmethod
    def symbol(cls, symbol: str) -> dict:
        symbols = cls.source()["symbols"]
        if symbol not in symbols:
            raise KeyError(f"threshold symbol {symbol!r} is not declared in thresholds.yaml")
        return symbols[symbol]

    @classmethod
    def value(cls, symbol: str):
        return cls.symbol(symbol)["value"]

    @classmethod
    def threshold_records(cls, symbols: Iterable[str]) -> list[dict]:
        return [
            {"symbol": item["symbol"], "value": item["value"], "unit": item["unit"]}
            for item in (cls.symbol(name) for name in symbols)
        ]

    # -- class attribute sets ----------------------------------------------
    @classmethod
    def leaf_attributes(cls, leaf: str) -> tuple[str, ...]:
        return cls.source()["classAttributes"].get(leaf, ())

    @classmethod
    def leaf_classes(cls) -> tuple[str, ...]:
        return tuple(cls.source()["classAttributes"])

    # -- provenance --------------------------------------------------------
    @classmethod
    def version(cls) -> dict:
        version = dict(cls.source()["version"])
        version["activeHeuristics"] = len(cls.active_ids())
        version["implementedHeuristics"] = len(cls.implemented_ids())
        return version

    @classmethod
    def coverage(cls) -> dict[str, list[str]]:
        """Active-vs-implemented difference, both ways.  Empty means 1:1."""
        active = set(cls.active_ids())
        implemented = set(cls.implemented_ids())
        return {
            "notImplemented": sorted(active - implemented),
            "notDeclared": sorted(implemented - active),
        }


def default_registry() -> type[BaseHeuristicRegistry]:
    """The dialect a caller gets when it does not name one.

    v5.1, because that is what the open-source line, the replay guard and
    every rule written before there was a second ontology mean.  Imported
    inside the function so the machinery carries no module-level dependency
    on any dialect's rule package.
    """
    from heuristics.v51.registry import HeuristicRegistry

    return HeuristicRegistry
