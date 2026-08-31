"""The v5.1 dialect: ``ontology/v5.1/source`` + ``heuristics.v51.rules``.

Two YAML files under ``ontology/v5.1/source`` are the single machine-readable
source of this dialect's heuristic set and of every named threshold and
tolerance:

``heuristics.yaml``   id, status, statement, grade, scope, appliesTo,
                      condition expression, external dependency, KPI links
``thresholds.yaml``   symbol -> value / unit / used_by / source

``classes.yaml`` supplies the third table the machinery asks for: which
attributes each leaf class carries, which is what lets the central gate tell
"this class does not have that attribute" from "this instance's value is
null".  Everything else is in ``heuristics.registry``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from heuristics.base import HeuristicRule
from heuristics.registry import BaseHeuristicRegistry, digest

SOURCE_DIR = Path(__file__).resolve().parents[3] / "ontology" / "v5.1" / "source"
HEURISTICS_YAML = SOURCE_DIR / "heuristics.yaml"
THRESHOLDS_YAML = SOURCE_DIR / "thresholds.yaml"
CLASSES_YAML = SOURCE_DIR / "classes.yaml"


@lru_cache(maxsize=1)
def _load_source() -> dict:
    heuristics = yaml.safe_load(HEURISTICS_YAML.read_text(encoding="utf-8"))
    thresholds = yaml.safe_load(THRESHOLDS_YAML.read_text(encoding="utf-8"))
    classes = yaml.safe_load(CLASSES_YAML.read_text(encoding="utf-8"))

    definitions = {}
    for entry in heuristics["heuristics"]:
        definitions[entry["id"]] = entry

    symbols: dict[str, dict] = {}
    for section, kind in (("process_thresholds", "threshold"), ("tolerances", "tolerance")):
        for item in thresholds.get(section) or []:
            symbols[item["symbol"]] = {
                "symbol": item["symbol"],
                "value": item["value"],
                "unit": item.get("unit"),
                "kind": kind,
                "usedBy": list(item.get("used_by") or []),
                "source": item.get("source"),
            }

    # ``pos`` is one page entry standing for the three coordinate attributes,
    # exactly as the generated attribute-set document expands it.
    class_attributes = {}
    for leaf in classes["leaves"]:
        names: list[str] = []
        for attribute in leaf["attributes"]:
            name = attribute["name"]
            names += ["posX", "posY", "posZ"] if name == "pos" else [name]
        class_attributes[leaf["name"]] = tuple(names)
    return {
        "definitions": definitions,
        "symbols": symbols,
        "classAttributes": class_attributes,
        "version": {
            "heuristicsYamlSha256": digest(HEURISTICS_YAML),
            "thresholdsYamlSha256": digest(THRESHOLDS_YAML),
        },
    }


class HeuristicRegistry(BaseHeuristicRegistry):
    """Frozen v5.1 definitions + the rule modules that implement them."""

    package = "heuristics.v51.rules"
    _loaded = False
    _rules: dict[str, HeuristicRule] = {}

    @classmethod
    def source(cls) -> dict:
        return _load_source()

    @classmethod
    def active_ids(cls) -> list[str]:
        return sorted(
            key for key, entry in cls.definitions().items() if entry["status"] != "retired"
        )

    @classmethod
    def retired_ids(cls) -> list[str]:
        return sorted(
            key for key, entry in cls.definitions().items() if entry["status"] == "retired"
        )


def register(rule):
    """Class decorator: instantiate and register one v5.1 heuristic rule."""

    HeuristicRegistry.register(rule() if isinstance(rule, type) else rule)
    return rule
