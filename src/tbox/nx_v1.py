"""The ``nx_v1`` ontology's machine-readable source, loaded.

``ontology/nx_v1/source/*.yaml`` is the single semantic standard of the NX
line, and two packages need to read it: the NX heuristic dialect wants the
frozen H set, the thresholds and each class's attribute set, and the NX
relation dialect wants the class tree so it can ask whether a class belongs to
one of NX's own families (``STEP``, ``HOLES``, ``POCKETS`` …).  Loading it
twice would be two places to be right about the same file, so it is loaded
here once.

This is *content*, not machinery: it names one ontology.  It sits in ``tbox``
rather than inside either dialect because neither owns it, and it is named
after the ontology rather than after a kernel line — ``nxline`` is who fills
the ``PartModel``, ``nx_v1`` is whose vocabulary the rules speak, and node
19' keeps those two questions apart.

Only the standard library and ``yaml`` are imported, so this is readable from
the NX journal process as well (its Python has the vendored pure-python yaml).
"""

from __future__ import annotations

import hashlib
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path

import yaml

SOURCE_DIR = Path(__file__).resolve().parents[2] / "ontology" / "nx_v1" / "source"
CLASSES_YAML = SOURCE_DIR / "classes.yaml"
ATTRIBUTES_YAML = SOURCE_DIR / "attributes.yaml"
HEURISTICS_YAML = SOURCE_DIR / "heuristics.yaml"
THRESHOLDS_YAML = SOURCE_DIR / "thresholds.yaml"
DERIVED_YAML = SOURCE_DIR / "derived.yaml"


def digest(path: Path) -> str:
    """Short content digest of one source file, or ``""`` when it is absent."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else ""


@lru_cache(maxsize=1)
def _classes() -> dict:
    """``{name: record}`` plus the derived parent chain and attribute sets.

    The effective attribute set of a class is its own declarations, its
    ancestors', and the run-time attributes NX attaches — exactly the
    reconstruction ``ontology/nx_v1/build/validate.py`` performs for its C2/C3
    conformance checks, so the pipeline and the ontology's own validator agree
    on what a class carries by construction rather than by coincidence.
    """
    classes = yaml.safe_load(CLASSES_YAML.read_text(encoding="utf-8"))
    attributes = yaml.safe_load(ATTRIBUTES_YAML.read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in classes["classes"]}

    root_runtime = {item["name"] for item in attributes["attributes"]
                    if item["source"] == "runtime" and item.get("attachment") == "root"}
    observed_runtime: dict[str, set[str]] = {}
    for item in attributes["attributes"]:
        if item["source"] == "runtime" and item.get("attachment") == "observed":
            for owner in item.get("observedOn") or []:
                observed_runtime.setdefault(owner, set()).add(item["name"])

    def chain(name: str) -> tuple[str, ...]:
        walk: list[str] = []
        cursor = name
        while cursor and cursor in by_name and cursor not in walk:
            walk.append(cursor)
            cursor = by_name[cursor]["parent"]
        return tuple(walk)

    ancestry = {name: chain(name) for name in by_name}
    effective = {}
    for name, walk in ancestry.items():
        names = set(root_runtime)
        for step in walk:
            names |= set(by_name[step]["ownAttributes"])
            names |= observed_runtime.get(step, set())
        effective[name] = tuple(sorted(names))

    return {
        "byName": by_name,
        "ancestry": ancestry,
        "effective": effective,
        "attributes": {item["name"]: item for item in attributes["attributes"]},
    }


@lru_cache(maxsize=1)
def _rules() -> dict:
    heuristics = yaml.safe_load(HEURISTICS_YAML.read_text(encoding="utf-8"))
    thresholds = yaml.safe_load(THRESHOLDS_YAML.read_text(encoding="utf-8"))

    definitions = {entry["id"]: entry for entry in heuristics["heuristics"]}
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
    return {"definitions": definitions, "symbols": symbols, "head": heuristics.get("head") or {}}


# --------------------------------------------------------------- class tree

def class_record(name: str) -> dict | None:
    return _classes()["byName"].get(name)


def ancestry(name: str) -> tuple[str, ...]:
    """The class and every ancestor of it, nearest first.  Empty if unknown."""
    return _classes()["ancestry"].get(name, ())


def in_family(name: str, family: str) -> bool:
    """Is *name* the family class itself or a descendant of it?

    NX's own tree answers this.  ``STEP`` and ``HOLES`` are classes NX
    declares, not groupings we invented, which is why a rule may name one.
    """
    return family in _classes()["ancestry"].get(name, ())


def effective_attributes(name: str) -> tuple[str, ...]:
    """Every attribute NX declares on this class, inherited ones included."""
    return _classes()["effective"].get(name, ())


def declares(name: str, attribute: str) -> bool:
    """Does this class declare an attribute of this name (globs allowed)?

    ``FACES_*`` is a glob because that is how the projection states a gap it
    cannot enumerate; a plain name matches itself.
    """
    return any(fnmatchcase(candidate, attribute)
               for candidate in _classes()["effective"].get(name, ()))


def attribute_record(name: str) -> dict | None:
    return _classes()["attributes"].get(name)


@lru_cache(maxsize=1)
def derived_attributes() -> dict[str, dict]:
    """``{name: record}`` of the ``nxd:`` namespace — what the pipeline may
    measure about a feature, as opposed to what NX said about it.

    Declared per quantity rather than per class: which instances carry one is
    decided by the rules that read it (``used_by`` → those heuristics'
    ``applies_to``), so the class tree never has to be annotated with names NX
    does not know.
    """
    payload = yaml.safe_load(DERIVED_YAML.read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in payload["attributes"]}


def derived_domain(name: str) -> tuple[str, ...]:
    """The ``applies_to`` patterns of every heuristic reading one derived
    quantity — the set of classes a measurement is owed on."""
    record = derived_attributes().get(name) or {}
    patterns: list[str] = []
    for h_id in record.get("used_by") or []:
        for pattern in (definitions().get(h_id) or {}).get("applies_to") or []:
            if pattern not in patterns:
                patterns.append(pattern)
    return tuple(patterns)


def reachable_classes() -> tuple[str, ...]:
    return tuple(sorted(name for name, entry in _classes()["byName"].items()
                        if entry.get("reachable")))


# ------------------------------------------------------------ H definitions

def definitions() -> dict[str, dict]:
    return _rules()["definitions"]


def symbols() -> dict[str, dict]:
    return _rules()["symbols"]


def head() -> dict:
    return _rules()["head"]


def source_version() -> dict:
    return {
        "heuristicsYamlSha256": digest(HEURISTICS_YAML),
        "thresholdsYamlSha256": digest(THRESHOLDS_YAML),
        "classesYamlSha256": digest(CLASSES_YAML),
        "derivedYamlSha256": digest(DERIVED_YAML),
    }
