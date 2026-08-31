"""Named cost presets, adjustable scalar weights and the configuration hash.

A *configuration* is the eight scalars ``CostConfig.describe()`` reports
(``W_SEM … C_INS``); a *preset* is a named dictionary of them.  Everything
else in ``CostConfig`` — the type matrix, the attribute, edge and fixed-node
callbacks — is the frozen representation / cost code and is not adjustable
from here.  The hash is computed over the **values only** (never the name),
so a custom weight set that happens to equal a preset shares its cache, and
"reset to default" lands on exactly the preset hash again.

Adding a preset = adding one entry to ``PRESET_WEIGHTS``; the CLI, the
endpoints and the page pick it up from ``list_presets()``.

The preset table is **shared by both representations** (node 21'): a weight
says how much a kind of difference matters, and that question is the same in
either ontology.  ``representation=`` selects which tables the resulting
config prices with; the hash stays the eight values, so the same slider
position reads the same hash on both lines - what keeps the two apart is the
per-representation semantics stamp on the cache, not the hash.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from typing import Any, Mapping

from ged.contract.representation import Representation, V5_1

from .config import CostConfig
from .default import default_config
from .fixed_cost import fixed_cost_upper_bound, make_fixed_node_cost


DEFAULT_PRESET = "v0_default"
CUSTOM_NAME = "custom"

# Canonical weight keys (the ``describe()`` names) -> the CostConfig field.
WEIGHT_FIELDS: dict[str, str] = {
    "W_SEM": "w_sem",
    "W_GEOM": "w_geom",
    "W_REL": "w_rel",
    "W_SCALE": "w_scale",
    "W_RES": "w_res",
    "W_LOCAL": "w_local",
    "C_DEL": "c_del",
    "C_INS": "c_ins",
}
WEIGHT_KEYS: tuple[str, ...] = tuple(WEIGHT_FIELDS)

# Named presets as weight overrides on top of ``default_config()``.
PRESET_WEIGHTS: dict[str, dict[str, float]] = {
    "v0_default": {},
    "v0_no_local": {"W_LOCAL": 0.0},
}

# Display contract for the eight scalars: label, range, step, the one-line
# explanation and whether the key belongs to the simple (four-slider) mode.
# Ranges are chosen so that the defaults sit at different positions of
# their tracks and leave room on both sides.  No page code names a key.
WEIGHT_SPECS: dict[str, dict[str, Any]] = {
    "W_SEM": {"label": "type", "min": 0.0, "max": 2.5, "step": 0.05, "simple": True,
              "hint": "mismatch between leaf types"},
    "W_GEOM": {"label": "dimensions", "min": 0.0, "max": 1.0, "step": 0.01, "simple": True,
               "hint": "attribute differences of matched features"},
    "W_REL": {"label": "relations", "min": 0.0, "max": 1.0, "step": 0.01, "simple": True,
              "hint": "changed, deleted or inserted relations"},
    "W_SCALE": {"label": "size", "min": 0.0, "max": 0.4, "step": 0.01, "simple": True,
                "hint": "overall size difference (L_ref ratio)"},
    "W_RES": {"label": "background", "min": 0.0, "max": 0.4, "step": 0.01, "simple": False,
              "hint": "surface background and unclaimed faces"},
    "W_LOCAL": {"label": "local structure", "min": 0.0, "max": 2.0, "step": 0.1, "simple": False,
                "hint": "incident relations steer the matching (0 = off)"},
    "C_DEL": {"label": "delete", "min": 0.0, "max": 1.0, "step": 0.01, "simple": False,
              "hint": "query feature without counterpart"},
    "C_INS": {"label": "insert", "min": 0.0, "max": 1.0, "step": 0.01, "simple": False,
              "hint": "candidate feature without counterpart"},
}


# ---------------------------------------------------------------- weights

def normalize_key(key: object) -> str:
    """``w_rel`` / ``W_REL`` / `` w_rel `` -> ``W_REL``; unknown -> KeyError."""
    canonical = str(key).strip().upper().replace("-", "_")
    if canonical not in WEIGHT_FIELDS:
        raise KeyError(f"unknown weight {key!r}; known: {', '.join(WEIGHT_KEYS)}")
    return canonical


def validate_weights(overrides: Mapping[str, object] | None) -> dict[str, float]:
    """Canonical keys, finite non-negative floats.  Unknown key -> KeyError,
    bad value -> ValueError."""
    values: dict[str, float] = {}
    for key, raw in (overrides or {}).items():
        canonical = normalize_key(key)
        if isinstance(raw, bool):
            raise ValueError(f"{canonical}: expected a number, got a boolean")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{canonical}: expected a number, got {raw!r}") from None
        if not math.isfinite(value):
            raise ValueError(f"{canonical}: must be finite, got {raw!r}")
        if value < 0:
            raise ValueError(f"{canonical}: must be >= 0, got {value}")
        values[canonical] = value
    return values


def weights_of(config: CostConfig) -> dict[str, float]:
    return {key: float(getattr(config, field)) for key, field in WEIGHT_FIELDS.items()}


def config_hash(config: CostConfig | Mapping[str, object]) -> str:
    """First 12 hex digits of sha256 over the eight values (never the name)."""
    weights = weights_of(config) if isinstance(config, CostConfig) else validate_weights(config)
    if set(weights) != set(WEIGHT_KEYS):
        raise ValueError("config hash needs all eight weights")
    canonical = {key: weights[key] for key in WEIGHT_KEYS}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def make_config(name: str, *, base: CostConfig | None = None,
                representation: Representation | None = None, **overrides: object) -> CostConfig:
    """Copy ``base`` (default: ``default_config()``) with some of the eight
    scalars overridden.  A new ``W_RES`` rebuilds the fixed-node cost and its
    upper bound, which are derived from it - and from the representation's
    own aggregate tables, so rebuilding one does not silently fall back to
    v5.1's background layer."""
    base = base or default_config(representation=representation or V5_1)
    values = validate_weights(overrides)
    fields: dict[str, object] = {WEIGHT_FIELDS[key]: value for key, value in values.items()}
    if "w_res" in fields:
        w_res = float(fields["w_res"])
        shares = base.representation.fixed_budget_shares
        fields["fixed_node_cost"] = make_fixed_node_cost(
            w_res, shares=shares,
            attribute_weights=base.representation.fixed_attribute_weights)
        fields["fixed_cost_upper_bound"] = fixed_cost_upper_bound(w_res, shares=shares)
    return dataclasses.replace(base, name=name, **fields)


# ---------------------------------------------------------------- presets

def get_config(name: str, *, representation: Representation | None = None) -> CostConfig:
    """The preset by name; unknown -> KeyError."""
    try:
        overrides = PRESET_WEIGHTS[name]
    except KeyError:
        raise KeyError(f"unknown cost preset {name!r}; known: {', '.join(PRESET_WEIGHTS)}") from None
    return make_config(name, representation=representation, **overrides)


def preset_name_for(weights: Mapping[str, float]) -> str | None:
    """The preset whose values equal ``weights`` exactly, else None."""
    wanted = config_hash(weights)
    for name in PRESET_WEIGHTS:
        if config_hash(get_config(name)) == wanted:
            return name
    return None


def describe_config(config: CostConfig, name: str | None = None) -> dict[str, Any]:
    weights = weights_of(config)
    preset = preset_name_for(weights)
    return {
        "name": name or preset or CUSTOM_NAME,
        "hash": config_hash(weights),
        "weights": weights,
        "preset": preset is not None,
    }


def list_presets() -> list[dict[str, Any]]:
    return [describe_config(get_config(name), name) for name in PRESET_WEIGHTS]


def resolve_config(
    name: str | None = None, weights: Mapping[str, object] | None = None,
    *, representation: Representation | None = None,
) -> tuple[CostConfig, dict[str, Any]]:
    """Base preset (default ``v0_default``) with optional overrides.

    Returns the config and its description ``{name, hash, weights, preset}``:
    the name is the preset's when the effective values equal a preset, else
    ``"custom"``.  Unknown preset -> KeyError, bad weight -> ValueError /
    KeyError (see ``validate_weights``).
    """
    base = get_config(name or DEFAULT_PRESET, representation=representation)
    overrides = validate_weights(weights)
    if not overrides:
        return base, describe_config(base, base.name)
    effective = dict(weights_of(base))
    effective.update(overrides)
    preset = preset_name_for(effective)
    config = make_config(preset or CUSTOM_NAME, base=base, **overrides)
    return config, describe_config(config, preset)


def parse_weights(text: str | None) -> dict[str, float]:
    """``'{"W_REL": 0.8}'`` or ``'W_REL=0.8,C_DEL=1'`` -> validated dict."""
    if text is None:
        return {}
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except ValueError as error:
            raise ValueError(f"weights is not valid JSON: {error}") from None
        if not isinstance(payload, dict):
            raise ValueError("weights JSON must be an object")
        return validate_weights(payload)
    pairs: dict[str, object] = {}
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"weights: expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        pairs[key.strip()] = value.strip()
    return validate_weights(pairs)


def cli_weights(weights: Mapping[str, float]) -> str:
    """The ``--weights`` argument that reproduces ``weights``."""
    return ",".join(f"{key}={weights[key]:g}" for key in WEIGHT_KEYS if key in weights)
