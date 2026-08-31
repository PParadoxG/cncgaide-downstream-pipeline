"""Central palette reader for exact ontology feature names."""

from __future__ import annotations

import json
from pathlib import Path


_FALLBACK = "#00BFFF"


def load_palette(project_root: str | Path) -> dict:
    path = Path(project_root) / "config" / "palette.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data.setdefault("features", {})
    data.setdefault("edges", {})
    return data


def feature_color(palette: dict, feature_type: str) -> str:
    return str(palette.get("features", {}).get(feature_type, _FALLBACK))


def edge_color(palette: dict, edge_type: str) -> str:
    return str(palette.get("edges", {}).get(edge_type, "#888888"))
