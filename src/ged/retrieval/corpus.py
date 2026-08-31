"""Load every PASS part of a ``runs/<tag>`` into projected graphs, per line.

A distance is only meaningful between graphs the same recogniser produced, so
a corpus is **one line's**.  Node 21' gave retrieval the line dimension the
module docstring used to promise: ``load_corpus(root, line=...)`` reads that
line's folders and projects them with that line's representation, and
``CORPUS_LINE`` remains the default so every existing caller keeps reading the
open-source line.  Two corpora are never merged and their distances are never
compared - the two ontologies do not share a scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator

from contract.registry import LINE_OCC
from ged.contract.projection import ProjectionAudit, load_part
from ged.contract.representation import for_line
from ged.contract.schema import GedGraph

CORPUS_LINE = LINE_OCC


@dataclass
class Corpus:
    run_root: Path
    line: str = CORPUS_LINE
    graphs: dict[str, GedGraph] = field(default_factory=dict)
    audits: dict[str, ProjectionAudit] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.graphs)

    def __iter__(self) -> Iterator[GedGraph]:
        return iter(self.graphs.values())

    def __getitem__(self, graph_id: str) -> GedGraph:
        return self.graphs[graph_id]

    def ids(self) -> list[str]:
        return sorted(self.graphs)

    def audit_summary(self) -> dict[str, Any]:
        return {
            "runRoot": str(self.run_root),
            "line": self.line,
            "loaded": len(self.graphs),
            "failed": dict(self.failed),
            "unknownLeafTypes": sorted({t for a in self.audits.values() for t in a.unknown_leaf_types}),
            "unknownRelationTypes": sorted({t for a in self.audits.values() for t in a.unknown_relation_types}),
            "attributesAbsent": sum(len(a.attributes_absent) for a in self.audits.values()),
            "attributesNull": sum(len(a.attributes_null) for a in self.audits.values()),
            "locatedOnUnresolved": sum(len(a.located_on_unresolved) for a in self.audits.values()),
            "areaSource": sorted({s for a in self.audits.values() for s in a.area_source.values()}),
        }


def _part_dirs(run_root: Path, line: str = CORPUS_LINE) -> list[Path]:
    manifest_path = run_root / "manifest.json"
    parts_dir = run_root / "parts"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        passed = {
            str(item["partId"])
            for item in manifest.get("statuses", [])
            if isinstance(item, dict) and item.get("status") == "PASS"
        }
        if passed:
            return sorted((parts_dir / part_id / line) for part_id in passed
                          if (parts_dir / part_id / line).is_dir())
    if not parts_dir.is_dir():
        return []
    return sorted(p / line for p in parts_dir.iterdir() if (p / line).is_dir())


def load_corpus(run_root: str | Path, *, ids: list[str] | None = None,
                line: str = CORPUS_LINE) -> Corpus:
    root = Path(run_root)
    corpus = Corpus(run_root=root, line=line)
    representation = for_line(line)
    wanted = set(ids) if ids else None
    for line_dir in _part_dirs(root, line):
        part_id = line_dir.parent.name
        if wanted is not None and part_id not in wanted:
            continue
        audit = ProjectionAudit()
        try:
            corpus.graphs[part_id] = load_part(line_dir, graph_id=part_id, audit=audit,
                                               representation=representation)
            corpus.audits[part_id] = audit
        except Exception as exc:  # noqa: BLE001 - report, don't abort the corpus
            corpus.failed[part_id] = f"{type(exc).__name__}: {exc}"
    return corpus
