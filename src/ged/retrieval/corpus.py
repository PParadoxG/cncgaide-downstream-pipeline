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


def run_label(run_root: Path) -> str:
    """``runs/<tag>`` - how the project names a run, on any machine.

    Taken from the two last segments of the path rather than from a project
    root, because a corpus is handed a run directory and is not told where the
    checkout is.  A run somewhere else is named by those two segments too,
    which is still a name and still not an absolute path.
    """
    root = Path(run_root)
    parent = root.parent.name
    return f"{parent}/{root.name}" if parent else root.name


@dataclass
class Corpus:
    run_root: Path
    line: str = CORPUS_LINE
    graphs: dict[str, GedGraph] = field(default_factory=dict)
    audits: dict[str, ProjectionAudit] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)
    #: Parts this line did not build, with the reason - so "not in the corpus"
    #: is a statement rather than an absence (node 26 F-02).
    not_built: dict[str, str] = field(default_factory=dict)

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
            # The run as the project names it (``runs/<tag>``), never this
            # machine's absolute path: the audit block is copied verbatim into
            # ``GED_VARIANT_REPORT.md`` (node 26 P-11).
            "runRoot": run_label(self.run_root),
            "line": self.line,
            "loaded": len(self.graphs),
            "failed": dict(self.failed),
            "notBuilt": dict(self.not_built),
            "unknownLeafTypes": sorted({t for a in self.audits.values() for t in a.unknown_leaf_types}),
            "unknownRelationTypes": sorted({t for a in self.audits.values() for t in a.unknown_relation_types}),
            "attributesAbsent": sum(len(a.attributes_absent) for a in self.audits.values()),
            "attributesNull": sum(len(a.attributes_null) for a in self.audits.values()),
            "locatedOnUnresolved": sum(len(a.located_on_unresolved) for a in self.audits.values()),
            "areaSource": sorted({s for a in self.audits.values() for s in a.area_source.values()}),
        }


def _part_dirs(run_root: Path, line: str = CORPUS_LINE,
               refused: dict[str, str] | None = None) -> list[Path]:
    """The folders of the parts *this line* built successfully.

    The verdict is read from each part's own ``record.json``, block ``line``
    (node 26 F-02: this used to read ``manifest.json``'s ``statuses``, which
    ``store/index.py`` folds from the **open-source** block only - so an nx
    corpus was the parts occ liked, and an all-failed run silently dropped
    the filter and took whatever directories existed).  ``SimilarityService.
    ref()`` answers the same question for the service; this is the corpus
    half of it, kept here because the corpus walks folders and the service
    walks ids.

    A run written before the store existed has no record.  Such a part is
    still loaded - that path predates the verdict and refusing it would
    break old runs - but it is *named* in ``refused`` rather than passing
    for a part that was checked.
    """
    parts_dir = run_root / "parts"
    if not parts_dir.is_dir():
        return []
    chosen: list[Path] = []
    for part_dir in sorted(parts_dir.iterdir()):
        if not part_dir.is_dir() or part_dir.name.startswith("."):
            continue
        line_dir = part_dir / line
        if not line_dir.is_dir():
            continue
        status, reason = _line_status(part_dir, line)
        if status in ("PASS", "UNRECORDED"):
            chosen.append(line_dir)
        elif refused is not None:
            refused[part_dir.name] = reason
    return chosen


def _line_status(part_dir: Path, line: str) -> tuple[str | None, str]:
    """``(status, why not)`` for one line of one part, from its own record."""
    record_path = part_dir / "record.json"
    if not record_path.is_file():
        # No record: a run older than the store.  Its result.json is all the
        # evidence there is, so it counts as built - and says so.
        if (part_dir / line / "result.json").is_file():
            return "UNRECORDED", ""
        return None, f"no record and no result for line {line}"
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, f"record unreadable: {type(error).__name__}"
    block = (record.get("lines") or {}).get(line) if isinstance(record, dict) else None
    if not isinstance(block, dict):
        return None, f"record has no block for line {line}"
    status = block.get("status")
    if status == "PASS":
        return "PASS", ""
    return status, f"line {line} is {status or 'unrecorded'}"


def load_corpus(run_root: str | Path, *, ids: list[str] | None = None,
                line: str = CORPUS_LINE) -> Corpus:
    root = Path(run_root)
    corpus = Corpus(run_root=root, line=line)
    representation = for_line(line)
    wanted = set(ids) if ids else None
    refused: dict[str, str] = {}
    for line_dir in _part_dirs(root, line, refused):
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
    # Why a part is not in this corpus is part of what the corpus says.
    for part_id, reason in sorted(refused.items()):
        if wanted is not None and part_id not in wanted:
            continue
        corpus.not_built[part_id] = reason
    return corpus
