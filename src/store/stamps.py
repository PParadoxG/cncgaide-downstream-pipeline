"""Change detection: STEP fingerprints, per-layer code stamps, two verdicts.

The two verdicts are deliberately separate, and the separation is the design:

* :func:`light` answers *has the source STEP changed*, and nothing else.  It
  is what the page shows next to a part, so it stays about the thing the user
  put in the input directory.
* :func:`plan_line` answers *what does this sync owe this line of this part*,
  which additionally depends on whether the code that line reads has moved.

A part whose STEP is untouched but whose rules moved therefore shows a calm
light and still gets rebuilt; the page says so once, at run level
(``rulesChanged``), rather than reporting a code edit as a change to every
part's source.

A *layer stamp* is a content hash over a fixed set of source files.  The table
below is the whole definition - it is deliberately a table, so "what makes a
part stale" can be read in one place and checked by a test.  ``LINE_LAYERS``
is the second half of it: which layers each line actually reads.  The NX line
does not read ``parse`` or ``aagnet``, so re-parsing STEP with a new OCC or
swapping AAGNet weights leaves every NX folder alone.

Three choices are worth stating:

* ``contract`` comes first because both sides of the seam depend on it: a
  change to the part model or to the kernel interface can change what any
  later layer computes, so it invalidates everything.  It is its own layer
  rather than part of ``core`` so the freshness chip can say *which* changed.
* ``viewer`` covers only the modules that *write per-part artifacts*
  (graph_view / id_map / palette / viewer_data + the palette file).  The
  server, the job runner and the instance registry live next to them but
  produce nothing under ``parts/``; counting them would invalidate every part
  each time the server is edited.  ``viewer_index.py`` is run-level and the
  index is rebuilt on every sync anyway.
* ``aagnet`` includes its vendored snapshot *and* the weights - a new weight
  file is exactly the kind of change that must rebuild every part.  The
  ``vendor/`` / ``web/`` exclusions the viewer needs therefore do not apply
  here; every layer names its own roots.

Hashes are content hashes and cached per file by ``(size, mtime_ns)``, so a
server polling the store every few seconds re-stats a few hundred files and
re-hashes only the ones that changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import nx_location
import occwl_provenance

# layer -> list of (relative path, recursive) or (relative path, recursive,
# excluded child dirs).  A non-recursive entry is one file.  Order inside a
# layer does not matter: entries are sorted before hashing.
LAYERS: dict[str, tuple[tuple, ...]] = {
    "contract": (("src/contract", True),),
    "core": (("src/core", True),),
    "parse": (("src/occline/parse", True), ("src/occline/kernel", True)),
    # ``occwl_provenance`` is listed here rather than left outside every layer
    # (the way ``nx_location`` is) because it decides three keys of
    # ``aagnet_evidence.json``: what it says is part of what this layer wrote.
    "aagnet": (("src/occline/aagnet", True), ("src/occwl_provenance.py", False)),
    # Pure B-Rep topology and arithmetic, read by both lines: the rim walk the
    # NX LOCATED_ON proof uses is the same code the v5.1 hole modules use.  It
    # is its own layer rather than part of ``features`` so the freshness chip
    # can say which of the two moved - and so that one day ``features`` can
    # become occ-only without taking the shared walk with it.
    "topology": (("src/topology", True),),
    "features": (("src/features", True),),
    # The relation and heuristic layers are split machine / content.  The two
    # packages' own modules - the executor, the edge structure, the four-value
    # gate, the witnesses that name no class - are read by every line.  The
    # rules are read only by the line whose nodes they can talk about, and are
    # filed by *ontology* rather than by kernel line: which vocabulary a rule
    # is written in is a different question from who filled the PartModel.
    "relations": (("src/relations", True, ("v51", "nx")),),
    "relations_v51": (("src/relations/v51", True),),
    "relations_nx": (
        ("src/relations/nx", True),
        ("src/tbox", True),
        ("ontology/nx_v1/source/relations.yaml", False),
        ("ontology/nx_v1/source/classes.yaml", False),
        ("ontology/nx_v1/source/attributes.yaml", False),
    ),
    "heuristics": (("src/heuristics", True, ("v51", "nx")),),
    "heuristics_v51": (
        ("src/heuristics/v51", True),
        ("ontology/v5.1/source/heuristics.yaml", False),
        ("ontology/v5.1/source/thresholds.yaml", False),
        ("ontology/v5.1/source/classes.yaml", False),
    ),
    "heuristics_nx": (
        ("src/heuristics/nx", True),
        ("src/tbox", True),
        ("ontology/nx_v1/source/heuristics.yaml", False),
        ("ontology/nx_v1/source/thresholds.yaml", False),
        ("ontology/nx_v1/source/classes.yaml", False),
        ("ontology/nx_v1/source/attributes.yaml", False),
        # The nxd: vocabulary the tier-E gate resolves derived names against.
        # Editing it re-runs the verdicts at seconds-tier; the measurements
        # themselves are repaid on the part's next full rebuild (nx layer).
        ("ontology/nx_v1/source/derived.yaml", False),
    ),
    # Both NX dialects read the ontology's classes and attributes - the tree
    # that says which family a class belongs to and which attributes it
    # carries - so the loader and those two files sit in both layers.  A
    # relation change forces a rebuild and a heuristic change does not, and
    # keeping them apart is what preserves the seconds-level path.
    "nx": (("src/nxline", True),),
    "viewer": (
        ("src/visualization/graph_view.py", False),
        ("src/visualization/id_map.py", False),
        ("src/visualization/palette.py", False),
        ("src/visualization/viewer_data.py", False),
        ("config/palette.json", False),
    ),
}

# Files outside the repository that a layer's meaning depends on, hashed into
# its stamp when present (an absent file hashes as "absent", so installing or
# removing the tool also changes the stamp).  ``nx`` covers the NX feature
# schema - the field catalogue every extraction is driven by - which also
# moves whenever the NX install itself changes version.  Where that install
# is comes from ``nx_location``, the one place both this module and
# ``nxline.config`` ask; this side may not import the kernel line, which is
# why the answer lives outside both.
EXTERNAL_LAYER_FILES: dict[str, tuple[str, ...]] = {
    "nx": (str(nx_location.nx_root() / "UGII" / "customization_base.xml"),),
}

# The same idea for a dependency that is installed rather than sitting at a
# path: a callable returning one canonical string, hashed into the layer's
# stamp.  ``aagnet`` covers occwl, whose graph extraction is the input to every
# per-face score - a different occwl means different evidence, and until node
# 28D nothing said so (node 27X left it open when occwl stopped being
# vendored).  The callable reads installer metadata; it never imports occwl.
EXTERNAL_LAYER_VALUES: dict[str, tuple[tuple[str, object], ...]] = {
    "aagnet": (("occwl", occwl_provenance.stamp_value),),
}

# Which layers each line reads.  Guarded by a test against LAYERS and against
# the line registry, so a new layer cannot be added without deciding who reads
# it and a new line cannot be added without a row here.
LINE_LAYERS: dict[str, frozenset[str]] = {
    "occ": frozenset({"contract", "core", "parse", "aagnet", "topology", "features",
                      "relations", "relations_v51",
                      "heuristics", "heuristics_v51", "viewer"}),
    "nx": frozenset({"contract", "core", "nx", "topology", "features",
                     "relations", "relations_nx",
                     "heuristics", "heuristics_nx", "viewer"}),
}

# Layers whose change does not force a full rebuild: a heuristic layer
# triggers a verdict recompute from the stored PartModel and result - no STEP
# parsing, no AAGNet, no NX session, which is why it works for either line
# unchanged.  A *relation* layer is not among them: edges live in result.json
# and only a rebuild regenerates them, which is the whole reason the rules of
# the two layers are stamped apart rather than in one "rules" layer.
HEURISTICS_LAYER = "heuristics"
PARTIAL_LAYERS = frozenset({"heuristics", "heuristics_v51", "heuristics_nx"})

EXCLUDED_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo", ".md"})

# Files that sit inside a layer's roots but produce nothing under ``parts/``:
# offline tools a human runs, never something a build reads.  Named one by one
# and guarded by a test, because a silent exclusion is a file whose changes
# nobody accounts for - and because the alternative (editing a diff tool and
# watching every NX folder go stale) is exactly what happened before this
# list existed.
EXCLUDED_FILES = frozenset({
    "src/nxline/partmodel_diff.py",  # offline occ-vs-nx PartModel comparison
})

_CHUNK = 1 << 20
_cache_lock = threading.Lock()
# Upper bound on remembered files.  The layers cover a few hundred sources, so
# this is roughly four times the working set: big enough that a long-lived
# viewer process never evicts anything it is about to ask for again, small
# enough that a mistake somewhere else cannot turn the cache into a leak.
_FILE_CACHE_MAX = 4096
# absolute path -> (size, mtime_ns, sha256), least-recently-used first
_file_cache: "OrderedDict[str, tuple[int, int, str]]" = OrderedDict()


# ------------------------------------------------------------------ hashing

def clear_file_cache() -> None:
    """Forget every remembered hash.

    Named rather than left to ``_file_cache.clear()``: a process that wants a
    cold read (a test measuring the cold path, a tool that just rewrote files
    under another process's nose) should not have to reach into a private
    dict to get one.
    """
    with _cache_lock:
        _file_cache.clear()


def file_sha256(path: Path) -> str:
    """Content hash of one file, cached by size + mtime.

    The hit criterion is *both* the size and the nanosecond mtime, so a
    rewrite that keeps both is served from the cache - the one gap this cache
    has, and out of reach in practice (NTFS resolves ``st_mtime_ns`` to 100 ns
    and two writes to one path within one tick would also have to be the same
    length).  Bounded at ``_FILE_CACHE_MAX`` entries, least-recently-used
    evicted first, so a long-lived viewer server cannot grow one entry per
    file it ever looked at.
    """
    stat = path.stat()
    key = str(path)
    with _cache_lock:
        cached = _file_cache.get(key)
        if cached is not None and cached[0] == stat.st_size and cached[1] == stat.st_mtime_ns:
            _file_cache.move_to_end(key)
            return cached[2]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    with _cache_lock:
        _file_cache[key] = (stat.st_size, stat.st_mtime_ns, value)
        _file_cache.move_to_end(key)
        while len(_file_cache) > _FILE_CACHE_MAX:
            _file_cache.popitem(last=False)
    return value


def fingerprint(step_path: str | Path) -> dict:
    """``{"sha256", "size", "mtime"}`` of a source STEP file."""
    path = Path(step_path)
    stat = path.stat()
    return {"sha256": file_sha256(path), "size": int(stat.st_size), "mtime": float(stat.st_mtime)}


# Never part of the artifact digest.  The record lives above the line folder,
# so a copy inside one would be a stray; excluding it by name keeps a stray
# from quietly changing a digest that is supposed to be about artifacts.
DIGEST_EXCLUDED = frozenset({"record.json"})

# Keys that record *when* a build ran rather than what it produced.  They are
# genuine provenance and stay in the artifacts; they are simply not what the
# digest asks about, and leaving them in would make every rebuild look
# non-deterministic for no reason.  Named, because a silently normalised field
# is a field nobody can be held to.
VOLATILE_KEYS = frozenset({
    "extraction_timestamp",  # PartMetadata: wall clock of the parse
    "_elapsed_sec",          # AAGNet evidence: how long inference took
})


def _strip_volatile(value):
    if isinstance(value, dict):
        return {key: _strip_volatile(item) for key, item in value.items()
                if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(item) for item in value]
    return value


def _content_hash(path: Path) -> str:
    """What this file *says*, independent of when it was written.

    JSON is compared as data - parsed, stripped of the volatile keys above and
    re-serialised canonically - so the digest answers "did the pipeline compute
    something different", not "was this file laid out differently".  How the
    writers lay a file out is code, and code already invalidates through the
    layer stamps.  Anything that is not readable JSON is hashed as bytes.
    """
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return file_sha256(path)
        canonical = json.dumps(_strip_volatile(payload), sort_keys=True,
                               separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return file_sha256(path)


def artifact_digest(line_root: str | Path) -> str | None:
    """One hash over everything a line wrote for a part.

    The point is regression detection, not identity: rebuild a part twice and
    the digest must not move.  When it does, hashing the folder file by file
    says which artifact stopped being deterministic, which is the only useful
    thing to know at that moment.
    """
    root = Path(line_root)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.is_file() or child.name in DIGEST_EXCLUDED or child.name.endswith(".tmp"):
            continue
        digest.update(child.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_content_hash(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def layer_files(project_root: str | Path, layer: str) -> list[Path]:
    """Every repository file a layer stamp covers, sorted by relative path."""
    root = Path(project_root)
    files: list[Path] = []
    for entry in LAYERS[layer]:
        relative, recursive = entry[0], entry[1]
        entry_excludes = frozenset(entry[2]) if len(entry) > 2 else frozenset()
        base = root / relative
        if not recursive:
            if base.is_file():
                files.append(base)
            continue
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                name for name in dirnames
                if name not in EXCLUDED_DIRS
                and not (Path(dirpath) == base and name in entry_excludes)
            )
            for name in filenames:
                candidate = Path(dirpath) / name
                if candidate.suffix.lower() in EXCLUDED_SUFFIXES:
                    continue
                if str(candidate.relative_to(root)).replace("\\", "/") in EXCLUDED_FILES:
                    continue
                files.append(candidate)
    return sorted(set(files), key=lambda p: str(p.relative_to(root)).replace("\\", "/").lower())


def layer_stamp(project_root: str | Path, layer: str) -> str:
    root = Path(project_root)
    digest = hashlib.sha256()
    for path in layer_files(root, layer):
        relative = str(path.relative_to(root)).replace("\\", "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    for external in EXTERNAL_LAYER_FILES.get(layer, ()):
        path = Path(external)
        digest.update(("external:" + external).encode("utf-8"))
        digest.update(b"\0")
        digest.update((file_sha256(path) if path.is_file() else "absent").encode("ascii"))
        digest.update(b"\n")
    for name, reader in EXTERNAL_LAYER_VALUES.get(layer, ()):
        digest.update(("value:" + name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(reader()).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# The last full answer, per project root: (monotonic time, stamps).  Only one
# root is ever live in a process, so this is a single slot rather than a cache
# that could grow.
_stamps_memo: dict[str, tuple[float, dict[str, str]]] = {}


def layer_stamps(project_root: str | Path, *, max_age: float = 0.0) -> dict[str, str]:
    """One content stamp per layer, in table order.

    A full answer walks and stats every file the table covers, which is the
    only way to notice an edit in place - a directory mtime does not move when
    a file's content changes.  ``max_age`` therefore defaults to 0: everything
    that *decides* something (``plan``, the sync) pays the full walk and can
    never act on a stale table.  A caller that only *displays* the answer may
    pass a window and reuse one computed that recently; the worst that can
    happen there is a "behind current rules" chip lagging by that many
    seconds, which is what ``store_status`` does for the viewer's poll.
    """
    key = str(Path(project_root))
    if max_age > 0:
        cached = _stamps_memo.get(key)
        if cached is not None and (time.monotonic() - cached[0]) <= max_age:
            return dict(cached[1])
    stamps = {layer: layer_stamp(project_root, layer) for layer in LAYERS}
    _stamps_memo.clear()
    _stamps_memo[key] = (time.monotonic(), dict(stamps))
    return stamps


def clear_stamps_memo() -> None:
    """Forget the last full answer.  Companion to :func:`clear_file_cache`."""
    _stamps_memo.clear()


# ------------------------------------------------------------------- lights

# What the page shows for a part.  Four values, all about the source STEP.
NEW = "new"                        # a STEP nobody has built yet
FRESH = "fresh"                    # built, and the STEP has not moved since
CHANGED = "changed"                # the STEP's content differs from what was built
SOURCE_MISSING = "source_missing"  # the STEP is gone from the input directory

LIGHTS = (NEW, FRESH, CHANGED, SOURCE_MISSING)

# A line that was not built for a reason that was never about this part - no
# licence, no vendor install.  Distinct from FAIL on purpose: FAIL means the
# part broke and must not be retried until something changes, NOT_BUILT means
# the environment was not ready and the next sync must try again.
NOT_BUILT = "NOT_BUILT"


@dataclass
class Light:
    state: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"state": self.state, "reasons": list(self.reasons)}


def light(record: dict | None, step_path: str | Path | None) -> Light:
    """The one status a part shows: has its source STEP changed?

    Deliberately blind to code stamps and to what either line managed to
    build.  Those are real, they are just not what this light is about.
    """
    if step_path is None:
        return Light(SOURCE_MISSING, ["source STEP not found under the input path"])
    if record is None or not (record.get("lines") or {}):
        return Light(NEW, ["STEP present, not built yet"])
    current = fingerprint(step_path)["sha256"]
    if (record.get("fingerprint") or {}).get("sha256") != current:
        return Light(CHANGED, ["source STEP content changed"])
    return Light(FRESH, [])


# ------------------------------------------------------------- line actions

# What a sync owes one line of one part.
BUILD = "build"                      # produce this line's folder from scratch
HEURISTICS_ONLY = "heuristics_only"  # re-run verdicts from the stored artifacts
SKIP = "skip"                        # nothing to do
FAILED = "failed"                    # it failed and nothing changed; do not retry

ACTIONS = (BUILD, HEURISTICS_ONLY, SKIP, FAILED)


@dataclass
class LineAction:
    line: str
    action: str
    reasons: list[str] = field(default_factory=list)
    staleLayers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"line": self.line, "action": self.action,
                "reasons": list(self.reasons), "staleLayers": list(self.staleLayers)}


def stale_layers(block: dict | None, line: str, current_stamps: dict[str, str]) -> list[str]:
    """Layers *this line reads* whose code moved since the line was built.

    A block with no stamps was never built - ``NOT_BUILT`` carries a reason
    and nothing else - and something never built is not "behind": it is
    missing, which the block's own status already says.  Reporting every
    watched layer for it would put "built under older rules" on a line that
    was never built under any.
    """
    if not isinstance(block, dict):
        return []
    built_under = block.get("stamps") or {}
    if not built_under:
        return []
    watched = LINE_LAYERS.get(line, frozenset())
    return [layer for layer in LAYERS
            if layer in watched and built_under.get(layer) != current_stamps.get(layer)]


def rules_changed(record: dict | None, current_stamps: dict[str, str]) -> list[str]:
    """Every layer any built line of this part is behind on, deduplicated.

    This is what turns the page's lights into the "rules moved" colour and
    feeds the run-level banner; it says nothing about the source STEP.
    """
    seen: list[str] = []
    for line_id, block in ((record or {}).get("lines") or {}).items():
        for layer in stale_layers(block, line_id, current_stamps):
            if layer not in seen:
                seen.append(layer)
    return [layer for layer in LAYERS if layer in seen]


#: ``CUDA_VISIBLE_DEVICES`` values that leave the CUDA runtime with no device
#: at all.  Any other value is a device *list*, and which member of it torch
#: ends up on is not something this module is going to guess.
CUDA_DISABLED_VALUES = frozenset({"", "-1"})

#: The device a line records when ``--no-aagnet`` switched the model off.  It
#: is not an inference device, and that is the point: it goes through
#: :func:`device_mismatch` like any other device string, so a part built
#: without the model and a part built with it are never mistaken for each
#: other in either direction.
AAGNET_DISABLED = "disabled"


def resolve_inference_device(device: str | None) -> tuple[str | None, str | None]:
    """Turn ``"auto"`` into the device it will resolve to - when that is certain.

    Returns ``(device, note)``.  *device* is a concrete device string, or
    ``None`` for **no opinion**, in which case *note* says why by name.

    ``auto`` means ``cuda if torch.cuda.is_available() else cpu``
    (``occline.aagnet.per_face._resolve_device``), and the plan may not import
    torch to find out: ``plan`` and ``store status`` decide what to do, they do
    not load the inference stack.  So the question is answered from what can be
    read off the environment, and only where the answer cannot be wrong:

    * ``CUDA_VISIBLE_DEVICES`` empty or ``-1`` hides every GPU from the CUDA
      runtime, so ``torch.cuda.is_available()`` is ``False`` -> ``cpu``;
    * a torch that was not built with CUDA (``torch/version.py`` says
      ``cuda = None``) has no CUDA support at all -> ``cpu``;
    * a CUDA build with GPUs visible **might** still fall back to cpu (no
      driver, a driver too old, a disabled device).  Establishing that needs
      the driver, i.e. torch, so this case gets no opinion.

    Guessing "cuda" from something weaker - ``nvidia-smi`` being on ``PATH``,
    say - is what is deliberately not done: a wrong guess makes every refresh
    rebuild every part for ever, because the build then records the device the
    guess said was wrong and the next plan disagrees with it again.  A missing
    opinion only leaves finding B-10's residual gap open, which is the same
    place it was before.
    """
    if not device or device != "auto":
        return device, None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in CUDA_DISABLED_VALUES:
        return "cpu", None
    built_with, why = _torch_cuda_build()
    if why is not None:
        return None, why
    if built_with is None:
        return "cpu", None
    return None, (
        f"device auto: this torch is a CUDA {built_with} build, and whether a "
        "usable GPU is present cannot be established without importing it"
    )


def _torch_cuda_build() -> tuple[str | None, str | None]:
    """The CUDA version torch was built against, read without importing torch.

    Returns ``(cuda_version_or_None, why_unreadable_or_None)``.  ``torch``'s
    ``version.py`` is a generated file of literal assignments, so it is read
    and parsed rather than executed - executing it would mean importing the
    package this function exists to avoid.
    """
    import ast
    import importlib.util

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError) as exc:
        return None, f"device auto: torch cannot be located ({exc})"
    locations = list(getattr(spec, "submodule_search_locations", None) or []) if spec else []
    if not locations:
        return None, "device auto: torch is not importable from here"
    version_file = Path(locations[0]) / "version.py"
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"device auto: torch's version.py cannot be read ({exc})"
    for line in text.splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() == "cuda":
            try:
                parsed = ast.literal_eval(value.strip())
            except (SyntaxError, ValueError):
                return None, "device auto: torch's version.py does not parse"
            return (str(parsed) if parsed is not None else None), None
    return None, "device auto: torch's version.py names no cuda build"


def device_mismatch(block: dict, line: str, device: str | None) -> str | None:
    """Why the device asked for makes this line out of date, or ``None``.

    Only for lines that read the ``aagnet`` layer - the NX line records
    ``device="nx"``, which names a session, not an inference device.  An
    unnamed request has no opinion, ``auto`` is resolved by
    :func:`resolve_inference_device` and has no opinion when that cannot be
    certain, and a line with no recorded device (built before the field
    existed) is not second-guessed.  ``cuda`` matches ``cuda:0``: the request
    is a family, the record is the member of it that was chosen.
    """
    device, _note = resolve_inference_device(device)
    if not device:
        return None
    if "aagnet" not in LINE_LAYERS.get(line, frozenset()):
        return None
    built_on = block.get("device")
    if not built_on:
        return None
    if built_on == device or built_on.startswith(device + ":"):
        return None
    return f"built on device {built_on}, this sync asks for {device}"


def plan_line(
    record: dict | None,
    line: str,
    step_path: str | Path | None,
    current_stamps: dict[str, str],
    *,
    want_viewer: bool = True,
    force: bool = False,
    device: str | None = None,
) -> LineAction:
    """What this sync owes one line of one part.

    The decision table, top to bottom:

    1. no source file                              -> ``skip``;
    2. this line has no block yet                  -> ``build``;
    3. ``force``                                   -> ``build``;
    4. the line was not built for an environment
       reason (no licence, no install)             -> ``build`` (always retry:
       the reason was never about this part);
    5. the STEP changed since the line was built   -> ``build``;
    6. the line failed: something changed since    -> ``build``,
       nothing changed                             -> ``failed`` (do not rerun);
    7. viewer artifacts wanted but never built     -> ``build``;
    8. the inference device asked for is not the
       one this line was built on                  -> ``build``;
    9. a layer outside ``PARTIAL_LAYERS`` moved    -> ``build``;
    10. only partial (heuristic) layers moved      -> ``heuristics_only``;
    11. otherwise                                  -> ``skip``.

    *device* is the inference device this sync was asked for.  The ``aagnet``
    layer stamps the code and the weights but cannot stamp the device, and the
    same weights on cuda and on cpu do not produce the same logits (finding
    B-10, measured: all 43 faces of Part_6 move, by up to 5.2e-3, with no top
    label changing).  So a line built on one device and asked for on the other
    is out of date, and says so.  ``None`` is *no opinion* - nobody named a
    device.  ``"auto"`` goes through :func:`resolve_inference_device`, which
    answers it from the environment without importing torch and returns no
    opinion where it cannot be certain.
    """
    if step_path is None:
        return LineAction(line, SKIP, ["source STEP not found under the input path"])

    block = ((record or {}).get("lines") or {}).get(line)
    if not isinstance(block, dict):
        return LineAction(line, BUILD, ["this line has never been built for this part"])

    stale = stale_layers(block, line, current_stamps)
    built_from = block.get("fingerprint")
    source_changed = built_from is not None and built_from != fingerprint(step_path)["sha256"]

    device_changed = device_mismatch(block, line, device)

    reasons: list[str] = []
    if stale:
        reasons.append("changed layer(s): " + ", ".join(stale))
    if source_changed:
        reasons.append("source STEP changed since this line was built")
    if device_changed:
        reasons.append(device_changed)

    if force:
        return LineAction(line, BUILD, ["forced"] + reasons, stale)

    status = block.get("status")
    if status == NOT_BUILT:
        return LineAction(
            line, BUILD,
            [f"not built last time: {block.get('reason') or 'no reason recorded'}"] + reasons,
            stale)
    if source_changed:
        return LineAction(line, BUILD, reasons, stale)
    if status != "PASS":
        if stale:
            return LineAction(line, BUILD, reasons, stale)
        return LineAction(line, FAILED, ["failed before and nothing changed since"])
    if want_viewer and not block.get("viewerEntry"):
        reasons.append("viewer artifacts were never built for this line")
        return LineAction(line, BUILD, reasons, stale)
    if device_changed:
        return LineAction(line, BUILD, reasons, stale)
    if any(layer not in PARTIAL_LAYERS for layer in stale):
        return LineAction(line, BUILD, reasons, stale)
    if stale:
        return LineAction(line, HEURISTICS_ONLY, reasons, stale)
    return LineAction(line, SKIP, [], [])
