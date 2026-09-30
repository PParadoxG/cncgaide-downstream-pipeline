"""Input discovery and unified v8 run layout helpers."""

from __future__ import annotations

import re
from pathlib import Path


STEP_EXTENSIONS = {".step", ".stp"}


def resolve_input_path(input_path: str | Path, project_root: str | Path) -> Path:
    path = Path(input_path)
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root).resolve() / path).resolve()


def collect_step_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path).resolve()
    if path.is_file():
        return [path] if path.suffix.lower() in STEP_EXTENSIONS else []
    if not path.exists():
        return []
    return sorted(
        (candidate for candidate in path.rglob("*") if candidate.is_file() and candidate.suffix.lower() in STEP_EXTENSIONS),
        key=lambda candidate: str(candidate).lower(),
    )


def make_part_id(step_path: str | Path, input_root: str | Path) -> str:
    step = Path(step_path).resolve()
    root = Path(input_root).resolve()
    if root.is_file():
        root = root.parent
    try:
        raw = "__".join(step.relative_to(root).with_suffix("").parts)
    except ValueError:
        raw = step.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def run_paths(project_root: str | Path, output_tag: str) -> dict[str, Path]:
    root = Path(project_root).resolve()
    run_root = root / "runs" / (output_tag or "latest")
    return {
        "run_root": run_root,
        "parts": run_root / "parts",
        "manifest": run_root / "manifest.json",
        "viewer_index": run_root / "viewer_index.json",
    }


def ensure_run_paths(project_root: str | Path, output_tag: str) -> dict[str, Path]:
    paths = run_paths(project_root, output_tag)
    paths["parts"].mkdir(parents=True, exist_ok=True)
    return paths


def part_dir(project_root: str | Path, output_tag: str, part_id: str) -> Path:
    """The part's own directory.  ``record.json`` lives here; each line's
    artifacts live in a folder below it."""
    return run_paths(project_root, output_tag)["parts"] / part_id


def part_paths(project_root: str | Path, output_tag: str, part_id: str, line: str) -> dict[str, Path]:
    """One line's artifact set for one part.

    Both lines write the *same file names* under different folders, so a
    reader switches lines by changing one path segment and nothing else.  Only
    the raw vendor output keeps its vendor's name (``aagnet_evidence.json``)
    - what the shared chain computes is unsuffixed, and a line writes only
    the files it has.
    """
    root = part_dir(project_root, output_tag, part_id) / line
    return {
        "root": root,
        "result": root / "result.json",
        "partmodel": root / "partmodel.json",
        "evidence": root / "aagnet_evidence.json",
        "native_groups": root / "aagnet_native_groups.audit.json",
        "tensor_shapes": root / "aagnet_tensor_shapes.audit.json",
        "heuristics": root / "heuristics.json",
        "id_map": root / "id_map.json",
        "graph_view": root / "graph_view.json",
        "mesh_bin": root / "mesh.bin",
        "mesh_index": root / "mesh.json",
        "viewer_view": root / "view.json",
    }


def mirror_part_paths(paths: dict[str, Path], target: str | Path) -> dict[str, Path]:
    """The same file names under a different directory.

    Used wherever a line's artifacts are produced somewhere other than where
    they will live - a staging directory that is published atomically
    afterwards - so the writer never has to know which of the two it is.
    """
    root = Path(target)
    mirrored = {key: root / value.name for key, value in paths.items() if key != "root"}
    mirrored["root"] = root
    return mirrored


def ensure_part_paths(project_root: str | Path, output_tag: str, part_id: str, line: str) -> dict[str, Path]:
    paths = part_paths(project_root, output_tag, part_id, line)
    paths["root"].mkdir(parents=True, exist_ok=True)
    return paths


def project_relative(project_root: str | Path, path: str | Path) -> str:
    """One file's location as the *project* names it, e.g. ``inputs\\Part_6.STEP``.

    ``sourceStep`` is written into ``result.json``, ``graph_view.json`` and
    ``view.json``, all of which are hashed into the line's artifact digest, so
    an absolute path here makes the digest answer "where is this checkout" as
    well as "did the result change" - and ``--input`` pointing outside the
    project is a documented, ordinary thing to do.  Falls back to the file
    name for exactly the reason :func:`run_relative` does, and never to an
    absolute path; :func:`project_relative_gap` is how a caller learns that
    the fall-back happened instead of having to compare strings.

    Forward slashes, like :func:`run_relative`: this is a reference the
    artifacts carry, not a path on any disk, and it used to go into a digest
    as ``inputs\\Part_6.STEP`` on Windows and ``inputs/Part_6.STEP`` elsewhere
    - two spellings of one fact, and one more thing the digest answered that
    nobody asked it (finding B-14).
    """
    root = Path(project_root).resolve()
    absolute = Path(path).resolve()
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError:
        return Path(path).name


def project_relative_gap(project_root: str | Path, path: str | Path) -> str | None:
    """Why :func:`project_relative` could only give a file name, or ``None``.

    A named gap rather than a silent shortening: the reference an artifact
    carries is then a file name that a reader knows is a file name, and the
    absolute location stays in ``record.json`` (``sourceStepAbs``), which no
    digest covers.
    """
    root = Path(project_root).resolve()
    absolute = Path(path).resolve()
    try:
        absolute.relative_to(root)
    except ValueError:
        return ("the source file is outside the project root, so it is named "
                "by file name only; its full location is in record.json "
                "(sourceStepAbs)")
    return None


def run_relative(project_root: str | Path, output_tag: str, path: str | Path) -> str:
    """One artifact's location as the *run* names it, e.g.
    ``parts/Part_6/occ/aagnet_native_groups.audit.json``.

    Artifacts that name their siblings are hashed into the line's artifact
    digest, so an absolute path in one of them makes the digest answer "where
    is this checkout" as well as "did the result change".  Forward slashes
    because this is a reference within the run, not a path on any disk.
    """
    run_root = run_paths(project_root, output_tag)["run_root"]
    absolute = Path(path).resolve()
    try:
        return absolute.relative_to(run_root).as_posix()
    except ValueError:
        return Path(path).name
