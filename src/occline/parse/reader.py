"""STEP file loading, AP version detection, metadata extraction, PMI extraction."""

import logging
import re
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import TopoDS_Shape

from occline.parse.pmi_registry import (
    PMI_ENTITY_REGISTRY,
    PMI_TOLERANCE_SUFFIX,
    is_registered,
    looks_like_pmi,
    matches_tolerance_suffix,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_section(filepath: str, section_name: str) -> str:
    """Read a named section (HEADER or DATA) from a STEP physical file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    match = re.search(rf"{section_name};(.*?)ENDSEC;", content, re.DOTALL)
    return match.group(1) if match else ""


def _parse_step_list(text: str) -> list[str]:
    """Split a STEP parenthesised argument list on top-level commas.

    Handles nested parentheses, single-quoted strings, and $ (missing) tokens.
    The caller must have already stripped the outermost enclosing parentheses.
    """
    result: list[str] = []
    current: list[str] = []
    depth = 0
    in_string = False

    for ch in text:
        if in_string:
            current.append(ch)
            if ch == "'":
                in_string = False
            continue

        if ch == "'":
            in_string = True
            current.append(ch)
            continue

        if ch == "(":
            depth += 1
            current.append(ch)
            continue

        if ch == ")":
            depth -= 1
            if depth > 0:
                current.append(ch)
            # if depth == 0 we have reached the outer close-paren (should not
            # happen when the caller already stripped outer parens, but handle
            # gracefully)
            continue

        if ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
            continue

        current.append(ch)

    if current:
        result.append("".join(current).strip())

    return result


def _find_matching_paren(text: str, start: int) -> int:
    """Return the index of the closing paren matching text[start] (must be '(')."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_step(filepath: str) -> TopoDS_Shape:
    """Load a STEP file via OCCT and return the root *TopoDS_Shape*.

    Raises *ValueError* with a descriptive message if loading or root-transfer
    fails.
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(filepath))

    if status != IFSelect_RetDone:
        raise ValueError(
            f"STEPControl_Reader.ReadFile failed on '{filepath}' "
            f"(status={status}, expected={IFSelect_RetDone})"
        )

    reader.TransferRoots()
    shape = reader.OneShape()

    if shape.IsNull():
        raise ValueError(
            f"Loaded shape from '{filepath}' is null — file may be empty or corrupt"
        )

    return shape


def detect_ap_version(filepath: str) -> str:
    """Parse ``FILE_SCHEMA`` in the header to return one of
    ``"AP203"``, ``"AP214"``, ``"AP242"``, or ``"UNKNOWN"``.

    Detection is tried in three tiers:

    1. ``10303 <N>`` OID in the schema identifier (authoritative)
    2. ``AP<N>`` prefix in the schema name (common for AP242 Ed.2)
    3. Substring match against known schema-name → AP mappings
       (``CONFIG_CONTROL_DESIGN`` → AP203, ``AUTOMOTIVE_DESIGN`` → AP214)

    ``FILE_DESCRIPTION`` is deliberately NOT consulted — its content is
    free-form text populated by the exporting system and may be empty or
    contain arbitrary user strings.

    Uses pure text parsing — no OCCT dependency.
    """
    header = _read_section(filepath, "HEADER")
    if not header:
        logger.warning("No HEADER section in %s", Path(filepath).name)
        return "UNKNOWN"

    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']*)'", header)
    if not m:
        logger.warning("No FILE_SCHEMA in %s", Path(filepath).name)
        return "UNKNOWN"

    schema_id = m.group(1)

    # Primary pattern:  ... 10303 214 ... (or 10303 442 for AP242 Ed.2)
    ap_match = re.search(r"10303\s+(\d+)", schema_id)
    if ap_match:
        num = ap_match.group(1)
        if num in {"203", "214", "242"}:
            return f"AP{num}"
        # AP242 Edition 2 uses EXPRESS resource 442; the schema name
        # still contains "AP242"
        if num == "442" and re.search(r"AP242", schema_id, re.IGNORECASE):
            return "AP242"
        logger.warning("Unrecognised AP number %s in '%s'", num, schema_id)
        return "UNKNOWN"

    # Fallback 2: explicit AP prefix in schema name
    ap_match = re.search(r"AP(\d+)", schema_id, re.IGNORECASE)
    if ap_match:
        return f"AP{ap_match.group(1)}"

    # Fallback 3: substring match against known schema-name → AP mapping.
    # The FILE_SCHEMA identifier is standardised — unlike FILE_DESCRIPTION
    # it cannot contain arbitrary user text.  This catches files whose
    # schema name omits both the 10303-<N> OID and the AP<N> prefix
    # (common for AP203/AP214 exported by older or lightweight writers).
    schema_upper = schema_id.upper()
    if "CONFIG_CONTROL_DESIGN" in schema_upper or "CONFIGURATION_CONTROL" in schema_upper:
        return "AP203"
    if "AUTOMOTIVE_DESIGN" in schema_upper:
        return "AP214"
    if "AP242" in schema_upper:
        return "AP242"

    logger.warning("Cannot determine AP version from FILE_SCHEMA '%s'", schema_id)
    return "UNKNOWN"


def extract_metadata(filepath: str) -> dict:
    """Return ``{"part_name": str, "originating_system": str, "timestamp": str}``
    from the ``FILE_NAME`` line in the header.  Missing fields become empty
    strings.  Uses pure text parsing.
    """
    result = {"part_name": "", "originating_system": "", "timestamp": ""}

    header = _read_section(filepath, "HEADER")
    if not header:
        return result

    # Locate FILE_NAME
    fn_pos = header.find("FILE_NAME")
    if fn_pos == -1:
        return result

    paren_start = header.find("(", fn_pos)
    if paren_start == -1:
        return result

    paren_end = _find_matching_paren(header, paren_start)
    if paren_end == -1:
        return result

    args_str = header[paren_start + 1 : paren_end]
    args = _parse_step_list(args_str)

    # FILE_NAME('name', 'timestamp', ('author',), ('org',),
    #           'system', 'preprocessor', 'authorizing')
    if len(args) >= 1:
        result["part_name"] = args[0].strip("'")
    if len(args) >= 2:
        result["timestamp"] = args[1].strip("'")
    if len(args) >= 5:
        result["originating_system"] = args[4].strip("'")

    return result


def _extract_pmi_entry(entity_name: str, raw_args: list[str], category: str) -> dict:
    """Build a minimal annotation dict from a recognised PMI entity instance.

    Stores the entity name and raw attribute list for downstream use.
    Attempts a best-effort extraction of a human-readable label from the
    first non-reference string argument, but does *not* follow reference
    chains to resolve numeric values (deferred to Stage 2).
    """
    entry: dict = {
        "type": category,
        "entity_name": entity_name,
        "raw_args": raw_args,
    }

    # Derive semantic kind from entity-name suffix for tolerances
    if category == "tolerance" and entity_name.endswith(PMI_TOLERANCE_SUFFIX):
        entry["kind"] = entity_name[: -len(PMI_TOLERANCE_SUFFIX)].lower()

    # Best-effort label extraction — first non-reference, non-empty string
    for arg in raw_args:
        stripped = arg.strip("'")
        if stripped and not stripped.startswith("#") and not stripped.startswith("."):
            if category == "dimension" and "name" not in entry:
                entry["name"] = stripped
            elif category == "tolerance" and "description" not in entry:
                entry["description"] = stripped
            elif category == "surface_finish" and "value" not in entry:
                entry["value"] = stripped
            elif category == "unknown_pmi":
                entry["label"] = stripped
            break

    return entry


def _collect_pmi_entities(data: str) -> list[tuple[str, str, list[str]]]:
    """Yield ``(entity_name, entity_id, raw_args)`` for every PMI-candidate
    entity instance found in *data*."""
    results: list[tuple[str, str, list[str]]] = []
    # Match every entity instance: #N = ENTITY_NAME ( ... )
    pattern = re.compile(r"#(\d+)\s*=\s*(\w+)\s*\(")

    for m in pattern.finditer(data):
        entity_id = m.group(1)
        entity_name = m.group(2)
        paren_start = m.end() - 1

        paren_end = _find_matching_paren(data, paren_start)
        if paren_end == -1:
            continue

        args_str = data[paren_start + 1 : paren_end]

        # Quick filter: skip entities that are clearly not PMI
        if not (is_registered(entity_name)
                or matches_tolerance_suffix(entity_name)
                or looks_like_pmi(entity_name)):
            continue

        try:
            raw_args = _parse_step_list(args_str)
        except Exception:
            logger.warning("Failed to parse args for %s #%s", entity_name, entity_id)
            continue

        results.append((entity_name, entity_id, raw_args))

    return results


def extract_pmi_annotations(filepath: str, ap_version: str) -> list[dict]:
    """Extract PMI annotations from the DATA section of AP242 files.

    Returns an empty list for AP203 / AP214 without touching the file body.
    For AP242, scans every entity instance and dispatches through a registry:

    * Exact match in ``PMI_ENTITY_REGISTRY`` → category from registry
    * Name ends with ``_TOLERANCE`` → category ``"tolerance"``
    * ``looks_like_pmi()`` heuristic match → category ``"unknown_pmi"``
      (logged at INFO level for incremental registry extension)

    PMI entities store their raw attribute list and entity name.  Numeric
    values hidden behind reference chains (e.g. TOLERANCE_VALUE →
    LENGTH_MEASURE_WITH_UNIT) are **not** resolved — that is deferred to
    Stage 2.  Never raises; parse errors are logged at WARNING level.
    """
    if ap_version != "AP242":
        return []

    data = _read_section(filepath, "DATA")
    if not data:
        logger.info("No DATA section in AP242 file: %s", Path(filepath).name)
        return []

    annotations: list[dict] = []
    seen_unknown: set[str] = set()

    for entity_name, entity_id, raw_args in _collect_pmi_entities(data):

        # 1. Exact registry match
        if is_registered(entity_name):
            category = PMI_ENTITY_REGISTRY[entity_name]["type"]
            entry = _extract_pmi_entry(entity_name, raw_args, category)
            annotations.append(entry)
            continue

        # 2. Tolerance suffix match
        if matches_tolerance_suffix(entity_name):
            entry = _extract_pmi_entry(entity_name, raw_args, "tolerance")
            annotations.append(entry)
            continue

        # 3. Heuristic PMI — log and store for future registry extension
        if looks_like_pmi(entity_name):
            if entity_name not in seen_unknown:
                logger.info(
                    "Unregistered PMI entity '%s' (#%s) in %s",
                    entity_name, entity_id, Path(filepath).name,
                )
                seen_unknown.add(entity_name)
            entry = _extract_pmi_entry(entity_name, raw_args, "unknown_pmi")
            annotations.append(entry)
            continue

    if not annotations:
        logger.info("No PMI entities found in AP242 file: %s", Path(filepath).name)

    return annotations

