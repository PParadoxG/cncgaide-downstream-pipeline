"""STEP file loading, AP version detection, metadata extraction, PMI extraction."""

import logging
import re
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static
from OCC.Core.TopoDS import TopoDS_Shape

from contract import refusals
from occline.parse.pmi_registry import (
    PMI_ENTITY_REGISTRY,
    PMI_TOLERANCE_SUFFIX,
    is_registered,
    looks_like_pmi,
    matches_tolerance_suffix,
)
from occline.parse.traversal import iter_solids

logger = logging.getLogger(__name__)


class MultiBodyStepUnsupported(ValueError):
    """Refusal: this STEP carries more than one solid body.

    Everything downstream - the face numbering, the shells, the solid
    classifier, the ray casts - is written for one body, and the pipeline has
    no notion of which body a feature belongs to.  Building the first body and
    saying nothing would publish a part that silently lost geometry, so the
    whole part is refused instead.  The wording is
    :data:`contract.refusals.MULTI_BODY_UNSUPPORTED` because the NX line
    refuses the same input, and two lines refusing in two sentences read as
    two different refusals.
    """

# The AP numbers ``detect_ap_version`` is allowed to return.  Its docstring
# promises one of ``AP203``/``AP214``/``AP242``/``UNKNOWN``, and every tier of
# the detection reads this same set so no tier can promise something else.
_KNOWN_AP_NUMBERS = frozenset({"203", "214", "242"})

# A STEP physical file states its own length unit; OCCT converts every length
# into the unit named by this process-wide XSTEP parameter while transferring.
# Every threshold in the v5.1 ontology is in millimetres, so the scale of the
# whole pipeline hangs off this one string.  OCCT's built-in default happens to
# be MM, but a default nobody sets is a default nobody can be held to: any code
# in the process - a future dependency included - can change it and every length
# then scales silently.  So the line sets it and reads it back on every load.
STEP_LENGTH_UNIT = "MM"
_CASCADE_UNIT_PARAMETER = "xstep.cascade.unit"

# ... and the other half of that story: what the file itself says its length
# unit is.  ISO 10303-41 writes it either as an SI unit with an optional
# prefix - ``SI_UNIT(.MILLI.,.METRE.)`` - or as a conversion-based unit that
# names itself and points at an SI one - ``CONVERSION_BASED_UNIT('INCH',#n)``
# inside a complex instance that also carries ``LENGTH_UNIT()``.  Both are
# read verbatim; neither is converted into anything (node 27B B-11 second
# half).
_SI_LENGTH_UNIT = re.compile(
    r"SI_UNIT\s*\(\s*(\$|\.\s*[A-Za-z_]+\s*\.)\s*,\s*\.\s*METRE\s*\.\s*\)",
    re.IGNORECASE)
_CONVERSION_UNIT_NAME = re.compile(
    r"CONVERSION_BASED_UNIT\s*\(\s*('(?:[^']|'')*')", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_section(filepath: str, section_name: str) -> str:
    """Read a named section (HEADER or DATA) from a STEP physical file.

    ISO 10303-21 physical files are ISO/IEC 8859-1 (ISO 646 plus the upper
    half), and anything outside that is written as an escape sequence inside
    the string, not as a multi-byte character.  Reading as UTF-8 therefore
    turned a perfectly legal ``\xe7`` into U+FFFD (finding B-09).  latin-1 maps
    every byte to exactly one code point and can never fail, so what the file
    says is what arrives here; :func:`_step_string` turns the escapes into
    text afterwards.
    """
    with open(filepath, "r", encoding="latin-1") as fh:
        content = fh.read()
    match = re.search(rf"{section_name};(.*?)ENDSEC;", content, re.DOTALL)
    return match.group(1) if match else ""


_X2_ESCAPE = re.compile(r"\\X2\\((?:[0-9A-Fa-f]{4})+)\\X0\\")
_X4_ESCAPE = re.compile(r"\\X4\\((?:[0-9A-Fa-f]{8})+)\\X0\\")
_X_ESCAPE = re.compile(r"\\X\\([0-9A-Fa-f]{2})")
_S_ESCAPE = re.compile(r"\\S\\(.)")


def _decode_step_escapes(text: str) -> str:
    """Turn ISO 10303-21 string escapes into the characters they stand for.

    Four forms, all of which a preprocessor may emit for a character outside
    ISO 646:

    * ``\\X2\\`` <UTF-16BE code units> ``\\X0\\`` - the common one;
    * ``\\X4\\`` <UTF-32BE code points> ``\\X0\\`` - the same for the
      astral planes;
    * ``\\X\\hh`` - one octet, read as ISO 8859-1;
    * ``\\S\\c`` - the character *c* with the eighth bit set.

    A sequence that does not parse is left exactly as it was written: this is
    metadata, and a name that cannot be decoded should arrive as the escape it
    is rather than as a guess.
    """
    def utf16(match: re.Match) -> str:
        try:
            return bytes.fromhex(match.group(1)).decode("utf-16-be")
        except (ValueError, UnicodeDecodeError):
            return match.group(0)

    def utf32(match: re.Match) -> str:
        try:
            return bytes.fromhex(match.group(1)).decode("utf-32-be")
        except (ValueError, UnicodeDecodeError):
            return match.group(0)

    text = _X2_ESCAPE.sub(utf16, text)
    text = _X4_ESCAPE.sub(utf32, text)
    text = _X_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    text = _S_ESCAPE.sub(lambda m: chr(ord(m.group(1)) + 128), text)
    return text


def _step_string(token: str) -> str:
    """One STEP string literal as text.

    ``.strip("'")`` was doing this job, and it strips *every* leading and
    trailing quote, so a name that really is quoted (``''Bracket''``) lost
    characters and an unterminated token lost its evidence (finding B-07).
    One matched pair comes off, doubled quotes inside collapse to one - that
    is how ISO 10303-21 writes an apostrophe - and the escapes are decoded.
    """
    token = token.strip()
    if len(token) >= 2 and token.startswith("'") and token.endswith("'"):
        token = token[1:-1]
    return _decode_step_escapes(token.replace("''", "'"))


def _strip_step_comments(text: str) -> str:
    """Remove ``/* ... */`` comments, but never inside a quoted string.

    A preprocessor may label the header's fields (``FILE_NAME(/* name */
    'model5.stp', ...)``), and the label was ending up in ``part_name``
    (finding B-07).  Quote state is tracked the same way the two functions
    above track it; an unterminated comment swallows the rest of the text,
    which is what it means.
    """
    out: list[str] = []
    index = 0
    in_string = False
    while index < len(text):
        ch = text[index]
        if in_string:
            out.append(ch)
            if ch == "'":
                in_string = False
            index += 1
            continue
        if ch == "'":
            in_string = True
            out.append(ch)
            index += 1
            continue
        if ch == "/" and text.startswith("/*", index):
            end = text.find("*/", index + 2)
            index = len(text) if end == -1 else end + 2
            continue
        out.append(ch)
        index += 1
    return "".join(out)


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
            if depth >= 0:
                current.append(ch)
            # depth < 0 is a close-paren with no opener in this text: the
            # caller was supposed to strip the outermost pair.  Drop it rather
            # than emit a token that does not balance.  ``depth == 0`` after
            # the decrement is the *nested* list closing - it belongs to the
            # token, and dropping it was how ``('author')`` came out as
            # ``('author'``.
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
    """Return the index of the closing paren matching text[start] (must be '(').

    Parentheses inside a single-quoted STEP string are text, not structure -
    ``'Bracket (rev B.prt'`` is a legal part name.  Counting them made a file
    name with one unmatched bracket return -1, and the caller then skipped the
    whole entity without a word.  The quote state is tracked the same way
    :func:`_parse_step_list` tracks it, doubled quotes included: the first of a
    ``''`` pair leaves the string and the second re-enters it, so the depth is
    unaffected either way.
    """
    depth = 0
    in_string = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if ch == "'":
                in_string = False
            continue
        if ch == "'":
            in_string = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _pin_length_unit() -> None:
    """Set the XSTEP length unit to millimetres and prove it took.

    The parameter only exists once an XSTEP controller has been instantiated,
    so this runs after the reader is constructed, not at import time.
    """
    Interface_Static.SetCVal(_CASCADE_UNIT_PARAMETER, STEP_LENGTH_UNIT)
    current = str(Interface_Static.CVal(_CASCADE_UNIT_PARAMETER) or "")
    if current.strip().upper() != STEP_LENGTH_UNIT:
        raise ValueError(
            f"OCCT length unit is '{current}', not '{STEP_LENGTH_UNIT}' - every "
            f"length this line reports would be scaled by an unknown factor"
        )


def load_step(filepath: str) -> TopoDS_Shape:
    """Load a STEP file via OCCT and return the root *TopoDS_Shape*.

    Lengths come back in millimetres; see :func:`_pin_length_unit`.

    Raises *ValueError* with a descriptive message if loading or root-transfer
    fails, or if the length unit could not be pinned, and
    :class:`MultiBodyStepUnsupported` when the file carries more than one
    solid body.
    """
    reader = STEPControl_Reader()
    _pin_length_unit()
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

    solid_count = sum(1 for _ in iter_solids(shape))
    if solid_count > 1:
        raise MultiBodyStepUnsupported(refusals.multi_body_unsupported(solid_count))

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
        if num in _KNOWN_AP_NUMBERS:
            return f"AP{num}"
        # AP242 Edition 2 uses EXPRESS resource 442; the schema name
        # still contains "AP242"
        if num == "442" and re.search(r"AP242", schema_id, re.IGNORECASE):
            return "AP242"
        logger.warning("Unrecognised AP number %s in '%s'", num, schema_id)
        return "UNKNOWN"

    # Fallback 2: explicit AP prefix in schema name.  An AP number outside the
    # promised set is not an answer - it falls through to the substring tier
    # and, failing that, to UNKNOWN, rather than inventing a fifth return value.
    ap_match = re.search(r"AP(\d+)", schema_id, re.IGNORECASE)
    if ap_match:
        num = ap_match.group(1)
        if num in _KNOWN_AP_NUMBERS:
            return f"AP{num}"
        logger.warning("Unsupported AP number %s in schema name '%s'", num, schema_id)

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
    """Return the ``FILE_NAME`` fields this reader knows how to name.

    ``{"part_name": str, "timestamp": str, "preprocessor_version": str,
    "originating_system": str}``; missing fields become empty strings.  Uses
    pure text parsing.

    ISO 10303-21 fixes the order of ``FILE_NAME``: *name*, *time_stamp*,
    *author*, *organization*, *preprocessor_version*, *originating_system*,
    *authorization*.  So ``args[4]`` is the preprocessor and ``args[5]`` is
    the originating system.  Until node 28 this function returned ``args[4]``
    under the name ``originating_system`` and never read ``args[5]`` at all -
    the two were one position out (node 27 Phase 2 §8 #1).  Only ``part_name``
    is consumed by the pipeline (``PartMetadata`` stores nothing else from
    here), so correcting the names moves no artifact byte.
    """
    result = {"part_name": "", "timestamp": "",
              "preprocessor_version": "", "originating_system": ""}

    header = _strip_step_comments(_read_section(filepath, "HEADER"))
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

    # FILE_NAME('name', 'time_stamp', ('author',), ('organization',),
    #           'preprocessor_version', 'originating_system', 'authorization')
    if len(args) >= 1:
        result["part_name"] = _step_string(args[0])
    if len(args) >= 2:
        result["timestamp"] = _step_string(args[1])
    if len(args) >= 5:
        result["preprocessor_version"] = _step_string(args[4])
    if len(args) >= 6:
        result["originating_system"] = _step_string(args[5])

    return result


def extract_length_unit(filepath: str) -> str:
    """The length unit the file declares, in the file's own words.

    ``"MILLI METRE"``, ``"METRE"``, ``"INCH"`` - the SI prefix and unit name
    as ISO 10303-41 spells them, or a conversion-based unit's own name.  A
    conversion-based declaration wins over the SI unit it is defined against,
    because a file whose measures are inches also carries the metre that inch
    is a conversion of.

    Empty string in three cases, and they are deliberately not distinguished:
    the file declares no length unit, it declares more than one and they
    disagree, or the DATA section could not be read.  Nothing computes with
    this value - the numbers in a PartModel are millimetres, pinned and
    asserted by :func:`_pin_length_unit` - so an unknown unit is provenance
    that is missing, not a measurement that is wrong.
    """
    data = _strip_step_comments(_read_section(filepath, "DATA"))
    if not data:
        return ""
    converted: set[str] = set()
    si: set[str] = set()
    for statement in data.split(";"):
        upper = statement.upper()
        if "LENGTH_UNIT" not in upper and "METRE" not in upper:
            continue
        conversion = _CONVERSION_UNIT_NAME.search(statement)
        if conversion is not None and "LENGTH_UNIT" in upper:
            name = _step_string(conversion.group(1)).strip().upper()
            if name:
                converted.add(name)
                continue
        match = _SI_LENGTH_UNIT.search(statement)
        if match is None:
            continue
        prefix = match.group(1).strip().strip(".").strip().upper()
        si.add(f"{prefix} METRE" if prefix and prefix != "$" else "METRE")
    for names in (converted, si):
        if len(names) == 1:
            return next(iter(names))
    return ""


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

