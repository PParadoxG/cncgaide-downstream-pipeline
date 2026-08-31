"""How the NX line numbers faces, edges and vertices.

The ids in ``partmodel.json`` are ordinals (``face_<i>``), so *what order the
entities are put in* is what makes the file reproducible.  Until node 16 the
order was the NX tag order, on probe j20's reading that tags are
process-stable.  Node 15 found the exception and probe j23 named the
mechanism: a tag reflects how many objects the session had already loaded, so
opening one .prt in a 21-part batch and opening it alone hand back two
different tag orderings for the same solid.  Every id in the file moved while
the solid did not.

Probe j23 also found the replacement.  ``uf.Tag.AskHandleOfTag`` returns NX's
persistent object handle (``RMPart_6.prt R0000028f00000022``), and it holds on
all three axes this line needs:

* stable across sessions (58/58 faces, 147/147 edges of ``stc_cad_rsb``,
  where the tag moved on every single one);
* stable across *session composition* - batch and single agree;
* stable across re-translation (j23b: the same STEP through a fresh
  ``step242`` run keeps every handle), so clearing the ``.prt`` cache does not
  shuffle the ids either.

The handle is the sort key, not the id: what the file publishes is still the
same dense ``face_<i>`` numbering, so nothing downstream changes shape.

**Never the tag.**  When handles are missing or not unique the fallback is a
geometric canonical key - quantized to a fixed number of decimals and
tie-broken by further fields - and the degradation is *named* in the model's
``normalization_log`` rather than silently taken.  Falling back to the tag
would reintroduce exactly the defect this module exists to remove.

Nothing here touches NX: ``assign_identity`` is a pure function of plain
records, which is what lets a test shuffle its input and demand the same
answer without a licence.
"""

from __future__ import annotations

import json

# Six decimals is nanometres in millimetres: far below any real distinction
# between two faces, far above the 1-ULP jitter that made coincident points
# disagree between sessions.
IDENTITY_DECIMALS = 6

MODE_HANDLE = "nx_handle"
MODE_GEOMETRIC = "geometric"


def quantize(value) -> float:
    """A float pinned to the identity grid.

    ``-0.0`` folds onto ``0.0`` so two spellings of the same coordinate cannot
    sort apart.
    """
    result = round(float(value), IDENTITY_DECIMALS)
    return 0.0 if result == 0.0 else result


def quantize_point(point) -> tuple[float, ...]:
    return tuple(quantize(v) for v in point)


def canonical(value) -> str:
    """A comparable string for an arbitrary parameter block.

    Sorting keys need a total order, and parameter dictionaries hold floats,
    lists, strings and ``None`` side by side.  Rounding every float on the way
    through keeps the string on the same grid as the rest of the key.
    """

    def scrub(item):
        if isinstance(item, bool):
            return item
        if isinstance(item, (int, float)):
            return quantize(item)
        if isinstance(item, dict):
            return {str(k): scrub(v) for k, v in sorted(item.items(), key=lambda kv: str(kv[0]))}
        if isinstance(item, (list, tuple)):
            return [scrub(v) for v in item]
        return str(item)

    return json.dumps(scrub(value), sort_keys=True, separators=(",", ":"), default=str)


def face_key(surface_type, area, centroid, bbox_min, bbox_max, parameters) -> tuple:
    """The geometric fallback key for one face.

    Ordered widest-agreement first: ``(surface_type, area, centroid)`` are the
    three quantities node 15 measured to be item-for-item identical between
    sessions, and they already separate all 826 faces of the corpus on their
    own.  The bounding box and the parameter block are tie-breakers, present so
    that a tie has a decided answer rather than an accidental one.
    """
    return (
        str(surface_type),
        quantize(area if area is not None else 0.0),
        quantize_point(centroid) if centroid else (),
        quantize_point(bbox_min or ()),
        quantize_point(bbox_max or ()),
        canonical(parameters or {}),
    )


def edge_key(curve_type, length, endpoints, parameters) -> tuple:
    """The geometric fallback key for one edge.

    ``endpoints`` is sorted, because which end NX calls first is not a property
    of the edge.
    """
    ends = tuple(sorted(quantize_point(point) for point in (endpoints or [])))
    return (
        str(curve_type),
        quantize(length if length is not None else 0.0),
        ends,
        canonical(parameters or {}),
    )


def assign_identity(records: list[dict]) -> tuple[list[int], str, list[str]]:
    """Decide the id order for one kind of entity.

    *records* is ``[{"handle": str | None, "key": tuple}, ...]`` in whatever
    order the kernel handed them over.  Returns ``(order, mode, notes)``:
    ``order[i]`` is the index of the record that becomes id ``i``, *mode* is
    which rule decided it, and *notes* are the named degradations - empty when
    nothing degraded.

    A pure function of *records*: permute the input and the record-to-id
    mapping is unchanged, which is the property the whole node is about.
    """
    count = len(records)
    if count == 0:
        return [], MODE_HANDLE, []

    handles = [str(record.get("handle") or "") for record in records]
    missing = sum(1 for handle in handles if not handle)
    distinct = len(set(handles))
    if missing == 0 and distinct == count:
        return sorted(range(count), key=lambda i: handles[i]), MODE_HANDLE, []

    notes = []
    if missing:
        notes.append(
            f"identity: {missing} of {count} entities carry no NX handle; ordering "
            f"fell back to the geometric key")
    else:
        notes.append(
            f"identity: NX handles are not unique ({distinct} distinct for {count} "
            f"entities); ordering fell back to the geometric key")

    # The handle joins the geometric key as its last tier rather than being
    # dropped: where it exists it still separates, and where it does not the
    # empty string sorts consistently.
    order = sorted(range(count), key=lambda i: (records[i]["key"], handles[i]))
    groups: dict[tuple, int] = {}
    for index in range(count):
        composite = (records[index]["key"], handles[index])
        groups[composite] = groups.get(composite, 0) + 1
    collisions = sum(size - 1 for size in groups.values() if size > 1)
    if collisions:
        notes.append(
            f"identity: {collisions} entities are indistinguishable in every recorded "
            f"field (geometric key and handle both equal); their relative order is the "
            f"kernel's enumeration and is not reproducible")
    return order, MODE_GEOMETRIC, notes


def vertex_table(points: list) -> tuple[dict[tuple, str], dict[str, tuple]]:
    """Turn every endpoint the kernel reported into the vertex table.

    Two jobs, both about reproducibility:

    * **One representative per location.**  Coincident points arriving through
      different edges differ in the last bit, and the old builder stored
      whichever it met first - which changed with the edge order.  The
      representative is the lexicographic minimum of the coincident group, so
      it depends on the set of points and not on the order they arrived in.
    * **Order-free numbering.**  ``vertex_<i>`` follows the sorted quantized
      key, not the order edges were walked.

    Returns ``(id_by_key, coordinates_by_id)``.
    """
    groups: dict[tuple, list[tuple]] = {}
    for point in points:
        exact = tuple(float(v) for v in point)
        groups.setdefault(quantize_point(exact), []).append(exact)
    id_by_key: dict[tuple, str] = {}
    coordinates: dict[str, tuple] = {}
    for index, key in enumerate(sorted(groups)):
        vertex_id = f"vertex_{index}"
        id_by_key[key] = vertex_id
        coordinates[vertex_id] = min(groups[key])
    return id_by_key, coordinates
