"""Asking NX the standard question.  Runs under ``run_journal.exe`` only.

One Parametric round with the Find Features dialog's factory defaults, and
whatever comes back is what we record.  Node 17' fixed this recipe against the
GUI item by item (2026-08-28, on ``084_818_012_solid_001``), so it can be
restated in one table:

    RecognitionType          Parametric      (the dialog's Type, default)
    rounds                   1               (the dialog runs one)
    GeometrySearchType       AllGeometry     (the dialog's Search Method)
    feature types            all registered  (the dialog's list, all ticked)
    MapFeatures              False           (the dialog's box, unticked)
    machining access dir.    not set         (the dialog's Method = None)
    limit search area        not set         (the dialog's Select Face = 0)
    AssignColor              not set
    AddCadFeatureAttributes  not set

Everything the old two-round extraction added is gone: no Legacy round (the
GUI's Type list has no such entry, so no GUI user ever sees it), and no
``IgnoreExistingFeatures`` - that flag only ever existed to stop the first
round's persisted features from suppressing the second one's.

Attributes are **enumerated**, not probed.  ``CAMFeature.Attributes`` is the
collection behind the Machining Feature Navigator's tooltip, and it hands over
every attribute NX carries on the feature; the installation schema
(``customization_base.xml``) no longer has to be parsed to guess a candidate
list.  What the enumeration cannot read is recorded by name and reason rather
than defaulted - which is how the ``FACES_*`` and ``CHAMFER`` slots stay
visible as gaps instead of disappearing.
"""

import NXOpen
import NXOpen.CAM

# ``CAMAttribute.ValueType``'s members, in declaration order.  Under this
# binding ``str(attribute.Type)`` yields the ordinal ("0", "1", ...) rather
# than the member name, so the ordinal is the key and the name only a
# fallback.  There is deliberately no entry for Face or Class: NX declares
# attributes of both kinds and this enum cannot express either, which is why
# they come back unreadable rather than wrong.
_ATTR_TYPES = ["Integer", "Double", "String", "Bool", "Point3", "Vector3"]
_ATTR_GETTER = {
    "Integer": "GetIntegerValue", "Double": "GetDoubleValue",
    "String": "GetStringValue", "Bool": "GetBoolValue",
    "Point3": "GetPoint3Value", "Vector3": "GetVector3Value",
}

_GEOM_SEARCH = ["CAM.FeatureRecognitionBuilderGeometrySearch",
                "CAM.FeatureRecognitionBuilder.GeometrySearch"]

RECIPE = {
    "recognitionType": "Parametric",
    "rounds": 1,
    "geometrySearchType": "AllGeometry",
    "mapFeatures": False,
    "featureTypes": "all registered",
    "machiningAccessDirection": "not set (dialog default None)",
    "limitSearchArea": "not set",
    "assignColor": "not set",
    "addCadFeatureAttributes": "not set",
    "camConfiguration": "cam_general",
    "camSetup": "mill_contour",
    "attributeSource": "CAMFeature.Attributes (enumerated)",
    "verifiedAgainstGui": "2026-08-28, node 17' stage 2",
}


def attribute_kind(attribute):
    """The attribute's value type as a name, whichever form the binding gives."""
    raw = str(attribute.Type).rsplit(".", 1)[-1]
    if raw.isdigit() and int(raw) < len(_ATTR_TYPES):
        return _ATTR_TYPES[int(raw)]
    return raw


def _enum_value(holders, member):
    """One enum member, whichever of the binding's spellings holds it."""
    for holder in holders:
        cursor = NXOpen
        try:
            for piece in holder.split("."):
                cursor = getattr(cursor, piece)
            return getattr(cursor, member)
        except Exception:
            continue
    return None


def read_attributes(feature):
    """Every attribute NX carries on *feature*, enumerated.

    Returns ``{"route", "values", "unreadable", "errors"}``.  ``route`` names
    how the collection enumerated, so a reader can always tell which path
    produced a row; ``unreadable`` names what NX declared and would not hand
    over, with NX's own reason.
    """
    out = {"route": None, "values": {}, "unreadable": [], "errors": []}
    try:
        collection = feature.Attributes
    except Exception as error:
        out["errors"].append("Attributes: %s" % str(error)[:120])
        return out
    items = None
    for route, call in (("iter", lambda: list(collection)),
                        ("GetContents", lambda: list(collection.GetContents()))):
        try:
            items = call()
            out["route"] = route
            break
        except Exception as error:
            out["errors"].append("%s: %s" % (route, str(error)[:100]))
    if items is None:
        return out
    for attribute in items:
        try:
            name = str(attribute.Name)
            kind = attribute_kind(attribute)
        except Exception as error:
            out["errors"].append("name/type: %s" % str(error)[:100])
            continue
        getter = _ATTR_GETTER.get(kind)
        if not getter:
            out["unreadable"].append({"name": name, "type": kind,
                                      "reason": "no scalar getter for this attribute type"})
            continue
        try:
            raw = getattr(attribute, getter)()
        except Exception as error:
            out["unreadable"].append({"name": name, "type": kind,
                                      "reason": str(error)[:100]})
            continue
        value = raw if isinstance(raw, (int, float, bool, str)) else str(raw)[:150]
        out["values"][name] = {"type": kind, "value": value}
    out["unreadable"].sort(key=lambda item: item["name"])
    return out


def read_csys(feature):
    try:
        csys = feature.CoordinateSystem
        element = csys.Orientation.Element
        matrix = [getattr(element, n) for n in
                  ("Xx", "Xy", "Xz", "Yx", "Yy", "Yz", "Zx", "Zy", "Zz")]
        return {"origin": [csys.Origin.X, csys.Origin.Y, csys.Origin.Z],
                "matrix": matrix, "zAxis": matrix[6:9]}
    except Exception as error:
        return {"error": str(error)[:120]}


def export_feature(feature):
    """One NX feature, in NX's own words, as plain records."""
    record = {}
    for source, key in (("Type", "type"), ("TypeDisplayName", "typeDisplayName"),
                        ("Name", "name"), ("SourceType", "sourceType"),
                        ("Status", "status")):
        try:
            record[key] = str(getattr(feature, source))
        except Exception as error:
            record[key] = None
            record.setdefault("readErrors", {})[source] = str(error)[:80]
    record["csys"] = read_csys(feature)
    try:
        record["faceTags"] = sorted(face.Tag for face in feature.GetFaces())
    except Exception as error:
        record["faceTags"] = []
        record["faceTagsError"] = str(error)[:120]
    attributes = read_attributes(feature)
    record["attributes"] = attributes["values"]
    record["unreadable"] = attributes["unreadable"]
    record["attributeRoute"] = attributes["route"]
    if attributes["errors"]:
        record["attributeErrors"] = attributes["errors"]
    return record


def find_features(work, bodies):
    """The standard question, asked once.

    *bodies* is accepted and deliberately unused for the search: the dialog's
    default Search Method is All Geometry, and setting the geometry explicitly
    is what the old recipe did instead of setting the search type.  It stays in
    the signature so the caller reads as "these are the bodies in play".

    The All Geometry enum is *required*, not best-effort.  The builder's own
    default is Workpiece, and a translated STEP has no workpiece assigned - so
    a binding whose enum this code cannot find would quietly return zero
    features on every part.  That is a broken question, not an empty answer,
    and it raises rather than being recorded as "NX found nothing".
    """
    setup = work.CreateCamSetup(RECIPE["camSetup"])
    enum = NXOpen.CAM.FeatureRecognitionBuilder.RecognitionEnum
    builder = setup.CreateFeatureRecognitionBuilder(NXOpen.CAM.CAMObject.Null)
    try:
        registered = list(builder.GetRegisteredFeatureTypes())
        builder.SetFeatureTypes(registered)
        builder.RecognitionType = enum.Parametric
        value = _enum_value(_GEOM_SEARCH, "AllGeometry")
        if value is None:
            raise RuntimeError(
                "this NXOpen binding spells the geometry-search enum somewhere other "
                "than %s; without All Geometry the builder defaults to Workpiece and a "
                "translated STEP has none, so the recipe cannot be asked" % (_GEOM_SEARCH,))
        builder.GeometrySearchType = value
        builder.MapFeatures = False
        found = list(builder.FindFeatures())
    finally:
        builder.Destroy()
    return found, registered


def extract(work, bodies, environment):
    """Everything NX says about this part, ready for ``nxline.native``."""
    found, registered = find_features(work, bodies)
    return {
        "status": "success",
        "recipe": dict(RECIPE),
        "environment": dict(environment),
        "registeredTypeCount": len(registered),
        "features": [export_feature(feature) for feature in found],
    }
