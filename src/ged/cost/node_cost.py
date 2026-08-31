"""Node attribute distance (geometry term) and node substitution cost."""

from __future__ import annotations

from ged.contract.attribute_policy import ANGLE_RANGE_DEG
from ged.contract.representation import Representation, V5_1
from ged.contract.schema import GedNode
from .config import CostConfig, GeometryTerm


EPS = 1.0e-9

# Category pairs that are closer than "different" (cost layer).
CATEGORY_DISTANCE: dict[frozenset[str], float] = {
    frozenset(("conical_118", "conical_135")): 0.25,
}


def bounded_diff(a: float, b: float) -> float:
    return abs(a - b) / (abs(a) + abs(b) + EPS)


def attribute_distance(kind: str, name: str, left: object, right: object,
                       angle_range: dict[str, float] | None = None) -> float:
    if kind in ("length", "area", "count"):
        return bounded_diff(float(left), float(right))  # type: ignore[arg-type]
    if kind == "angle":
        span = (ANGLE_RANGE_DEG if angle_range is None else angle_range).get(name, 180.0)
        return min(1.0, abs(float(left) - float(right)) / span)  # type: ignore[arg-type]
    if kind == "category":
        if left == right:
            return 0.0
        return CATEGORY_DISTANCE.get(frozenset((str(left), str(right))), 1.0)
    raise ValueError(f"no distance for attribute kind {kind!r}")


def make_geometry_distance(representation: Representation, *, skip_both_zero: bool = False):
    """The geometry term for one representation.

    ``skip_both_zero`` implements node 19' D7 (a) for nx_v1 and is off for
    v5.1.  An NX class publishes every slot its whole inheritance chain
    declares - a ``STEP6HOLE_THREAD`` carries 67 of them - and the unused ones
    read 0.  Two different counterbores agree on every unused slot, and a mean
    over the union would dilute the sections that actually differ by more than
    half.  Skipping the pairs that are 0 on **both** sides removes exactly the
    slots neither feature uses; it is visible in the edit path as
    ``skipped_both_zero``, and what it costs is stated in 19' §3.4: "neither
    of us has a top chamfer" is itself a similarity, and it is not counted.
    """
    angle_range = dict(representation.angle_range_deg)

    def geometry_distance(left: GedNode, right: GedNode) -> GeometryTerm:
        """Mean bounded distance over attributes both nodes publish with a value.

        Attributes only one class owns are not compared (the type cost carries
        that difference); an attribute that is null on either side is skipped
        and reported in ``details`` so the edit path can say why.
        """
        details: list[dict[str, object]] = []
        total = 0.0
        compared = 0
        for name in sorted(set(left.attributes) & set(right.attributes)):
            a = left.attributes[name]
            b = right.attributes[name]
            if a.state != "valid" or b.state != "valid":
                details.append(
                    {
                        "attribute": name,
                        "status": "skipped_null",
                        "left": a.raw,
                        "right": b.raw,
                        "left_reason": a.reason,
                        "right_reason": b.reason,
                    }
                )
                continue
            if skip_both_zero and _both_zero(a.value, b.value):
                details.append(
                    {
                        "attribute": name,
                        "status": "skipped_both_zero",
                        "left": a.raw,
                        "right": b.raw,
                    }
                )
                continue
            distance = attribute_distance(a.kind, name, a.value, b.value, angle_range)
            total += distance
            compared += 1
            details.append(
                {
                    "attribute": name,
                    "kind": a.kind,
                    "left": a.raw,
                    "right": b.raw,
                    "distance": distance,
                }
            )
        return GeometryTerm(
            distance=total / compared if compared else 0.0,
            compared=compared,
            details=tuple(details),
        )

    return geometry_distance


def _both_zero(left: object, right: object) -> bool:
    return (isinstance(left, (int, float)) and isinstance(right, (int, float))
            and abs(float(left)) <= EPS and abs(float(right)) <= EPS)


#: The v5.1 geometry term, unchanged: every shared attribute is compared.
geometry_distance = make_geometry_distance(V5_1)


def substitution_cost(left: GedNode, right: GedNode, config: CostConfig) -> tuple[float, float, GeometryTerm]:
    """Return (cost, type_cost, geometry_term)."""
    type_part = config.type_cost(left.type_key, right.type_key)
    geometry = config.geometry_distance(left, right)
    return config.w_sem * type_part + config.w_geom * geometry.distance, type_part, geometry
