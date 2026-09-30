"""Three-component vector arithmetic and canonical numeric forms.

Every function here is a pure function of its arguments: no model, no state, no
tolerance that means anything beyond "this number is too small to divide by".

Two predicates, deliberately kept apart
---------------------------------------
``valid_vector`` accepts a three-element sequence whose items *are* numbers.
``numeric_vector`` accepts one whose items *convert* to numbers.  They are not
the same rule and the modules that grew their own copies did not agree on which
one they wanted, so both are named here rather than silently merged (Node 26
finding C-12).
"""

from __future__ import annotations

import math

#: A direction shorter than this has no direction; dividing by it is noise.
UNIT_TOL = 1.0e-12


def valid_vector(value: object) -> bool:
    """Three finite numbers, as a list or tuple - the strict reading."""
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)


def numeric_vector(value: object) -> list[float] | None:
    """The three floats in *value*, or ``None`` - the coercing reading.

    Unlike :func:`valid_vector` this accepts anything ``float()`` understands,
    which is what the band-frame and band-endpoint walks have always done with
    values read back out of a diagnostics dictionary.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        components = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return components if all(math.isfinite(item) for item in components) else None


def is_numeric_vector(value: object) -> bool:
    """:func:`numeric_vector` as a predicate."""
    return numeric_vector(value) is not None


def unit_vector(value: object) -> list[float] | None:
    """The unit vector along *value*, or ``None`` if it is not a direction."""
    if not valid_vector(value):
        return None
    magnitude = math.sqrt(sum(float(item) ** 2 for item in value))
    if magnitude <= UNIT_TOL:
        return None
    return [float(item) / magnitude for item in value]


def coerced_unit_vector(value: object, tolerance: float = UNIT_TOL) -> list[float] | None:
    """:func:`unit_vector` under the coercing predicate, with a caller tolerance."""
    components = numeric_vector(value)
    if components is None:
        return None
    magnitude = math.sqrt(sum(item * item for item in components))
    if magnitude <= tolerance:
        return None
    return [item / magnitude for item in components]


def dot(left: object, right: object) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def cross(left: object, right: object) -> list[float]:
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


def norm(vector: object) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in vector))


def distance_sq(left: object, right: object) -> float:
    return sum((float(left[axis]) - float(right[axis])) ** 2 for axis in range(3))


def rounded_scalar(value: object) -> float | None:
    """A finite number rounded to the published six decimals, or ``None``."""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return round(float(value), 6)


def canonical_axis(value: object) -> list[float] | None:
    """The axis of *value* with a sign convention: dominant component positive."""
    axis = unit_vector(value)
    if axis is None:
        return None
    dominant = max(range(3), key=lambda index: abs(axis[index]))
    if axis[dominant] < 0.0:
        axis = [-component for component in axis]
    return [round(0.0 if abs(component) < 0.5e-6 else component, 6) for component in axis]


def perpendicular_basis(axis: object) -> tuple[list[float], list[float]]:
    """Two orthonormal vectors spanning the plane perpendicular to *axis*.

    The reference direction is chosen away from the axis, so for any unit axis
    the first cross product is already long enough.  The rest is the degenerate
    guard, and three copies of this function disagreed about it (Node 26
    finding C-12): two retried with a second reference, one returned the world
    axes.  Both survive, in the order that makes each right.  Retrying is what
    stays perpendicular to a short-but-nonzero *axis*, so it goes first; the
    world axes answer only the case where retrying cannot - a zero direction
    has no perpendicular plane, and the copy that did not retry was the only
    one that did not divide by zero there.
    """
    reference = [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0.0, 0.0]
    u = cross(reference, axis)
    magnitude = norm(u)
    if magnitude <= UNIT_TOL:
        reference = [0.0, 1.0, 0.0]
        u = cross(reference, axis)
        magnitude = norm(u)
    if magnitude == 0.0:
        return [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    u = [value / magnitude for value in u]
    v = cross(axis, u)
    return u, v


def distance_to_axis(point: object, location: object, axis: object) -> float:
    """Perpendicular distance from *point* to the line (*location*, *axis*)."""
    delta = [float(point[index]) - float(location[index]) for index in range(3)]
    return norm(cross(delta, axis))


def point_angle(point: object, location: object, axis: object,
                basis: tuple[list[float], list[float]]) -> float:
    """Angle of *point* around the axis line, in the perpendicular plane."""
    delta = [float(point[index]) - float(location[index]) for index in range(3)]
    axial = dot(delta, axis)
    perpendicular = [
        delta[0] - axial * axis[0],
        delta[1] - axial * axis[1],
        delta[2] - axial * axis[2],
    ]
    u, v = basis
    return math.atan2(dot(perpendicular, v), dot(perpendicular, u))


def sample_angular_coverage(samples: object, centre: object, axis: object) -> float:
    """How much of a full turn a sampled curve covers around an axis, in radians.

    Conservative: consecutive samples contribute their short-way angular step,
    so a curve sampled too coarsely reports less coverage than it has, never
    more.  Used as the fallback when an edge carries no usable length.
    """
    if not isinstance(samples, list) or len(samples) < 2 or not valid_vector(centre):
        return 0.0
    basis = perpendicular_basis(axis)
    angles: list[float] = []
    for point in samples:
        if not valid_vector(point):
            continue
        delta = [float(point[index]) - float(centre[index]) for index in range(3)]
        axial = dot(delta, axis)
        radial = [delta[index] - axial * axis[index] for index in range(3)]
        angles.append(math.atan2(dot(radial, basis[1]), dot(radial, basis[0])))
    if len(angles) < 2:
        return 0.0
    total = 0.0
    for left, right in zip(angles, angles[1:]):
        delta = abs(right - left)
        while delta > math.pi:
            delta = abs(delta - 2.0 * math.pi)
        total += delta
    return total
