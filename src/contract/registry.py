"""The provider registry: which kernel line fills which of the two slots.

A *line* is a pairing of two providers:

* the **geometry** slot parses the source file into a ``PartModel`` and
  implements :class:`contract.kernel.KernelServices`;
* the **recognition** slot produces face semantics — either per-face
  evidence votes (AAGNet) or the vendor's own typed feature records
  (NX FBM).

The two lines that exist are data, not branches: the open-source line is
(occline, aagnet), the full NX line is (nxline, nx_fbm).  The composition
root (``core`` / ``store.sync``) looks the line up here and imports
the named packages itself; nothing in this module imports a provider.

There is deliberately no (occline, nx_fbm) pairing.  It existed as the
"hybrid lattice" through node 13 and was removed by user decision on
2026-08-28: the open-source line must stay purely open source, and one part
carrying two recognitions over one geometry made the store layout and the
viewer's file naming answer to two different questions at once.  The
controlled comparison it provided (same PartModel, two recognisers) is a
known loss, recorded in ``NODE_REPORT/Node14_TwoLines_StoreAdoption_Plan_v1.md``.

Capabilities are declared, not probed, and **refusal is the design**: a
consumer that needs a capability the provider does not declare must record a
named gap and stop, never adapt the data into a lookalike (a vendor's own
feature record is not a softmax, and pretending it is one is how a second
source degrades silently — the exact failure the seed-modularization
decision exists to prevent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

SLOT_GEOMETRY = "geometry"
SLOT_RECOGNITION = "recognition"

# Recognition evidence kinds, the capability consumers dispatch on.
EVIDENCE_PER_FACE_SOFTMAX = "per_face_softmax"       # complete per-face label distribution
# The recognizer's own typed features - its class name, its attributes, its
# face tags - taken as they come.  Until node 18' this line restated them as
# FaceAssertion documents in a shared vocabulary; that translation layer was
# retired in node 23, and the declaration says so rather than naming a
# document type no provider produces any more.
EVIDENCE_NATIVE_FEATURES = "native_typed_features"


@dataclass(frozen=True)
class ProviderSpec:
    """One provider: its slot, and what it declares it can do."""

    provider_id: str
    slot: str
    # Declared capabilities, e.g. {"evidence": EVIDENCE_NATIVE_FEATURES}.
    # Absence of a key is a refusal, and consumers must treat it as one.
    capabilities: dict = field(default_factory=dict)
    # The package a composition root imports to drive this provider.  Stored
    # as a string precisely so this module never imports a kernel line.
    package: str = ""


@dataclass(frozen=True)
class Line:
    """One runnable pairing of a geometry provider and a recognition provider.

    ``line_id`` doubles as the folder a part's artifacts are written into
    (``runs/<tag>/parts/<id>/<line_id>/``), which is why the ids are short and
    filesystem-safe.  Both folders hold the *same file names*, so one reader
    reads either line by changing the folder and nothing else.
    """

    line_id: str
    geometry: str
    recognition: str


PROVIDERS: dict[str, ProviderSpec] = {
    "occline": ProviderSpec(
        provider_id="occline",
        slot=SLOT_GEOMETRY,
        capabilities={"partmodel": True, "kernel_services": True, "tessellate": True},
        package="occline",
    ),
    "nxline": ProviderSpec(
        provider_id="nxline",
        slot=SLOT_GEOMETRY,
        # Landed in node 12 M4: NX fills the PartModel and implements all six
        # KernelServices methods, so it declares what occline declares.
        capabilities={"partmodel": True, "kernel_services": True, "tessellate": True},
        package="nxline",
    ),
    "aagnet": ProviderSpec(
        provider_id="aagnet",
        slot=SLOT_RECOGNITION,
        capabilities={"evidence": EVIDENCE_PER_FACE_SOFTMAX},
        package="occline.aagnet",
    ),
    "nx_fbm": ProviderSpec(
        provider_id="nx_fbm",
        slot=SLOT_RECOGNITION,
        capabilities={"evidence": EVIDENCE_NATIVE_FEATURES},
        package="nxline",
    ),
}

LINE_OCC = "occ"
LINE_NX = "nx"

LINES: dict[str, Line] = {
    LINE_OCC: Line(LINE_OCC, geometry="occline", recognition="aagnet"),
    LINE_NX: Line(LINE_NX, geometry="nxline", recognition="nx_fbm"),
}

# Iteration order wherever both lines are built or reported: the open-source
# line first, because it needs neither a licence nor a vendor install and so
# is the one that always produces something.
LINE_ORDER: tuple[str, ...] = (LINE_OCC, LINE_NX)


def provider(provider_id: str) -> ProviderSpec:
    spec = PROVIDERS.get(provider_id)
    if spec is None:
        raise KeyError(f"unknown provider {provider_id!r}; known: {sorted(PROVIDERS)}")
    return spec


def line(line_id: str) -> Line:
    spec = LINES.get(line_id)
    if spec is None:
        raise KeyError(f"unknown line {line_id!r}; known: {sorted(LINES)}")
    geometry = provider(spec.geometry)
    recognition = provider(spec.recognition)
    if geometry.slot != SLOT_GEOMETRY:
        raise ValueError(f"{spec.geometry!r} is not a geometry provider")
    if recognition.slot != SLOT_RECOGNITION:
        raise ValueError(f"{spec.recognition!r} is not a recognition provider")
    return spec


def evidence_kind(recognition_provider_id: str) -> str | None:
    """The declared evidence capability, or ``None`` — a named refusal."""
    return provider(recognition_provider_id).capabilities.get("evidence")
