"""Shared transition-face evidence (chamfer / fillet-round) derived from AAGNet.

AAGNet decides no feature here, and its instance grouping is not used.  It
supplies exactly one thing: per-face evidence that a face is a *transition*
face — a chamfer or a fillet/round — so that endpoint traversal may cross it
instead of stopping on it.  Nothing in this module registers a Chamfer or
Fillet feature; the hint exists only to let the manufacturing-feature modules
see past a mouth treatment to the wall behind it.

Two levels of judgement are exposed, because both are needed:

- **face level** — a single face carries transition evidence;
- **region level** — a *whole set* of faces (typically a complete side-wall
  band) acts as a transition region.

The region level is the load-bearing one.  A pocket mouth chamfer or fillet
frequently forms a complete, legal closed wall band of its own that sits
directly against the real pocket wall band, so "is the face in front of the
band a chamfer" is not a sufficient question — "is this entire band a
transition ring" has to be answerable too.

Scoring
-------
A merged ``PartModel`` face carries one AAGNet vote per source CAD face.  The
per-vote transition score is ``P(chamfer) + P(round)`` read from the full
softmax vector (``top_k`` is the same distribution, truncated to five entries,
and is kept for diagnostics).  The face score is the **mean** across that
face's votes, which is deliberately not the maximum: a seam-merged hole wall
whose one member happens to poll ``round`` must not become a transition face
on the strength of that single member.

Region scoring is the **area-weighted mean of the continuous face score**, not
a second threshold applied to a count of already-thresholded faces.  Area
weighting is what separates "a chamfer ring" from "a pocket wall whose corners
are filleted": the corner rounds of a rounded pocket are genuine ``round``
faces, but carry a small share of the band's area.  Keeping the score
continuous is what separates a real ring from a heterogeneous band containing
one large *ambiguous* face — measured on this corpus, genuine rings score
0.83-1.00 while the worst mixed band reaches 0.52, and a binary-then-fraction
rule mistakenly passed that band because its one 0.57-scoring face happened to
be the biggest.

A face-count floor is kept as a second guard so that a single dominant face
cannot carry a region whose other faces have no transition evidence at all.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from contract.partmodel import PartModel

from contract.evidence import FEATURE_NAMES

TRANSITION_LABELS = ("chamfer", "round")
_TRANSITION_INDEXES = tuple(FEATURE_NAMES.index(label) for label in TRANSITION_LABELS)

# Measured separation on the current corpus: faces that are visually chamfers
# or fillets score 0.86-1.00, while the highest-scoring non-transition face
# reaches 0.41.  The threshold sits in the middle of that gap rather than
# being fitted to any one part.
_FACE_MIN_SCORE = 0.5
_REGION_MIN_MEAN_SCORE = 0.7
_REGION_MIN_FACE_FRACTION = 0.5


@dataclass(frozen=True)
class FaceTransitionHint:
    """Aggregated transition evidence for one active PartModel face."""

    face_id: str
    score: float
    max_vote_score: float
    min_vote_score: float
    vote_count: int
    dominant_label: str | None
    is_transition: bool


@dataclass
class RegionTransitionVerdict:
    """Whether a whole face set acts as a transition region."""

    is_transition: bool
    mean_score: float
    face_fraction: float
    transition_face_ids: list[str] = field(default_factory=list)
    other_face_ids: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "isTransition": self.is_transition,
            "areaWeightedMeanScore": round(self.mean_score, 4),
            "faceFraction": round(self.face_fraction, 4),
            "transitionFaceIds": list(self.transition_face_ids),
            "otherFaceIds": list(self.other_face_ids),
            "reason": self.reason,
        }


class TransitionEvidence:
    """Per-part transition hints; safe to use when AAGNet produced nothing."""

    def __init__(self, model: PartModel, evidence_document: dict | None) -> None:
        self._model = model
        self._hints: dict[str, FaceTransitionHint] = {}
        # Faces a geometric proof elsewhere has established as transitions.  They do not come
        # from AAGNet and are not scored by it, so they carry full confidence: the proof either
        # holds or the face was never added.
        self._proven: frozenset[str] = frozenset()
        document = evidence_document if isinstance(evidence_document, dict) else {}
        self.available = document.get("status") == "success"
        per_face = document.get("per_face_evidence") or []

        votes: dict[str, list[float]] = {}
        labels: dict[str, list[float]] = {}
        for item in per_face:
            face_id = item.get("partmodel_face_id")
            if not face_id:
                continue
            scores = _label_scores(item)
            if scores is None:
                continue
            votes.setdefault(str(face_id), []).append(sum(scores))
            existing = labels.setdefault(str(face_id), [0.0, 0.0])
            for index, value in enumerate(scores):
                existing[index] += value

        for face_id, values in votes.items():
            mean = sum(values) / len(values)
            totals = labels[face_id]
            dominant = TRANSITION_LABELS[0] if totals[0] >= totals[1] else TRANSITION_LABELS[1]
            self._hints[face_id] = FaceTransitionHint(
                face_id=face_id,
                score=round(mean, 6),
                max_vote_score=round(max(values), 6),
                min_vote_score=round(min(values), 6),
                vote_count=len(values),
                dominant_label=dominant if mean >= _FACE_MIN_SCORE else None,
                is_transition=mean >= _FACE_MIN_SCORE,
            )

    # -- face level ---------------------------------------------------------

    def with_extra_faces(self, face_ids) -> "TransitionEvidence":
        """A view of this evidence in which *face_ids* are transitions by proof, not by vote.

        Returns a copy so the shared per-part evidence stays exactly what AAGNet said; only the
        caller that owns the proof sees the widened set.  The hints dictionary is shared, not
        copied - it is never mutated after construction.
        """
        extra = frozenset(str(face_id) for face_id in (face_ids or ()))
        if not extra:
            return self
        clone = copy.copy(self)
        clone._proven = self._proven | extra
        return clone

    def face(self, face_id: str) -> FaceTransitionHint | None:
        return self._hints.get(face_id)

    def is_transition_face(self, face_id: str) -> bool:
        if face_id in self._proven:
            return True
        hint = self._hints.get(face_id)
        return bool(hint and hint.is_transition)

    def transition_face_ids(self) -> list[str]:
        voted = {face_id for face_id, hint in self._hints.items() if hint.is_transition}
        return sorted(voted | self._proven)

    # -- region level -------------------------------------------------------

    def region_verdict(self, face_ids, *, apply_face_fraction_guard: bool = True) -> RegionTransitionVerdict:
        """Decide whether a whole face set acts as one transition region.

        Formal transition grouping passes an already eligible face set, so its
        ``face_fraction`` is descriptive rather than whole-band coverage.  It
        can disable the fraction guard explicitly while retaining the default
        legacy guard for callers that pass a mixed band (such as SWB).
        """
        faces = sorted(set(face_ids))
        if not faces:
            return RegionTransitionVerdict(False, 0.0, 0.0, reason="empty_region")
        # A region made entirely of geometrically proven faces needs no AAGNet run behind it;
        # asking whether the classifier was available would reject a proof for lack of a hint.
        if not self.available and not set(faces) <= self._proven:
            return RegionTransitionVerdict(False, 0.0, 0.0, other_face_ids=faces, reason="no_aagnet_evidence")

        transition = [face_id for face_id in faces if self.is_transition_face(face_id)]
        other = [face_id for face_id in faces if face_id not in set(transition)]
        total_area = sum(self._area(face_id) for face_id in faces)
        if total_area > 0.0:
            mean_score = sum(self._area(face_id) * self._score(face_id) for face_id in faces) / total_area
        else:
            mean_score = sum(self._score(face_id) for face_id in faces) / len(faces)
        face_fraction = len(transition) / len(faces)

        if not transition:
            reason = "no_transition_face_in_region"
        elif mean_score < _REGION_MIN_MEAN_SCORE:
            reason = "transition_mean_score_below_threshold"
        elif apply_face_fraction_guard and face_fraction < _REGION_MIN_FACE_FRACTION:
            reason = "transition_face_fraction_below_threshold"
        else:
            reason = "transition_region"
        return RegionTransitionVerdict(
            is_transition=reason == "transition_region",
            mean_score=mean_score,
            face_fraction=face_fraction,
            transition_face_ids=transition,
            other_face_ids=other,
            reason=reason,
        )

    def _score(self, face_id: str) -> float:
        if face_id in self._proven:
            return 1.0
        hint = self._hints.get(face_id)
        return hint.score if hint is not None else 0.0

    def _area(self, face_id: str) -> float:
        face = self._model.faces.get(face_id)
        area = getattr(face, "area", None) if face is not None else None
        try:
            value = float(area)
        except (TypeError, ValueError):
            return 0.0
        return value if value > 0.0 else 0.0

    # -- diagnostics --------------------------------------------------------

    def diagnostics(self) -> dict:
        transition = self.transition_face_ids()
        return {
            "source": "aagnet_per_face_top_k",
            "available": self.available,
            "labels": list(TRANSITION_LABELS),
            "faceScore": "mean over the face's source-CAD votes of P(chamfer)+P(round)",
            "faceMinScore": _FACE_MIN_SCORE,
            "regionScore": "area-weighted mean of the face scores",
            "regionMinMeanScore": _REGION_MIN_MEAN_SCORE,
            "regionMinFaceFraction": _REGION_MIN_FACE_FRACTION,
            "regionFaceFractionGuardDefault": True,
            "scoredFaceCount": len(self._hints),
            "transitionFaceCount": len(transition),
            "transitionFaceIds": transition,
            "geometricallyProvenFaceIds": sorted(self._proven),
        }


def build_transition_evidence(model: PartModel, evidence_document: dict | None) -> TransitionEvidence:
    return TransitionEvidence(model, evidence_document)


def _label_scores(item: dict) -> tuple[float, float] | None:
    """``(P(chamfer), P(round))`` from the full softmax, else from ``top_k``."""
    softmax = item.get("softmax")
    if isinstance(softmax, list) and len(softmax) > max(_TRANSITION_INDEXES):
        try:
            return tuple(float(softmax[index]) for index in _TRANSITION_INDEXES)  # type: ignore[return-value]
        except (TypeError, ValueError):
            pass
    scores = {label: 0.0 for label in TRANSITION_LABELS}
    for entry in item.get("top_k") or []:
        if isinstance(entry, dict) and entry.get("label") in scores:
            try:
                scores[entry["label"]] = float(entry.get("softmax_score", 0.0))
            except (TypeError, ValueError):
                continue
    return (scores[TRANSITION_LABELS[0]], scores[TRANSITION_LABELS[1]])
