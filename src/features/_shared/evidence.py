"""Read AAGNet per-face evidence without making it feature ownership."""

from __future__ import annotations

from contract.evidence import best_evidence_for_face, evidence_by_partmodel_face, member_votes


def grouped_evidence(evidence_document: dict) -> dict[str, list[dict]]:
    return evidence_by_partmodel_face(evidence_document)


def best_hint(face_id: str, by_partmodel_face: dict[str, list[dict]]) -> dict | None:
    return best_evidence_for_face(face_id, by_partmodel_face)


def votes(face_id: str, by_partmodel_face: dict[str, list[dict]]) -> list[dict]:
    return member_votes(face_id, by_partmodel_face)
