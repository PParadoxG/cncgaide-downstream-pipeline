"""Enumerate the differences between an OCC-parsed and an NX-parsed PartModel.

This is the measurement the two-kernel architecture exists for: same solid,
same table, two kernels filling it.  Faces are paired geometrically (bbox
centre + diagonal + surface type + radius, greedy mutual nearest — the same
signature family the provider's face matcher uses), then compared field by
field.  Nothing is judged here; the output is the enumerable diff.

Both models come out of the same run: a part directory holds one folder per
line, so the two kernels' answers about one solid sit side by side.

Usage:
    python -m nxline.partmodel_diff [--tag latest] [ID ...]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from contract.partmodel import PartModel, SurfaceType
from contract.partmodel_json import from_json

_PAIR_THRESHOLD = 0.25


def _signature(face) -> tuple | None:
    bbox = face.bbox
    if not isinstance(bbox, dict):
        return None
    low, high = bbox.get("min"), bbox.get("max")
    centre = [(float(low[i]) + float(high[i])) / 2.0 for i in range(3)]
    diagonal = math.dist([float(v) for v in low], [float(v) for v in high])
    radius = face.parameters.get("radius")
    radius = float(radius) if isinstance(radius, (int, float)) else None
    return centre, diagonal, face.surface_type, radius


def pair_faces(occ: PartModel, nx: PartModel) -> tuple[dict[str, str], list[str], list[str]]:
    occ_faces = {fid: _signature(face) for fid, face in occ.faces.items()}
    nx_faces = {fid: _signature(face) for fid, face in nx.faces.items()}
    scale = 1.0
    all_centres = [sig[0] for sig in occ_faces.values() if sig]
    if all_centres:
        lows = [min(c[i] for c in all_centres) for i in range(3)]
        highs = [max(c[i] for c in all_centres) for i in range(3)]
        scale = max(1.0e-9, math.dist(lows, highs))
    pairs = []
    for nx_id, nx_sig in nx_faces.items():
        if nx_sig is None:
            continue
        for occ_id, occ_sig in occ_faces.items():
            if occ_sig is None:
                continue
            cost = (math.dist(nx_sig[0], occ_sig[0]) / scale
                    + abs(nx_sig[1] - occ_sig[1]) / scale)
            if nx_sig[2] != occ_sig[2]:
                cost += 0.5
            if nx_sig[3] is not None and occ_sig[3] is not None:
                cost += abs(nx_sig[3] - occ_sig[3]) / max(1.0, occ_sig[3])
            if cost <= _PAIR_THRESHOLD:
                pairs.append((cost, nx_id, occ_id))
    pairs.sort()
    matched: dict[str, str] = {}
    used_occ: set[str] = set()
    for _cost, nx_id, occ_id in pairs:
        if nx_id in matched or occ_id in used_occ:
            continue
        matched[nx_id] = occ_id
        used_occ.add(occ_id)
    unmatched_nx = sorted(set(nx_faces) - set(matched))
    unmatched_occ = sorted(set(occ_faces) - used_occ)
    return matched, unmatched_nx, unmatched_occ


def _angle(a, b) -> float | None:
    ua = _unit(a)
    ub = _unit(b)
    if ua is None or ub is None:
        return None
    dot = abs(sum(ua[i] * ub[i] for i in range(3)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _unit(vector):
    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        return None
    magnitude = math.sqrt(sum(float(v) ** 2 for v in vector))
    if magnitude <= 1.0e-12:
        return None
    return [float(v) / magnitude for v in vector]


def _stored_outward(face) -> list[float] | None:
    """A plane's outward normal by the contract's reading of what is stored.

    ``parameters['normal']`` is the natural surface normal and ``is_reversed``
    says the outward one is its negation (``contract.partmodel.FaceData``).
    ``_angle`` above deliberately cannot see a sign - it answers "same line?",
    which is the right question for an axis - so the direction is compared
    here instead, and it is the only place a 180-degree disagreement shows up.
    """
    if face is None or face.surface_type != SurfaceType.PLANE:
        return None
    normal = _unit(face.parameters.get("normal"))
    if normal is None:
        return None
    return [-v for v in normal] if bool(face.is_reversed) else normal


def _sampled_outward(face) -> list[float] | None:
    """The same direction as the interior samples see it, averaged."""
    total = [0.0, 0.0, 0.0]
    for sample in (getattr(face, "interior_samples", None) or []):
        direction = sample.get("normal")
        if isinstance(direction, (list, tuple)) and len(direction) == 3:
            total = [a + float(b) for a, b in zip(total, direction)]
    return _unit(total)


def _self_check(face) -> str:
    """Whether one line's own two answers about a plane's outward normal agree."""
    stored = _stored_outward(face)
    sampled = _sampled_outward(face)
    if stored is None or sampled is None:
        return "unavailable"
    return "agree" if sum(stored[i] * sampled[i] for i in range(3)) > 0.0 else "flip"


def diff_part(occ: PartModel, nx: PartModel) -> dict:
    matched, unmatched_nx, unmatched_occ = pair_faces(occ, nx)
    report: dict = {
        "counts": {
            "faces": {"occ": len(occ.faces), "nx": len(nx.faces)},
            "edges": {"occ": len(occ.edges), "nx": len(nx.edges)},
            "vertices": {"occ": len(occ.vertices), "nx": len(nx.vertices)},
            "loops": {"occ": len(occ.loops), "nx": len(nx.loops)},
        },
        "facePairing": {
            "matched": len(matched),
            "unmatchedNx": unmatched_nx,
            "unmatchedOcc": unmatched_occ,
        },
        "faces": [],
    }
    concavity_agree = concavity_total = 0
    for nx_id, occ_id in sorted(matched.items(), key=lambda kv: int(kv[0].rsplit("_", 1)[-1])):
        nf, of = nx.faces[nx_id], occ.faces[occ_id]
        entry: dict = {"nx": nx_id, "occ": occ_id}
        entry["surfaceType"] = {"occ": of.surface_type.value, "nx": nf.surface_type.value,
                                "equal": of.surface_type == nf.surface_type}
        entry["isReversed"] = {"occ": of.is_reversed, "nx": nf.is_reversed,
                               "equal": bool(of.is_reversed) == bool(nf.is_reversed)}
        if isinstance(of.area, (int, float)) and isinstance(nf.area, (int, float)) and of.area:
            entry["areaRelDelta"] = round(abs(float(nf.area) - float(of.area)) / abs(float(of.area)), 8)
        occ_radius, nx_radius = of.parameters.get("radius"), nf.parameters.get("radius")
        if isinstance(occ_radius, (int, float)) and isinstance(nx_radius, (int, float)):
            entry["radiusDelta"] = round(abs(float(nx_radius) - float(occ_radius)), 8)
        axis_keys = ("axis_direction", "normal", "axis")
        for key in axis_keys:
            if key in of.parameters and key in nf.parameters:
                angle = _angle(of.parameters[key], nf.parameters[key])
                if angle is not None:
                    entry[f"{key}AngleDeg"] = round(angle, 6)
                break
        occ_outward, nx_outward = _stored_outward(of), _stored_outward(nf)
        if occ_outward is not None and nx_outward is not None:
            entry["outwardNormal"] = {
                "occ": [round(v, 6) for v in occ_outward],
                "nx": [round(v, 6) for v in nx_outward],
                "agree": sum(occ_outward[i] * nx_outward[i] for i in range(3)) > 0.0,
            }
        entry["normalSelfCheck"] = {"occ": _self_check(of), "nx": _self_check(nf)}
        if of.centroid and nf.centroid:
            entry["centroidDelta"] = round(math.dist(of.centroid, nf.centroid), 8)
        entry["interiorSamples"] = {"occ": len(of.interior_samples), "nx": len(nf.interior_samples)}

        occ_neighbours = {}
        for adjacency in of.adjacencies:
            occ_neighbours[adjacency.adjacent_face_id] = adjacency.connection_type.value
        agreements = mismatches = unpaired = 0
        for adjacency in nf.adjacencies:
            paired_occ = matched.get(adjacency.adjacent_face_id)
            if paired_occ is None or paired_occ not in occ_neighbours:
                unpaired += 1
                continue
            concavity_total += 1
            if occ_neighbours[paired_occ] == adjacency.connection_type.value:
                agreements += 1
                concavity_agree += 1
            else:
                mismatches += 1
        entry["adjacency"] = {"occ": len(of.adjacencies), "nx": len(nf.adjacencies),
                              "concavityAgree": agreements, "concavityMismatch": mismatches,
                              "unpairedNeighbours": unpaired}
        report["faces"].append(entry)
    report["summary"] = {
        "surfaceTypeMatches": sum(1 for f in report["faces"] if f["surfaceType"]["equal"]),
        "isReversedMatches": sum(1 for f in report["faces"] if f["isReversed"]["equal"]),
        "maxAreaRelDelta": max((f.get("areaRelDelta", 0.0) for f in report["faces"]), default=None),
        "maxRadiusDelta": max((f.get("radiusDelta", 0.0) for f in report["faces"]), default=None),
        "maxCentroidDelta": max((f.get("centroidDelta", 0.0) for f in report["faces"]), default=None),
        "concavityAgreementRate": (round(concavity_agree / concavity_total, 4)
                                   if concavity_total else None),
        "outwardNormalAgreements": sum(1 for f in report["faces"]
                                       if f.get("outwardNormal", {}).get("agree") is True),
        "outwardNormalFlips": sum(1 for f in report["faces"]
                                  if f.get("outwardNormal", {}).get("agree") is False),
        "normalSelfCheck": {
            line: {
                verdict: sum(1 for f in report["faces"]
                             if f["normalSelfCheck"][line] == verdict)
                for verdict in ("agree", "flip", "unavailable")
            }
            for line in ("occ", "nx")
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest",
                        help="the run holding both lines (default: latest)")
    parser.add_argument("--out", default=None, help="write the full JSON report here")
    parser.add_argument("ids", nargs="*")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    parts = root / "runs" / args.tag / "parts"
    ids = args.ids or sorted(p.name for p in parts.iterdir()
                             if (p / "nx" / "partmodel.json").is_file())
    reports = {}
    for part_id in ids:
        occ_path = parts / part_id / "occ" / "partmodel.json"
        nx_path = parts / part_id / "nx" / "partmodel.json"
        if not (occ_path.is_file() and nx_path.is_file()):
            print(f"{part_id:45s} SKIP (missing partmodel on one side)")
            continue
        occ = from_json(occ_path.read_text(encoding="utf-8"))
        nx = from_json(nx_path.read_text(encoding="utf-8"))
        report = diff_part(occ, nx)
        reports[part_id] = report
        s = report["summary"]
        p = report["facePairing"]
        c = report["counts"]
        print(f"{part_id:45s} faces occ={c['faces']['occ']:3d} nx={c['faces']['nx']:3d} "
              f"paired={p['matched']:3d} (unmatched occ={len(p['unmatchedOcc'])} nx={len(p['unmatchedNx'])})  "
              f"type={s['surfaceTypeMatches']}/{p['matched']} rev={s['isReversedMatches']}/{p['matched']} "
              f"maxAreaRel={s['maxAreaRelDelta']} maxRad={s['maxRadiusDelta']} "
              f"concavityAgree={s['concavityAgreementRate']} "
              f"normalFlips={s['outwardNormalFlips']}/"
              f"{s['outwardNormalFlips'] + s['outwardNormalAgreements']}")
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1, ensure_ascii=False), encoding="utf-8")
        print("report ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
