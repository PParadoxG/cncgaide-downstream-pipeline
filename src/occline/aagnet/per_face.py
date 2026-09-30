"""AAGNet per-face semantic evidence adapter."""

from __future__ import annotations

import json
import contextlib
import io
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from occline.aagnet.vendor_import import vendor_top_level


_PACKAGE_ROOT = Path(__file__).resolve().parent
_AAGNET_ROOT = _PACKAGE_ROOT / "vendor" / "AAGNet"

# AAGNet is vendored - a research repository, not a package - so its tree
# imports itself by the bare top-level names ``dataset`` / ``models`` /
# ``utils``.  ``vendor_top_level`` binds exactly those three, exactly here, and
# takes them back again: putting the vendor root on ``sys.path`` and leaving it
# there made ``import utils`` mean AAGNet for the whole process (node 26
# finding B-20).
# occwl is *not* vendored: it is installed from its upstream repository at a
# pinned commit by ``install/environment.yml``, the same way torch and dgl are,
# and is imported from the environment.  AAGNet's graph extractor is what
# reaches for it, so a missing occwl surfaces on the import below.
try:
    with vendor_top_level(_AAGNET_ROOT):
        from dataset.AAGExtractor import AAGExtractor
        from models.inst_segmentors import AAGNetSegmentor
        from utils.data_utils import load_one_graph, load_statistics, standardization
except ImportError as _exc:
    if ((getattr(_exc, "name", "") or "").split(".")[0]) != "occwl":
        raise
    # Not a kernel refusal and not a degraded mode - the environment is simply
    # not the one install/SETUP.md builds.  Say so, by name.
    raise ImportError(
        "occwl is not installed, and AAGNet's graph extractor imports it. "
        "See install/SETUP.md, step 1: install/environment.yml pins it to "
        "git+https://github.com/AutodeskAILab/occwl"
        "@8b536ea8b3cf977dbafc1cf0a89eaa28fa996bba. "
        "It is an installed dependency, not a snapshot in this repository."
    ) from _exc

from contract.evidence import FEATURE_NAMES
from occwl_provenance import occwl_provenance



STOCK_CLASS_IDX = 24
_DEFAULT_WEIGHT_PATH = str(_AAGNET_ROOT / "weights" / "weight_on_MFInstseg.pth")
_DEFAULT_ATTR_STAT_PATH = str(_AAGNET_ROOT / "weights" / "attr_stat.json")
_DEFAULT_ATTR_SCHEMA_PATH = str(_AAGNET_ROOT / "feature_lists" / "all.json")
_INST_THRESHOLD = 0.5
_BOTTOM_THRESHOLD = 0.5


def load_aagnet_bundle(device: str = "auto") -> dict[str, Any]:
    torch_device = _resolve_device(device)
    with open(_DEFAULT_ATTR_SCHEMA_PATH, "r", encoding="utf-8") as fh:
        attr_schema = json.load(fh)
    stat = load_statistics(_DEFAULT_ATTR_STAT_PATH)
    model = _build_model(len(FEATURE_NAMES), _DEFAULT_WEIGHT_PATH, torch_device)
    return {
        "device": str(torch_device),
        "model": model,
        "stat": stat,
        "attr_schema": attr_schema,
    }


def run_aagnet_per_face(
    step_path: str,
    *,
    device: str = "auto",
    model: AAGNetSegmentor | None = None,
    stat: dict | None = None,
    attr_schema: dict | None = None,
    face_mappings: dict | None = None,
    save_native_groups_path: str | None = None,
    save_tensor_shapes_path: str | None = None,
    source_ref: str | None = None,
    native_groups_ref: str | None = None,
) -> dict[str, Any]:
    # Where a file *is* on this machine is not part of what AAGNet computed,
    # and this evidence is hashed into the line's artifact digest - so an
    # absolute path here reports "the result changed" when the only thing that
    # moved was the checkout (node 15 found exactly that).  The caller passes
    # the logical names; the absolute paths stay local to the writing.
    source_ref = source_ref or str(step_path)
    native_groups_ref = native_groups_ref or (
        Path(save_native_groups_path).name if save_native_groups_path else None)
    result: dict[str, Any] = {
        "schema_version": "aagnet_per_face_evidence.v1",
        "recognizer": "AAGNet",
        "recognizer_version": "vendor_snapshot",
        # AAGNet is a snapshot in this tree; the graph it is given comes from
        # an occwl that is *not*, so the evidence names that one too.
        **occwl_provenance(),
        "part_id": Path(step_path).stem,
        "source_step": source_ref,
        "face_id_space": "cad_face_id + partmodel_face_id",
        "feature_classes": FEATURE_NAMES,
        "score_semantics": {
            "logits": "raw_model_logits",
            "softmax": "relative_softmax_over_25_classes",
            "top_softmax_score": "not_calibrated_probability",
        },
        "status": "failure",
        "device": None,
        "per_face_evidence": [],
        "mismatch_diagnostics": {},
        "audit": {},
        "error": None,
    }
    t_start = time.time()
    torch_device = _resolve_device(device)
    result["device"] = str(torch_device)
    try:
        if attr_schema is None:
            with open(_DEFAULT_ATTR_SCHEMA_PATH, "r", encoding="utf-8") as fh:
                attr_schema = json.load(fh)
        if stat is None:
            stat = load_statistics(_DEFAULT_ATTR_STAT_PATH)

        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            aag_data = AAGExtractor(str(step_path), attr_schema).process()
            num_nodes = aag_data["graph"]["num_nodes"]
            num_edges = len(aag_data["graph"]["edges"][0])
            sample = load_one_graph(str(step_path), aag_data)
            sample = standardization(sample, stat)
            one_graph = sample["graph"].to(torch_device)

        if model is None:
            model = _build_model(len(FEATURE_NAMES), _DEFAULT_WEIGHT_PATH, torch_device)

        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                seg_out, inst_out, bottom_out = model(one_graph)

        tensor_shapes = {
            "seg_out": list(seg_out.shape),
            "inst_out": [list(t.shape) for t in inst_out],
            "bottom_out": list(bottom_out.shape),
        }
        if save_tensor_shapes_path:
            Path(save_tensor_shapes_path).write_text(
                json.dumps(tensor_shapes, indent=2, ensure_ascii=False),
                encoding="utf-8", newline="",
            )

        mismatch = {
            "aagnet_graph_nodes": num_nodes,
            "aagnet_graph_edges": num_edges,
            "seg_out_rows": seg_out.shape[0],
            "nodes_match_seg_out": seg_out.shape[0] == num_nodes,
        }
        if face_mappings:
            cad_face_count = len(face_mappings.get("cad_face_to_partmodel_face", {}))
            mismatch["partmodel_cad_face_count"] = cad_face_count
            mismatch["nodes_vs_partmodel_cad_match"] = num_nodes == cad_face_count
        result["mismatch_diagnostics"] = mismatch
        result["per_face_evidence"] = _build_per_face_evidence(seg_out, num_nodes, face_mappings)

        native_groups = _extract_native_groups(seg_out, inst_out, bottom_out)
        if save_native_groups_path:
            out = {
                "schema_version": "aagnet_native_groups.audit.v1",
                "source_file": source_ref,
                "recognizer": "AAGNet",
                "note": "AUDIT/COMPARISON ONLY; v6 seed recognition does not use these groups.",
                "native_feature_count": len(native_groups),
                "native_feature_groups": native_groups,
            }
            Path(save_native_groups_path).write_text(
                json.dumps(out, indent=2, ensure_ascii=False),
                encoding="utf-8", newline="",
            )
        result["audit"] = {
            "native_feature_groups_saved": save_native_groups_path is not None,
            "native_feature_group_path": native_groups_ref,
            "native_feature_count": len(native_groups),
            "tensor_shapes": tensor_shapes,
        }
        result["status"] = "success"
    except Exception as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    result["_elapsed_sec"] = round(time.time() - t_start, 3)
    return result


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_spec)


def _build_model(num_classes: int, weight_path: str, device: torch.device) -> AAGNetSegmentor:
    model = AAGNetSegmentor(
        arch="AAGNetGraphEncoder",
        num_classes=num_classes,
        edge_attr_dim=12,
        node_attr_dim=10,
        edge_attr_emb=64,
        node_attr_emb=64,
        edge_grid_dim=0,
        node_grid_dim=7,
        edge_grid_emb=0,
        node_grid_emb=64,
        num_layers=3,
        delta=2,
        mlp_ratio=2,
        drop=0.0,
        drop_path=0.0,
        head_hidden_dim=64,
        conv_on_edge=False,
    )
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model = model.to(device)
    model.eval()
    return model


def _build_per_face_evidence(seg_out: torch.Tensor, num_graph_nodes: int, mappings: dict | None) -> list[dict]:
    logits_np = seg_out.cpu().numpy()
    softmax_np = _softmax(logits_np)
    cad_to_partmodel = (mappings or {}).get("cad_face_to_partmodel_face", {})

    evidence = []
    for i in range(num_graph_nodes):
        logits_i = logits_np[i].tolist()
        softmax_i = softmax_np[i].tolist()
        top_idx = int(np.argmax(logits_np[i]))
        sorted_indices = np.argsort(-logits_np[i])[:5]
        top_k = [
            {
                "label": FEATURE_NAMES[int(idx)] if int(idx) < len(FEATURE_NAMES) else f"class_{int(idx)}",
                "class_index": int(idx),
                "softmax_score": round(softmax_i[int(idx)], 6),
                "logit": round(logits_i[int(idx)], 6),
            }
            for idx in sorted_indices
        ]
        cad_key = str(i)
        evidence.append({
            "cad_face_id": i,
            "partmodel_face_id": cad_to_partmodel.get(cad_key),
            "logits": [round(v, 6) for v in logits_i],
            "softmax": [round(v, 6) for v in softmax_i],
            "top_label": FEATURE_NAMES[top_idx] if top_idx < len(FEATURE_NAMES) else "unknown",
            "top_class_index": top_idx,
            "top_softmax_score": round(softmax_i[top_idx], 6),
            "top_k": top_k,
        })
    return evidence


def _extract_native_groups(seg_out: torch.Tensor, inst_out: list[torch.Tensor], bottom_out: torch.Tensor) -> list[dict]:
    face_logits = seg_out.cpu().numpy()
    inst = inst_out[0].sigmoid()
    adj = (inst > _INST_THRESHOLD).cpu().numpy().astype("int32")
    bottom_logits = (bottom_out.sigmoid() > _BOTTOM_THRESHOLD).cpu().numpy()
    used = np.zeros(adj.shape[0], dtype=bool)
    proposals = []
    for row_idx in range(adj.shape[0]):
        if used[row_idx] or adj[row_idx].sum() <= 1e-6:
            continue
        proposal = [col_idx for col_idx in range(adj.shape[0]) if not used[col_idx] and adj[row_idx, col_idx]]
        for idx in proposal:
            used[idx] = True
        if proposal:
            proposals.append(proposal)

    groups = []
    for proposal in proposals:
        sum_logit = face_logits[proposal].sum(axis=0)
        inst_class = int(np.argmax(sum_logit))
        if inst_class == STOCK_CLASS_IDX:
            continue
        groups.append({
            "feature_type": FEATURE_NAMES[inst_class],
            "cad_face_ids": [int(i) for i in proposal],
            "cad_bottom_face_ids": [int(i) for i in proposal if bottom_logits[i]],
            "score": round(float(sum_logit[inst_class]), 6),
            "score_type": "sum_voting_logit",
            "score_is_calibrated_probability": False,
        })
    return groups


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)
