#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from explain_evidence import EvidenceAgg, TargetMeta, collect_top_edge_refs, read_edge_details, scan_evidence, top_items
from llm_rationale_gnn.data import append_node_evidence_features, load_evidence, load_graph
from llm_rationale_gnn.model import build_model
from train_sampled import BidirectionalNeighborSampler, parse_fanouts, predict_nodes, softmax_positive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model predictions with auditable evidence explanations.")
    parser.add_argument("--data-dir", required=True, help="Directory containing nodes.csv and edges.csv.")
    parser.add_argument("--evidence", required=True, help="Evidence CSV produced by generate_evidence.py.")
    parser.add_argument("--checkpoint", required=True, help="Path to sampled SAGE best.pt.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="test", help="train/val/test/all; only labeled addresses are exported.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of addresses to export. 0 means all matching addresses.")
    parser.add_argument("--top-k", type=int, default=5, help="Top evidence edges to keep per address.")
    parser.add_argument("--chunksize", type=int, default=500000)
    parser.add_argument("--threshold", type=float, default=None, help="Prediction threshold. Defaults to metrics.json best_threshold or 0.5.")
    parser.add_argument("--fanouts", default=None, help="Override fanouts. Defaults to checkpoint args.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def threshold_from_metrics(checkpoint: Path, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    metrics_path = checkpoint.parent / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        for key_path in (
            ("best_threshold",),
            ("best_test_at_val_threshold", "threshold"),
            ("last", "test_at_val_threshold", "threshold"),
        ):
            current: Any = metrics
            for key in key_path:
                if not isinstance(current, dict) or key not in current:
                    current = None
                    break
                current = current[key]
            if current is not None and not (isinstance(current, float) and math.isnan(current)):
                return float(current)
    return 0.5


def split_indices(data: Any, split: str) -> np.ndarray:
    wanted = split.strip().lower()
    labeled = data.y.detach().cpu().numpy() >= 0
    if wanted == "all":
        mask = labeled
    elif wanted == "train":
        mask = data.train_mask.detach().cpu().numpy() & labeled
    elif wanted in {"val", "valid", "validation"}:
        mask = data.val_mask.detach().cpu().numpy() & labeled
    elif wanted == "test":
        mask = data.test_mask.detach().cpu().numpy() & labeled
    else:
        raise ValueError("--split must be train, val, test, or all")
    return np.flatnonzero(mask)


def make_targets(node_ids: np.ndarray, labels: torch.Tensor, indices: np.ndarray, split: str) -> dict[int, TargetMeta]:
    y = labels.detach().cpu().numpy()
    return {
        int(idx): TargetMeta(address=str(node_ids[int(idx)]), label=int(y[int(idx)]), split=split)
        for idx in indices
    }


def mean_score(agg: EvidenceAgg) -> float:
    return float(agg.score_sum / agg.count) if agg.count else 0.0


def dominant_direction(agg: EvidenceAgg) -> str:
    if agg.incoming_score_sum >= agg.outgoing_score_sum:
        return "incoming"
    return "outgoing"


def make_prediction_explanation(prob: float, pred: int, agg: EvidenceAgg) -> str:
    label_text = "钓鱼" if pred == 1 else "非钓鱼"
    if agg.count == 0:
        return f"模型预测钓鱼概率为 {prob:.3f}，判为{label_text}；未检索到超过阈值的结构化 evidence 边。"
    top_support = ", ".join(name for name, _ in agg.support_type_counts.most_common(3)) or "无"
    top_counter = ", ".join(name for name, _ in agg.counter_type_counts.most_common(3)) or "无"
    direction = "入向" if agg.incoming_score_sum >= agg.outgoing_score_sum else "出向"
    reasons = ", ".join(name for name, _ in agg.reason_counts.most_common(3)) or "无额外原因码"
    return (
        f"模型预测钓鱼概率为 {prob:.3f}，判为{label_text}。"
        f"该地址关联 {agg.count} 条可审计 evidence 边，最高证据分 {agg.max_score:.3f}，"
        f"其中 support={agg.support_count} 条、counter={agg.counter_count} 条。"
        f"主要正证据类型为 {top_support}，主要反证类型为 {top_counter}，"
        f"证据强度以{direction}交易为主，主要原因码包括 {reasons}。"
    )


def top_counter_scores(counter: Counter, limit: int = 3) -> str:
    if not counter:
        return ""
    return ";".join(f"{key}:{float(value):.6f}" for key, value in counter.most_common(limit))


def write_outputs(
    out_dir: Path,
    targets: dict[int, TargetMeta],
    probabilities: dict[int, float],
    threshold: float,
    aggs: dict[int, EvidenceAgg],
    refs: dict[int, list[tuple[float, int, str, str, str, str]]],
    edge_details: dict[int, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    address_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for node_idx, meta in targets.items():
        prob = float(probabilities[node_idx])
        pred = int(prob >= threshold)
        agg = aggs.get(node_idx, EvidenceAgg())
        address_rows.append(
            {
                "node_idx": node_idx,
                "address": meta.address,
                "split": meta.split,
                "true_label": meta.label,
                "pred_label": pred,
                "phish_probability": round(prob, 8),
                "threshold": round(float(threshold), 8),
                "is_correct": int(pred == meta.label),
                "evidence_count": agg.count,
                "incoming_count": agg.incoming_count,
                "outgoing_count": agg.outgoing_count,
                "support_count": agg.support_count,
                "counter_count": agg.counter_count,
                "score_sum": round(agg.score_sum, 6),
                "support_score_sum": round(agg.support_score_sum, 6),
                "counter_score_sum": round(agg.counter_score_sum, 6),
                "mean_score": round(mean_score(agg), 6),
                "max_score": round(agg.max_score, 6),
                "dominant_direction": dominant_direction(agg) if agg.count else "",
                "top_evidence_types": top_items(agg.type_counts),
                "top_evidence_type_scores": top_counter_scores(agg.type_scores),
                "top_support_evidence_types": top_items(agg.support_type_counts),
                "top_counter_evidence_types": top_items(agg.counter_type_counts),
                "top_reason_codes": top_items(agg.reason_counts),
                "evidence_explanation": make_prediction_explanation(prob, pred, agg),
            }
        )
        for score, edge_pos, role, evidence_type, reason_codes, polarity in refs.get(node_idx, []):
            edge = edge_details.get(edge_pos, {})
            edge_rows.append(
                {
                    "node_idx": node_idx,
                    "address": meta.address,
                    "true_label": meta.label,
                    "pred_label": pred,
                    "phish_probability": round(prob, 8),
                    "role": role,
                    "edge_pos": edge_pos,
                    "evidence_score": round(float(score), 6),
                    "evidence_type": evidence_type,
                    "polarity": polarity,
                    "reason_codes": reason_codes,
                    "src": edge.get("src", ""),
                    "dst": edge.get("dst", ""),
                    "amount": edge.get("amount", ""),
                    "timestamp": edge.get("timestamp", ""),
                    "block_number": edge.get("block_number", ""),
                    "tx_type": edge.get("tx_type", ""),
                    "tx_hash": edge.get("tx_hash", ""),
                }
            )

    pd.DataFrame(address_rows).sort_values(
        ["pred_label", "phish_probability", "evidence_count"], ascending=[False, False, False]
    ).to_csv(out_dir / "prediction_explanations.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(edge_rows).sort_values(["address", "evidence_score"], ascending=[True, False]).to_csv(
        out_dir / "top_evidence_edges.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )

    y_true = np.array([row["true_label"] for row in address_rows], dtype=np.int64)
    y_pred = np.array([row["pred_label"] for row in address_rows], dtype=np.int64)
    metadata.update(
        {
            "threshold": float(threshold),
            "num_addresses": len(address_rows),
            "num_predicted_positive": int(y_pred.sum()) if len(y_pred) else 0,
            "num_true_positive_label": int(y_true.sum()) if len(y_true) else 0,
            "num_correct": int((y_true == y_pred).sum()) if len(y_true) else 0,
            "num_addresses_with_evidence": sum(1 for node_idx in targets if aggs.get(node_idx, EvidenceAgg()).count > 0),
            "num_addresses_with_support_evidence": sum(
                1 for node_idx in targets if aggs.get(node_idx, EvidenceAgg()).support_count > 0
            ),
            "num_addresses_with_counter_evidence": sum(
                1 for node_idx in targets if aggs.get(node_idx, EvidenceAgg()).counter_count > 0
            ),
            "num_top_edges": len(edge_rows),
        }
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    started = time.time()
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(json.dumps({"phase": "load_graph_start", "data_dir": args.data_dir}, ensure_ascii=False), flush=True)
    loaded = load_graph(args.data_dir, seed=args.seed, build_edge_attr=False)
    data = loaded.data
    data.edge_attr = None

    checkpoint = load_checkpoint(checkpoint_path)
    ckpt_args = dict(checkpoint.get("args", {}))
    use_evidence_features = bool(ckpt_args.get("use_evidence_features", False))
    evidence_types: list[str] = []
    if use_evidence_features:
        print(json.dumps({"phase": "load_evidence_features_start", "evidence": args.evidence}, ensure_ascii=False), flush=True)
        evidence_score, evidence_types = load_evidence(args.evidence, data.num_edges)
        data = append_node_evidence_features(data, evidence_score)

    indices = split_indices(data, args.split)
    if args.limit and args.limit > 0:
        indices = indices[: args.limit]
    targets = make_targets(loaded.node_ids, data.y, indices, args.split)

    hidden = int(ckpt_args.get("hidden", 64))
    layers = int(ckpt_args.get("layers", 2))
    dropout = float(ckpt_args.get("dropout", 0.30))
    fanouts = parse_fanouts(args.fanouts or str(ckpt_args.get("fanouts", "10,5")), layers)
    num_classes = int(data.y[data.y >= 0].max().item() + 1)
    model = build_model(
        "sage",
        in_channels=data.x.size(-1),
        num_classes=num_classes,
        hidden_channels=hidden,
        layers=layers,
        heads=1,
        dropout=dropout,
        edge_dim=None,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    print(
        json.dumps(
            {
                "phase": "predict_start",
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.num_edges),
                "num_targets": int(len(indices)),
                "num_features": int(data.x.size(-1)),
                "fanouts": fanouts,
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    sampler = BidirectionalNeighborSampler(data.edge_index, data.num_nodes, fanouts, args.seed)
    logits = predict_nodes(model, sampler, data.x, indices, args.batch_size, device)
    probs = softmax_positive(logits)
    probabilities = {int(node_idx): float(prob) for node_idx, prob in zip(indices, probs)}
    threshold = threshold_from_metrics(checkpoint_path, args.threshold)

    print(json.dumps({"phase": "scan_evidence_start", "targets": len(targets)}, ensure_ascii=False), flush=True)
    aggs = scan_evidence(Path(args.evidence), targets, args.top_k, args.chunksize)
    refs = collect_top_edge_refs(aggs)
    wanted_positions = {edge_pos for items in refs.values() for _, edge_pos, _, _, _, _ in items}
    print(json.dumps({"phase": "read_edge_details_start", "top_edges": len(wanted_positions)}, ensure_ascii=False), flush=True)
    edge_details = read_edge_details(Path(args.data_dir) / "edges.csv", wanted_positions, args.chunksize)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "data_dir": str(args.data_dir),
        "evidence": str(args.evidence),
        "split": args.split,
        "top_k": int(args.top_k),
        "use_evidence_features": use_evidence_features,
        "evidence_types": evidence_types,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    write_outputs(out_dir, targets, probabilities, threshold, aggs, refs, edge_details, metadata)
    print(json.dumps({"phase": "done", "out_dir": str(out_dir), "elapsed_seconds": round(time.time() - started, 2)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
