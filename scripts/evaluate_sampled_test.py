#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for item in (ROOT, SCRIPTS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from llm_rationale_gnn.data import append_node_evidence_features, load_evidence, load_graph
from llm_rationale_gnn.model import build_model
from train_sampled import BidirectionalNeighborSampler, parse_fanouts, predict_nodes


def softmax_positive(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp[:, 1] / exp.sum(axis=1)


def metrics_from_prob(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, prob))
        out["ap"] = float(average_precision_score(y_true, prob))
    else:
        out["auc"] = 0.0
        out["ap"] = 0.0
    return out


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(prob, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    best_t = 0.5
    best = metrics_from_prob(y_true, prob, best_t)
    for threshold in thresholds:
        item = metrics_from_prob(y_true, prob, float(threshold))
        if item["f1"] > best["f1"]:
            best_t = float(threshold)
            best = item
    best["threshold"] = best_t
    return best_t, best


def load_cached_graph(data_dir: str, graph_cache: str | None, seed: int):
    cache_path = Path(graph_cache) if graph_cache else None
    if cache_path is not None and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return SimpleNamespace(
            data=payload["data"],
            feature_columns=payload.get("feature_columns", []),
            edge_feature_columns=payload.get("edge_feature_columns", []),
            node_ids=payload.get("node_ids"),
        )
    return load_graph(data_dir, seed=seed, build_edge_attr=False)


def sample_test_indices(data, neg_ratio: float, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    test_idx = torch.nonzero(data.test_mask, as_tuple=False).view(-1).cpu().numpy()
    y = data.y[torch.as_tensor(test_idx, dtype=torch.long)].cpu().numpy()
    pos = test_idx[y == 1]
    neg = test_idx[y == 0]
    neg_take = min(len(neg), int(round(len(pos) * neg_ratio)))
    sampled_neg = rng.choice(neg, size=neg_take, replace=False) if neg_take < len(neg) else neg
    sample = np.concatenate([pos, sampled_neg]).astype(np.int64, copy=False)
    rng.shuffle(sample)
    return sample, {
        "test_pos_total": int(len(pos)),
        "test_neg_total": int(len(neg)),
        "sample_total": int(len(sample)),
        "sample_pos": int(len(pos)),
        "sample_neg": int(len(sampled_neg)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained sampled GraphSAGE model on a fixed positive:negative sampled test set.")
    parser.add_argument("--metrics", required=True, help="metrics.json from train_sampled.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--neg-ratio", type=float, default=2.0)
    parser.add_argument("--sample-seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    metrics_path = Path(args.metrics)
    train_metrics = json.load(open(metrics_path, encoding="utf-8"))
    train_args = train_metrics["args"]
    out_dir = Path(train_args["out_dir"])
    best_path = out_dir / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(best_path)

    seed = int(train_args.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    loaded = load_cached_graph(train_args["data_dir"], train_args.get("graph_cache"), seed)
    data = loaded.data
    data.edge_attr = None
    evidence_types: list[str] = []
    if train_args.get("evidence") and train_args.get("use_evidence_features"):
        evidence_score, evidence_types = load_evidence(train_args["evidence"], data.num_edges)
        data = append_node_evidence_features(data, evidence_score)

    fanouts = parse_fanouts(train_args.get("fanouts", "15,10"), int(train_args.get("layers", 2)))
    sampler = BidirectionalNeighborSampler(data.edge_index, data.num_nodes, fanouts, seed)
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    num_classes = int(data.y[data.y >= 0].max().item() + 1)
    model = build_model(
        "sage",
        in_channels=data.x.size(-1),
        num_classes=num_classes,
        hidden_channels=int(train_args.get("hidden", 128)),
        layers=int(train_args.get("layers", 2)),
        heads=1,
        dropout=float(train_args.get("dropout", 0.3)),
        edge_dim=None,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    sample_idx, sample_info = sample_test_indices(data, args.neg_ratio, args.sample_seed)
    logits = predict_nodes(model, sampler, data.x, sample_idx, int(train_args.get("eval_batch_size", 131072)), device)
    y_sample = data.y[torch.as_tensor(sample_idx, dtype=torch.long)].cpu().numpy()
    prob = softmax_positive(logits)
    val_threshold = float(train_metrics.get("best_threshold", 0.5))
    _, sample_best = best_threshold(y_sample, prob)
    result = {
        "metrics_path": str(metrics_path),
        "model_path": str(best_path),
        "data_dir": train_args["data_dir"],
        "evidence": train_args.get("evidence"),
        "uses_rule_evidence": bool(train_args.get("evidence") and train_args.get("use_evidence_features")),
        "neg_ratio": args.neg_ratio,
        "sample_seed": args.sample_seed,
        **sample_info,
        "threshold_0_5": metrics_from_prob(y_sample, prob, 0.5),
        "val_threshold": {"threshold": val_threshold, **metrics_from_prob(y_sample, prob, val_threshold)},
        "sample_best_threshold": sample_best,
        "full_test_at_val_threshold": train_metrics.get("best_test_at_val_threshold"),
        "history_best_full_test_at_val_threshold": train_metrics.get("history_best_test_at_val_threshold", {}).get("test_at_val_threshold"),
        "evidence_types": evidence_types,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"phase": "done", "out": str(out_path), **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
