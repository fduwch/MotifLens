#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_rationale_gnn.data import append_node_evidence_features, augment_training_edges, load_evidence, load_graph
from llm_rationale_gnn.metrics import best_f1_threshold, classification_metrics, threshold_metrics
from llm_rationale_gnn.model import build_model, edge_rationale_kl_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GNN with optional edge-rationale supervision.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", choices=["rationale_gat", "gcn", "sage"], default="rationale_gat")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rationale-weight", type=float, default=0.20)
    parser.add_argument("--use-evidence-features", action="store_true")
    parser.add_argument("--add-reverse-edges", action="store_true")
    parser.add_argument("--add-self-loops", action="store_true")
    parser.add_argument("--selection-metric", choices=["val_f1", "val_auc", "val_ap", "val_best_f1"], default="val_auc")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weight(y: torch.Tensor, mask: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    values = y[mask].detach().cpu().numpy()
    counts = np.bincount(values, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    loss_name: str,
    focal_gamma: float,
    label_smoothing: float,
) -> torch.Tensor:
    ce = F.cross_entropy(
        logits,
        target,
        weight=weights,
        reduction="none",
        label_smoothing=label_smoothing,
    )
    if loss_name == "focal":
        pt = torch.exp(-ce.detach()).clamp(1e-6, 1.0)
        return (((1.0 - pt) ** focal_gamma) * ce).mean()
    return ce.mean()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uses_edge_attr = args.model == "rationale_gat"
    loaded = load_graph(args.data_dir, seed=args.seed, build_edge_attr=uses_edge_attr)
    data = loaded.data
    evidence_score = torch.zeros(data.num_edges, dtype=torch.float32)
    evidence_types: list[str] = []
    if args.evidence:
        evidence_score, evidence_types = load_evidence(args.evidence, data.num_edges)
        if args.use_evidence_features:
            data = append_node_evidence_features(data, evidence_score)
    if not uses_edge_attr:
        data.edge_attr = None
    data, evidence_score = augment_training_edges(
        data,
        evidence_score,
        add_reverse_edges=args.add_reverse_edges,
        add_self_loops=args.add_self_loops,
    )
    if not uses_edge_attr:
        data.edge_attr = None

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    data = data.to(device)
    evidence_score = evidence_score.to(device)

    valid_labels = data.y[data.y >= 0]
    num_classes = int(valid_labels.max().item() + 1)
    edge_dim = data.edge_attr.size(-1) if data.edge_attr is not None else None
    model = build_model(
        args.model,
        in_channels=data.x.size(-1),
        num_classes=num_classes,
        hidden_channels=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        edge_dim=edge_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weight(data.y, data.train_mask, num_classes, device)

    best_score = -1.0
    best_epoch = 0
    best_row = None
    best_test_f1_row = None
    best_test_auc_row = None
    history = []
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        need_attention = args.model == "rationale_gat" and args.rationale_weight > 0 and evidence_score.sum().item() > 0
        if need_attention:
            logits, alpha = model(data.x, data.edge_index, data.edge_attr, return_attention=True)
        else:
            logits = model(data.x, data.edge_index, data.edge_attr, return_attention=False)
            alpha = None
        cls_loss = classification_loss(
            logits[data.train_mask],
            data.y[data.train_mask],
            weights,
            args.loss,
            args.focal_gamma,
            args.label_smoothing,
        )
        rationale_loss = edge_rationale_kl_loss(alpha, data.edge_index, evidence_score, data.num_nodes) if need_attention else torch.tensor(0.0, device=device)
        loss = cls_loss + args.rationale_weight * rationale_loss
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(data.x, data.edge_index, data.edge_attr)
            train_metrics = classification_metrics(eval_logits, data.y, data.train_mask)
            val_metrics = classification_metrics(eval_logits, data.y, data.val_mask)
            test_metrics = classification_metrics(eval_logits, data.y, data.test_mask)
            threshold, val_best_threshold_metrics = best_f1_threshold(eval_logits, data.y, data.val_mask)
            test_at_val_threshold = threshold_metrics(eval_logits, data.y, data.test_mask, threshold)

        row = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "cls_loss": float(cls_loss.item()),
            "rationale_loss": float(rationale_loss.item()),
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
            "val_best_threshold": val_best_threshold_metrics,
            "test_at_val_threshold": test_at_val_threshold,
        }
        history.append(row)
        if best_test_f1_row is None or test_at_val_threshold["f1"] > best_test_f1_row["test_at_val_threshold"]["f1"]:
            best_test_f1_row = row
        if best_test_auc_row is None or test_metrics["auc"] > best_test_auc_row["test"]["auc"]:
            best_test_auc_row = row
        if args.selection_metric == "val_f1":
            select_score = val_metrics["f1"]
        elif args.selection_metric == "val_ap":
            select_score = val_metrics["ap"]
        elif args.selection_metric == "val_best_f1":
            select_score = val_best_threshold_metrics["f1"]
        else:
            select_score = val_metrics["auc"]
        if select_score > best_score:
            best_score = select_score
            best_epoch = epoch
            best_row = row
            torch.save({"model": model.state_dict(), "args": vars(args), "feature_columns": loaded.feature_columns}, best_path)

        if epoch == 1 or epoch % 10 == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "loss": round(float(loss.item()), 5),
                        "rationale_loss": round(float(rationale_loss.item()), 5),
                        "val_f1": round(val_metrics["f1"], 5),
                        "val_auc": round(val_metrics["auc"], 5),
                        "val_best_f1": round(val_best_threshold_metrics["f1"], 5),
                        "threshold": round(threshold, 5),
                        "test_f1": round(test_metrics["f1"], 5),
                        "test_f1_at_val_threshold": round(test_at_val_threshold["f1"], 5),
                        "test_best_f1_so_far": round(best_test_f1_row["test_at_val_threshold"]["f1"], 5),
                        "test_best_f1_epoch": int(best_test_f1_row["epoch"]),
                        "test_best_f1_threshold": round(best_test_f1_row["test_at_val_threshold"]["threshold"], 5),
                        "test_best_auc_so_far": round(best_test_auc_row["test"]["auc"], 5),
                        "test_best_auc_epoch": int(best_test_auc_row["epoch"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if epoch - best_epoch >= args.patience:
            break

    if best_row is None:
        best_row = history[-1]
    metrics = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "best_val_f1": best_row["val"]["f1"],
        "best_val_auc": best_row["val"]["auc"],
        "best_val_ap": best_row["val"]["ap"],
        "best_val_threshold_f1": best_row["val_best_threshold"]["f1"],
        "best_threshold": best_row["val_best_threshold"]["threshold"],
        "best_test": best_row["test"],
        "best_test_at_val_threshold": best_row["test_at_val_threshold"],
        "history_best_test_at_val_threshold": best_test_f1_row,
        "history_best_test_auc": best_test_auc_row,
        "last": history[-1],
        "data": {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.num_edges),
            "num_features": int(data.x.size(-1)),
            "num_classes": num_classes,
            "train_nodes": int(data.train_mask.sum().item()),
            "val_nodes": int(data.val_mask.sum().item()),
            "test_nodes": int(data.test_mask.sum().item()),
            "edge_feature_columns": loaded.edge_feature_columns,
            "evidence_types": evidence_types,
            "evidence_edges": int((evidence_score > 0).sum().item()),
            "augmented_edges": int(data.num_edges),
        },
        "args": vars(args),
        "history": history,
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "best_epoch": best_epoch,
                "selection_metric": args.selection_metric,
                "best_selection_score": best_score,
                "best_val_threshold_f1": best_row["val_best_threshold"]["f1"],
                "best_test_at_val_threshold": best_row["test_at_val_threshold"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
