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
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_rationale_gnn.data import load_graph
from llm_rationale_gnn.model import build_model
from scripts.train_sampled import (
    balanced_epoch_indices,
    best_f1_threshold_np,
    class_weight,
    classification_loss,
    iter_batches,
    metrics_np,
    parse_fanouts,
    predict_nodes,
    train_eval_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train generic fraud GNN baselines on the cleaned ResearchH graphs.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=["care_gnn", "graphconsis_gnn", "pc_gnn"], required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--fanouts", default="10,5")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--pc-pos-repeat", type=int, default=16)
    parser.add_argument("--pc-neg-ratio", type=float, default=4.0)
    parser.add_argument("--care-candidate-multiplier", type=int, default=4)
    parser.add_argument("--care-candidate-cap", type=int, default=96)
    parser.add_argument("--graphconsis-lambda", type=float, default=0.10)
    parser.add_argument("--graphconsis-max-edges", type=int, default=4096)
    parser.add_argument("--train-eval-max-nodes", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--selection-metric", choices=["val_f1", "val_auc", "val_ap", "val_best_f1"], default="val_best_f1")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-split-column", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--graph-cache", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cached_or_graph(data_dir: str, seed: int, ignore_split_column: bool, graph_cache: str | None):
    cache_path = Path(graph_cache) if graph_cache else None
    if cache_path and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return SimpleNamespace(
            data=payload["data"],
            feature_columns=payload.get("feature_columns", []),
            edge_feature_columns=payload.get("edge_feature_columns", []),
            node_ids=payload.get("node_ids"),
            cache_used=str(cache_path),
        )
    loaded = load_graph(data_dir, seed=seed, build_edge_attr=False, use_split_column=not ignore_split_column)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "data": loaded.data,
                "feature_columns": loaded.feature_columns,
                "edge_feature_columns": loaded.edge_feature_columns,
                "seed": seed,
                "data_dir": data_dir,
                "use_split_column": not ignore_split_column,
            },
            cache_path,
        )
    return loaded


class CARENeighborSampler:
    """Single-relation CARE-GNN style neighbor sampler.

    The original CARE-GNN learns relation-aware neighbor filters. These datasets
    are homogeneous after preprocessing, so this scalable baseline applies the
    same fraud-GNN idea as a feature-similarity neighbor filter before GraphSAGE
    aggregation. Labels are never used in neighbor selection.
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        fanouts: list[int],
        seed: int,
        features: torch.Tensor,
        candidate_multiplier: int,
        candidate_cap: int,
    ) -> None:
        src = edge_index[0].detach().cpu().numpy().astype(np.int32, copy=False)
        dst = edge_index[1].detach().cpu().numpy().astype(np.int32, copy=False)
        values = np.ones(len(src), dtype=np.uint8)
        self.in_csr = csr_matrix((values, (dst, src)), shape=(num_nodes, num_nodes), dtype=np.uint8)
        self.out_csr = csr_matrix((values, (src, dst)), shape=(num_nodes, num_nodes), dtype=np.uint8)
        self.fanouts = fanouts
        self.rng = np.random.default_rng(seed)
        self.features = features.detach().cpu().numpy()
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.candidate_cap = max(8, int(candidate_cap))

    def _neighbors(self, node: int, fanout: int) -> np.ndarray:
        in_start, in_end = self.in_csr.indptr[node], self.in_csr.indptr[node + 1]
        out_start, out_end = self.out_csr.indptr[node], self.out_csr.indptr[node + 1]
        neigh = np.concatenate(
            [
                self.in_csr.indices[in_start:in_end],
                self.out_csr.indices[out_start:out_end],
            ]
        )
        if len(neigh) == 0:
            return np.array([node], dtype=np.int64)
        neigh = np.unique(neigh).astype(np.int64, copy=False)
        budget = min(max(fanout * self.candidate_multiplier, fanout + 1), self.candidate_cap)
        if len(neigh) > budget:
            neigh = self.rng.choice(neigh, size=budget, replace=False).astype(np.int64, copy=False)

        center = self.features[node].astype(np.float32, copy=False)
        cand = self.features[neigh].astype(np.float32, copy=False)
        denom = (np.linalg.norm(cand, axis=1) * (np.linalg.norm(center) + 1e-12)) + 1e-12
        sim = np.nan_to_num((cand @ center) / denom, nan=-1.0, posinf=1.0, neginf=-1.0)
        take = max(1, fanout - 1)
        if len(neigh) > take:
            idx = np.argpartition(-sim, take - 1)[:take]
            neigh = neigh[idx]
        neigh = np.unique(np.concatenate([neigh, np.array([node], dtype=np.int64)]))
        if len(neigh) > fanout:
            if node in neigh:
                rest = neigh[neigh != node]
                rest = self.rng.choice(rest, size=fanout - 1, replace=False).astype(np.int64, copy=False)
                neigh = np.concatenate([rest, np.array([node], dtype=np.int64)])
            else:
                neigh = self.rng.choice(neigh, size=fanout, replace=False).astype(np.int64, copy=False)
        return neigh.astype(np.int64, copy=False)

    def sample(self, seeds: np.ndarray) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
        seeds = np.asarray(seeds, dtype=np.int64)
        frontier = seeds
        all_parts = [seeds]
        edge_src: list[int] = []
        edge_dst: list[int] = []
        for fanout in self.fanouts:
            next_parts: list[np.ndarray] = []
            for node in frontier:
                neigh = self._neighbors(int(node), fanout)
                edge_src.extend(int(x) for x in neigh)
                edge_dst.extend([int(node)] * len(neigh))
                next_parts.append(neigh)
            if not next_parts:
                break
            frontier = np.unique(np.concatenate(next_parts))
            all_parts.append(frontier)

        nodes = np.unique(np.concatenate(all_parts))
        local = {int(node): idx for idx, node in enumerate(nodes)}
        local_src = np.fromiter((local[int(x)] for x in edge_src), dtype=np.int64, count=len(edge_src))
        local_dst = np.fromiter((local[int(x)] for x in edge_dst), dtype=np.int64, count=len(edge_dst))
        edge_index = torch.tensor(np.vstack([local_src, local_dst]), dtype=torch.long)
        seed_local = np.fromiter((local[int(x)] for x in seeds), dtype=np.int64, count=len(seeds))
        return nodes, edge_index, seed_local


def make_sampler(args: argparse.Namespace, data, fanouts: list[int]):
    if args.method == "care_gnn":
        return CARENeighborSampler(
            data.edge_index,
            data.num_nodes,
            fanouts,
            args.seed,
            data.x,
            args.care_candidate_multiplier,
            args.care_candidate_cap,
        )
    from scripts.train_sampled import BidirectionalNeighborSampler

    return BidirectionalNeighborSampler(data.edge_index, data.num_nodes, fanouts, args.seed)


def graphconsis_loss(logits: torch.Tensor, edge_index: torch.Tensor, max_edges: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return logits.new_tensor(0.0)
    src, dst = edge_index[0], edge_index[1]
    keep = src != dst
    src = src[keep]
    dst = dst[keep]
    if src.numel() == 0:
        return logits.new_tensor(0.0)
    if max_edges > 0 and src.numel() > max_edges:
        perm = torch.randperm(src.numel(), device=src.device)[:max_edges]
        src = src[perm]
        dst = dst[perm]
    prob = torch.softmax(logits, dim=-1)
    return torch.mean((prob[src] - prob[dst]).pow(2).sum(dim=-1))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    started = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    loaded = load_cached_or_graph(args.data_dir, args.seed, args.ignore_split_column, args.graph_cache)
    data = loaded.data
    data.edge_attr = None
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1).cpu().numpy()
    val_idx = torch.nonzero(data.val_mask, as_tuple=False).view(-1).cpu().numpy()
    test_idx = torch.nonzero(data.test_mask, as_tuple=False).view(-1).cpu().numpy()
    valid_labels = data.y[data.y >= 0]
    num_classes = int(valid_labels.max().item() + 1)
    fanouts = parse_fanouts(args.fanouts, args.layers)
    sampler = make_sampler(args, data, fanouts)

    model = build_model(
        "sage",
        in_channels=data.x.size(-1),
        num_classes=num_classes,
        hidden_channels=args.hidden,
        layers=args.layers,
        heads=4,
        dropout=args.dropout,
        edge_dim=None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weight(data.y, train_idx, num_classes, device)
    rng = np.random.default_rng(args.seed)
    if args.method == "pc_gnn":
        weights = torch.ones(num_classes, dtype=torch.float32, device=device)
        loss_name = "focal"
    else:
        loss_name = args.loss

    train_metric_idx = train_eval_indices(data.y, train_idx, args.train_eval_max_nodes, rng)
    y_train = data.y[torch.as_tensor(train_metric_idx, dtype=torch.long)].cpu().numpy()
    y_val = data.y[torch.as_tensor(val_idx, dtype=torch.long)].cpu().numpy()
    y_test = data.y[torch.as_tensor(test_idx, dtype=torch.long)].cpu().numpy()

    print(
        json.dumps(
            {
                "phase": "load_done",
                "method": args.method,
                "data_dir": args.data_dir,
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.num_edges),
                "num_features": int(data.x.size(-1)),
                "train_nodes": int(len(train_idx)),
                "val_nodes": int(len(val_idx)),
                "test_nodes": int(len(test_idx)),
                "fanouts": fanouts,
                "device": str(device),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    best_score = -1.0
    best_epoch = 0
    best_row = None
    best_test_f1_row = None
    best_test_auc_row = None
    history: list[dict] = []
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        if args.method == "pc_gnn":
            epoch_train_idx = balanced_epoch_indices(data.y, train_idx, args.pc_pos_repeat, args.pc_neg_ratio, rng)
        else:
            epoch_train_idx = train_idx

        for seeds in iter_batches(epoch_train_idx, args.batch_size, shuffle=True, rng=rng):
            nodes, edge_index, seed_local = sampler.sample(seeds)
            x_sub = data.x[torch.as_tensor(nodes, dtype=torch.long)].to(device)
            edge_index = edge_index.to(device)
            target = data.y[torch.as_tensor(seeds, dtype=torch.long)].to(device)

            optimizer.zero_grad()
            logits = model(x_sub, edge_index, None, return_attention=False)
            seed_logits = logits[torch.as_tensor(seed_local, dtype=torch.long, device=device)]
            loss = classification_loss(seed_logits, target, weights, loss_name, args.focal_gamma, args.label_smoothing)
            if args.method == "graphconsis_gnn" and args.graphconsis_lambda > 0:
                loss = loss + args.graphconsis_lambda * graphconsis_loss(logits, edge_index, args.graphconsis_max_edges)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        should_eval = epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs
        if not should_eval:
            continue

        train_logits = predict_nodes(model, sampler, data.x, train_metric_idx, args.eval_batch_size, device)
        val_logits = predict_nodes(model, sampler, data.x, val_idx, args.eval_batch_size, device)
        test_logits = predict_nodes(model, sampler, data.x, test_idx, args.eval_batch_size, device)

        train_metrics = metrics_np(y_train, train_logits, 0.5)
        val_metrics = metrics_np(y_val, val_logits, 0.5)
        test_metrics = metrics_np(y_test, test_logits, 0.5)
        threshold, val_best_threshold_metrics = best_f1_threshold_np(y_val, val_logits)
        test_at_val_threshold = metrics_np(y_test, test_logits, threshold)
        test_at_val_threshold["threshold"] = float(threshold)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
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
        elif args.selection_metric == "val_auc":
            select_score = val_metrics["auc"]
        else:
            select_score = val_best_threshold_metrics["f1"]
        if select_score > best_score:
            best_score = select_score
            best_epoch = epoch
            best_row = row
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "feature_columns": list(getattr(loaded, "feature_columns", []) or []),
                    "model_name": args.method,
                    "base_model": "sage",
                    "fraud_gnn_variant": args.method,
                },
                best_path,
            )

        if epoch == 1 or epoch % 5 == 0:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "epoch_train_nodes": int(len(epoch_train_idx)),
                        "loss": round(row["loss"], 5),
                        "val_f1": round(val_metrics["f1"], 5),
                        "val_auc": round(val_metrics["auc"], 5),
                        "val_best_f1": round(val_best_threshold_metrics["f1"], 5),
                        "threshold": round(float(threshold), 5),
                        "test_f1": round(test_metrics["f1"], 5),
                        "test_f1_at_val_threshold": round(test_at_val_threshold["f1"], 5),
                        "test_best_f1_so_far": round(best_test_f1_row["test_at_val_threshold"]["f1"], 5),
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
        "method": args.method,
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
            "train_nodes": int(len(train_idx)),
            "train_metric_nodes": int(len(train_metric_idx)),
            "val_nodes": int(len(val_idx)),
            "test_nodes": int(len(test_idx)),
            "fanouts": fanouts,
            "batch_size": int(args.batch_size),
            "model_name": args.method,
            "base_model": "sage",
            "feature_columns": list(getattr(loaded, "feature_columns", []) or []),
            "uses_evidence": False,
            "fraud_gnn_variant": args.method,
            "graphconsis_lambda": float(args.graphconsis_lambda) if args.method == "graphconsis_gnn" else 0.0,
            "graphconsis_max_edges": int(args.graphconsis_max_edges) if args.method == "graphconsis_gnn" else 0,
        },
        "args": vars(args),
        "history": history,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "best_epoch": best_epoch,
                "best_val_threshold_f1": metrics["best_val_threshold_f1"],
                "best_test_at_val_threshold": metrics["best_test_at_val_threshold"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
