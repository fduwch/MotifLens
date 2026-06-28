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
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_rationale_gnn.data import append_node_evidence_features, append_typed_node_evidence_features, load_evidence, load_graph
from llm_rationale_gnn.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphSAGE with CPU neighbor sampling for very large full graphs.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--llm-evidence", default=None, help="Optional motif evidence CSV used by motif_channel/scalar_plus_motif modes.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--model", choices=["sage", "gcn", "gat"], default="sage")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads for the sampled GAT baseline.")
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--fanouts", default="10,5", help="Comma-separated neighbor fanouts, one per SAGE layer.")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--balanced-train", action="store_true", help="Use class-balanced epoch batches while keeping full graph and full val/test evaluation.")
    parser.add_argument("--balanced-pos-repeat", type=int, default=16, help="Positive repetitions per balanced epoch.")
    parser.add_argument("--balanced-neg-ratio", type=float, default=4.0, help="Negative examples per repeated positive in balanced epochs.")
    parser.add_argument("--train-eval-max-nodes", type=int, default=0, help="Optional train-set nodes for logging metrics only. 0 evaluates all train nodes.")
    parser.add_argument("--eval-every", type=int, default=1, help="Run full train/val/test evaluation every N epochs. Epoch 1 is always evaluated.")
    parser.add_argument("--use-evidence-features", action="store_true")
    parser.add_argument(
        "--evidence-mode",
        choices=["scalar", "motif_channel", "scalar_plus_motif"],
        default="scalar",
        help="scalar keeps the legacy six evidence features; motif_channel expands LLM motifs into typed channels; scalar_plus_motif uses both.",
    )
    parser.add_argument("--evidence-gate", action="store_true", help="Use an evidence-gated SAGE fusion model.")
    parser.add_argument(
        "--motif-types",
        default="rapid_fanout_after_inflow,mule_collector,scatter_gather_laundering,pass_through_relay,peeling_chain_like,victim_drain_then_cashout,service_like_false_positive",
        help="Comma-separated evidence_type values to encode as motif channels.",
    )
    parser.add_argument("--evidence-polarities", default="support,counter", help="Comma-separated polarity values to encode.")
    parser.add_argument("--evidence-chunksize", type=int, default=1_000_000)
    parser.add_argument("--selection-metric", choices=["val_f1", "val_auc", "val_ap", "val_best_f1"], default="val_best_f1")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-split-column", action="store_true", help="Ignore split/mask columns and create a stratified split from labels using --seed.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--graph-cache",
        default=None,
        help="Optional torch cache for the loaded graph tensors; creates it on first use and reuses it later.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_fanouts(value: str, layers: int) -> list[int]:
    fanouts = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not fanouts:
        raise ValueError("--fanouts cannot be empty")
    if len(fanouts) < layers:
        fanouts.extend([fanouts[-1]] * (layers - len(fanouts)))
    return fanouts[:layers]


def class_weight(y: torch.Tensor, train_idx: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    values = y[torch.as_tensor(train_idx, dtype=torch.long)].detach().cpu().numpy()
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
    ce = F.cross_entropy(logits, target, weight=weights, reduction="none", label_smoothing=label_smoothing)
    if loss_name == "focal":
        pt = torch.exp(-ce.detach()).clamp(1e-6, 1.0)
        return (((1.0 - pt) ** focal_gamma) * ce).mean()
    return ce.mean()


class BidirectionalNeighborSampler:
    def __init__(self, edge_index: torch.Tensor, num_nodes: int, fanouts: list[int], seed: int) -> None:
        src = edge_index[0].detach().cpu().numpy().astype(np.int32, copy=False)
        dst = edge_index[1].detach().cpu().numpy().astype(np.int32, copy=False)
        values = np.ones(len(src), dtype=np.uint8)
        self.in_csr = csr_matrix((values, (dst, src)), shape=(num_nodes, num_nodes), dtype=np.uint8)
        self.out_csr = csr_matrix((values, (src, dst)), shape=(num_nodes, num_nodes), dtype=np.uint8)
        self.fanouts = fanouts
        self.rng = np.random.default_rng(seed)

    def _neighbors(self, node: int, fanout: int) -> np.ndarray:
        in_start, in_end = self.in_csr.indptr[node], self.in_csr.indptr[node + 1]
        out_start, out_end = self.out_csr.indptr[node], self.out_csr.indptr[node + 1]
        neigh = np.concatenate(
            [
                self.in_csr.indices[in_start:in_end],
                self.out_csr.indices[out_start:out_end],
                np.array([node], dtype=np.int32),
            ]
        )
        if len(neigh) == 0:
            return neigh
        if len(neigh) > fanout:
            neigh = self.rng.choice(neigh, size=fanout, replace=False)
        neigh = np.unique(neigh)
        if len(neigh) > fanout:
            neigh = self.rng.choice(neigh, size=fanout, replace=False)
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
                if len(neigh) == 0:
                    continue
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


def softmax_positive(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp[:, 1] / exp.sum(axis=1)


def metrics_np(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prob = softmax_positive(logits)
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
        out["ap"] = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        out["auc"] = 0.0
        out["ap"] = 0.0
    return out


def best_f1_threshold_np(y_true: np.ndarray, logits: np.ndarray) -> tuple[float, dict[str, float]]:
    prob = softmax_positive(logits)
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(prob, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    best_threshold = 0.5
    best_metrics = metrics_np(y_true, logits, 0.5)
    for threshold in thresholds:
        item = metrics_np(y_true, logits, float(threshold))
        if item["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = item
    best_metrics["threshold"] = float(best_threshold)
    return best_threshold, best_metrics


def iter_batches(values: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    order = values.copy()
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def balanced_epoch_indices(
    y: torch.Tensor,
    train_idx: np.ndarray,
    pos_repeat: int,
    neg_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    labels = y[torch.as_tensor(train_idx, dtype=torch.long)].detach().cpu().numpy()
    pos = train_idx[labels == 1]
    neg = train_idx[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return train_idx
    pos_repeat = max(1, int(pos_repeat))
    repeated_pos = np.repeat(pos, pos_repeat)
    neg_take = max(1, int(round(len(repeated_pos) * max(0.0, float(neg_ratio)))))
    sampled_neg = rng.choice(neg, size=neg_take, replace=neg_take > len(neg))
    epoch_idx = np.concatenate([repeated_pos, sampled_neg])
    rng.shuffle(epoch_idx)
    return epoch_idx.astype(np.int64, copy=False)


def train_eval_indices(y: torch.Tensor, train_idx: np.ndarray, max_nodes: int, rng: np.random.Generator) -> np.ndarray:
    if max_nodes <= 0 or len(train_idx) <= max_nodes:
        return train_idx
    labels = y[torch.as_tensor(train_idx, dtype=torch.long)].detach().cpu().numpy()
    pos = train_idx[labels == 1]
    neg = train_idx[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return rng.choice(train_idx, size=max_nodes, replace=False).astype(np.int64, copy=False)
    pos_take = min(len(pos), max_nodes // 2)
    neg_take = max_nodes - pos_take
    chosen = np.concatenate(
        [
            rng.choice(pos, size=pos_take, replace=False),
            rng.choice(neg, size=neg_take, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def predict_nodes(
    model: torch.nn.Module,
    sampler: BidirectionalNeighborSampler,
    x_cpu: torch.Tensor,
    node_idx: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    logits_parts: list[np.ndarray] = []
    with torch.no_grad():
        for seeds in iter_batches(node_idx, batch_size, shuffle=False, rng=sampler.rng):
            nodes, edge_index, seed_local = sampler.sample(seeds)
            x_sub = x_cpu[torch.as_tensor(nodes, dtype=torch.long)].to(device)
            edge_index = edge_index.to(device)
            logits = model(x_sub, edge_index, None, return_attention=False)
            logits_parts.append(logits[torch.as_tensor(seed_local, dtype=torch.long, device=device)].detach().cpu().numpy())
    return np.concatenate(logits_parts, axis=0)


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

    cache_path = Path(args.graph_cache) if args.graph_cache else None
    if cache_path is not None and cache_path.exists():
        print(json.dumps({"phase": "load_graph_cache_start", "graph_cache": str(cache_path)}, ensure_ascii=False), flush=True)
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        loaded = SimpleNamespace(
            data=payload["data"],
            feature_columns=payload.get("feature_columns", []),
            edge_feature_columns=payload.get("edge_feature_columns", []),
            node_ids=payload.get("node_ids"),
        )
        print(
            json.dumps(
                {
                    "phase": "load_graph_cache_done",
                    "num_nodes": int(loaded.data.num_nodes),
                    "num_edges": int(loaded.data.num_edges),
                    "elapsed_seconds": round(time.time() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print(json.dumps({"phase": "load_graph_start", "data_dir": args.data_dir}, ensure_ascii=False), flush=True)
        loaded = load_graph(args.data_dir, seed=args.seed, build_edge_attr=False, use_split_column=not args.ignore_split_column)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "data": loaded.data,
                    "feature_columns": loaded.feature_columns,
                    "edge_feature_columns": loaded.edge_feature_columns,
                    "seed": args.seed,
                    "data_dir": args.data_dir,
                    "use_split_column": not args.ignore_split_column,
                },
                cache_path,
            )
            print(
                json.dumps(
                    {
                        "phase": "save_graph_cache_done",
                        "graph_cache": str(cache_path),
                        "elapsed_seconds": round(time.time() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    data = loaded.data
    data.edge_attr = None
    print(
        json.dumps(
            {
                "phase": "load_graph_done",
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.num_edges),
                "num_features": int(data.x.size(-1)),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    base_num_features = int(data.x.size(-1))
    evidence_feature_names: list[str] = []
    evidence_types: list[str] = []
    evidence_stats: dict[str, object] = {}
    if args.use_evidence_features and (args.evidence or args.llm_evidence):
        if args.evidence_mode in {"scalar", "scalar_plus_motif"}:
            if not args.evidence:
                raise ValueError("--evidence is required for scalar evidence modes.")
            print(
                json.dumps(
                    {"phase": "load_scalar_evidence_start", "evidence": args.evidence},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            evidence_score, scalar_evidence_types = load_evidence(args.evidence, data.num_edges)
            data = append_node_evidence_features(data, evidence_score)
            evidence_types.extend(f"scalar:{name}" for name in scalar_evidence_types)
            scalar_feature_names = [
                "scalar_evidence_in_sum",
                "scalar_evidence_out_sum",
                "scalar_evidence_in_count",
                "scalar_evidence_out_count",
                "scalar_evidence_in_max",
                "scalar_evidence_out_max",
            ]
            evidence_feature_names.extend(scalar_feature_names)
            evidence_stats.update(
                {
                    "scalar_evidence_edges": int((evidence_score > 0).sum().item()),
                    "scalar_evidence_types": scalar_evidence_types,
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "load_scalar_evidence_done",
                        "evidence_edges": int((evidence_score > 0).sum().item()),
                        "num_features": int(data.x.size(-1)),
                        "elapsed_seconds": round(time.time() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        if args.evidence_mode in {"motif_channel", "scalar_plus_motif"}:
            motif_path = args.llm_evidence or args.evidence
            if not motif_path:
                raise ValueError("--llm-evidence or --evidence is required for motif_channel evidence mode.")
            motif_types = parse_csv_list(args.motif_types)
            polarities = parse_csv_list(args.evidence_polarities)
            print(
                json.dumps(
                    {
                        "phase": "load_typed_evidence_start",
                        "evidence": motif_path,
                        "motif_types": motif_types,
                        "polarities": polarities,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            data, typed_feature_names, typed_stats = append_typed_node_evidence_features(
                data,
                motif_path,
                evidence_types=motif_types,
                polarities=polarities,
                chunksize=args.evidence_chunksize,
            )
            evidence_feature_names.extend(typed_feature_names)
            evidence_types.extend(f"motif:{name}" for name in motif_types)
            evidence_stats.update(typed_stats)
            print(
                json.dumps(
                    {
                        "phase": "load_typed_evidence_done",
                        **typed_stats,
                        "num_features": int(data.x.size(-1)),
                        "elapsed_seconds": round(time.time() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        evidence_dim = int(data.x.size(-1)) - base_num_features
        print(
            json.dumps(
                {
                    "phase": "load_evidence_done",
                    "evidence_mode": args.evidence_mode,
                    "evidence_dim": evidence_dim,
                    "num_features": int(data.x.size(-1)),
                    "elapsed_seconds": round(time.time() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        evidence_dim = 0

    if args.evidence_gate and evidence_dim <= 0:
        raise ValueError("--evidence-gate requires --use-evidence-features and non-empty evidence features.")

    train_idx = torch.nonzero(data.train_mask, as_tuple=False).view(-1).cpu().numpy()
    val_idx = torch.nonzero(data.val_mask, as_tuple=False).view(-1).cpu().numpy()
    test_idx = torch.nonzero(data.test_mask, as_tuple=False).view(-1).cpu().numpy()
    valid_labels = data.y[data.y >= 0]
    num_classes = int(valid_labels.max().item() + 1)
    fanouts = parse_fanouts(args.fanouts, args.layers)

    print(json.dumps({"phase": "build_sampler_start", "fanouts": fanouts}, ensure_ascii=False), flush=True)
    sampler = BidirectionalNeighborSampler(data.edge_index, data.num_nodes, fanouts, args.seed)
    print(json.dumps({"phase": "build_sampler_done", "elapsed_seconds": round(time.time() - started, 2)}, ensure_ascii=False), flush=True)

    if args.evidence_gate and args.model != "sage":
        raise ValueError("--evidence-gate is only implemented for --model sage.")
    model_name = "evidence_gated_sage" if args.evidence_gate else {"sage": "sage", "gcn": "gcn", "gat": "rationale_gat"}[args.model]
    model = build_model(
        model_name,
        in_channels=data.x.size(-1),
        num_classes=num_classes,
        hidden_channels=args.hidden,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        edge_dim=None,
        evidence_start=base_num_features if args.evidence_gate else None,
        evidence_dim=evidence_dim if args.evidence_gate else None,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    weights = class_weight(data.y, train_idx, num_classes, device)
    if args.balanced_train:
        weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    rng = np.random.default_rng(args.seed)

    best_score = -1.0
    best_epoch = 0
    best_row = None
    best_test_f1_row = None
    best_test_auc_row = None
    history: list[dict] = []
    best_path = out_dir / "best.pt"

    train_metric_idx = train_eval_indices(data.y, train_idx, args.train_eval_max_nodes, rng)
    y_train = data.y[torch.as_tensor(train_metric_idx, dtype=torch.long)].cpu().numpy()
    y_val = data.y[torch.as_tensor(val_idx, dtype=torch.long)].cpu().numpy()
    y_test = data.y[torch.as_tensor(test_idx, dtype=torch.long)].cpu().numpy()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        epoch_train_idx = (
            balanced_epoch_indices(data.y, train_idx, args.balanced_pos_repeat, args.balanced_neg_ratio, rng)
            if args.balanced_train
            else train_idx
        )
        for seeds in iter_batches(epoch_train_idx, args.batch_size, shuffle=True, rng=rng):
            nodes, edge_index, seed_local = sampler.sample(seeds)
            x_sub = data.x[torch.as_tensor(nodes, dtype=torch.long)].to(device)
            edge_index = edge_index.to(device)
            target = data.y[torch.as_tensor(seeds, dtype=torch.long)].to(device)

            optimizer.zero_grad()
            logits = model(x_sub, edge_index, None, return_attention=False)
            seed_logits = logits[torch.as_tensor(seed_local, dtype=torch.long, device=device)]
            loss = classification_loss(seed_logits, target, weights, args.loss, args.focal_gamma, args.label_smoothing)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        should_eval = epoch == 1 or epoch % max(1, args.eval_every) == 0 or epoch == args.epochs
        if not should_eval:
            if epoch % 5 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "epoch_train_nodes": int(len(epoch_train_idx)),
                            "loss": round(float(np.mean(losses)) if losses else 0.0, 5),
                            "phase": "train_only",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
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
                    "feature_columns": loaded.feature_columns,
                    "base_num_features": base_num_features,
                    "evidence_dim": evidence_dim,
                    "evidence_feature_names": evidence_feature_names,
                    "model_name": model_name,
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
            "train_nodes": int(len(train_idx)),
            "train_metric_nodes": int(len(train_metric_idx)),
            "val_nodes": int(len(val_idx)),
            "test_nodes": int(len(test_idx)),
            "fanouts": fanouts,
            "batch_size": args.batch_size,
            "evidence_types": evidence_types,
            "base_num_features": int(base_num_features),
            "evidence_dim": int(evidence_dim),
            "uses_evidence": bool(args.use_evidence_features and (args.evidence or args.llm_evidence)),
            "evidence_mode": args.evidence_mode if args.use_evidence_features and (args.evidence or args.llm_evidence) else "none",
            "evidence_path": args.evidence,
            "llm_evidence_path": args.llm_evidence,
            "evidence_feature_names": evidence_feature_names,
            "evidence_stats": evidence_stats,
            "model_name": model_name,
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
