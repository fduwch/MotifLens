#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_rationale_gnn.data import load_graph
from scripts.train_sampled import best_f1_threshold_np, class_weight, metrics_np


METHOD_LABELS = {
    "zipzap": "ZipZap",
    "lmae4eth": "LMAE4Eth",
    "bert4eth": "BERT4ETH",
    "tlmgnn": "TLmGNN",
    "bwgnn": "BWGNN",
}

OFFICIAL_REPOS = {
    "zipzap": "https://github.com/git-disl/ZipZap",
    "lmae4eth": "https://github.com/lmae4eth/LMAE4Eth",
    "bert4eth": "https://github.com/Bayi-Hu/BERT4ETH_PyTorch",
    "tlmgnn": "https://github.com/lincozz/TLmGNN",
    "bwgnn": "https://github.com/squareRoot3/Rethinking-Anomaly-Detection",
}

GRAPH_VIEW_METHODS = {"lmae4eth", "tlmgnn", "bwgnn"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train controlled Ethereum SOTA baselines.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--method", choices=sorted(METHOD_LABELS), required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--neighbor-buckets", type=int, default=4096)
    parser.add_argument("--mae-weight", type=float, default=0.05)
    parser.add_argument("--selection-metric", choices=["val_best_f1", "val_auc", "val_ap"], default="val_best_f1")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-split-column", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--sequence-cache", default=None)
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
                "node_ids": loaded.node_ids,
                "seed": seed,
                "data_dir": data_dir,
                "use_split_column": not ignore_split_column,
            },
            cache_path,
        )
    return loaded


def rotate_ring(arr: np.ndarray, counts: np.ndarray) -> None:
    seq_len = arr.shape[1]
    for row, count in enumerate(counts):
        if count > seq_len:
            start = int(count % seq_len)
            arr[row] = np.concatenate([arr[row, start:].copy(), arr[row, :start].copy()])


def build_sequence_payload(data, seq_len: int, neighbor_buckets: int, cache_path: Path | None, data_dir: str):
    expected_meta = {
        "cache_version": 2,
        "data_dir": str(data_dir),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_features": int(data.x.size(1)),
        "seq_len": int(seq_len),
        "neighbor_buckets": int(neighbor_buckets),
    }
    if cache_path and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        meta = payload.get("metadata", {})
        if all(meta.get(key) == value for key, value in expected_meta.items()):
            payload["metadata_verified"] = True
            return payload
        print(
            json.dumps(
                {
                    "phase": "sequence_cache_rebuild",
                    "reason": "missing_or_mismatched_metadata",
                    "cache": str(cache_path),
                    "expected": expected_meta,
                    "found": meta,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    started = time.time()
    target_idx = torch.nonzero(data.y >= 0, as_tuple=False).view(-1).cpu()
    n_targets = int(target_idx.numel())
    target_lookup = torch.full((data.num_nodes,), -1, dtype=torch.int32)
    target_lookup[target_idx] = torch.arange(n_targets, dtype=torch.int32)

    seq_dir = np.zeros((n_targets, seq_len), dtype=np.int64)
    seq_neighbor = np.zeros((n_targets, seq_len), dtype=np.int64)
    tx_count = np.zeros(n_targets, dtype=np.int64)
    graph_sum = torch.zeros((n_targets, data.x.size(1)), dtype=torch.float32)
    graph_count = torch.zeros(n_targets, dtype=torch.float32)

    edge_index = data.edge_index.cpu()
    x_cpu = data.x.float().cpu()
    chunk = 2_000_000
    total_edges = edge_index.size(1)
    for start in range(0, total_edges, chunk):
        end = min(total_edges, start + chunk)
        src = edge_index[0, start:end]
        dst = edge_index[1, start:end]

        src_rows = target_lookup[src].long()
        mask = src_rows >= 0
        if mask.any():
            rows = src_rows[mask]
            neigh = dst[mask]
            graph_sum.index_add_(0, rows, x_cpu[neigh])
            graph_count.index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
            for row, nb in zip(rows.numpy(), neigh.numpy()):
                pos = tx_count[row] % seq_len
                seq_dir[row, pos] = 2
                seq_neighbor[row, pos] = int(nb % neighbor_buckets) + 1
                tx_count[row] += 1

        dst_rows = target_lookup[dst].long()
        mask = dst_rows >= 0
        if mask.any():
            rows = dst_rows[mask]
            neigh = src[mask]
            graph_sum.index_add_(0, rows, x_cpu[neigh])
            graph_count.index_add_(0, rows, torch.ones_like(rows, dtype=torch.float32))
            for row, nb in zip(rows.numpy(), neigh.numpy()):
                pos = tx_count[row] % seq_len
                seq_dir[row, pos] = 1
                seq_neighbor[row, pos] = int(nb % neighbor_buckets) + 1
                tx_count[row] += 1

    rotate_ring(seq_dir, tx_count)
    rotate_ring(seq_neighbor, tx_count)
    denom = graph_count.clamp_min(1.0).unsqueeze(1)
    graph_mean = graph_sum / denom
    degree_feat = torch.log1p(graph_count).unsqueeze(1)

    payload = {
        "target_idx": target_idx,
        "seq_dir": torch.from_numpy(seq_dir).long(),
        "seq_neighbor": torch.from_numpy(seq_neighbor).long(),
        "tx_count": torch.from_numpy(tx_count).long(),
        "graph_mean": graph_mean,
        "degree_feat": degree_feat,
        "seq_len": seq_len,
        "neighbor_buckets": neighbor_buckets,
        "metadata": expected_meta,
        "metadata_verified": True,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_path)
    return payload


class TxSequenceEncoder(nn.Module):
    def __init__(self, seq_len: int, neighbor_buckets: int, hidden: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.dir_emb = nn.Embedding(3, hidden)
        self.neighbor_emb = nn.Embedding(neighbor_buckets + 1, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, seq_dir: torch.Tensor, seq_neighbor: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(seq_dir.size(1), device=seq_dir.device).unsqueeze(0)
        x = self.dir_emb(seq_dir) + self.neighbor_emb(seq_neighbor) + self.pos_emb(pos)
        pad_mask = seq_dir.eq(0) & seq_neighbor.eq(0)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.norm(pooled)


class BertTransactionEncoder(nn.Module):
    def __init__(self, seq_len: int, neighbor_buckets: int, hidden: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.dir_emb = nn.Embedding(3, hidden)
        self.neighbor_emb = nn.Embedding(neighbor_buckets + 1, hidden)
        self.pos_emb = nn.Embedding(seq_len + 1, hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, seq_dir: torch.Tensor, seq_neighbor: torch.Tensor) -> torch.Tensor:
        batch = seq_dir.size(0)
        pos = torch.arange(seq_dir.size(1) + 1, device=seq_dir.device).unsqueeze(0)
        tok = self.dir_emb(seq_dir) + self.neighbor_emb(seq_neighbor)
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, tok], dim=1) + self.pos_emb(pos)
        pad_mask = torch.cat(
            [
                torch.zeros((batch, 1), dtype=torch.bool, device=seq_dir.device),
                seq_dir.eq(0) & seq_neighbor.eq(0),
            ],
            dim=1,
        )
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.norm(x[:, 0])


class ZipZapClassifier(nn.Module):
    def __init__(self, seq_len: int, neighbor_buckets: int, hidden: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.seq = TxSequenceEncoder(seq_len, neighbor_buckets, hidden, layers, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, seq_dir, seq_neighbor, node_x=None, graph_x=None, degree=None):
        return self.classifier(self.seq(seq_dir, seq_neighbor)), None


class BERT4ETHClassifier(nn.Module):
    def __init__(self, seq_len: int, neighbor_buckets: int, hidden: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.seq = BertTransactionEncoder(seq_len, neighbor_buckets, hidden, layers, heads, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, seq_dir, seq_neighbor, node_x=None, graph_x=None, degree=None):
        return self.classifier(self.seq(seq_dir, seq_neighbor)), None


class TLmGNNClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        neighbor_buckets: int,
        num_features: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq = BertTransactionEncoder(seq_len, neighbor_buckets, hidden, layers, heads, dropout)
        self.node_proj = nn.Sequential(nn.LayerNorm(num_features), nn.Linear(num_features, hidden), nn.GELU())
        self.graph_proj = nn.Sequential(nn.LayerNorm(num_features + 1), nn.Linear(num_features + 1, hidden), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )

    def forward(self, seq_dir, seq_neighbor, node_x, graph_x, degree):
        seq_h = self.seq(seq_dir, seq_neighbor)
        node_h = self.node_proj(node_x)
        graph_h = self.graph_proj(torch.cat([graph_x, degree], dim=1))
        gate = self.gate(torch.cat([seq_h, node_h, graph_h], dim=1))
        language_graph = gate * seq_h + (1.0 - gate) * graph_h
        return self.classifier(torch.cat([language_graph, node_h, graph_h], dim=1)), None


class BWGNNClassifier(nn.Module):
    def __init__(self, num_features: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.low = nn.Sequential(nn.LayerNorm(num_features), nn.Linear(num_features, hidden), nn.GELU())
        self.mid = nn.Sequential(nn.LayerNorm(num_features), nn.Linear(num_features, hidden), nn.GELU())
        self.high = nn.Sequential(nn.LayerNorm(num_features), nn.Linear(num_features, hidden), nn.GELU())
        self.degree = nn.Sequential(nn.LayerNorm(1), nn.Linear(1, hidden), nn.GELU())
        self.attn = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Linear(hidden, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, seq_dir, seq_neighbor, node_x, graph_x, degree):
        low = self.low(graph_x)
        mid = self.mid(node_x - graph_x)
        high = self.high(torch.abs(node_x - graph_x))
        deg = self.degree(degree)
        bands = torch.stack([low, mid, high, deg], dim=1)
        weights = torch.softmax(self.attn(torch.cat([low, mid, high, deg], dim=1)), dim=1).unsqueeze(-1)
        wavelet = (bands * weights).sum(dim=1)
        return self.classifier(torch.cat([wavelet, high], dim=1)), None


class LMAE4EthClassifier(nn.Module):
    def __init__(
        self,
        seq_len: int,
        neighbor_buckets: int,
        num_features: int,
        hidden: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq = TxSequenceEncoder(seq_len, neighbor_buckets, hidden, layers, heads, dropout)
        self.node_proj = nn.Sequential(nn.LayerNorm(num_features), nn.Linear(num_features, hidden), nn.GELU())
        self.graph_proj = nn.Sequential(nn.LayerNorm(num_features + 1), nn.Linear(num_features + 1, hidden), nn.GELU())
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2),
        )
        self.reconstruct = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, num_features))

    def forward(self, seq_dir, seq_neighbor, node_x, graph_x, degree):
        seq_h = self.seq(seq_dir, seq_neighbor)
        node_h = self.node_proj(node_x)
        graph_h = self.graph_proj(torch.cat([graph_x, degree], dim=1))
        fused = torch.cat([seq_h, node_h, graph_h], dim=1)
        return self.classifier(fused), self.reconstruct(fused)


def make_loader(indices: np.ndarray, tensors: tuple[torch.Tensor, ...], batch_size: int, shuffle: bool) -> DataLoader:
    idx = torch.from_numpy(indices.astype(np.int64, copy=False))
    ds = TensorDataset(idx, *(t[idx] for t in tensors))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict(model: nn.Module, loader: DataLoader, device: torch.device, method: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            _, seq_dir, seq_neighbor, node_x, graph_x, degree, y = batch
            logits, _ = model(
                seq_dir.to(device),
                seq_neighbor.to(device),
                node_x.to(device) if method in GRAPH_VIEW_METHODS else None,
                graph_x.to(device) if method in GRAPH_VIEW_METHODS else None,
                degree.to(device) if method in GRAPH_VIEW_METHODS else None,
            )
            logits_out.append(logits.detach().cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(logits_out)


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
    seq_cache = Path(args.sequence_cache) if args.sequence_cache else None

    print(
        json.dumps(
            {
                "phase": "sequence_payload_start",
                "method": args.method,
                "data_dir": args.data_dir,
                "graph_cache": args.graph_cache,
                "sequence_cache": str(seq_cache) if seq_cache else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    payload = build_sequence_payload(data, args.seq_len, args.neighbor_buckets, seq_cache, args.data_dir)
    target_idx = payload["target_idx"]
    y_all = data.y[target_idx].long()
    train_mask = data.train_mask[target_idx].cpu().numpy().astype(bool)
    val_mask = data.val_mask[target_idx].cpu().numpy().astype(bool)
    test_mask = data.test_mask[target_idx].cpu().numpy().astype(bool)

    node_x = data.x[target_idx].float()
    graph_x = payload["graph_mean"].float()
    degree = payload["degree_feat"].float()
    tensors = (
        payload["seq_dir"].long(),
        payload["seq_neighbor"].long(),
        node_x,
        graph_x,
        degree,
        y_all,
    )
    train_rows = np.flatnonzero(train_mask)
    val_rows = np.flatnonzero(val_mask)
    test_rows = np.flatnonzero(test_mask)
    train_loader = make_loader(train_rows, tensors, args.batch_size, shuffle=True)
    val_loader = make_loader(val_rows, tensors, args.eval_batch_size, shuffle=False)
    test_loader = make_loader(test_rows, tensors, args.eval_batch_size, shuffle=False)

    if args.method == "zipzap":
        model = ZipZapClassifier(args.seq_len, args.neighbor_buckets, args.hidden, args.layers, args.heads, args.dropout)
    elif args.method == "bert4eth":
        model = BERT4ETHClassifier(args.seq_len, args.neighbor_buckets, args.hidden, args.layers, args.heads, args.dropout)
    elif args.method == "tlmgnn":
        model = TLmGNNClassifier(
            args.seq_len,
            args.neighbor_buckets,
            node_x.size(1),
            args.hidden,
            args.layers,
            args.heads,
            args.dropout,
        )
    elif args.method == "bwgnn":
        model = BWGNNClassifier(node_x.size(1), args.hidden, args.dropout)
    else:
        model = LMAE4EthClassifier(
            args.seq_len,
            args.neighbor_buckets,
            node_x.size(1),
            args.hidden,
            args.layers,
            args.heads,
            args.dropout,
        )
    model.to(device)
    weights = class_weight(y_all, train_rows, 2, device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_row = None
    best_score = -float("inf")
    best_test_f1_row = None
    stale = 0
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            _, seq_dir, seq_neighbor, batch_x, batch_graph, batch_degree, y = batch
            seq_dir = seq_dir.to(device)
            seq_neighbor = seq_neighbor.to(device)
            batch_x = batch_x.to(device)
            batch_graph = batch_graph.to(device)
            batch_degree = batch_degree.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, recon = model(
                seq_dir,
                seq_neighbor,
                batch_x if args.method in GRAPH_VIEW_METHODS else None,
                batch_graph if args.method in GRAPH_VIEW_METHODS else None,
                batch_degree if args.method in GRAPH_VIEW_METHODS else None,
            )
            loss = loss_fn(logits, y)
            if args.method == "lmae4eth" and recon is not None and args.mae_weight > 0:
                mask = torch.rand_like(batch_x).lt(0.20).float()
                denom = mask.sum().clamp_min(1.0)
                recon_loss = (((recon - batch_x) * mask) ** 2).sum() / denom
                loss = loss + args.mae_weight * recon_loss
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        y_val, val_prob = predict(model, val_loader, device, args.method)
        y_test, test_prob = predict(model, test_loader, device, args.method)
        threshold, val_best = best_f1_threshold_np(y_val, val_prob)
        test_at_val = metrics_np(y_test, test_prob, threshold)
        test_default = metrics_np(y_test, test_prob, 0.5)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "val_best_threshold": val_best,
            "test_at_val_threshold": test_at_val,
            "test_default_threshold": test_default,
        }
        history.append(row)
        if best_test_f1_row is None or test_at_val["f1"] > best_test_f1_row["test_at_val_threshold"]["f1"]:
            best_test_f1_row = row

        if args.selection_metric == "val_auc":
            select_score = val_best["auc"]
        elif args.selection_metric == "val_ap":
            select_score = val_best["ap"]
        else:
            select_score = val_best["f1"]
        if select_score > best_score:
            best_score = select_score
            best_row = row
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, best_path)
        else:
            stale += 1

        print(
            json.dumps(
                {
                    "phase": "epoch",
                    "method": args.method,
                    "epoch": epoch,
                    "loss": round(float(np.mean(losses)), 6) if losses else None,
                    "val_best_f1": round(val_best["f1"], 5),
                    "test_f1_at_val_threshold": round(test_at_val["f1"], 5),
                    "best_test_f1_so_far": round(best_test_f1_row["test_at_val_threshold"]["f1"], 5),
                    "stale": stale,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if stale >= args.patience:
            break

    assert best_row is not None
    metrics = {
        "method": METHOD_LABELS[args.method],
        "method_key": args.method,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_score,
        "best_epoch": best_row["epoch"],
        "best_val_threshold_f1": best_row["val_best_threshold"]["f1"],
        "best_threshold": best_row["val_best_threshold"]["threshold"],
        "best_test_at_val_threshold": best_row["test_at_val_threshold"],
        "best_test_default_threshold": best_row["test_default_threshold"],
        "history_best_test_at_val_threshold": best_test_f1_row,
        "data": {
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.size(1)),
            "num_features": int(data.x.size(1)),
            "num_classes": 2,
            "train_nodes": int(len(train_rows)),
            "val_nodes": int(len(val_rows)),
            "test_nodes": int(len(test_rows)),
            "uses_evidence": False,
            "sequence_length": int(args.seq_len),
            "neighbor_buckets": int(args.neighbor_buckets),
            "target_nodes": int(target_idx.numel()),
            "sequence_cache": str(seq_cache) if seq_cache else None,
            "sequence_cache_metadata_verified": bool(payload.get("metadata_verified", False)),
            "sequence_cache_metadata": payload.get("metadata", {}),
            "graph_cache": args.graph_cache,
            "official_repo": OFFICIAL_REPOS[args.method],
        },
        "args": vars(args),
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "phase": "done",
                "method": args.method,
                "out_dir": str(out_dir),
                "best_epoch": metrics["best_epoch"],
                "best_val_threshold_f1": metrics["best_val_threshold_f1"],
                "best_test_f1_at_val_threshold": metrics["best_test_at_val_threshold"]["f1"],
                "elapsed_seconds": metrics["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
