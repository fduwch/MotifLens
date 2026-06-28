#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_phishcombine import DATASETS, norm_addr, safe_float, safe_int, split_labels


RELATION_FILES = [
    "trans_eoa_eoa.csv",
    "trans_eoa_ca.csv",
    "trans_ca_eoa.csv",
    "trans_ca_ca.csv",
    "call_eoa_ca.csv",
    "call_ca_ca.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert existing *-Meta_Format aggregate graph into standard graph CSVs.")
    parser.add_argument("--root", default="data/raw/PhishCombine")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--edge-scope", choices=["labeled_incident", "all"], default="labeled_incident")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=500000)
    return parser.parse_args()


def load_label_meta(root: Path, dataset: str) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    raw = json.load(open(root / DATASETS[dataset]["label_file"], "r", encoding="utf-8"))
    labels: dict[str, int] = {}
    meta: dict[str, dict[str, Any]] = {}
    for address, item in raw.items():
        address = norm_addr(address)
        label = int(item.get("VerifiedLabel", -1))
        if address and label in (0, 1):
            labels[address] = label
            meta[address] = {
                "label": label,
                "transaction_types": safe_int(item.get("transaction_types"), 0),
                "span_label": item.get("SpanLabel", ""),
                "title_label": item.get("TitleLabel", ""),
            }
    return labels, meta


def read_feature_table(root: Path, dataset: str) -> pd.DataFrame:
    feature_path = root / DATASETS[dataset]["feature_dir"] / "node_hete_EOA.csv"
    if not feature_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(feature_path)
    if "Address" not in df.columns:
        return pd.DataFrame()
    df["node_id"] = df["Address"].astype(str).str.lower()
    return df.drop(columns=["Address"])


def write_edges(root: Path, dataset: str, out_path: Path, labeled: set[str], edge_scope: str, progress_every: int) -> tuple[Counter, set[str]]:
    edge_dir = root / DATASETS[dataset]["feature_dir"] / "edge"
    stats: Counter = Counter()
    nodes: set[str] = set(labeled)
    relation_to_id = {name: idx for idx, name in enumerate(RELATION_FILES)}
    fieldnames = ["src", "dst", "timestamp", "amount", "tx_count", "relation_id", "relation_name"]

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for relation_name in RELATION_FILES:
            path = edge_dir / relation_name
            if not path.exists():
                stats[f"missing_{relation_name}"] += 1
                continue
            with path.open("r", encoding="utf-8", errors="replace", newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                for row in reader:
                    src = norm_addr(row.get(":START_ID") or row.get("START_ID") or row.get("src"))
                    dst = norm_addr(row.get(":END_ID") or row.get("END_ID") or row.get("dst"))
                    if not src or not dst:
                        stats["bad_rows"] += 1
                        continue
                    if edge_scope == "labeled_incident" and src not in labeled and dst not in labeled:
                        stats["filtered_edges"] += 1
                        continue
                    writer.writerow(
                        {
                            "src": src,
                            "dst": dst,
                            "timestamp": 0,
                            "amount": safe_float(row.get("sum"), 0.0),
                            "tx_count": safe_float(row.get("count"), 0.0),
                            "relation_id": relation_to_id[relation_name],
                            "relation_name": relation_name.replace(".csv", ""),
                        }
                    )
                    nodes.add(src)
                    nodes.add(dst)
                    stats["edges"] += 1
                    stats[f"edges_{relation_name}"] += 1
                    if progress_every and stats["edges"] % progress_every == 0:
                        print(json.dumps({"written_edges": stats["edges"], "relation": relation_name}, ensure_ascii=False), flush=True)
    return stats, nodes


def write_nodes(root: Path, dataset: str, out_path: Path, nodes: set[str], label_meta: dict[str, dict[str, Any]], split: dict[str, str]) -> None:
    rows = []
    for node in sorted(nodes):
        item = label_meta.get(node)
        rows.append(
            {
                "node_id": node,
                "label": int(item["label"]) if item else -1,
                "split": split.get(node, ""),
                "is_target": 1 if item else 0,
                "label_transaction_types": int(item.get("transaction_types", 0)) if item else 0,
                "span_label": item.get("span_label", "") if item else "",
                "title_label": item.get("title_label", "") if item else "",
            }
        )
    df = pd.DataFrame(rows)
    features = read_feature_table(root, dataset)
    if not features.empty:
        df = df.merge(features, on="node_id", how="left")
    df.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, label_meta = load_label_meta(root, args.dataset)
    addresses = sorted(labels)
    split = split_labels(addresses, labels, args.train_ratio, args.val_ratio, args.seed)
    print(
        json.dumps(
            {
                "phase": "start",
                "dataset": args.dataset,
                "labeled_addresses": len(addresses),
                "label_counts": Counter(labels.values()),
                "edge_scope": args.edge_scope,
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )
    stats, nodes = write_edges(root, args.dataset, out_dir / "edges.csv", set(addresses), args.edge_scope, args.progress_every)
    write_nodes(root, args.dataset, out_dir / "nodes.csv", nodes, label_meta, split)
    summary = {
        "dataset": args.dataset,
        "out_dir": str(out_dir),
        "labeled_addresses": len(addresses),
        "label_counts": {str(k): int(v) for k, v in Counter(labels.values()).items()},
        "edge_scope": args.edge_scope,
        "nodes": len(nodes),
        "stats": {str(k): int(v) for k, v in stats.items()},
    }
    with (out_dir / "conversion_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"phase": "done", **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
