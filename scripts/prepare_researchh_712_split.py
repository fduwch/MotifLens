#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ResearchH 7:1:2 split directories without rewriting edge files.")
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    return parser.parse_args()


def split_targets(rows: list[dict[str, str]], seed: int, train_ratio: float, val_ratio: float) -> dict[str, str]:
    by_label: dict[str, list[str]] = {}
    for row in rows:
        label = row.get("label", "")
        if label in {"0", "1"}:
            by_label.setdefault(label, []).append(row["node_id"])

    rng = random.Random(seed)
    split: dict[str, str] = {}
    for label, node_ids in sorted(by_label.items()):
        ids = node_ids[:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        for node_id in ids[:n_train]:
            split[node_id] = "train"
        for node_id in ids[n_train : n_train + n_val]:
            split[node_id] = "val"
        for node_id in ids[n_train + n_val :]:
            split[node_id] = "test"
    return split


def main() -> None:
    args = parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (src / "nodes.csv").open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if "node_id" not in fieldnames or "label" not in fieldnames:
        raise ValueError("nodes.csv must contain node_id and label columns")
    if "split" not in fieldnames:
        fieldnames.append("split")

    split = split_targets(rows, args.seed, args.train_ratio, args.val_ratio)
    for row in rows:
        label = row.get("label", "")
        row["split"] = split.get(row["node_id"], "") if label in {"0", "1"} else ""

    with (out / "nodes.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    edge_src = src / "edges.csv"
    edge_dst = out / "edges.csv"
    if edge_dst.exists() or edge_dst.is_symlink():
        edge_dst.unlink()
    try:
        os.symlink(edge_src.resolve(), edge_dst)
    except OSError:
        shutil.copy2(edge_src, edge_dst)

    summary: dict[str, object] = {
        "src_dir": str(src),
        "out_dir": str(out),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": round(1.0 - args.train_ratio - args.val_ratio, 10),
        "splits": {},
    }
    for row in rows:
        label = row.get("label", "")
        split_name = row.get("split", "")
        if label not in {"0", "1"} or not split_name:
            continue
        bucket = summary["splits"].setdefault(split_name, {"n": 0, "labels": {"0": 0, "1": 0}})
        bucket["n"] += 1
        bucket["labels"][label] += 1

    with (out / "split_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
