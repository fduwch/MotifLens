#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small synthetic Ethereum-like graph for smoke tests.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--nodes", type=int, default=800)
    parser.add_argument("--fraud", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    node_ids = np.array([f"0x{i:040x}" for i in range(args.nodes)])
    fraud_nodes = set(rng.choice(args.nodes, size=args.fraud, replace=False).tolist())
    labels = np.array([1 if i in fraud_nodes else 0 for i in range(args.nodes)])
    split = np.array(["train"] * args.nodes, dtype=object)
    idx = np.arange(args.nodes)
    rng.shuffle(idx)
    split[idx[int(0.7 * args.nodes) : int(0.85 * args.nodes)]] = "val"
    split[idx[int(0.85 * args.nodes) :]] = "test"

    edges = []
    t = 1_700_000_000
    for i in range(args.nodes * 4):
        src = int(rng.integers(0, args.nodes))
        dst = int(rng.integers(0, args.nodes))
        if src == dst:
            continue
        amount = float(rng.lognormal(mean=0.0, sigma=1.0))
        edges.append((node_ids[src], node_ids[dst], t + int(i * rng.integers(20, 600)), amount))

    benign_pool = [i for i in range(args.nodes) if i not in fraud_nodes]
    for f in fraud_nodes:
        collector = int(rng.choice(benign_pool))
        base_t = t + int(rng.integers(0, 100_000))
        amount = float(rng.lognormal(mean=3.0, sigma=0.4))
        edges.append((node_ids[collector], node_ids[f], base_t, amount))
        receivers = rng.choice(benign_pool, size=8, replace=False)
        for j, r in enumerate(receivers):
            edges.append((node_ids[f], node_ids[int(r)], base_t + 60 * (j + 1), amount / 9.0 * float(rng.uniform(0.6, 1.2))))

    nodes = pd.DataFrame({"node_id": node_ids, "label": labels, "split": split})
    edge_df = pd.DataFrame(edges, columns=["src", "dst", "timestamp", "amount"])
    nodes.to_csv(out / "nodes.csv", index=False)
    edge_df.to_csv(out / "edges.csv", index=False)
    print({"out": str(out), "nodes": len(nodes), "edges": len(edge_df), "fraud": int(labels.sum())})


if __name__ == "__main__":
    main()
