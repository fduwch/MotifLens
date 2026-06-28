#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASETS = {
    "Meta": {
        "label_file": "address_Meta.json",
        "feature_dir": "MetaID-Meta_Format",
    },
    "ZipZap": {
        "label_file": "address_ZipZap.json",
        "feature_dir": "ZipZap-Meta_Format",
    },
    "2DynEth": {
        "label_file": "address_2DynEth.json",
        "feature_dir": "2DynEth-Meta_Format",
    },
}

TX_TYPE_DIRS = {
    "normal": "Normal",
    "internal": "Internal",
    "erc20": "ERC20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PhishCombine datasets into standard nodes.csv/edges.csv format.")
    parser.add_argument("--root", default="data/raw/PhishCombine")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--include-two-hop", action="store_true", help="Also read RelatedAddressTransactions as hop=2 context.")
    parser.add_argument("--tx-types", default="normal,internal,erc20", help="Comma-separated subset: normal,internal,erc20.")
    parser.add_argument("--include-meta-features", action="store_true", help="Merge existing node_hete_EOA.csv features when available.")
    parser.add_argument("--max-addresses", type=int, default=0, help="Optional cap for smoke tests. 0 means all labeled addresses.")
    parser.add_argument("--max-rows-per-file", type=int, default=0, help="Optional per-address transaction row cap. 0 means all rows.")
    parser.add_argument("--max-file-mb", type=float, default=0.0, help="Skip a transaction CSV above this size. 0 means no size cap.")
    parser.add_argument("--max-edges-per-pair", type=int, default=0, help="Cap written transaction edges per directed node pair. 0 means no pair cap.")
    parser.add_argument(
        "--pair-cap-scope",
        choices=["pair", "pair_tx_type_hop"],
        default="pair_tx_type_hop",
        help="Whether the pair cap is shared by a directed pair or separated by tx_type/hop.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def norm_addr(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def normalized_amount(row: dict[str, str], tx_type: str) -> float:
    raw = safe_float(row.get("value"), 0.0)
    if tx_type == "erc20":
        dec = safe_int(row.get("tokenDecimal"), 18)
        dec = max(0, min(36, dec))
        return raw / (10.0 ** dec)
    return raw / 1e18


def split_labels(addresses: list[str], labels: dict[str, int], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {0: [], 1: []}
    for address in addresses:
        label = labels[address]
        by_label.setdefault(label, []).append(address)

    split: dict[str, str] = {}
    for label, values in by_label.items():
        rng.shuffle(values)
        n = len(values)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        for address in values[:n_train]:
            split[address] = "train"
        for address in values[n_train : n_train + n_val]:
            split[address] = "val"
        for address in values[n_train + n_val :]:
            split[address] = "test"
    return split


def load_labels(root: Path, dataset: str, max_addresses: int, seed: int) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    info = DATASETS[dataset]
    label_path = root / info["label_file"]
    raw = json.load(open(label_path, "r", encoding="utf-8"))
    meta: dict[str, dict[str, Any]] = {}
    for address, item in raw.items():
        address_norm = norm_addr(address)
        if not address_norm:
            continue
        label = int(item.get("VerifiedLabel", -1))
        if label not in (0, 1):
            continue
        meta[address_norm] = {
            "label": label,
            "transaction_types": safe_int(item.get("transaction_types"), 0),
            "span_label": item.get("SpanLabel", ""),
            "title_label": item.get("TitleLabel", ""),
        }
    labels = {address: item["label"] for address, item in meta.items()}

    if max_addresses and max_addresses < len(labels):
        rng = random.Random(seed)
        positives = [a for a, y in labels.items() if y == 1]
        negatives = [a for a, y in labels.items() if y == 0]
        pos_take = min(len(positives), max_addresses // 2)
        neg_take = min(len(negatives), max_addresses - pos_take)
        chosen = set(rng.sample(positives, pos_take) + rng.sample(negatives, neg_take))
        labels = {a: labels[a] for a in sorted(chosen)}
        meta = {a: meta[a] for a in sorted(chosen)}
    return labels, meta


def read_feature_table(root: Path, dataset: str) -> pd.DataFrame:
    feature_path = root / DATASETS[dataset]["feature_dir"] / "node_hete_EOA.csv"
    if not feature_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(feature_path)
    if "Address" not in df.columns:
        return pd.DataFrame()
    df["node_id"] = df["Address"].astype(str).str.lower()
    df = df.drop(columns=["Address"])
    return df


def transaction_paths(root: Path, address: str, tx_types: Iterable[str], include_two_hop: bool) -> list[tuple[Path, str, int]]:
    bases = [("RelatedTransactions", 1)]
    if include_two_hop:
        bases.append(("RelatedAddressTransactions", 2))
    out: list[tuple[Path, str, int]] = []
    for base, hop in bases:
        for tx_type in tx_types:
            type_dir = TX_TYPE_DIRS[tx_type]
            out.append((root / base / type_dir / f"{address}.csv", tx_type, hop))
    return out


def iter_csv_rows(path: Path, max_rows: int) -> Iterable[dict[str, str]]:
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            if max_rows and idx >= max_rows:
                break
            yield row


def edge_from_row(row: dict[str, str], tx_type: str, hop: int) -> Optional[dict[str, Any]]:
    src = norm_addr(row.get("from"))
    dst = norm_addr(row.get("to"))
    if not src or not dst:
        return None
    timestamp = safe_int(row.get("timeStamp"), 0)
    raw_value = safe_float(row.get("value"), 0.0)
    amount = normalized_amount(row, tx_type)
    gas = safe_float(row.get("gas"), 0.0)
    gas_price = safe_float(row.get("gasPrice"), 0.0)
    gas_used = safe_float(row.get("gasUsed"), 0.0)
    token_decimal = safe_int(row.get("tokenDecimal"), 0 if tx_type != "erc20" else 18)
    block_number = safe_int(row.get("blockNumber"), 0)
    is_error = safe_int(row.get("isError"), 0)
    tx_hash = str(row.get("hash", "")).strip()
    return {
        "src": src,
        "dst": dst,
        "timestamp": timestamp,
        "amount": amount,
        "raw_value": raw_value,
        "gas": gas,
        "gas_price": gas_price,
        "gas_used": gas_used,
        "token_decimal": token_decimal,
        "block_number": block_number,
        "is_error": is_error,
        "tx_type": tx_type,
        "hop": hop,
        "tx_hash": tx_hash,
    }


def pair_cap_key(edge: dict[str, Any], scope: str) -> tuple[Any, ...]:
    if scope == "pair":
        return (edge["src"], edge["dst"])
    return (edge["src"], edge["dst"], edge["tx_type"], edge["hop"])


def write_edges(
    root: Path,
    addresses: list[str],
    tx_types: list[str],
    include_two_hop: bool,
    out_path: Path,
    max_rows_per_file: int,
    max_file_mb: float,
    max_edges_per_pair: int,
    pair_cap_scope: str,
    progress_every: int,
) -> tuple[Counter, set[str]]:
    fieldnames = [
        "src",
        "dst",
        "timestamp",
        "amount",
        "raw_value",
        "gas",
        "gas_price",
        "gas_used",
        "token_decimal",
        "block_number",
        "is_error",
        "tx_type",
        "hop",
        "tx_hash",
    ]
    seen: set[tuple[Any, ...]] = set()
    pair_counts: Counter = Counter()
    nodes: set[str] = set(addresses)
    stats: Counter = Counter()
    started = time.time()

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, address in enumerate(addresses, 1):
            for path, tx_type, hop in transaction_paths(root, address, tx_types, include_two_hop):
                if not path.exists():
                    stats[f"missing_{tx_type}_hop{hop}"] += 1
                    continue
                size_mb = path.stat().st_size / (1024 * 1024)
                if max_file_mb and size_mb > max_file_mb:
                    stats[f"skipped_large_{tx_type}_hop{hop}"] += 1
                    continue
                stats[f"files_{tx_type}_hop{hop}"] += 1
                try:
                    for row in iter_csv_rows(path, max_rows_per_file):
                        edge = edge_from_row(row, tx_type, hop)
                        if edge is None:
                            stats["bad_rows"] += 1
                            continue
                        tx_hash = edge["tx_hash"]
                        if tx_hash:
                            key = (tx_hash, edge["src"], edge["dst"], edge["tx_type"])
                        else:
                            key = (edge["src"], edge["dst"], edge["timestamp"], edge["raw_value"], edge["tx_type"])
                        if key in seen:
                            stats["duplicate_edges"] += 1
                            continue
                        pair_key = pair_cap_key(edge, pair_cap_scope)
                        if max_edges_per_pair and pair_counts[pair_key] >= max_edges_per_pair:
                            stats["skipped_pair_cap"] += 1
                            stats[f"skipped_pair_cap_{tx_type}_hop{hop}"] += 1
                            continue
                        seen.add(key)
                        pair_counts[pair_key] += 1
                        writer.writerow(edge)
                        nodes.add(edge["src"])
                        nodes.add(edge["dst"])
                        stats["edges"] += 1
                        stats[f"edges_{tx_type}_hop{hop}"] += 1
                except Exception as exc:
                    stats[f"errors_{tx_type}_hop{hop}"] += 1
                    stats[f"error::{path.name}::{type(exc).__name__}"] += 1
            if progress_every and idx % progress_every == 0:
                elapsed = time.time() - started
                print(
                    json.dumps(
                        {
                            "processed_addresses": idx,
                            "total_addresses": len(addresses),
                            "edges": stats["edges"],
                            "skipped_pair_cap": stats["skipped_pair_cap"],
                            "elapsed_seconds": round(elapsed, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return stats, nodes


def write_nodes(
    root: Path,
    dataset: str,
    out_path: Path,
    all_nodes: set[str],
    label_meta: dict[str, dict[str, Any]],
    split: dict[str, str],
    include_meta_features: bool,
) -> None:
    rows = []
    for node in sorted(all_nodes):
        item = label_meta.get(node)
        if item:
            rows.append(
                {
                    "node_id": node,
                    "label": int(item["label"]),
                    "split": split.get(node, ""),
                    "is_target": 1,
                    "label_transaction_types": int(item.get("transaction_types", 0)),
                    "span_label": item.get("span_label", ""),
                    "title_label": item.get("title_label", ""),
                }
            )
        else:
            rows.append(
                {
                    "node_id": node,
                    "label": -1,
                    "split": "",
                    "is_target": 0,
                    "label_transaction_types": 0,
                    "span_label": "",
                    "title_label": "",
                }
            )
    df = pd.DataFrame(rows)
    if include_meta_features:
        features = read_feature_table(root, dataset)
        if not features.empty:
            df = df.merge(features, on="node_id", how="left")
    df.to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tx_types = [x.strip().lower() for x in args.tx_types.split(",") if x.strip()]
    bad = [x for x in tx_types if x not in TX_TYPE_DIRS]
    if bad:
        raise ValueError(f"Unknown tx types: {bad}")

    labels, label_meta = load_labels(root, args.dataset, args.max_addresses, args.seed)
    addresses = sorted(labels)
    split = split_labels(addresses, labels, args.train_ratio, args.val_ratio, args.seed)

    print(
        json.dumps(
            {
                "phase": "start",
                "dataset": args.dataset,
                "labeled_addresses": len(addresses),
                "label_counts": Counter(labels.values()),
                "include_two_hop": args.include_two_hop,
                "tx_types": tx_types,
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )

    edge_path = out_dir / "edges.csv"
    stats, all_nodes = write_edges(
        root=root,
        addresses=addresses,
        tx_types=tx_types,
        include_two_hop=args.include_two_hop,
        out_path=edge_path,
        max_rows_per_file=args.max_rows_per_file,
        max_file_mb=args.max_file_mb,
        max_edges_per_pair=args.max_edges_per_pair,
        pair_cap_scope=args.pair_cap_scope,
        progress_every=args.progress_every,
    )
    node_path = out_dir / "nodes.csv"
    write_nodes(root, args.dataset, node_path, all_nodes, label_meta, split, args.include_meta_features)

    summary = {
        "dataset": args.dataset,
        "root": str(root),
        "out_dir": str(out_dir),
        "labeled_addresses": len(addresses),
        "label_counts": {str(k): int(v) for k, v in Counter(labels.values()).items()},
        "nodes": len(all_nodes),
        "stats": {str(k): int(v) for k, v in stats.items() if not str(k).startswith("error::")},
        "errors": {str(k): int(v) for k, v in stats.items() if str(k).startswith("error::")},
        "include_two_hop": args.include_two_hop,
        "tx_types": tx_types,
        "max_rows_per_file": args.max_rows_per_file,
        "max_file_mb": args.max_file_mb,
        "max_edges_per_pair": args.max_edges_per_pair,
        "pair_cap_scope": args.pair_cap_scope,
    }
    with (out_dir / "conversion_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"phase": "done", **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
