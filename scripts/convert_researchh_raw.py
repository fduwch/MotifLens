#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


PHISH_TRANS_COLUMNS = [
    "tx_hash",
    "status_or_nonce",
    "block_hash",
    "block_number",
    "tx_index",
    "from",
    "to",
    "value",
    "gas",
    "gas_price",
    "input",
    "timestamp",
    "unused_12",
    "unused_13",
    "unused_14",
]


def norm_addr(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("0x") and len(text) == 42:
        return text
    return ""


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def split_labels(addresses: list[str], labels: dict[str, int], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = {}
    for address in addresses:
        by_label.setdefault(int(labels[address]), []).append(address)

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


def iter_phish_zip_rows(path: Path):
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith(".csv"):
                continue
            relation = "out" if "out" in name.lower() else "in"
            with archive.open(name) as fh:
                for raw in fh:
                    parts = raw.decode("utf-8", "replace").rstrip("\r\n").split(",")
                    if len(parts) < len(PHISH_TRANS_COLUMNS):
                        parts.extend([""] * (len(PHISH_TRANS_COLUMNS) - len(parts)))
                    yield relation, dict(zip(PHISH_TRANS_COLUMNS, parts[: len(PHISH_TRANS_COLUMNS)]))


def convert_phish_trans(args: argparse.Namespace) -> None:
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    nodes: set[str] = set()
    phish: set[str] = set()
    relation_rows: Counter[str] = Counter()
    for relation, row in iter_phish_zip_rows(source):
        src = norm_addr(row.get("from"))
        dst = norm_addr(row.get("to"))
        if src:
            nodes.add(src)
        if dst:
            nodes.add(dst)
        if relation == "out" and src:
            phish.add(src)
        if relation == "in" and dst:
            phish.add(dst)
        relation_rows[relation] += 1

    labels = {node: (1 if node in phish else 0) for node in sorted(nodes)}
    split = split_labels(sorted(labels), labels, args.train_ratio, args.val_ratio, args.seed)

    with (out_dir / "nodes.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["node_id", "label", "split", "is_target"])
        writer.writeheader()
        for node in sorted(labels):
            writer.writerow({"node_id": node, "label": labels[node], "split": split[node], "is_target": 1})

    edge_fields = [
        "src",
        "dst",
        "timestamp",
        "amount",
        "raw_value",
        "gas",
        "gas_price",
        "block_number",
        "tx_index",
        "relation_id",
        "relation_name",
    ]
    bad_edges = 0
    written = 0
    with (out_dir / "edges.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=edge_fields)
        writer.writeheader()
        for relation, row in iter_phish_zip_rows(source):
            src = norm_addr(row.get("from"))
            dst = norm_addr(row.get("to"))
            if not src or not dst:
                bad_edges += 1
                continue
            raw_value = safe_float(row.get("value"), 0.0)
            writer.writerow(
                {
                    "src": src,
                    "dst": dst,
                    "timestamp": safe_int(row.get("timestamp"), 0),
                    "amount": raw_value / 1e18,
                    "raw_value": raw_value,
                    "gas": safe_float(row.get("gas"), 0.0),
                    "gas_price": safe_float(row.get("gas_price"), 0.0),
                    "block_number": safe_int(row.get("block_number"), 0),
                    "tx_index": safe_int(row.get("tx_index"), 0),
                    "relation_id": 0 if relation == "out" else 1,
                    "relation_name": f"phisher_transaction_{relation}",
                }
            )
            written += 1

    label_counts = Counter(labels.values())
    summary = {
        "dataset": "phish_trans",
        "source": str(source),
        "out_dir": str(out_dir),
        "label_rule": "phishing = out.from union in.to; normal = other visible from/to addresses",
        "expected_description": {
            "phishing_accounts": args.expected_phishing,
            "normal_accounts": args.expected_normal,
        },
        "observed": {
            "nodes": len(nodes),
            "edges": written,
            "bad_edges": bad_edges,
            "relation_rows": dict(relation_rows),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "matches_expected": {
                "phishing": args.expected_phishing == 0 or label_counts[1] == args.expected_phishing,
                "normal": args.expected_normal == 0 or label_counts[0] == args.expected_normal,
            },
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (out_dir / "conversion_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"phase": "done", **summary}, ensure_ascii=False, indent=2), flush=True)


def convert_bert4eth_phishtrans(args: argparse.Namespace) -> None:
    phish_source = Path(args.phish_source)
    normal_source = Path(args.normal_source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    nodes: set[str] = set()
    phish_centers: set[str] = set()
    normal_centers: set[str] = set()
    rows_by_file: Counter[str] = Counter()
    bad_rows_by_file: Counter[str] = Counter()

    def iter_sources():
        yield phish_source, 1, "phish"
        yield normal_source, 0, "normal"

    def usable_member(name: str) -> bool:
        base = Path(name).name
        return name.lower().endswith(".csv") and not name.startswith("__MACOSX/") and not base.startswith("._")

    def relation_from_name(name: str) -> str:
        lower = Path(name).name.lower()
        if "_out" in lower or "transaction_out" in lower:
            return "out"
        if "_in" in lower or "transaction_in" in lower:
            return "in"
        return "unknown"

    def iter_zip_rows(path: Path):
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if not usable_member(name):
                    continue
                relation = relation_from_name(name)
                with archive.open(name) as fh:
                    for raw in fh:
                        parts = raw.decode("utf-8", "replace").rstrip("\r\n").split(",")
                        if len(parts) < len(PHISH_TRANS_COLUMNS):
                            parts.extend([""] * (len(PHISH_TRANS_COLUMNS) - len(parts)))
                        yield name, relation, dict(zip(PHISH_TRANS_COLUMNS, parts[: len(PHISH_TRANS_COLUMNS)]))

    edge_fields = [
        "src",
        "dst",
        "timestamp",
        "amount",
        "raw_value",
        "gas",
        "gas_price",
        "block_number",
        "tx_index",
        "relation_id",
        "relation_name",
        "account_class",
    ]
    written = 0
    with (out_dir / "edges.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=edge_fields)
        writer.writeheader()
        for path, label, account_class in iter_sources():
            for name, relation, row in iter_zip_rows(path):
                rows_by_file[f"{account_class}:{name}"] += 1
                src = norm_addr(row.get("from"))
                dst = norm_addr(row.get("to"))
                if src:
                    nodes.add(src)
                if dst:
                    nodes.add(dst)
                center = src if relation == "out" else dst if relation == "in" else ""
                if center:
                    if label == 1:
                        phish_centers.add(center)
                    else:
                        normal_centers.add(center)
                if not src or not dst:
                    bad_rows_by_file[f"{account_class}:{name}"] += 1
                    continue
                raw_value = safe_float(row.get("value"), 0.0)
                writer.writerow(
                    {
                        "src": src,
                        "dst": dst,
                        "timestamp": safe_int(row.get("timestamp"), 0),
                        "amount": raw_value / 1e18,
                        "raw_value": raw_value,
                        "gas": safe_float(row.get("gas"), 0.0),
                        "gas_price": safe_float(row.get("gas_price"), 0.0),
                        "block_number": safe_int(row.get("block_number"), 0),
                        "tx_index": safe_int(row.get("tx_index"), 0),
                        "relation_id": {"phish_in": 0, "phish_out": 1, "normal_in": 2, "normal_out": 3}.get(
                            f"{account_class}_{relation}",
                            9,
                        ),
                        "relation_name": f"{account_class}_transaction_{relation}",
                        "account_class": account_class,
                    }
                )
                written += 1
                if args.progress_every and written % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "edge_progress",
                                "written": written,
                                "nodes": len(nodes),
                                "phish_centers": len(phish_centers),
                                "normal_centers": len(normal_centers),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    overlap = phish_centers & normal_centers
    supervised = (phish_centers | normal_centers) & nodes
    labels = {node: (1 if node in phish_centers else 0) for node in sorted(supervised)}
    split = split_labels(sorted(labels), labels, args.train_ratio, args.val_ratio, args.seed)

    with (out_dir / "nodes.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["node_id", "label", "split", "is_target", "label_source"])
        writer.writeheader()
        for node in sorted(nodes):
            if node in phish_centers:
                label = 1
                label_source = "phish_center"
            elif node in normal_centers:
                label = 0
                label_source = "normal_center"
            else:
                label = -1
                label_source = "context"
            writer.writerow(
                {
                    "node_id": node,
                    "label": label,
                    "split": split.get(node, ""),
                    "is_target": 1 if label >= 0 else 0,
                    "label_source": label_source,
                }
            )

    label_counts = Counter(labels.values())
    summary = {
        "dataset": "BERT4ETH_PhishTrans",
        "phish_source": str(phish_source),
        "normal_source": str(normal_source),
        "out_dir": str(out_dir),
        "label_rule": (
            "supervised labels only for central accounts: phish out.from/in.to -> 1; "
            "normal out.from/in.to -> 0; other counterparties kept as context label=-1"
        ),
        "expected_description": {
            "phishing_accounts": args.expected_phishing,
            "normal_accounts": args.expected_normal,
        },
        "observed": {
            "nodes_total": len(nodes),
            "context_nodes": len(nodes - supervised),
            "edges": written,
            "rows_by_file": dict(rows_by_file),
            "bad_rows_by_file": dict(bad_rows_by_file),
            "phish_centers": len(phish_centers),
            "normal_centers": len(normal_centers),
            "center_overlap": len(overlap),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "matches_expected": {
                "phishing": args.expected_phishing == 0 or label_counts[1] == args.expected_phishing,
                "normal": args.expected_normal == 0 or label_counts[0] == args.expected_normal,
            },
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (out_dir / "conversion_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"phase": "done", **summary}, ensure_ascii=False, indent=2), flush=True)


def convert_muldigraph(args: argparse.Namespace) -> None:
    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    with source.open("rb") as fh:
        graph = pickle.load(fh)

    labels: dict[str, int] = {}
    for node, attrs in graph.nodes(data=True):
        labels[str(node).lower()] = int(attrs.get("isp", -1))
    split = split_labels(sorted(labels), labels, args.train_ratio, args.val_ratio, args.seed)

    with (out_dir / "nodes.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["node_id", "label", "split", "is_target"])
        writer.writeheader()
        for node in sorted(labels):
            writer.writerow({"node_id": node, "label": labels[node], "split": split[node], "is_target": 1})

    bad_edges = 0
    written = 0
    with (out_dir / "edges.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["src", "dst", "timestamp", "amount"])
        writer.writeheader()
        for src, dst, attrs in graph.edges(data=True):
            src = norm_addr(src)
            dst = norm_addr(dst)
            if not src or not dst:
                bad_edges += 1
                continue
            writer.writerow(
                {
                    "src": src,
                    "dst": dst,
                    "timestamp": safe_float(attrs.get("timestamp"), 0.0),
                    "amount": safe_float(attrs.get("amount"), 0.0),
                }
            )
            written += 1
            if args.progress_every and written % args.progress_every == 0:
                print(json.dumps({"phase": "edge_progress", "written": written}, ensure_ascii=False), flush=True)

    label_counts = Counter(labels.values())
    summary = {
        "dataset": "muldigraph",
        "source": str(source),
        "out_dir": str(out_dir),
        "label_rule": "node attribute isp",
        "observed": {
            "nodes": len(labels),
            "edges": written,
            "bad_edges": bad_edges,
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (out_dir / "conversion_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"phase": "done", **summary}, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ResearchH raw phishing datasets into nodes.csv/edges.csv.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", required=True)
    common.add_argument("--out-dir", required=True)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--train-ratio", type=float, default=0.70)
    common.add_argument("--val-ratio", type=float, default=0.15)

    p1 = sub.add_parser("phish-trans", parents=[common])
    p1.add_argument("--expected-phishing", type=int, default=3220)
    p1.add_argument("--expected-normal", type=int, default=594038)
    p1.set_defaults(func=convert_phish_trans)

    p_full = sub.add_parser("bert4eth-phishtrans")
    p_full.add_argument("--phish-source", required=True)
    p_full.add_argument("--normal-source", required=True)
    p_full.add_argument("--out-dir", required=True)
    p_full.add_argument("--seed", type=int, default=42)
    p_full.add_argument("--train-ratio", type=float, default=0.70)
    p_full.add_argument("--val-ratio", type=float, default=0.15)
    p_full.add_argument("--expected-phishing", type=int, default=3220)
    p_full.add_argument("--expected-normal", type=int, default=594038)
    p_full.add_argument("--progress-every", type=int, default=1_000_000)
    p_full.set_defaults(func=convert_bert4eth_phishtrans)

    p2 = sub.add_parser("muldigraph", parents=[common])
    p2.add_argument("--progress-every", type=int, default=1_000_000)
    p2.set_defaults(func=convert_muldigraph)
    return parser.parse_args()


def main() -> None:
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
