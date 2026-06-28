#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def norm_addr(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("0x") and len(text) == 42:
        return text
    return ""


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def split_labels(addresses: list[str], labels: dict[str, int], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for address in addresses:
        by_label[int(labels[address])].append(address)
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


def is_numeric_text(value: str) -> bool:
    try:
        if value in ("", None):
            return False
        float(str(value))
        return True
    except Exception:
        return False


def load_label_rows(path: Path) -> tuple[dict[str, int], dict[str, str], dict[str, dict[str, str]], list[str]]:
    labels: dict[str, int] = {}
    label_source: dict[str, str] = {}
    features: dict[str, dict[str, str]] = {}
    numeric_feature_candidates: Counter = Counter()
    rows_seen = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_seen += 1
            address = norm_addr(row.get("node_id") or row.get("address") or row.get("Address"))
            label = safe_int(row.get("label") or row.get("FLAG"), -1)
            if not address or label not in (0, 1):
                continue
            labels[address] = label
            label_source[address] = row.get("label_source", "")
            feature_row: dict[str, str] = {}
            for key, value in row.items():
                if key in {"node_id", "address", "Address", "label", "FLAG", "split", "is_target", "label_source"}:
                    continue
                text = str(value or "").strip()
                if is_numeric_text(text):
                    feature_row[key] = text
                    numeric_feature_candidates[key] += 1
            features[address] = feature_row
    feature_columns = [key for key, count in numeric_feature_candidates.items() if count > 0]
    return labels, label_source, features, feature_columns


def relation_id(event_type: str, direction: str) -> int:
    key = f"{event_type}:{direction}"
    mapping = {
        "evm_transaction:out": 0,
        "evm_transaction:in": 1,
        "evm_trace:out": 2,
        "evm_trace:in": 3,
        "token_transfer:out": 4,
        "token_transfer:in": 5,
    }
    return mapping.get(key, 9)


def amount_value(row: dict[str, Any]) -> tuple[float, str]:
    raw = str(row.get("value_raw") or "")
    asset = str(row.get("asset") or "")
    value = safe_float(raw, 0.0)
    if asset == "native":
        return value / 1e18, raw
    return value, raw


def iter_response_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                yield line_no, {"status": "decode_error", "error": str(exc)}
                continue
            yield line_no, payload


def extract_result_rows(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [row for row in response if isinstance(row, dict)]
    if not isinstance(response, dict):
        return []
    for key in ("result", "data", "items", "transactions"):
        result = response.get(key)
        if isinstance(result, list):
            return [row for row in result if isinstance(row, dict)]
        if isinstance(result, dict):
            rows = result.get("rows")
            if isinstance(rows, list):
                if not rows:
                    return []
                if isinstance(rows[0], dict):
                    return [row for row in rows if isinstance(row, dict)]
                columns = result.get("columns")
                if isinstance(columns, list):
                    normalized = []
                    for row in rows:
                        if isinstance(row, (list, tuple)):
                            normalized.append({str(col): row[idx] if idx < len(row) else "" for idx, col in enumerate(columns)})
                    return normalized
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert platform address-transaction JSONL into standard graph CSVs.")
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    labels, label_source, node_features, feature_columns = load_label_rows(Path(args.labels))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes: set[str] = set(labels)
    stats: Counter = Counter()
    edge_fields = [
        "src",
        "dst",
        "timestamp",
        "amount",
        "raw_value",
        "block_number",
        "tx_hash",
        "relation_id",
        "relation_name",
        "event_type",
        "direction",
        "asset",
        "detail",
        "root_address",
        "root_label",
    ]
    with (out_dir / "edges.csv").open("w", encoding="utf-8", newline="") as edge_fh:
        writer = csv.DictWriter(edge_fh, fieldnames=edge_fields)
        writer.writeheader()
        for line_no, payload in iter_response_rows(Path(args.responses_jsonl)):
            if payload.get("status") != "ok":
                stats[f"payload_{payload.get('status', 'bad')}"] += 1
                continue
            root = norm_addr(payload.get("address"))
            root_label = safe_int(payload.get("label"), -1)
            result = extract_result_rows(payload.get("response") or {})
            if not result:
                stats["empty_result"] += 1
                continue
            for item in result:
                src = norm_addr(item.get("from_address"))
                dst = norm_addr(item.get("to_address"))
                if not src or not dst:
                    stats["bad_edge_address"] += 1
                    continue
                amount, raw_value = amount_value(item)
                event_type = str(item.get("event_type") or "")
                direction = str(item.get("direction") or "")
                writer.writerow(
                    {
                        "src": src,
                        "dst": dst,
                        "timestamp": item.get("block_time") or "",
                        "amount": amount,
                        "raw_value": raw_value,
                        "block_number": safe_int(item.get("block_number"), 0),
                        "tx_hash": item.get("tx_hash") or "",
                        "relation_id": relation_id(event_type, direction),
                        "relation_name": f"{event_type}_{direction}".strip("_"),
                        "event_type": event_type,
                        "direction": direction,
                        "asset": item.get("asset") or "",
                        "detail": item.get("detail") or "",
                        "root_address": root,
                        "root_label": root_label,
                    }
                )
                nodes.add(src)
                nodes.add(dst)
                stats["edges"] += 1
                stats[f"event_{event_type}"] += 1
                if args.progress_every and stats["edges"] % args.progress_every == 0:
                    print(json.dumps({"phase": "edge_progress", "edges": stats["edges"], "nodes": len(nodes)}, ensure_ascii=False), flush=True)
            stats["responses"] += 1

    split = split_labels(sorted(labels), labels, args.train_ratio, args.val_ratio, args.seed)
    node_fields = ["node_id", "label", "split", "is_target", "label_source"] + feature_columns
    with (out_dir / "nodes.csv").open("w", encoding="utf-8", newline="") as node_fh:
        writer = csv.DictWriter(node_fh, fieldnames=node_fields, extrasaction="ignore")
        writer.writeheader()
        for node in sorted(nodes):
            label = labels.get(node, -1)
            row = {
                "node_id": node,
                "label": label,
                "split": split.get(node, ""),
                "is_target": 1 if label in (0, 1) else 0,
                "label_source": label_source.get(node, "context"),
            }
            row.update(node_features.get(node, {}))
            writer.writerow(row)

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "responses_jsonl": args.responses_jsonl,
        "labels": args.labels,
        "out_dir": str(out_dir),
        "nodes": len(nodes),
        "labeled_nodes": len(labels),
        "label_counts": {str(k): int(v) for k, v in Counter(labels.values()).items()},
        "stats": {str(k): int(v) for k, v in stats.items()},
        "feature_columns": feature_columns,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (out_dir / "conversion_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"phase": "done", **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
