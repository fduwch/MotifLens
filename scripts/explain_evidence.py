#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create address-level, auditable evidence explanations.")
    parser.add_argument("--data-dir", required=True, help="Directory containing nodes.csv and edges.csv.")
    parser.add_argument("--evidence", required=True, help="Evidence CSV produced by generate_evidence.py.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="test", help="train/val/test/all; only labeled addresses are explained.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of addresses to explain. 0 means all matching addresses.")
    parser.add_argument("--top-k", type=int, default=5, help="Top evidence edges to keep per address.")
    parser.add_argument("--chunksize", type=int, default=500000)
    return parser.parse_args()


@dataclass
class TargetMeta:
    address: str
    label: int
    split: str


@dataclass
class EvidenceAgg:
    count: int = 0
    incoming_count: int = 0
    outgoing_count: int = 0
    score_sum: float = 0.0
    incoming_score_sum: float = 0.0
    outgoing_score_sum: float = 0.0
    support_count: int = 0
    counter_count: int = 0
    support_score_sum: float = 0.0
    counter_score_sum: float = 0.0
    max_score: float = 0.0
    type_counts: Counter = field(default_factory=Counter)
    type_scores: Counter = field(default_factory=Counter)
    support_type_counts: Counter = field(default_factory=Counter)
    counter_type_counts: Counter = field(default_factory=Counter)
    reason_counts: Counter = field(default_factory=Counter)
    top_heap: list[tuple[float, int, str, str, str, str]] = field(default_factory=list)


def normalize_split(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()


def load_targets(nodes_path: Path, split: str, limit: int, chunksize: int) -> dict[int, TargetMeta]:
    targets: dict[int, TargetMeta] = {}
    wanted_split = split.strip().lower()
    usecols = ["node_id", "label", "split"]
    for chunk in pd.read_csv(nodes_path, usecols=usecols, chunksize=chunksize):
        labels = pd.to_numeric(chunk["label"], errors="coerce").fillna(-1).astype(int)
        splits = chunk["split"].map(normalize_split)
        mask = labels >= 0
        if wanted_split != "all":
            mask &= splits == wanted_split
        if not mask.any():
            continue
        for idx in np.flatnonzero(mask.to_numpy()):
            node_idx = int(chunk.index[idx])
            targets[node_idx] = TargetMeta(
                address=str(chunk.iloc[idx]["node_id"]),
                label=int(labels.iloc[idx]),
                split=str(splits.iloc[idx]),
            )
            if limit and len(targets) >= limit:
                return targets
    return targets


def update_agg(
    agg: EvidenceAgg,
    *,
    role: str,
    edge_pos: int,
    score: float,
    evidence_type: str,
    reason_codes: str,
    polarity: str,
    top_k: int,
) -> None:
    polarity = polarity if polarity in {"support", "counter", "neutral"} else "support"
    agg.count += 1
    agg.score_sum += score
    agg.max_score = max(agg.max_score, score)
    agg.type_counts[evidence_type] += 1
    agg.type_scores[evidence_type] += score
    if polarity == "counter":
        agg.counter_count += 1
        agg.counter_score_sum += score
        agg.counter_type_counts[evidence_type] += 1
    elif polarity == "support":
        agg.support_count += 1
        agg.support_score_sum += score
        agg.support_type_counts[evidence_type] += 1
    for reason in str(reason_codes or "").split("|"):
        reason = reason.strip()
        if reason and reason != "none":
            agg.reason_counts[reason] += 1
    if role == "incoming":
        agg.incoming_count += 1
        agg.incoming_score_sum += score
    else:
        agg.outgoing_count += 1
        agg.outgoing_score_sum += score
    item = (score, edge_pos, role, evidence_type, reason_codes, polarity)
    if len(agg.top_heap) < top_k:
        heapq.heappush(agg.top_heap, item)
    elif score > agg.top_heap[0][0]:
        heapq.heapreplace(agg.top_heap, item)


def scan_evidence(evidence_path: Path, targets: dict[int, TargetMeta], top_k: int, chunksize: int) -> dict[int, EvidenceAgg]:
    target_set = set(targets)
    aggs: dict[int, EvidenceAgg] = defaultdict(EvidenceAgg)
    header = pd.read_csv(evidence_path, nrows=0)
    usecols = ["edge_pos", "src_idx", "dst_idx", "evidence_score", "evidence_type", "reason_codes"]
    if "polarity" in header.columns:
        usecols.append("polarity")
    for chunk in pd.read_csv(evidence_path, usecols=usecols, chunksize=chunksize):
        if "polarity" not in chunk.columns:
            chunk["polarity"] = "support"
        src = pd.to_numeric(chunk["src_idx"], errors="coerce").fillna(-1).astype(np.int64)
        dst = pd.to_numeric(chunk["dst_idx"], errors="coerce").fillna(-1).astype(np.int64)
        src_mask = src.isin(target_set).to_numpy()
        dst_mask = dst.isin(target_set).to_numpy()
        if not (src_mask.any() or dst_mask.any()):
            continue
        for pos in np.flatnonzero(src_mask):
            node_idx = int(src.iloc[pos])
            update_agg(
                aggs[node_idx],
                role="outgoing",
                edge_pos=int(chunk.iloc[pos]["edge_pos"]),
                score=float(chunk.iloc[pos]["evidence_score"]),
                evidence_type=str(chunk.iloc[pos]["evidence_type"]),
                reason_codes=str(chunk.iloc[pos].get("reason_codes", "")),
                polarity=str(chunk.iloc[pos].get("polarity", "support")),
                top_k=top_k,
            )
        for pos in np.flatnonzero(dst_mask):
            node_idx = int(dst.iloc[pos])
            update_agg(
                aggs[node_idx],
                role="incoming",
                edge_pos=int(chunk.iloc[pos]["edge_pos"]),
                score=float(chunk.iloc[pos]["evidence_score"]),
                evidence_type=str(chunk.iloc[pos]["evidence_type"]),
                reason_codes=str(chunk.iloc[pos].get("reason_codes", "")),
                polarity=str(chunk.iloc[pos].get("polarity", "support")),
                top_k=top_k,
            )
    return aggs


def collect_top_edge_refs(aggs: dict[int, EvidenceAgg]) -> dict[int, list[tuple[float, int, str, str, str, str]]]:
    refs: dict[int, list[tuple[float, int, str, str, str, str]]] = {}
    for node_idx, agg in aggs.items():
        refs[node_idx] = sorted(agg.top_heap, reverse=True)
    return refs


def read_edge_details(edges_path: Path, wanted_positions: set[int], chunksize: int) -> dict[int, dict[str, Any]]:
    details: dict[int, dict[str, Any]] = {}
    if not wanted_positions:
        return details
    offset = 0
    for chunk in pd.read_csv(edges_path, chunksize=chunksize):
        end = offset + len(chunk)
        hits = [pos for pos in wanted_positions if offset <= pos < end]
        if hits:
            for edge_pos in hits:
                row = chunk.iloc[edge_pos - offset].to_dict()
                details[int(edge_pos)] = row
        offset = end
        if len(details) == len(wanted_positions):
            break
    return details


def top_items(counter: Counter, limit: int = 3) -> str:
    if not counter:
        return ""
    return ";".join(f"{k}:{v}" for k, v in counter.most_common(limit))


def make_explanation(meta: TargetMeta, agg: EvidenceAgg) -> str:
    if agg.count == 0:
        return "未检索到超过阈值的结构化证据边，模型判断主要依赖图邻域与地址统计特征。"
    top_support = ", ".join(name for name, _ in agg.support_type_counts.most_common(3)) or "无"
    top_counter = ", ".join(name for name, _ in agg.counter_type_counts.most_common(3)) or "无"
    direction = "入向" if agg.incoming_score_sum >= agg.outgoing_score_sum else "出向"
    reasons = ", ".join(name for name, _ in agg.reason_counts.most_common(3)) or "规则分数较高但无额外原因码"
    return (
        f"该地址共有 {agg.count} 条可审计 evidence 边，最高证据分 {agg.max_score:.3f}。"
        f"其中 support={agg.support_count} 条、counter={agg.counter_count} 条。"
        f"主要正证据类型为 {top_support}，主要反证类型为 {top_counter}，证据强度以{direction}交易为主。"
        f"主要原因码包括 {reasons}。"
    )


def write_outputs(
    out_dir: Path,
    targets: dict[int, TargetMeta],
    aggs: dict[int, EvidenceAgg],
    refs: dict[int, list[tuple[float, int, str, str, str, str]]],
    edge_details: dict[int, dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for node_idx, meta in targets.items():
        agg = aggs.get(node_idx, EvidenceAgg())
        summary_rows.append(
            {
                "node_idx": node_idx,
                "address": meta.address,
                "label": meta.label,
                "split": meta.split,
                "evidence_count": agg.count,
                "incoming_count": agg.incoming_count,
                "outgoing_count": agg.outgoing_count,
                "score_sum": round(agg.score_sum, 6),
                "incoming_score_sum": round(agg.incoming_score_sum, 6),
                "outgoing_score_sum": round(agg.outgoing_score_sum, 6),
                "support_count": agg.support_count,
                "counter_count": agg.counter_count,
                "support_score_sum": round(agg.support_score_sum, 6),
                "counter_score_sum": round(agg.counter_score_sum, 6),
                "max_score": round(agg.max_score, 6),
                "top_evidence_types": top_items(agg.type_counts),
                "top_support_evidence_types": top_items(agg.support_type_counts),
                "top_counter_evidence_types": top_items(agg.counter_type_counts),
                "top_reason_codes": top_items(agg.reason_counts),
                "explanation": make_explanation(meta, agg),
            }
        )
        for score, edge_pos, role, evidence_type, reason_codes, polarity in refs.get(node_idx, []):
            edge = edge_details.get(edge_pos, {})
            edge_rows.append(
                {
                    "node_idx": node_idx,
                    "address": meta.address,
                    "role": role,
                    "edge_pos": edge_pos,
                    "evidence_score": round(float(score), 6),
                    "evidence_type": evidence_type,
                    "polarity": polarity,
                    "reason_codes": reason_codes,
                    "src": edge.get("src", ""),
                    "dst": edge.get("dst", ""),
                    "amount": edge.get("amount", ""),
                    "timestamp": edge.get("timestamp", ""),
                    "block_number": edge.get("block_number", ""),
                    "tx_type": edge.get("tx_type", ""),
                    "tx_hash": edge.get("tx_hash", ""),
                }
            )
    pd.DataFrame(summary_rows).sort_values(["evidence_count", "score_sum"], ascending=[False, False]).to_csv(
        out_dir / "address_explanations.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    pd.DataFrame(edge_rows).sort_values(["address", "evidence_score"], ascending=[True, False]).to_csv(
        out_dir / "top_evidence_edges.csv", index=False, quoting=csv.QUOTE_MINIMAL
    )
    metadata = {
        "num_addresses": len(targets),
        "num_addresses_with_evidence": sum(1 for k in targets if aggs.get(k, EvidenceAgg()).count > 0),
        "num_addresses_with_support_evidence": sum(1 for k in targets if aggs.get(k, EvidenceAgg()).support_count > 0),
        "num_addresses_with_counter_evidence": sum(1 for k in targets if aggs.get(k, EvidenceAgg()).counter_count > 0),
        "num_top_edges": len(edge_rows),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    nodes_path = data_dir / "nodes.csv"
    edges_path = data_dir / "edges.csv"
    evidence_path = Path(args.evidence)

    print(json.dumps({"phase": "load_targets_start", "split": args.split, "limit": args.limit}, ensure_ascii=False), flush=True)
    targets = load_targets(nodes_path, args.split, args.limit, args.chunksize)
    print(json.dumps({"phase": "load_targets_done", "num_targets": len(targets)}, ensure_ascii=False), flush=True)
    print(json.dumps({"phase": "scan_evidence_start"}, ensure_ascii=False), flush=True)
    aggs = scan_evidence(evidence_path, targets, args.top_k, args.chunksize)
    refs = collect_top_edge_refs(aggs)
    wanted_positions = {edge_pos for values in refs.values() for _, edge_pos, _, _, _, _ in values}
    print(
        json.dumps(
            {
                "phase": "scan_evidence_done",
                "targets_with_evidence": len(aggs),
                "top_edge_positions": len(wanted_positions),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(json.dumps({"phase": "read_edge_details_start"}, ensure_ascii=False), flush=True)
    edge_details = read_edge_details(edges_path, wanted_positions, args.chunksize)
    print(json.dumps({"phase": "read_edge_details_done", "edge_details": len(edge_details)}, ensure_ascii=False), flush=True)
    write_outputs(out_dir, targets, aggs, refs, edge_details)
    print(json.dumps({"phase": "done", "out_dir": str(out_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
