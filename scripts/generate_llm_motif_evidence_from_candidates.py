#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
import time
from itertools import count
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_llm_motif_evidence import (  # noqa: E402
    MotifConfig,
    aggregate_observations,
    call_openai_compatible,
    compact_edge,
    evidence_rows_from_results,
    heuristic_result,
    load_cached_responses,
    normalize_result,
    write_jsonl,
)


NEEDED_EVIDENCE_COLUMNS = {
    "edge_pos",
    "src_idx",
    "dst_idx",
    "amount_percentile",
    "src_out_degree",
    "dst_in_degree",
    "receiver_new_like",
    "time_since_last_inflow",
    "evidence_score",
    "evidence_type",
    "reason_codes",
}
PUSH_COUNTER = count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate motif evidence from an existing deterministic candidate evidence CSV. "
            "This avoids rereading very large raw edge CSV files."
        )
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--observation-evidence", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cards-out", default=None)
    parser.add_argument("--responses-out", default=None)
    parser.add_argument("--split", default="all")
    parser.add_argument("--max-cards", type=int, default=0)
    parser.add_argument("--max-edges-per-card", type=int, default=6)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--teacher", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--llm-batch-size", type=int, default=2)
    parser.add_argument("--llm-card-edges", type=int, default=6, help="Number of candidate edges to include in each LLM prompt card.")
    parser.set_defaults(llm_full_card=True)
    parser.add_argument("--llm-full-card", dest="llm_full_card", action="store_true", help="Send the same full card schema as generate_llm_motif_evidence.py.")
    parser.add_argument("--llm-compact-card", dest="llm_full_card", action="store_false", help="Use a shorter card schema for context-limited smoke tests.")
    parser.add_argument("--llm-sleep", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--prompt-no-think", action="store_true")
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Allow heuristic fallback for failed single-card LLM calls. Disabled by default so LLM outages stop and can be resumed cleanly.",
    )
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    return parser.parse_args()


def split_matches(value: Any, wanted: str) -> bool:
    if wanted == "all":
        return True
    item = str(value).strip().lower()
    if wanted in {"val", "valid", "validation"}:
        return item in {"val", "valid", "validation"}
    return item == wanted


def load_targets(nodes_path: Path, split: str, max_cards: int) -> tuple[list[int], dict[int, str]]:
    wanted = split.strip().lower()
    target_indices: list[int] = []
    target_addresses: dict[int, str] = {}
    usecols = lambda c: c in {"node_id", "label", "split"}  # noqa: E731
    for chunk in pd.read_csv(nodes_path, usecols=usecols, chunksize=1_000_000):
        labels = pd.to_numeric(chunk.get("label"), errors="coerce").fillna(-1)
        mask = labels >= 0
        if wanted != "all" and "split" in chunk.columns:
            mask &= chunk["split"].map(lambda x: split_matches(x, wanted))
        selected = chunk.loc[mask]
        for idx, row in selected.iterrows():
            node_idx = int(idx)
            target_indices.append(node_idx)
            target_addresses[node_idx] = str(row["node_id"])
            if max_cards and len(target_indices) >= max_cards:
                return target_indices, target_addresses
    return target_indices, target_addresses


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return int(value)
    except Exception:
        return default


def push_candidate(
    heaps: dict[int, list[tuple[tuple[float, int], int, int, dict[str, Any]]]],
    target_idx: int,
    row: pd.Series,
    role: str,
    max_edges: int,
) -> None:
    score = float(pd.to_numeric(pd.Series([row.get("evidence_score")]), errors="coerce").fillna(0).iloc[0])
    edge_pos = safe_int(row.get("edge_pos"))
    item = dict(row)
    item["_role"] = role
    item["edge_pos"] = edge_pos
    item["src_idx"] = safe_int(row.get("src_idx"))
    item["dst_idx"] = safe_int(row.get("dst_idx"))
    key = (score, -edge_pos)
    heap = heaps.setdefault(target_idx, [])
    packed = (key, edge_pos, next(PUSH_COUNTER), item)
    if len(heap) < max_edges:
        heapq.heappush(heap, packed)
    elif key > heap[0][0]:
        heapq.heapreplace(heap, packed)


def build_cards_from_candidate_evidence(
    evidence_path: Path,
    target_indices: list[int],
    target_addresses: dict[int, str],
    max_edges_per_card: int,
    chunksize: int,
) -> tuple[list[dict[str, Any]], int, int]:
    target_set = set(target_indices)
    partials: list[pd.DataFrame] = []
    scanned = 0
    matched = 0
    reader = pd.read_csv(
        evidence_path,
        usecols=lambda c: c in NEEDED_EVIDENCE_COLUMNS,
        chunksize=chunksize,
    )
    for chunk_id, chunk in enumerate(reader, start=1):
        scanned += len(chunk)
        for column in NEEDED_EVIDENCE_COLUMNS:
            if column not in chunk.columns:
                chunk[column] = ""
        for column in ("edge_pos", "src_idx", "dst_idx"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").fillna(-1).astype("int64")
        chunk["evidence_score"] = pd.to_numeric(chunk["evidence_score"], errors="coerce").fillna(0.0)

        src_match = chunk["src_idx"].isin(target_set)
        dst_match = chunk["dst_idx"].isin(target_set)
        raw_matches = int(src_match.sum() + dst_match.sum())
        matched += raw_matches

        event_frames: list[pd.DataFrame] = []
        if src_match.any():
            outgoing = chunk.loc[src_match].copy()
            outgoing["_target_idx"] = outgoing["src_idx"]
            outgoing["_role"] = "outgoing"
            event_frames.append(outgoing)
        if dst_match.any():
            incoming = chunk.loc[dst_match].copy()
            incoming["_target_idx"] = incoming["dst_idx"]
            incoming["_role"] = "incoming"
            event_frames.append(incoming)
        if event_frames:
            events = pd.concat(event_frames, ignore_index=True)
            events = events.sort_values(
                ["_target_idx", "evidence_score", "edge_pos"],
                ascending=[True, False, True],
            )
            partials.append(events.groupby("_target_idx", sort=False).head(max_edges_per_card))

        if chunk_id == 1 or chunk_id % 5 == 0:
            print(
                json.dumps(
                    {
                        "phase": "candidate_stream_progress",
                        "chunks": chunk_id,
                        "rows_scanned": scanned,
                        "rows_matched": matched,
                        "partial_top_rows": int(sum(len(part) for part in partials)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if partials:
        top_rows = pd.concat(partials, ignore_index=True)
        top_rows = top_rows.sort_values(
            ["_target_idx", "evidence_score", "edge_pos"],
            ascending=[True, False, True],
        )
        top_rows = top_rows.groupby("_target_idx", sort=False).head(max_edges_per_card)
        grouped = {int(target_idx): frame for target_idx, frame in top_rows.groupby("_target_idx", sort=False)}
    else:
        grouped = {}

    cards: list[dict[str, Any]] = []
    for node_idx in target_indices:
        frame = grouped.get(int(node_idx))
        candidate_edges: list[dict[str, Any]] = []
        if frame is not None:
            for i, (_, row) in enumerate(frame.iterrows()):
                role = str(row["_role"])
                edge = compact_edge(row, role, f"e{i}")
                edge["src_idx"] = int(row["src_idx"])
                edge["dst_idx"] = int(row["dst_idx"])
                candidate_edges.append(edge)
        cards.append(
            {
                "target_id": f"n{int(node_idx)}",
                "target_node_idx": int(node_idx),
                "address": target_addresses[int(node_idx)],
                "task": "Choose a behavioral motif evidence type from the taxonomy. Do not predict the fraud label.",
                "observations": aggregate_observations(candidate_edges),
                "candidate_edges": candidate_edges,
            }
        )
    return cards, scanned, matched


def fill_src_dst(evidence: pd.DataFrame, cards: list[dict[str, Any]]) -> pd.DataFrame:
    if not len(evidence):
        return evidence
    lookup_rows = []
    seen = set()
    for card in cards:
        for edge in card["candidate_edges"]:
            edge_pos = int(edge["edge_pos"])
            if edge_pos in seen:
                continue
            seen.add(edge_pos)
            lookup_rows.append({"edge_pos": edge_pos, "src_idx": edge["src_idx"], "dst_idx": edge["dst_idx"]})
    src_dst = pd.DataFrame(lookup_rows)
    evidence = evidence.drop(columns=["src_idx", "dst_idx"], errors="ignore").merge(src_dst, on="edge_pos", how="left")
    preferred = [
        "edge_pos",
        "src_idx",
        "dst_idx",
        "target_idx",
        "address",
        "local_edge_id",
        "evidence_score",
        "evidence_type",
        "reason_codes",
        "polarity",
        "observation_score",
        "observation_type",
    ]
    return evidence[preferred]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_card_for_llm(card: dict[str, Any], max_edges: int) -> dict[str, Any]:
    edges = []
    for edge in card.get("candidate_edges", [])[:max_edges]:
        edges.append(
            {
                "edge_id": edge.get("edge_id"),
                "role": edge.get("role"),
                "score": edge.get("observation_score"),
                "type": edge.get("observation_type"),
                "amount": edge.get("amount_bin"),
                "delay": edge.get("delay_bin"),
                "low_receiver": edge.get("receiver_low_degree"),
                "reasons": edge.get("reason_codes", []),
            }
        )
    return {
        "target_id": card.get("target_id"),
        "observations": card.get("observations", []),
        "candidate_edges": edges,
    }


def full_card_for_llm(card: dict[str, Any], max_edges: int) -> dict[str, Any]:
    selected = dict(card)
    selected["candidate_edges"] = list(card.get("candidate_edges", []))[:max_edges]
    selected["observations"] = aggregate_observations(selected["candidate_edges"])
    return selected


def card_for_llm(card: dict[str, Any], max_edges: int, full_card: bool) -> dict[str, Any]:
    if full_card:
        return full_card_for_llm(card, max_edges)
    return compact_card_for_llm(card, max_edges)


def call_llm_resilient(
    batch: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    config: MotifConfig,
) -> list[dict[str, Any]]:
    llm_batch = [card_for_llm(card, args.llm_card_edges, args.llm_full_card) for card in batch]
    items: list[dict[str, Any]] = []
    try:
        items = call_openai_compatible(
            llm_batch,
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            config=config,
        )
        result_ids = {
            str(item.get("result", {}).get("target_id", ""))
            for item in items
            if isinstance(item.get("result"), dict)
        }
        missing = [card for card in batch if str(card.get("target_id")) not in result_ids]
        if not missing:
            return items
        print(
            json.dumps(
                {
                    "phase": "llm_batch_missing_results_retry_single",
                    "batch_size": len(batch),
                    "returned_results": len(items),
                    "missing_target_ids": [card.get("target_id") for card in missing],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        error_text = repr(exc)
        if "maximum context length" in error_text:
            phase = "llm_batch_context_limit_retry_single"
            error_payload = {"reason": "context_length"}
        else:
            phase = "llm_batch_failed_retry_single"
            error_payload = {"error": error_text}
        print(
            json.dumps(
                {
                    "phase": phase,
                    "batch_size": len(batch),
                    "target_ids": [card.get("target_id") for card in batch],
                    **error_payload,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        missing = list(batch)

    for card in missing:
        llm_card = [card_for_llm(card, args.llm_card_edges, args.llm_full_card)]
        try:
            single_items = call_openai_compatible(
                llm_card,
                base_url=args.llm_base_url,
                api_key=args.llm_api_key,
                model=args.llm_model,
                config=config,
            )
            if len(single_items) == 1 and isinstance(single_items[0].get("result"), dict):
                single_items[0]["result"]["target_id"] = card["target_id"]
            items.extend(single_items)
        except Exception as exc:
            if not args.allow_heuristic_fallback:
                print(
                    json.dumps(
                        {
                            "phase": "llm_single_failed_stop",
                            "target_id": card.get("target_id"),
                            "error": repr(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                raise
            print(
                json.dumps(
                    {
                        "phase": "llm_single_failed_heuristic_fallback",
                        "target_id": card.get("target_id"),
                        "error": repr(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            items.append(
                {
                    "raw_content": f"LLM call or parse failed; heuristic fallback used. error={exc!r}",
                    "result": heuristic_result(card),
                }
            )
    return items


def main() -> None:
    args = parse_args()
    started = time.time()
    data_dir = Path(args.data_dir)
    nodes_path = data_dir / "nodes.csv"
    out_path = Path(args.out)
    cards_path = Path(args.cards_out) if args.cards_out else out_path.with_suffix(".cards.jsonl")
    responses_path = Path(args.responses_out) if args.responses_out else out_path.with_suffix(".responses.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if cards_path.exists() and cards_path.stat().st_size > 0:
        print(json.dumps({"phase": "load_cards_cache_start", "cards_out": str(cards_path)}, ensure_ascii=False), flush=True)
        cards = read_jsonl(cards_path)
        if args.max_cards:
            cards = cards[: args.max_cards]
        target_indices = [int(card["target_node_idx"]) for card in cards]
        scanned = 0
        matched = 0
        print(
            json.dumps(
                {
                    "phase": "load_cards_cache_done",
                    "num_cards": len(cards),
                    "elapsed_seconds": round(time.time() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print(json.dumps({"phase": "load_targets_start", "nodes": str(nodes_path)}, ensure_ascii=False), flush=True)
        target_indices, target_addresses = load_targets(nodes_path, args.split, args.max_cards)
        print(
            json.dumps(
                {
                    "phase": "load_targets_done",
                    "num_targets": len(target_indices),
                    "elapsed_seconds": round(time.time() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        print(json.dumps({"phase": "candidate_stream_start", "evidence": args.observation_evidence}, ensure_ascii=False), flush=True)
        cards, scanned, matched = build_cards_from_candidate_evidence(
            Path(args.observation_evidence),
            target_indices,
            target_addresses,
            args.max_edges_per_card,
            args.chunksize,
        )
        write_jsonl(cards_path, cards)
        print(
            json.dumps(
                {
                    "phase": "cards_done",
                    "num_cards": len(cards),
                    "rows_scanned": scanned,
                    "rows_matched": matched,
                    "cards_out": str(cards_path),
                    "elapsed_seconds": round(time.time() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    config = MotifConfig(
        max_cards=args.max_cards,
        max_edges_per_card=args.max_edges_per_card,
        min_confidence=args.min_confidence,
        llm_batch_size=args.llm_batch_size,
        llm_sleep=args.llm_sleep,
        max_tokens=args.max_tokens,
        prompt_no_think=args.prompt_no_think,
    )
    card_by_id = {card["target_id"]: card for card in cards}
    cached_rows, cached_target_ids = load_cached_responses(responses_path, card_by_id)
    normalized: list[dict[str, Any]] = [dict(row["result"]) for row in cached_rows if isinstance(row.get("result"), dict)]
    cards_to_process = [card for card in cards if card["target_id"] not in cached_target_ids]
    if cached_target_ids:
        print(
            json.dumps(
                {
                    "phase": "motif_teacher_resume",
                    "cached_cards": len(cached_target_ids),
                    "remaining_cards": len(cards_to_process),
                    "responses_out": str(responses_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(json.dumps({"phase": "motif_teacher_start", "teacher": args.teacher}, ensure_ascii=False), flush=True)
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with responses_path.open("a", encoding="utf-8") as response_fh:
        for start in range(0, len(cards_to_process), max(1, args.llm_batch_size)):
            batch = cards_to_process[start : start + max(1, args.llm_batch_size)]
            if args.teacher == "heuristic":
                batch_items = [{"raw_content": "", "result": heuristic_result(card)} for card in batch]
            else:
                if not args.llm_base_url:
                    raise ValueError("--llm-base-url is required when --teacher llm")
                batch_items = call_llm_resilient(batch, args=args, config=config)
            for item in batch_items:
                result = item["result"]
                target_id = str(result.get("target_id", ""))
                card = card_by_id.get(target_id)
                if card is None and len(batch) == 1:
                    card = batch[0]
                    result["target_id"] = card["target_id"]
                if card is None:
                    continue
                norm = normalize_result(result, card)
                normalized.append(norm)
                response_fh.write(
                    json.dumps({"target_id": norm["target_id"], "raw_content": item.get("raw_content", ""), "result": norm}, ensure_ascii=False)
                    + "\n"
                )
            response_fh.flush()
            if args.llm_sleep > 0:
                time.sleep(args.llm_sleep)
            processed = len(cached_target_ids) + min(start + len(batch), len(cards_to_process))
            if start == 0 or (start // max(1, args.llm_batch_size) + 1) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "motif_teacher_progress",
                            "processed_cards": processed,
                            "total_cards": len(cards),
                            "elapsed_seconds": round(time.time() - started, 2),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    evidence = evidence_rows_from_results(cards, normalized, args.min_confidence)
    evidence = fill_src_dst(evidence, cards)
    evidence.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary = {
        "data_dir": str(data_dir),
        "observation_evidence": str(args.observation_evidence),
        "out": str(out_path),
        "cards_out": str(cards_path),
        "responses_out": str(responses_path),
        "teacher": args.teacher,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "split": args.split,
        "num_targets": int(len(target_indices)),
        "num_cards": int(len(cards)),
        "num_observation_rows_scanned": int(scanned),
        "num_observation_rows_matched": int(matched),
        "num_evidence_edges": int(len(evidence)),
        "mean_evidence_score": float(evidence["evidence_score"].mean()) if len(evidence) else 0.0,
        "motif_types": evidence["evidence_type"].value_counts().to_dict() if len(evidence) else {},
        "elapsed_seconds": round(time.time() - started, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
