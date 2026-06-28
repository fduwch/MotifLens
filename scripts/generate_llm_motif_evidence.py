#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_rationale_gnn.data import load_graph
from llm_rationale_gnn.evidence import EvidenceConfig, rule_score_candidates


MOTIF_TYPES = (
    "rapid_fanout_after_inflow",
    "pass_through_relay",
    "scatter_gather_laundering",
    "mule_collector",
    "peeling_chain_like",
    "victim_drain_then_cashout",
    "service_like_false_positive",
    "insufficient_evidence",
)

SUPPORT_MOTIFS = {
    "rapid_fanout_after_inflow",
    "pass_through_relay",
    "scatter_gather_laundering",
    "mule_collector",
    "peeling_chain_like",
    "victim_drain_then_cashout",
}
COUNTER_MOTIFS = {
    "service_like_false_positive",
}


@dataclass
class MotifConfig:
    max_cards: int
    max_edges_per_card: int
    min_confidence: float
    llm_batch_size: int
    llm_sleep: float
    max_tokens: int
    prompt_no_think: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LLM motif evidence from deterministic transaction observation cards."
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing nodes.csv and edges.csv.")
    parser.add_argument("--out", required=True, help="Training-compatible edge evidence CSV.")
    parser.add_argument("--cards-out", default=None, help="Optional JSONL observation cards.")
    parser.add_argument("--responses-out", default=None, help="Optional JSONL raw LLM motif responses.")
    parser.add_argument("--split", default="all", help="train/val/test/all; only labeled target addresses are carded.")
    parser.add_argument("--max-cards", type=int, default=0, help="Limit target cards. 0 means no limit.")
    parser.add_argument("--max-edges-per-card", type=int, default=12)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--teacher", choices=["llm", "heuristic"], default="llm")
    parser.add_argument("--llm-base-url", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--llm-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--llm-batch-size", type=int, default=2)
    parser.add_argument("--llm-sleep", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--prompt-no-think", action="store_true")
    parser.add_argument("--rapid-seconds", type=float, default=3600.0)
    parser.add_argument("--min-rule-score", type=float, default=0.20)
    parser.add_argument("--top-edges-per-source", type=int, default=20)
    return parser.parse_args()


def split_indices(data: Any, split: str) -> np.ndarray:
    wanted = split.strip().lower()
    labeled = data.y.detach().cpu().numpy() >= 0
    if wanted == "all":
        mask = labeled
    elif wanted == "train":
        mask = data.train_mask.detach().cpu().numpy() & labeled
    elif wanted in {"val", "valid", "validation"}:
        mask = data.val_mask.detach().cpu().numpy() & labeled
    elif wanted == "test":
        mask = data.test_mask.detach().cpu().numpy() & labeled
    else:
        raise ValueError("--split must be train, val, test, or all")
    return np.flatnonzero(mask)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def amount_bin(percentile: float) -> str:
    if percentile >= 0.95:
        return "very_high"
    if percentile >= 0.80:
        return "high"
    if percentile >= 0.50:
        return "medium"
    return "low"


def delay_bin(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "no_prior_inflow"
    if seconds <= 600:
        return "under_10m"
    if seconds <= 3600:
        return "under_1h"
    if seconds <= 86400:
        return "under_1d"
    return "long"


def degree_bin(value: float) -> str:
    if value <= 1:
        return "new_or_single"
    if value <= 5:
        return "low"
    if value <= 50:
        return "medium"
    return "high"


def reason_list(value: Any) -> list[str]:
    reasons = []
    for item in str(value or "").split("|"):
        item = item.strip()
        if item and item != "none":
            reasons.append(item)
    return reasons


def compact_edge(row: pd.Series, role: str, local_id: str) -> dict[str, Any]:
    delay = safe_float(row.get("time_since_last_inflow"), math.inf)
    amount_pct = safe_float(row.get("amount_percentile"))
    src_out_degree = safe_float(row.get("src_out_degree"))
    dst_in_degree = safe_float(row.get("dst_in_degree"))
    return {
        "edge_id": local_id,
        "edge_pos": int(row["edge_pos"]),
        "role": role,
        "observation_subject": "target_as_sender" if role == "outgoing" else "counterparty_as_sender",
        "observation_score": round(safe_float(row.get("evidence_score")), 4),
        "observation_type": str(row.get("evidence_type", "none")),
        "amount_bin": amount_bin(amount_pct),
        "delay_bin": delay_bin(delay),
        "src_out_degree_bin": degree_bin(src_out_degree),
        "dst_in_degree_bin": degree_bin(dst_in_degree),
        "receiver_low_degree": bool(safe_float(row.get("receiver_new_like")) > 0),
        "reason_codes": reason_list(row.get("reason_codes")),
    }


def aggregate_observations(edges: list[dict[str, Any]]) -> list[str]:
    if not edges:
        return ["no candidate transaction observations above the deterministic threshold"]
    types = Counter(str(edge["observation_type"]) for edge in edges)
    reasons = Counter(reason for edge in edges for reason in edge.get("reason_codes", []))
    roles = Counter(str(edge["role"]) for edge in edges)
    high_value = sum(1 for edge in edges if edge["amount_bin"] in {"high", "very_high"})
    outgoing = [edge for edge in edges if edge["role"] == "outgoing"]
    incoming = [edge for edge in edges if edge["role"] == "incoming"]
    target_rapid_outgoing = sum(1 for edge in outgoing if edge["delay_bin"] in {"under_10m", "under_1h"})
    counterparty_rapid_incoming = sum(1 for edge in incoming if edge["delay_bin"] in {"under_10m", "under_1h"})
    target_high_value_outgoing = sum(1 for edge in outgoing if edge["amount_bin"] in {"high", "very_high"})
    target_low_degree_receivers = sum(1 for edge in outgoing if edge["receiver_low_degree"])
    incoming_to_low_degree_target = sum(1 for edge in incoming if edge["receiver_low_degree"])
    low_receivers = sum(1 for edge in edges if edge["receiver_low_degree"])
    observations = [
        f"{len(edges)} candidate edges around the target address",
        f"direction mix: incoming={roles.get('incoming', 0)}, outgoing={roles.get('outgoing', 0)}",
        f"dominant low-level observations: {', '.join(name for name, _ in types.most_common(3))}",
        f"all-candidate high-value edges={high_value}, low-degree receiver edges={low_receivers}",
        f"target-as-sender signals: rapid outgoing after target inflow={target_rapid_outgoing}, high-value outgoing={target_high_value_outgoing}, outgoing to low-degree receivers={target_low_degree_receivers}",
        f"incoming counterparty signals: incoming edges whose sender recently relayed funds={counterparty_rapid_incoming}, incoming edges where target is low-degree receiver={incoming_to_low_degree_target}",
    ]
    if reasons:
        observations.append("frequent reason codes: " + ", ".join(name for name, _ in reasons.most_common(5)))
    return observations


def build_cards(
    observations: pd.DataFrame,
    node_ids: np.ndarray,
    target_indices: np.ndarray,
    max_cards: int,
    max_edges_per_card: int,
) -> list[dict[str, Any]]:
    by_src: dict[int, list[int]] = defaultdict(list)
    by_dst: dict[int, list[int]] = defaultdict(list)
    for pos, row in enumerate(observations[["src_idx", "dst_idx"]].itertuples(index=False)):
        by_src[int(row.src_idx)].append(pos)
        by_dst[int(row.dst_idx)].append(pos)

    cards: list[dict[str, Any]] = []
    for node_idx in target_indices:
        if max_cards and len(cards) >= max_cards:
            break
        positions = set(by_src.get(int(node_idx), [])) | set(by_dst.get(int(node_idx), []))
        if not positions:
            candidate_edges: list[dict[str, Any]] = []
        else:
            rows = observations.iloc[list(positions)].copy()
            rows["_role"] = np.where(rows["src_idx"].to_numpy() == int(node_idx), "outgoing", "incoming")
            rows = rows.sort_values(["evidence_score", "edge_pos"], ascending=[False, True]).head(max_edges_per_card)
            candidate_edges = [
                compact_edge(row, str(row["_role"]), f"e{i}")
                for i, (_, row) in enumerate(rows.iterrows())
            ]
        cards.append(
            {
                "target_id": f"n{int(node_idx)}",
                "target_node_idx": int(node_idx),
                "address": str(node_ids[int(node_idx)]),
                "task": "Choose a behavioral motif evidence type from the taxonomy. Do not predict the fraud label.",
                "observations": aggregate_observations(candidate_edges),
                "candidate_edges": candidate_edges,
            }
        )
    return cards


def motif_system_prompt(prompt_no_think: bool) -> str:
    prefix = "/no_think\n" if prompt_no_think else ""
    return (
        prefix
        + "You are a cautious blockchain AML motif annotator. "
        "You do not classify whether an address is fraudulent. "
        "You convert deterministic transaction observations into structured behavioral evidence motifs for a GNN. "
        "Use only the supplied observations and candidate_edges. "
        "Return strict JSON with a top-level key results. "
        "Each result must contain target_id, motif_type, confidence in [0,1], polarity, supporting_edge_ids, and reason_codes. "
        "motif_type must be one of: "
        + ", ".join(MOTIF_TYPES)
        + ". polarity must be support, counter, or neutral. "
        "Edge role matters: outgoing edges describe the target as sender; incoming edges describe the counterparty as sender and the target as receiver. "
        "For rapid_fanout_after_inflow, pass_through_relay, peeling_chain_like, and victim_drain_then_cashout, supporting edges should be outgoing target_as_sender edges. "
        "Do not choose rapid_fanout_after_inflow unless the card has target-as-sender rapid outgoing evidence. "
        "Do not choose pass_through_relay or victim_drain_then_cashout unless outgoing target_as_sender edges show short-delay onward flow. "
        "Incoming counterparty edges alone must not support target fan-out or pass-through motifs. "
        "For mule_collector, prefer incoming edges where the target receives/collects, or outgoing edges only if they clearly support collection behavior. "
        "If the observations are dominated by sender_high_out_degree or burst_activity but lack rapid outgoing after target inflow, "
        "lack high-value outgoing transfers, and look like routine high-volume address operation, prefer service_like_false_positive with polarity counter. "
        "For insufficient_evidence, use polarity neutral, confidence <= 0.5, and an empty supporting_edge_ids list. "
        "Use service_like_false_positive with polarity counter for observations that look like benign high-volume service behavior, "
        "and include the supporting edges that make the target look service-like. "
        "Do not use polarity counter merely because evidence is missing; use insufficient_evidence and neutral for missing evidence. "
        "Use insufficient_evidence if the card lacks a coherent multi-edge motif."
    )


def motif_messages(cards: list[dict[str, Any]], prompt_no_think: bool) -> list[dict[str, str]]:
    user = {
        "instruction": "Annotate each target card with one motif. Prefer multi-edge behavioral patterns over single threshold facts.",
        "taxonomy": {
            "rapid_fanout_after_inflow": "funds arrive then quickly split to multiple receivers",
            "pass_through_relay": "address mostly relays value onward shortly after receiving it",
            "scatter_gather_laundering": "funds scatter to several addresses or gather through consolidation-like flows",
            "mule_collector": "address collects from low-degree senders or newly active counterparties",
            "peeling_chain_like": "repeated partial onward transfers look like peeling or staged cash-out",
            "victim_drain_then_cashout": "large or sudden outgoing flow after suspicious inflow pattern",
            "service_like_false_positive": "pattern is more consistent with benign service/exchange-like activity",
            "insufficient_evidence": "no coherent motif can be inferred from the card",
        },
        "cards": cards,
    }
    return [
        {"role": "system", "content": motif_system_prompt(prompt_no_think)},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    return text


def parse_json_payload(text: str) -> Any:
    text = strip_think(text)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise ValueError("LLM response does not contain parseable JSON")


def normalize_results(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, dict):
        for key in ("results", "motifs", "evidence"):
            if isinstance(parsed.get(key), list):
                return [dict(item) for item in parsed[key] if isinstance(item, dict)]
        if "target_id" in parsed:
            return [parsed]
    if isinstance(parsed, list):
        return [dict(item) for item in parsed if isinstance(item, dict)]
    raise ValueError("LLM JSON must be a result object or a list of result objects")


def call_openai_compatible(
    cards: list[dict[str, Any]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    config: MotifConfig,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=motif_messages(cards, config.prompt_no_think),
        temperature=0.0,
        max_tokens=config.max_tokens,
    )
    content = response.choices[0].message.content or "{}"
    parsed = parse_json_payload(content)
    results = normalize_results(parsed)
    return [{"raw_content": content, "result": item} for item in results]


def heuristic_result(card: dict[str, Any]) -> dict[str, Any]:
    reasons = Counter(reason for edge in card["candidate_edges"] for reason in edge.get("reason_codes", []))
    edge_ids = [edge["edge_id"] for edge in card["candidate_edges"][:4]]
    if not card["candidate_edges"]:
        motif, confidence, polarity = "insufficient_evidence", 0.0, "neutral"
    elif reasons["outflow_shortly_after_inflow"] and len(edge_ids) >= 2:
        motif, confidence, polarity = "rapid_fanout_after_inflow", 0.72, "support"
    elif reasons["receiver_consolidation_like"]:
        motif, confidence, polarity = "scatter_gather_laundering", 0.64, "support"
    elif reasons["receiver_has_low_degree"]:
        motif, confidence, polarity = "mule_collector", 0.58, "support"
    elif reasons["sender_high_out_degree"]:
        motif, confidence, polarity = "service_like_false_positive", 0.45, "counter"
    else:
        motif, confidence, polarity = "insufficient_evidence", 0.25, "neutral"
    return {
        "target_id": card["target_id"],
        "motif_type": motif,
        "confidence": confidence,
        "polarity": polarity,
        "supporting_edge_ids": edge_ids,
        "reason_codes": [name for name, _ in reasons.most_common(5)],
    }


def normalize_reason_code(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def allowed_support_edge_ids(card: dict[str, Any], motif: str) -> set[str]:
    edges = card["candidate_edges"]
    if motif in {"rapid_fanout_after_inflow", "pass_through_relay", "peeling_chain_like", "victim_drain_then_cashout"}:
        return {edge["edge_id"] for edge in edges if edge.get("role") == "outgoing"}
    if motif == "mule_collector":
        return {edge["edge_id"] for edge in edges if edge.get("role") == "incoming"}
    if motif == "scatter_gather_laundering":
        return {edge["edge_id"] for edge in edges}
    return set()


def allowed_counter_edge_ids(card: dict[str, Any], motif: str) -> set[str]:
    edges = card["candidate_edges"]
    if motif == "service_like_false_positive":
        return {edge["edge_id"] for edge in edges}
    return set()


def motif_has_required_signal(card: dict[str, Any], motif: str, edge_ids: list[str]) -> bool:
    edge_by_id = {edge["edge_id"]: edge for edge in card["candidate_edges"]}
    selected = [edge_by_id[edge_id] for edge_id in edge_ids if edge_id in edge_by_id]
    if motif in {"rapid_fanout_after_inflow", "pass_through_relay", "victim_drain_then_cashout"}:
        rapid_outgoing = [
            edge
            for edge in selected
            if edge.get("role") == "outgoing"
            and (
                edge.get("delay_bin") in {"under_10m", "under_1h"}
                or "outflow_shortly_after_inflow" in edge.get("reason_codes", [])
            )
        ]
        return len(rapid_outgoing) > 0
    if motif == "peeling_chain_like":
        return sum(1 for edge in selected if edge.get("role") == "outgoing") >= 2
    if motif == "mule_collector":
        return len(selected) > 0 and all(edge.get("role") == "incoming" for edge in selected)
    if motif == "scatter_gather_laundering":
        reasons = {reason for edge in selected for reason in edge.get("reason_codes", [])}
        roles = {edge.get("role") for edge in selected}
        return "receiver_consolidation_like" in reasons or (len(selected) >= 3 and len(roles) >= 2)
    return motif in SUPPORT_MOTIFS


def normalize_result(item: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    motif = str(item.get("motif_type", "insufficient_evidence"))
    if motif not in MOTIF_TYPES:
        motif = "insufficient_evidence"
    polarity = str(item.get("polarity", "neutral")).lower()
    if polarity not in {"support", "counter", "neutral"}:
        polarity = "neutral"
    confidence = max(0.0, min(1.0, safe_float(item.get("confidence"), 0.0)))
    if motif == "insufficient_evidence":
        polarity = "neutral"
        confidence = min(confidence, 0.5)
    edge_ids = item.get("supporting_edge_ids", [])
    if isinstance(edge_ids, str):
        edge_ids = [edge_ids]
    valid_ids = {edge["edge_id"] for edge in card["candidate_edges"]}
    edge_ids = [str(edge_id) for edge_id in edge_ids if str(edge_id) in valid_ids]
    if polarity == "support":
        allowed_ids = allowed_support_edge_ids(card, motif)
    elif polarity == "counter":
        allowed_ids = allowed_counter_edge_ids(card, motif)
    else:
        allowed_ids = set()
    edge_ids = [edge_id for edge_id in edge_ids if edge_id in allowed_ids]
    if not edge_ids and polarity == "support" and motif in SUPPORT_MOTIFS:
        edge_ids = [edge["edge_id"] for edge in card["candidate_edges"] if edge["edge_id"] in allowed_ids][:3]
    if not edge_ids and polarity == "counter" and motif in COUNTER_MOTIFS:
        edge_ids = [edge["edge_id"] for edge in card["candidate_edges"] if edge["edge_id"] in allowed_ids][:3]
    post_filter_reason = ""
    if polarity == "support" and motif in SUPPORT_MOTIFS and not motif_has_required_signal(card, motif, edge_ids):
        motif = "insufficient_evidence"
        polarity = "neutral"
        confidence = min(confidence, 0.4)
        edge_ids = []
        post_filter_reason = "post_filter_missing_required_signal"
    if not ((polarity == "support" and motif in SUPPORT_MOTIFS) or (polarity == "counter" and motif in COUNTER_MOTIFS)):
        edge_ids = []
    reasons = item.get("reason_codes", [])
    if isinstance(reasons, str):
        reasons = reason_list(reasons)
    reasons = [normalize_reason_code(reason) for reason in reasons if normalize_reason_code(reason)]
    if post_filter_reason:
        reasons.append(post_filter_reason)
    return {
        "target_id": card["target_id"],
        "target_node_idx": int(card["target_node_idx"]),
        "address": card["address"],
        "motif_type": motif,
        "confidence": confidence,
        "polarity": polarity,
        "supporting_edge_ids": edge_ids,
        "reason_codes": reasons,
        "rationale": str(item.get("rationale", ""))[:500],
    }


def write_jsonl(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_cached_responses(path: Path | None, card_by_id: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    if path is None or not path.exists():
        return [], set()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            result = item.get("result", item)
            target_id = str(result.get("target_id", ""))
            if target_id in card_by_id and target_id not in seen:
                rows.append(item)
                seen.add(target_id)
    return rows, seen


def evidence_rows_from_results(cards: list[dict[str, Any]], results: list[dict[str, Any]], min_confidence: float) -> pd.DataFrame:
    card_by_id = {card["target_id"]: card for card in cards}
    rows: list[dict[str, Any]] = []
    for result in results:
        card = card_by_id.get(result["target_id"])
        if card is None:
            continue
        edge_by_id = {edge["edge_id"]: edge for edge in card["candidate_edges"]}
        keep_support = result["polarity"] == "support" and result["motif_type"] in SUPPORT_MOTIFS
        keep_counter = result["polarity"] == "counter" and result["motif_type"] in COUNTER_MOTIFS
        if not (keep_support or keep_counter):
            continue
        if result["confidence"] < min_confidence:
            continue
        reasons = result["reason_codes"] or [result["motif_type"]]
        reason_codes = "|".join(reasons)
        for edge_id in result["supporting_edge_ids"]:
            edge = edge_by_id.get(edge_id)
            if not edge:
                continue
            rows.append(
                {
                    "edge_pos": int(edge["edge_pos"]),
                    "src_idx": "",
                    "dst_idx": "",
                    "target_idx": int(result["target_node_idx"]),
                    "address": result["address"],
                    "local_edge_id": edge_id,
                    "evidence_score": float(result["confidence"]),
                    "evidence_type": result["motif_type"],
                    "reason_codes": reason_codes,
                    "polarity": result["polarity"],
                    "observation_score": edge["observation_score"],
                    "observation_type": edge["observation_type"],
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
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
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["edge_pos", "target_idx"]).drop_duplicates(["edge_pos", "target_idx"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    started = time.time()
    out_path = Path(args.out)
    cards_path = Path(args.cards_out) if args.cards_out else out_path.with_suffix(".cards.jsonl")
    responses_path = Path(args.responses_out) if args.responses_out else out_path.with_suffix(".responses.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"phase": "load_graph_start", "data_dir": args.data_dir}, ensure_ascii=False), flush=True)
    loaded = load_graph(args.data_dir, build_edge_attr=False)
    target_indices = split_indices(loaded.data, args.split)
    print(
        json.dumps(
            {
                "phase": "load_graph_done",
                "num_nodes": int(loaded.data.num_nodes),
                "num_edges": int(loaded.data.num_edges),
                "num_targets": int(len(target_indices)),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    rule_config = EvidenceConfig(
        rapid_seconds=args.rapid_seconds,
        min_rule_score=args.min_rule_score,
        top_edges_per_source=args.top_edges_per_source,
    )
    print(json.dumps({"phase": "observation_candidates_start"}, ensure_ascii=False), flush=True)
    observations = rule_score_candidates(loaded.edge_frame, rule_config)
    print(
        json.dumps(
            {
                "phase": "observation_candidates_done",
                "num_observation_edges": int(len(observations)),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    cards = build_cards(
        observations,
        loaded.node_ids,
        target_indices,
        max_cards=args.max_cards,
        max_edges_per_card=args.max_edges_per_card,
    )
    write_jsonl(cards_path, cards)
    print(json.dumps({"phase": "cards_done", "num_cards": len(cards), "cards_out": str(cards_path)}, ensure_ascii=False), flush=True)

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
                batch_items = call_openai_compatible(
                    batch,
                    base_url=args.llm_base_url,
                    api_key=args.llm_api_key,
                    model=args.llm_model,
                    config=config,
                )
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
    if len(evidence):
        src_dst = observations[["edge_pos", "src_idx", "dst_idx"]].drop_duplicates("edge_pos")
        evidence = evidence.drop(columns=["src_idx", "dst_idx"]).merge(src_dst, on="edge_pos", how="left")
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
        evidence = evidence[preferred]
    evidence.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "data_dir": str(args.data_dir),
        "out": str(out_path),
        "cards_out": str(cards_path),
        "responses_out": str(responses_path),
        "teacher": args.teacher,
        "llm_base_url": args.llm_base_url,
        "llm_model": args.llm_model,
        "split": args.split,
        "num_edges_total": int(loaded.data.num_edges),
        "num_targets": int(len(target_indices)),
        "num_cards": int(len(cards)),
        "num_observation_edges": int(len(observations)),
        "num_evidence_edges": int(len(evidence)),
        "mean_evidence_score": float(evidence["evidence_score"].mean()) if len(evidence) else 0.0,
        "motif_types": evidence["evidence_type"].value_counts().to_dict() if len(evidence) else {},
        "polarities": evidence["polarity"].value_counts().to_dict() if len(evidence) and "polarity" in evidence.columns else {},
        "elapsed_seconds": round(time.time() - started, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
