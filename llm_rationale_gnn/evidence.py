from __future__ import annotations

import json
import os
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


EVIDENCE_TYPES = (
    "rapid_fanout",
    "high_value_transfer",
    "new_counterparty",
    "burst_activity",
    "consolidation",
    "none",
)


@dataclass
class EvidenceConfig:
    rapid_seconds: float = 3600.0
    min_rule_score: float = 0.20
    top_edges_per_source: int = 20
    llm_batch_size: int = 20
    llm_sleep: float = 0.0


def percentile_rank(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values.astype(np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(values), endpoint=True)
    return ranks


def compute_time_since_last_inflow(edge_df: pd.DataFrame) -> np.ndarray:
    if "timestamp_num" not in edge_df.columns:
        return np.full(len(edge_df), np.inf, dtype=np.float64)

    incoming: dict[int, list[float]] = {}
    for dst, ts in edge_df[["dst_idx", "timestamp_num"]].itertuples(index=False):
        if np.isfinite(ts):
            incoming.setdefault(int(dst), []).append(float(ts))
    for times in incoming.values():
        times.sort()

    delays = np.full(len(edge_df), np.inf, dtype=np.float64)
    for idx, src, ts in edge_df[["src_idx", "timestamp_num"]].itertuples(index=True, name=None):
        if not np.isfinite(ts):
            continue
        times = incoming.get(int(src))
        if not times:
            continue
        pos = bisect_left(times, float(ts))
        if pos > 0:
            delays[idx] = float(ts) - times[pos - 1]
    return delays


def candidate_feature_frame(edge_df: pd.DataFrame) -> pd.DataFrame:
    amount = pd.to_numeric(edge_df.get("amount_num", 1.0), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=np.float64)
    src = edge_df["src_idx"].to_numpy(dtype=np.int64)
    dst = edge_df["dst_idx"].to_numpy(dtype=np.int64)
    num_nodes = int(max(src.max(initial=0), dst.max(initial=0)) + 1)

    out_degree = np.bincount(src, minlength=num_nodes).astype(np.float64)
    in_degree = np.bincount(dst, minlength=num_nodes).astype(np.float64)
    total_degree = out_degree + in_degree

    delay = compute_time_since_last_inflow(edge_df)
    amount_pct = percentile_rank(np.log1p(amount))
    out_degree_pct = percentile_rank(out_degree)[src]
    in_degree_pct = percentile_rank(in_degree)[dst]

    return pd.DataFrame(
        {
            "edge_pos": edge_df["edge_pos"].to_numpy(dtype=np.int64),
            "src_idx": src,
            "dst_idx": dst,
            "amount_percentile": amount_pct,
            "src_out_degree": out_degree[src],
            "dst_in_degree": in_degree[dst],
            "src_total_degree": total_degree[src],
            "dst_total_degree": total_degree[dst],
            "src_out_degree_percentile": out_degree_pct,
            "dst_in_degree_percentile": in_degree_pct,
            "receiver_new_like": (total_degree[dst] <= 2).astype(float),
            "time_since_last_inflow": delay,
        }
    )


def rule_score_candidates(edge_df: pd.DataFrame, config: EvidenceConfig) -> pd.DataFrame:
    df = candidate_feature_frame(edge_df)
    rapid = ((df["time_since_last_inflow"] >= 0) & (df["time_since_last_inflow"] <= config.rapid_seconds)).to_numpy(dtype=np.float64)
    high_value = df["amount_percentile"].to_numpy(dtype=np.float64)
    new_counterparty = df["receiver_new_like"].to_numpy(dtype=np.float64)
    burst = df["src_out_degree_percentile"].to_numpy(dtype=np.float64)
    consolidation = (df["dst_in_degree_percentile"].to_numpy(dtype=np.float64) * high_value)

    components = np.vstack(
        [
            0.35 * rapid * np.maximum(high_value, 0.20),
            0.25 * high_value,
            0.15 * new_counterparty,
            0.15 * burst,
            0.10 * consolidation,
        ]
    )
    score = np.clip(components.sum(axis=0), 0.0, 1.0)
    type_indices = np.argmax(
        np.vstack(
            [
                rapid * np.maximum(high_value, 0.20),
                high_value,
                new_counterparty,
                burst,
                consolidation,
            ]
        ),
        axis=0,
    )
    type_names = np.array(EVIDENCE_TYPES[:-1], dtype=object)[type_indices]
    type_names[score < config.min_rule_score] = "none"

    keep = score >= config.min_rule_score
    out = df.loc[
        keep,
        [
            "edge_pos",
            "src_idx",
            "dst_idx",
            "amount_percentile",
            "src_out_degree",
            "dst_in_degree",
            "receiver_new_like",
            "time_since_last_inflow",
        ],
    ].copy()
    out["_row_pos"] = np.flatnonzero(keep)
    out["evidence_score"] = score[keep]
    out["evidence_type"] = type_names[keep]

    if config.top_edges_per_source > 0:
        out = (
            out.sort_values(["src_idx", "evidence_score"], ascending=[True, False])
            .groupby("src_idx", as_index=False, group_keys=False)
            .head(config.top_edges_per_source)
        )
    row_pos = out["_row_pos"].to_numpy(dtype=np.int64)
    reasons: list[str] = []
    for i in row_pos:
        r: list[str] = []
        if rapid[i] > 0:
            r.append("outflow_shortly_after_inflow")
        if high_value[i] >= 0.90:
            r.append("high_value_percentile")
        if new_counterparty[i] > 0:
            r.append("receiver_has_low_degree")
        if burst[i] >= 0.90:
            r.append("sender_high_out_degree")
        if consolidation[i] >= 0.80:
            r.append("receiver_consolidation_like")
        reasons.append("|".join(r) if r else "none")
    out["reason_codes"] = reasons
    out = out.drop(columns=["_row_pos"]).sort_values("edge_pos").reset_index(drop=True)
    return out


def compact_candidate_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
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
    ]
    records: list[dict[str, Any]] = []
    for row in df[fields].to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                clean[key] = None
            elif isinstance(value, (np.integer, np.floating)):
                clean[key] = value.item()
            else:
                clean[key] = value
        records.append(clean)
    return records


def llm_prompt(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        "You are a cautious blockchain AML evidence annotator. "
        "You do not decide whether an account is fraudulent. "
        "You only rescore candidate transaction edges as weak evidence for GNN training. "
        "Return strict JSON: a list of objects with edge_pos, evidence_score in [0,1], "
        "evidence_type, and reason_codes. Use evidence_type from: "
        + ", ".join(EVIDENCE_TYPES)
        + "."
    )
    user = {
        "task": "Rescore these candidate Ethereum transaction edges using only the structured statistics.",
        "rules": [
            "Favor rapid fan-out after inflow, high value transfer, receiver with low degree, burst activity, and consolidation-like receivers.",
            "Do not use fraud labels or infer facts not present in the record.",
            "Keep scores calibrated; use 0.0 for no evidence.",
        ],
        "candidates": records,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_llm_json(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        for key in ("edges", "evidence", "results"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    if isinstance(parsed, list):
        return parsed
    raise ValueError("LLM response is not a JSON list.")


def rescore_with_openai_compatible(
    candidates: pd.DataFrame,
    model: str,
    config: EvidenceConfig,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package is required for --teacher llm") from exc

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )

    cached: dict[int, dict[str, Any]] = {}
    if cache_path and cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    item = json.loads(line)
                    cached[int(item["edge_pos"])] = item

    rows: list[dict[str, Any]] = []
    records = compact_candidate_records(candidates)
    missing = [r for r in records if int(r["edge_pos"]) not in cached]

    for start in range(0, len(missing), config.llm_batch_size):
        batch = missing[start : start + config.llm_batch_size]
        messages = llm_prompt(batch)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        content = response.choices[0].message.content or "[]"
        parsed = parse_llm_json(content)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as fh:
                for item in parsed:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        for item in parsed:
            cached[int(item["edge_pos"])] = item
        if config.llm_sleep > 0:
            time.sleep(config.llm_sleep)

    for record in records:
        edge_pos = int(record["edge_pos"])
        item = cached.get(edge_pos, record)
        score = float(item.get("evidence_score", record.get("evidence_score", 0.0)))
        evidence_type = str(item.get("evidence_type", record.get("evidence_type", "none")))
        reason_codes = item.get("reason_codes", record.get("reason_codes", "none"))
        if isinstance(reason_codes, list):
            reason_codes = "|".join(str(x) for x in reason_codes)
        rows.append(
            {
                **record,
                "evidence_score": max(0.0, min(1.0, score)),
                "evidence_type": evidence_type if evidence_type in EVIDENCE_TYPES else "none",
                "reason_codes": str(reason_codes),
            }
        )

    return pd.DataFrame(rows).sort_values("edge_pos").reset_index(drop=True)
