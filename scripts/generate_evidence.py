#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_rationale_gnn.data import load_graph
from llm_rationale_gnn.evidence import EvidenceConfig, rescore_with_openai_compatible, rule_score_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate edge-level weak evidence for rationale-aware GNN training.")
    parser.add_argument("--data-dir", required=True, help="Directory containing edges.csv and optional nodes.csv/labels.csv.")
    parser.add_argument("--out", required=True, help="Output evidence CSV path.")
    parser.add_argument("--teacher", choices=["rule", "llm"], default="rule")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--max-llm-edges", type=int, default=1000)
    parser.add_argument("--cache", default=None, help="Optional JSONL cache for LLM results.")
    parser.add_argument("--rapid-seconds", type=float, default=3600.0)
    parser.add_argument("--min-rule-score", type=float, default=0.20)
    parser.add_argument("--top-edges-per-source", type=int, default=20)
    parser.add_argument("--llm-batch-size", type=int, default=20)
    parser.add_argument("--llm-sleep", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    print(json.dumps({"phase": "load_graph_start", "data_dir": args.data_dir}, ensure_ascii=False), flush=True)
    loaded = load_graph(args.data_dir, build_edge_attr=False)
    print(
        json.dumps(
            {
                "phase": "load_graph_done",
                "num_nodes": int(loaded.data.num_nodes),
                "num_edges": int(loaded.data.num_edges),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    config = EvidenceConfig(
        rapid_seconds=args.rapid_seconds,
        min_rule_score=args.min_rule_score,
        top_edges_per_source=args.top_edges_per_source,
        llm_batch_size=args.llm_batch_size,
        llm_sleep=args.llm_sleep,
    )
    print(json.dumps({"phase": "rule_evidence_start"}, ensure_ascii=False), flush=True)
    evidence = rule_score_candidates(loaded.edge_frame, config)
    print(
        json.dumps(
            {
                "phase": "rule_evidence_done",
                "num_evidence_edges": int(len(evidence)),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if args.teacher == "llm":
        evidence = evidence.sort_values("evidence_score", ascending=False).head(args.max_llm_edges).sort_values("edge_pos")
        cache_path = Path(args.cache) if args.cache else out_path.with_suffix(".llm_cache.jsonl")
        evidence = rescore_with_openai_compatible(evidence, args.llm_model, config, cache_path)

    evidence.to_csv(out_path, index=False)
    summary = {
        "data_dir": str(args.data_dir),
        "out": str(out_path),
        "teacher": args.teacher,
        "num_edges_total": int(loaded.data.num_edges),
        "num_evidence_edges": int(len(evidence)),
        "mean_evidence_score": float(evidence["evidence_score"].mean()) if len(evidence) else 0.0,
        "evidence_types": evidence["evidence_type"].value_counts().to_dict() if len(evidence) else {},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
