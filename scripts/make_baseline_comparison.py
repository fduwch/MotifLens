#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORTED_BASELINES = [
    {
        "dataset": "MulDiGraph",
        "method": "BERT4ETH",
        "source": "reported:LMAE4Eth README",
        "precision": 0.4469,
        "recall": 0.7344,
        "f1": 0.5557,
        "balanced_accuracy": 0.6400,
    },
    {
        "dataset": "MulDiGraph",
        "method": "ZipZap",
        "source": "reported:LMAE4Eth README",
        "precision": 0.4537,
        "recall": 0.7298,
        "f1": 0.5595,
        "balanced_accuracy": 0.6452,
    },
    {
        "dataset": "MulDiGraph",
        "method": "TLmGNN",
        "source": "reported:TLmGNN README",
        "precision": None,
        "recall": None,
        "f1": 0.9041,
        "balanced_accuracy": None,
    },
    {
        "dataset": "MulDiGraph",
        "method": "LMAE4Eth",
        "source": "reported:LMAE4Eth README",
        "precision": 0.9024,
        "recall": 0.8889,
        "f1": 0.8960,
        "balanced_accuracy": 0.9204,
    },
    {
        "dataset": "PhishTrans/B4E",
        "method": "BERT4ETH",
        "source": "reported:LMAE4Eth README",
        "precision": 0.7421,
        "recall": 0.6125,
        "f1": 0.6711,
        "balanced_accuracy": 0.7530,
    },
    {
        "dataset": "PhishTrans/B4E",
        "method": "ZipZap",
        "source": "reported:LMAE4Eth README",
        "precision": 0.7374,
        "recall": 0.6132,
        "f1": 0.6696,
        "balanced_accuracy": 0.7520,
    },
    {
        "dataset": "PhishTrans/B4E",
        "method": "TLmGNN",
        "source": "reported:TLmGNN README",
        "precision": None,
        "recall": None,
        "f1": 0.8123,
        "balanced_accuracy": None,
    },
    {
        "dataset": "PhishTrans/B4E",
        "method": "LMAE4Eth",
        "source": "reported:LMAE4Eth README",
        "precision": 0.7903,
        "recall": 0.8397,
        "f1": 0.8143,
        "balanced_accuracy": 0.8641,
    },
]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def metric_row(dataset: str, method: str, path: Path, metric_key: str, source: str) -> dict[str, Any]:
    obj = load_json(path)
    metrics = obj[metric_key]
    return {
        "dataset": dataset,
        "method": method,
        "source": source,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "accuracy": metrics.get("accuracy"),
        "auc": metrics.get("auc"),
        "ap": metrics.get("ap"),
        "threshold": metrics.get("threshold"),
        "path": str(path),
    }


def full_row(dataset: str, method: str, path: Path, source: str) -> dict[str, Any]:
    obj = load_json(path)
    metrics = obj["best_test_at_val_threshold"]
    return {
        "dataset": dataset,
        "method": method,
        "source": source,
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "balanced_accuracy": None,
        "accuracy": metrics.get("accuracy"),
        "auc": metrics.get("auc"),
        "ap": metrics.get("ap"),
        "threshold": metrics.get("threshold"),
        "path": str(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comparison tables for first two datasets.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--phish-no-evidence-sampled", required=True)
    parser.add_argument("--phish-rule-sampled", required=True)
    parser.add_argument("--muldi-no-evidence-sampled", required=True)
    parser.add_argument("--muldi-rule-sampled", required=True)
    parser.add_argument("--phish-no-evidence-full", required=True)
    parser.add_argument("--phish-rule-full", required=True)
    parser.add_argument("--muldi-no-evidence-full", required=True)
    parser.add_argument("--muldi-rule-full", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampled_rows = list(REPORTED_BASELINES)
    sampled_rows.extend(
        [
            metric_row("PhishTrans/B4E", "Ours-no-evidence", Path(args.phish_no_evidence_sampled), "sample_best_threshold", "ours:unified-sampled-test"),
            metric_row("PhishTrans/B4E", "Ours-rule", Path(args.phish_rule_sampled), "sample_best_threshold", "ours:unified-sampled-test"),
            metric_row("MulDiGraph", "Ours-no-evidence", Path(args.muldi_no_evidence_sampled), "sample_best_threshold", "ours:unified-sampled-test"),
            metric_row("MulDiGraph", "Ours-rule", Path(args.muldi_rule_sampled), "sample_best_threshold", "ours:unified-sampled-test"),
        ]
    )
    full_rows = [
        full_row("PhishTrans/B4E", "Ours-no-evidence", Path(args.phish_no_evidence_full), "ours:full-test-val-threshold"),
        full_row("PhishTrans/B4E", "Ours-rule", Path(args.phish_rule_full), "ours:full-test-val-threshold"),
        full_row("MulDiGraph", "Ours-no-evidence", Path(args.muldi_no_evidence_full), "ours:full-test-val-threshold"),
        full_row("MulDiGraph", "Ours-rule", Path(args.muldi_rule_full), "ours:full-test-val-threshold"),
    ]
    out = {
        "notes": [
            "Reported baseline rows are copied from public README tables, not rerun in this job.",
            "Ours-rule uses only rule evidence features; no LLM evidence is used.",
            "sampled_comparison uses test positives plus 2x sampled test negatives and sample_best_threshold.",
            "full_test_ours uses the full test split and the validation-selected threshold from train_sampled.py.",
        ],
        "sampled_comparison": sampled_rows,
        "full_test_ours": full_rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"phase": "done", "out": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
