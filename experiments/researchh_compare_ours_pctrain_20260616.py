#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = ROOT / "outputs/paper_5seed_qwen_counter_clean_20260615"
PCTRAIN_ROOT = ROOT / "outputs/paper_5seed_qwen_counter_clean_pctrain_20260616"
OUT_DIR = PCTRAIN_ROOT / "summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["Meta", "ZipZap", "Illicit-ETH", "EPSD-Ponzi"]
SEEDS = [42, 43, 44, 45, 46]
METHODS = [
    ("sage_no_evidence", "Base GraphSAGE"),
    ("sage_rule_evidence", "GNN + rule evidence"),
    ("sage_qwen_motif_channel", "GNN + Qwen motif"),
    ("sage_rule_scalar_qwen_motif_channel", "GNN + rule + Qwen motif"),
]


def metric_path(root: Path, dataset: str, seed: int, method: str) -> Path:
    return root / dataset / f"seed_{seed}" / method / "metrics.json"


def load_f1(path: Path) -> float | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    row = data.get("best_test_at_val_threshold") or {}
    value = row.get("f1")
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


rows: list[dict] = []
aggregate: list[dict] = []
warnings: list[dict] = []

for dataset in DATASETS:
    for method, label in METHODS:
        deltas = []
        base_values = []
        pctrain_values = []
        for seed in SEEDS:
            base_f1 = load_f1(metric_path(BASE_ROOT, dataset, seed, method))
            pctrain_f1 = load_f1(metric_path(PCTRAIN_ROOT, dataset, seed, method))
            row = {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "method_label": label,
                "base_f1": base_f1,
                "pctrain_f1": pctrain_f1,
                "delta": None if base_f1 is None or pctrain_f1 is None else pctrain_f1 - base_f1,
                "base_metrics_path": str(metric_path(BASE_ROOT, dataset, seed, method).relative_to(ROOT)),
                "pctrain_metrics_path": str(metric_path(PCTRAIN_ROOT, dataset, seed, method).relative_to(ROOT)),
            }
            rows.append(row)
            if base_f1 is not None:
                base_values.append(base_f1)
            if pctrain_f1 is not None:
                pctrain_values.append(pctrain_f1)
            if row["delta"] is not None:
                deltas.append(float(row["delta"]))
        base_mean, base_std = summarize(base_values)
        pc_mean, pc_std = summarize(pctrain_values)
        delta_mean, delta_std = summarize(deltas)
        agg = {
            "dataset": dataset,
            "method": method,
            "method_label": label,
            "base_completed": len(base_values),
            "pctrain_completed": len(pctrain_values),
            "expected": len(SEEDS),
            "base_f1_mean": base_mean,
            "base_f1_std": base_std,
            "pctrain_f1_mean": pc_mean,
            "pctrain_f1_std": pc_std,
            "delta_f1_mean": delta_mean,
            "delta_f1_std": delta_std,
        }
        aggregate.append(agg)
        if len(pctrain_values) < len(SEEDS):
            warnings.append(
                {
                    "severity": "info",
                    "type": "incomplete_pctrain",
                    "dataset": dataset,
                    "method": method,
                    "message": f"PC-train has {len(pctrain_values)}/{len(SEEDS)} seeds; do not use as final yet.",
                }
            )
        elif method == "sage_rule_scalar_qwen_motif_channel" and delta_mean is not None:
            if delta_mean < -0.005:
                warnings.append(
                    {
                        "severity": "warning",
                        "type": "pctrain_hurts_ours",
                        "dataset": dataset,
                        "method": method,
                        "delta_f1_mean": delta_mean,
                        "message": "PC-train lowers the final Ours row; inspect before replacing main table root.",
                    }
                )
            elif delta_mean > 0.005:
                warnings.append(
                    {
                        "severity": "info",
                        "type": "pctrain_improves_ours",
                        "dataset": dataset,
                        "method": method,
                        "delta_f1_mean": delta_mean,
                        "message": "PC-train improves the final Ours row; candidate for replacing main table root after anomaly checks.",
                    }
                )


def write_csv(path: Path, data: list[dict]) -> None:
    keys: list[str] = []
    for row in data:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


write_csv(OUT_DIR / "compare_to_clean_per_seed.csv", rows)
write_csv(OUT_DIR / "compare_to_clean_aggregate.csv", aggregate)

payload = {
    "base_root": str(BASE_ROOT.relative_to(ROOT)),
    "pctrain_root": str(PCTRAIN_ROOT.relative_to(ROOT)),
    "aggregate": aggregate,
    "warnings": warnings,
}
(OUT_DIR / "compare_to_clean.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

md: list[str] = []
md.append("# Ours PC-Train vs Clean Ours")
md.append("")
md.append("Primary metric: test F1 at the threshold selected by validation-best F1.")
md.append("")
md.append("| Dataset | Method | Done | Clean F1 | PC-train F1 | Delta |")
md.append("|---|---|---:|---:|---:|---:|")
for row in aggregate:
    md.append(
        "| {dataset} | {method} | {done}/{expected} | {base}+-{bases} | {pc}+-{pcs} | {delta}+-{deltas} |".format(
            dataset=row["dataset"],
            method=row["method_label"],
            done=row["pctrain_completed"],
            expected=row["expected"],
            base=fmt(row["base_f1_mean"]),
            bases=fmt(row["base_f1_std"]),
            pc=fmt(row["pctrain_f1_mean"]),
            pcs=fmt(row["pctrain_f1_std"]),
            delta=fmt(row["delta_f1_mean"]),
            deltas=fmt(row["delta_f1_std"]),
        )
    )
md.append("")
md.append("## Warnings")
md.append("")
if warnings:
    for item in warnings:
        md.append(f"- [{item['severity']}] {item['dataset']} {item['method']}: {item['message']}")
else:
    md.append("No warnings.")
md.append("")
md.append("## Files")
md.append("")
md.append("- `outputs/paper_5seed_qwen_counter_clean_pctrain_20260616/summary/compare_to_clean_per_seed.csv`")
md.append("- `outputs/paper_5seed_qwen_counter_clean_pctrain_20260616/summary/compare_to_clean_aggregate.csv`")
md.append("- `outputs/paper_5seed_qwen_counter_clean_pctrain_20260616/summary/compare_to_clean.json`")
(OUT_DIR / "compare_to_clean.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("compare_dir", OUT_DIR)
print("pctrain_completed", sum(1 for row in rows if row["pctrain_f1"] is not None), "/", len(rows))
for row in aggregate:
    if row["pctrain_completed"]:
        print(
            row["dataset"],
            row["method"],
            f"{row['pctrain_completed']}/{row['expected']}",
            "clean",
            fmt(row["base_f1_mean"]),
            "pctrain",
            fmt(row["pctrain_f1_mean"]),
            "delta",
            fmt(row["delta_f1_mean"]),
        )

