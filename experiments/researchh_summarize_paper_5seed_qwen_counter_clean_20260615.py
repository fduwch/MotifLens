#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/paper_5seed_qwen_counter_clean_20260615"
SUMMARY_DIR = OUT_ROOT / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["Meta", "ZipZap", "Illicit-ETH", "EPSD-Ponzi"]
CORE_SEEDS = [42, 43, 44, 45, 46]
METHOD_LABELS = {
    "sage_no_evidence": "Base GNN",
    "sage_rule_evidence": "GNN + rule",
    "sage_qwen_motif_channel": "GNN + Qwen motif",
    "sage_rule_scalar_qwen_motif_channel": "GNN + rule + Qwen motif",
}
METHOD_SEEDS = {
    "sage_no_evidence": CORE_SEEDS,
    "sage_rule_evidence": CORE_SEEDS,
    "sage_qwen_motif_channel": CORE_SEEDS,
    "sage_rule_scalar_qwen_motif_channel": CORE_SEEDS,
}


def get_metric(metrics: dict, path: str) -> float | None:
    cur = metrics
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    return float(cur)


def fmt(v: float | None, nd: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return f"{v:.{nd}f}"


def summarize_values(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


rows: list[dict] = []
aggregate: list[dict] = []
missing: list[dict] = []
loaded: dict[tuple[str, int, str], dict] = {}

for dataset in DATASETS:
    for method, seeds in METHOD_SEEDS.items():
        for seed in seeds:
            path = OUT_ROOT / dataset / f"seed_{seed}" / method / "metrics.json"
            if not path.exists():
                missing.append({"dataset": dataset, "seed": seed, "method": method, "path": str(path.relative_to(ROOT))})
                continue
            metrics = json.loads(path.read_text(encoding="utf-8"))
            loaded[(dataset, seed, method)] = metrics
            row = {
                "dataset": dataset,
                "seed": seed,
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "best_epoch": metrics.get("best_epoch"),
                "val_best_f1": get_metric(metrics, "best_val_threshold_f1"),
                "test_f1_at_val": get_metric(metrics, "best_test_at_val_threshold.f1"),
                "test_auc_at_val": get_metric(metrics, "best_test_at_val_threshold.auc"),
                "test_ap_at_val": get_metric(metrics, "best_test_at_val_threshold.ap"),
                "test_precision_at_val": get_metric(metrics, "best_test_at_val_threshold.precision"),
                "test_recall_at_val": get_metric(metrics, "best_test_at_val_threshold.recall"),
                "threshold": get_metric(metrics, "best_test_at_val_threshold.threshold"),
                "history_best_test_f1_at_val": get_metric(metrics, "history_best_test_at_val_threshold.test_at_val_threshold.f1"),
                "num_nodes": get_metric(metrics, "data.num_nodes"),
                "num_edges": get_metric(metrics, "data.num_edges"),
                "train_nodes": get_metric(metrics, "data.train_nodes"),
                "val_nodes": get_metric(metrics, "data.val_nodes"),
                "test_nodes": get_metric(metrics, "data.test_nodes"),
                "metrics_path": str(path.relative_to(ROOT)),
            }
            rows.append(row)

for dataset in DATASETS:
    for method, seeds in METHOD_SEEDS.items():
        method_rows = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
        out = {
            "dataset": dataset,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "completed": len(method_rows),
            "expected": len(seeds),
        }
        for key in [
            "val_best_f1",
            "test_f1_at_val",
            "test_auc_at_val",
            "test_ap_at_val",
            "test_precision_at_val",
            "test_recall_at_val",
            "history_best_test_f1_at_val",
        ]:
            values = [float(r[key]) for r in method_rows if r.get(key) is not None]
            m, s = summarize_values(values)
            out[f"{key}_mean"] = m
            out[f"{key}_std"] = s
        aggregate.append(out)

delta_specs = [
    ("qwen_minus_rule", "sage_qwen_motif_channel", "sage_rule_evidence", CORE_SEEDS),
    ("rule_qwen_minus_rule", "sage_rule_scalar_qwen_motif_channel", "sage_rule_evidence", CORE_SEEDS),
    ("rule_qwen_minus_qwen", "sage_rule_scalar_qwen_motif_channel", "sage_qwen_motif_channel", CORE_SEEDS),
    ("qwen_minus_base", "sage_qwen_motif_channel", "sage_no_evidence", CORE_SEEDS),
]
deltas: list[dict] = []
row_index = {(r["dataset"], int(r["seed"]), r["method"]): r for r in rows}
for dataset in DATASETS:
    for name, a, b, seeds in delta_specs:
        values = []
        completed = 0
        for seed in seeds:
            ra = row_index.get((dataset, seed, a))
            rb = row_index.get((dataset, seed, b))
            if not ra or not rb:
                continue
            if ra.get("test_f1_at_val") is None or rb.get("test_f1_at_val") is None:
                continue
            completed += 1
            values.append(float(ra["test_f1_at_val"]) - float(rb["test_f1_at_val"]))
        m, s = summarize_values(values)
        deltas.append(
            {
                "dataset": dataset,
                "delta": name,
                "method_a": a,
                "method_b": b,
                "completed": completed,
                "expected": len(seeds),
                "test_f1_delta_mean": m,
                "test_f1_delta_std": s,
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


write_csv(SUMMARY_DIR / "per_seed_metrics.csv", rows)
write_csv(SUMMARY_DIR / "aggregate_metrics.csv", aggregate)
write_csv(SUMMARY_DIR / "f1_deltas.csv", deltas)
write_csv(SUMMARY_DIR / "missing_runs.csv", missing)

payload = {
    "out_root": str(OUT_ROOT.relative_to(ROOT)),
    "rows": rows,
    "aggregate": aggregate,
    "deltas": deltas,
    "missing": missing,
}
(SUMMARY_DIR / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

md: list[str] = []
md.append("# ResearchH 5-Seed Qwen Counter Summary")
md.append("")
md.append("Primary metric: test F1/AUC/AP at the threshold selected by validation-best F1.")
md.append("")
md.append("## Aggregate Metrics")
md.append("")
md.append("| Dataset | Method | Done | Test F1 | Test AUC | Test AP | Val F1 |")
md.append("|---|---|---:|---:|---:|---:|---:|")
for row in aggregate:
    md.append(
        "| {dataset} | {method} | {done}/{expected} | {f1}±{f1s} | {auc}±{aucs} | {ap}±{aps} | {val}±{vals} |".format(
            dataset=row["dataset"],
            method=row["method_label"],
            done=row["completed"],
            expected=row["expected"],
            f1=fmt(row.get("test_f1_at_val_mean")),
            f1s=fmt(row.get("test_f1_at_val_std")),
            auc=fmt(row.get("test_auc_at_val_mean")),
            aucs=fmt(row.get("test_auc_at_val_std")),
            ap=fmt(row.get("test_ap_at_val_mean")),
            aps=fmt(row.get("test_ap_at_val_std")),
            val=fmt(row.get("val_best_f1_mean")),
            vals=fmt(row.get("val_best_f1_std")),
        )
    )
md.append("")
md.append("## F1 Deltas")
md.append("")
md.append("| Dataset | Delta | Done | Mean | Std |")
md.append("|---|---|---:|---:|---:|")
for row in deltas:
    md.append(
        f"| {row['dataset']} | {row['delta']} | {row['completed']}/{row['expected']} | {fmt(row.get('test_f1_delta_mean'))} | {fmt(row.get('test_f1_delta_std'))} |"
    )
md.append("")
md.append("## Missing Runs")
md.append("")
if missing:
    for item in missing[:80]:
        md.append(f"- {item['dataset']} seed={item['seed']} method={item['method']}")
    if len(missing) > 80:
        md.append(f"- ... {len(missing) - 80} more")
else:
    md.append("All expected runs completed.")
md.append("")
md.append("## Files")
md.append("")
md.append("- `outputs/paper_5seed_qwen_counter_clean_20260615/summary/per_seed_metrics.csv`")
md.append("- `outputs/paper_5seed_qwen_counter_clean_20260615/summary/aggregate_metrics.csv`")
md.append("- `outputs/paper_5seed_qwen_counter_clean_20260615/summary/f1_deltas.csv`")
md.append("- `outputs/paper_5seed_qwen_counter_clean_20260615/summary/missing_runs.csv`")
(SUMMARY_DIR / "summary.md").write_text("\\n".join(md) + "\\n", encoding="utf-8")

print("summary_dir", SUMMARY_DIR)
print("completed_metrics", len(rows))
print("missing_runs", len(missing))
for row in aggregate:
    if row["completed"]:
        print(
            row["dataset"],
            row["method"],
            f"{row['completed']}/{row['expected']}",
            "f1",
            fmt(row.get("test_f1_at_val_mean")),
            "+/-",
            fmt(row.get("test_f1_at_val_std")),
        )

