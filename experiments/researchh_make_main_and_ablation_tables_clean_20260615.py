#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
OURS_ROOT = ROOT / "outputs/paper_5seed_qwen_counter_clean_20260615"
IMBALANCE_OURS_ROOT = ROOT / "outputs/paper_5seed_qwen_counter_clean_pctrain_20260616"
BASE_ROOT = ROOT / "outputs/paper_main_baselines_clean_20260615"
SOTA_ROOT = ROOT / "outputs/paper_sota_baselines_20260615"
OUT = ROOT / "outputs/paper_tables_clean_20260615"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["Meta", "ZipZap", "Illicit-ETH", "EPSD-Ponzi"]
SEEDS = [42, 43, 44, 45, 46]

MAIN_METHODS = [
    ("logreg", "Logistic Regression", "Raw tabular", BASE_ROOT, "tabular/logreg"),
    ("mlp", "MLP", "Raw tabular", BASE_ROOT, "tabular/mlp"),
    ("rf", "Random Forest", "Raw tabular", BASE_ROOT, "tabular/rf"),
    ("lightgbm", "LightGBM", "Raw tabular", BASE_ROOT, "tabular/lightgbm"),
    ("gcn", "GCN", "GNN", BASE_ROOT, "sampled_gcn"),
    ("gat", "GAT", "GNN", BASE_ROOT, "sampled_gat"),
    ("graphsage", "GraphSAGE", "GNN", OURS_ROOT, "sage_no_evidence"),
    ("care_gnn", "CARE-GNN", "Generic Fraud GNN", BASE_ROOT, "sampled_care_gnn"),
    ("graphconsis_gnn", "GraphConsis", "Generic Fraud GNN", BASE_ROOT, "sampled_graphconsis_gnn"),
    ("pc_gnn", "PC-GNN", "Generic Fraud GNN", BASE_ROOT, "sampled_pc_gnn"),
    ("zipzap", "ZipZap", "Ethereum SOTA", SOTA_ROOT, "zipzap"),
    ("lmae4eth", "LMAE4Eth", "Ethereum SOTA", SOTA_ROOT, "lmae4eth"),
    ("bert4eth", "BERT4ETH", "Ethereum SOTA", SOTA_ROOT, "bert4eth"),
    ("tlmgnn", "TLmGNN", "Ethereum SOTA", SOTA_ROOT, "tlmgnn"),
    ("bwgnn", "BWGNN", "Graph Anomaly SOTA", SOTA_ROOT, "bwgnn"),
    ("ours", "Ours", "Proposed", OURS_ROOT, "sage_rule_scalar_qwen_motif_channel"),
    ("ours_imbalance", "Ours+Imbalance", "Proposed", IMBALANCE_OURS_ROOT, "sage_rule_scalar_qwen_motif_channel"),
]

ABLATION_METHODS = [
    ("graphsage", "Base GraphSAGE", "base", OURS_ROOT, "sage_no_evidence", SEEDS),
    ("rule", "GNN + rule evidence", "evidence", OURS_ROOT, "sage_rule_evidence", SEEDS),
    ("qwen", "GNN + Qwen motif", "evidence", OURS_ROOT, "sage_qwen_motif_channel", SEEDS),
    ("rule_qwen", "GNN + rule + Qwen motif", "evidence", OURS_ROOT, "sage_rule_scalar_qwen_motif_channel", SEEDS),
]

SOTA_STATUS = [
    {
        "method": "BERT4ETH",
        "role": "classic Ethereum transaction-language baseline",
        "repo": "https://github.com/Bayi-Hu/BERT4ETH_PyTorch / https://github.com/git-disl/BERT4ETH",
        "status": "in_progress_this_round",
        "reason": "official PyTorch repository cloned; controlled four-dataset 5-seed run is queued in the shared SOTA environment",
    },
    {
        "method": "ZipZap",
        "role": "efficient Ethereum transaction-language baseline",
        "repo": "https://github.com/git-disl/ZipZap",
        "status": "in_progress_this_round",
        "reason": "official repository cloned; isolated Python environment created; controlled four-dataset 5-seed run is being prepared",
    },
    {
        "method": "TLmGNN",
        "role": "SOTA transaction language model + GNN baseline",
        "repo": "https://github.com/lincozz/TLmGNN",
        "status": "in_progress_this_round",
        "reason": "official repository cloned; controlled four-dataset 5-seed run is queued in the shared SOTA environment",
    },
    {
        "method": "BWGNN",
        "role": "general graph anomaly detection SOTA baseline",
        "repo": "https://github.com/squareRoot3/Rethinking-Anomaly-Detection",
        "status": "in_progress_this_round",
        "reason": "official repository cloned; controlled four-dataset 5-seed run is queued in the shared SOTA environment",
    },
    {
        "method": "LMAE4Eth",
        "role": "SOTA multi-view Ethereum fraud baseline",
        "repo": "https://github.com/lmae4eth/LMAE4Eth",
        "status": "in_progress_this_round",
        "reason": "official repository cloned; isolated Python environment is being created; controlled four-dataset 5-seed run is being prepared",
    },
]


def metric_path(root: Path, dataset: str, seed: int, suffix: str) -> Path:
    return root / dataset / f"seed_{seed}" / suffix / "metrics.json"


def get_metric(metrics: dict, path: str) -> float | None:
    cur = metrics
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return None if cur is None else float(cur)


def fmt(v: float | None, nd: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return f"{v:.{nd}f}"


def summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def read_rows(method_specs: list[tuple], main: bool) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    missing: list[dict] = []
    for dataset in DATASETS:
        for spec in method_specs:
            if main:
                method, label, group, root, suffix = spec
                seeds = SEEDS
            else:
                method, label, group, root, suffix, seeds = spec
            for seed in seeds:
                path = metric_path(root, dataset, seed, suffix)
                if not path.exists():
                    missing.append({"dataset": dataset, "seed": seed, "method": method, "path": str(path.relative_to(ROOT))})
                    continue
                metrics = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "method": method,
                        "method_label": label,
                        "group": group,
                        "test_f1": get_metric(metrics, "best_test_at_val_threshold.f1"),
                        "test_auc": get_metric(metrics, "best_test_at_val_threshold.auc"),
                        "test_ap": get_metric(metrics, "best_test_at_val_threshold.ap"),
                        "precision": get_metric(metrics, "best_test_at_val_threshold.precision"),
                        "recall": get_metric(metrics, "best_test_at_val_threshold.recall"),
                        "val_f1": get_metric(metrics, "best_val_threshold_f1"),
                        "metrics_path": str(path.relative_to(ROOT)),
                    }
                )
    return rows, missing


def aggregate(rows: list[dict], method_specs: list[tuple], main: bool) -> list[dict]:
    out: list[dict] = []
    for dataset in DATASETS:
        for spec in method_specs:
            if main:
                method, label, group, *_ = spec
                expected = len(SEEDS)
            else:
                method, label, group, *_rest = spec
                expected = len(spec[-1])
            subset = [r for r in rows if r["dataset"] == dataset and r["method"] == method]
            row = {"dataset": dataset, "method": method, "method_label": label, "group": group, "completed": len(subset), "expected": expected}
            for key in ["test_f1", "test_auc", "test_ap", "precision", "recall", "val_f1"]:
                vals = [float(r[key]) for r in subset if r.get(key) is not None]
                m, s = summarize(vals)
                row[f"{key}_mean"] = m
                row[f"{key}_std"] = s
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_main_md(path: Path, rows: list[dict]) -> None:
    md = ["# Main Comparison Table", "", "Ablation methods are intentionally excluded from this table.", ""]
    md.append("Only F1 is shown. Best and second-best cells are ranked by displayed F1 values for each dataset; incomplete rows remain marked in Done.")
    md.append("")
    md.append("| Method | Group | Meta F1 | ZipZap F1 | Illicit-ETH F1 | EPSD-Ponzi F1 | Done |")
    md.append("|---|---|---:|---:|---:|---:|---:|")

    ranks: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        ranked_rows = [
            r for r in rows
            if r["dataset"] == dataset
            and r.get("test_f1_mean") is not None
            and r.get("completed", 0) > 0
        ]
        ranked_rows.sort(key=lambda r: float(r["test_f1_mean"]), reverse=True)
        ranks[dataset] = {r["method"]: idx for idx, r in enumerate(ranked_rows[:2])}

    for method, label, group, *_ in MAIN_METHODS:
        cells = []
        done = []
        for dataset in DATASETS:
            row = next((r for r in rows if r["dataset"] == dataset and r["method"] == method), None)
            if not row or not row["completed"]:
                cells.append("")
                done.append("0/5")
            else:
                cell = f"{fmt(row['test_f1_mean'])}+/-{fmt(row['test_f1_std'])}"
                rank = ranks.get(dataset, {}).get(method)
                if rank == 0:
                    cell = f"**{cell}**"
                elif rank == 1:
                    cell = f"\\underline{{{cell}}}"
                cells.append(cell)
                done.append(f"{row['completed']}/{row['expected']}")
        md.append(f"| {label} | {group} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {'; '.join(done)} |")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")

def write_ablation_md(path: Path, rows: list[dict]) -> None:
    md = ["# Ablation Table", "", "This table is separate from the main comparison table.", ""]
    md.append("| Method | Type | Meta F1/AUC | ZipZap F1/AUC | Illicit-ETH F1/AUC | EPSD-Ponzi F1/AUC | Done |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for method, label, group, *_ in ABLATION_METHODS:
        cells = []
        done = []
        for dataset in DATASETS:
            row = next((r for r in rows if r["dataset"] == dataset and r["method"] == method), None)
            if not row or not row["completed"]:
                cells.append("")
                done.append("0")
            else:
                cells.append(f"{fmt(row['test_f1_mean'])}+/-{fmt(row['test_f1_std'])} / {fmt(row['test_auc_mean'])}+/-{fmt(row['test_auc_std'])}")
                done.append(f"{row['completed']}/{row['expected']}")
        md.append(f"| {label} | {group} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {'; '.join(done)} |")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    main_rows, main_missing = read_rows(MAIN_METHODS, main=True)
    ablation_rows, ablation_missing = read_rows(ABLATION_METHODS, main=False)
    main_agg = aggregate(main_rows, MAIN_METHODS, main=True)
    ablation_agg = aggregate(ablation_rows, ABLATION_METHODS, main=False)

    write_csv(OUT / "main_comparison_per_seed.csv", main_rows)
    write_csv(OUT / "main_comparison_aggregate.csv", main_agg)
    write_csv(OUT / "main_comparison_missing.csv", main_missing)
    write_csv(OUT / "ablation_per_seed.csv", ablation_rows)
    write_csv(OUT / "ablation_aggregate.csv", ablation_agg)
    write_csv(OUT / "ablation_missing.csv", ablation_missing)
    write_csv(OUT / "sota_status.csv", SOTA_STATUS)
    write_main_md(OUT / "main_comparison_table.md", main_agg)
    write_ablation_md(OUT / "ablation_table.md", ablation_agg)

    payload = {
        "main_aggregate": main_agg,
        "main_missing": main_missing,
        "ablation_aggregate": ablation_agg,
        "ablation_missing": ablation_missing,
        "sota_status": SOTA_STATUS,
    }
    (OUT / "tables.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("table_dir", OUT)
    print("main_completed", len(main_rows), "main_missing", len(main_missing))
    print("ablation_completed", len(ablation_rows), "ablation_missing", len(ablation_missing))
    for row in main_agg:
        if row["completed"]:
            print(row["dataset"], row["method"], f"{row['completed']}/{row['expected']}", "f1", fmt(row["test_f1_mean"]), "+/-", fmt(row["test_f1_std"]))


if __name__ == "__main__":
    main()

