#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "outputs/paper_tables_clean_20260615"
BASE = ROOT / "outputs/paper_main_baselines_clean_20260615"
OURS = ROOT / "outputs/paper_5seed_qwen_counter_clean_20260615"
SOTA = ROOT / "outputs/paper_sota_baselines_20260615"

DATASETS = ["Meta", "ZipZap", "Illicit-ETH", "EPSD-Ponzi"]
SEEDS = [42, 43, 44, 45, 46]
EXPECTED_MAIN = 340
EXPECTED_ABLATION = 80

TABULAR_METHODS = ["logreg", "mlp", "rf", "lightgbm"]
GENERIC_GNN_METHODS = ["sampled_gcn", "sampled_gat", "sampled_care_gnn", "sampled_pc_gnn"]
SOTA_METHODS = ["zipzap", "lmae4eth", "bert4eth", "tlmgnn", "bwgnn"]
OURS_METHODS = [
    "sage_no_evidence",
    "sage_rule_evidence",
    "sage_qwen_motif_channel",
    "sage_rule_scalar_qwen_motif_channel",
]

CODE_ISSUE_HINTS = {
    "raw_tabular_policy": {
        "suspect_files": [
            "scripts/train_feature_baselines.py",
            "logs/researchh_run_main_baselines_clean_20260615.sh",
        ],
        "checks": [
            "Confirm --raw-node-features-only is passed for Logistic Regression/MLP/LightGBM.",
            "Confirm DERIVED_NODE_FEATURE_COLUMNS are filtered from cached data.x.",
            "Move invalid tabular result directories to stale_or_invalid and rerun only affected seeds.",
        ],
    },
    "baseline_used_evidence": {
        "suspect_files": [
            "scripts/train_sampled.py",
            "scripts/train_fraud_gnn_baselines.py",
            "logs/researchh_run_main_baselines_clean_20260615.sh",
            "logs/researchh_run_sota_baselines_20260615.sh",
        ],
        "checks": [
            "Non-Ours baselines must not pass --evidence, --llm-evidence, or --use-evidence-features.",
            "metrics.json data.uses_evidence must be false or absent for all baselines.",
        ],
    },
    "sota_method_mismatch": {
        "suspect_files": [
            "scripts/train_sota_baselines.py",
            "logs/researchh_run_sota_baselines_20260615.sh",
            "logs/researchh_make_main_and_ablation_tables_clean_20260615.py",
        ],
        "checks": [
            "The output suffix must equal args.method for SOTA rows.",
            "The table script suffix must point to the same method directory.",
            "Move mismatched SOTA metrics to stale_or_invalid and rerun that dataset/seed/method.",
        ],
    },
    "dataset_path_mismatch": {
        "suspect_files": [
            "logs/researchh_run_sota_baselines_20260615.sh",
            "scripts/train_sota_baselines.py",
        ],
        "checks": [
            "dataset_path() must map Meta/ZipZap/Illicit-ETH/EPSD-Ponzi to the correct data directories.",
            "metrics.json args.data_dir should include the expected dataset directory.",
        ],
    },
    "cache_mismatch_or_unverifiable": {
        "suspect_files": [
            "scripts/train_sota_baselines.py",
            "llm_rationale_gnn/data.py",
        ],
        "checks": [
            "Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.",
            "Graph cache metadata should match dataset and seed.",
            "If cache metadata is missing or mismatched, rebuild cache before accepting the run.",
        ],
    },
    "implausibly_strong": {
        "suspect_files": [
            "llm_rationale_gnn/data.py",
            "scripts/train_feature_baselines.py",
            "scripts/train_sota_baselines.py",
            "logs/researchh_make_main_and_ablation_tables_clean_20260615.py",
        ],
        "checks": [
            "Check feature columns for label/split/source/root/target leakage.",
            "Check train/val/test masks are disjoint and stratified by seed.",
            "Check metrics are read from the intended output root and suffix.",
        ],
    },
    "threshold_or_metric_issue": {
        "suspect_files": [
            "scripts/train_sampled.py",
            "scripts/train_fraud_gnn_baselines.py",
            "scripts/train_sota_baselines.py",
        ],
        "checks": [
            "Verify best_val_threshold_f1 selects threshold only on validation data.",
            "Verify best_test_at_val_threshold uses the validation-selected threshold, not test-tuned threshold.",
        ],
    },
    "llm_delta_issue": {
        "suspect_files": [
            "llm_rationale_gnn/data.py",
            "scripts/train_sampled.py",
            "logs/researchh_summarize_paper_5seed_qwen_counter_clean_20260615.py",
        ],
        "checks": [
            "Check Qwen motif evidence coverage and typed-evidence deduplication.",
            "Check Rule+Qwen uses scalar_plus_motif evidence mode with correct evidence files.",
            "If LLM hurts most seeds, inspect evidence quality rather than hiding the result.",
        ],
    },
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        out = float(value)
        return None if math.isnan(out) or math.isinf(out) else out
    except Exception:
        return None


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def metric_path(root: Path, dataset: str, seed: int, suffix: str) -> Path:
    return root / dataset / f"seed_{seed}" / suffix / "metrics.json"


def add_issue(
    findings: list[dict[str, Any]],
    severity: str,
    kind: str,
    message: str,
    *,
    dataset: str | None = None,
    method: str | None = None,
    seed: int | None = None,
    evidence: dict[str, Any] | None = None,
    code_hint: str | None = None,
) -> None:
    item: dict[str, Any] = {
        "severity": severity,
        "type": kind,
        "message": message,
    }
    if dataset is not None:
        item["dataset"] = dataset
    if method is not None:
        item["method"] = method
    if seed is not None:
        item["seed"] = seed
    if evidence:
        item["evidence"] = evidence
    if code_hint:
        hint = CODE_ISSUE_HINTS.get(code_hint, {})
        item["likely_code_issue"] = {
            "category": code_hint,
            "suspect_files": hint.get("suspect_files", []),
            "checks": hint.get("checks", []),
        }
    findings.append(item)


def mean_by_method(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        v = fnum(row.get("test_f1_mean"))
        if v is not None:
            grouped[(row["dataset"], row["method"])].append(v)
    return {k: vals[0] for k, vals in grouped.items() if vals}


def main() -> None:
    TABLE.mkdir(parents=True, exist_ok=True)
    main_rows = read_csv(TABLE / "main_comparison_per_seed.csv")
    main_agg = read_csv(TABLE / "main_comparison_aggregate.csv")
    main_missing = read_csv(TABLE / "main_comparison_missing.csv")
    ab_rows = read_csv(TABLE / "ablation_per_seed.csv")
    ab_agg = read_csv(TABLE / "ablation_aggregate.csv")
    ab_missing = read_csv(TABLE / "ablation_missing.csv")

    findings: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "main_completed": len(main_rows),
        "main_expected": EXPECTED_MAIN,
        "main_missing": len(main_missing),
        "ablation_completed": len(ab_rows),
        "ablation_expected": EXPECTED_ABLATION,
        "ablation_missing": len(ab_missing),
    }

    if len(main_rows) < EXPECTED_MAIN:
        add_issue(
            findings,
            "info",
            "incomplete_main",
            f"Main table incomplete: {len(main_rows)}/{EXPECTED_MAIN}; missing {len(main_missing)} runs. Incomplete rows are not final anomalies.",
        )
    if len(ab_rows) != EXPECTED_ABLATION or ab_missing:
        add_issue(
            findings,
            "warning",
            "ablation_incomplete",
            f"Ablation table incomplete: {len(ab_rows)}/{EXPECTED_ABLATION}; missing {len(ab_missing)}.",
        )

    missing_by_method: dict[str, int] = defaultdict(int)
    for row in main_missing:
        missing_by_method[row.get("method", "?")] += 1
    summary["missing_by_method"] = dict(sorted(missing_by_method.items()))

    # Per-seed statistical anomalies.
    by_key: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in main_rows:
        v = fnum(row.get("test_f1"))
        if v is not None:
            by_key[(row["dataset"], row["method"])].append((int(row["seed"]), v))
    for (dataset, method), values in sorted(by_key.items()):
        nums = [v for _, v in values]
        if len(nums) < 3:
            continue
        mean_v = statistics.mean(nums)
        std_v = statistics.stdev(nums) if len(nums) > 1 else 0.0
        threshold = 0.04 if dataset in {"Illicit-ETH", "EPSD-Ponzi"} else 0.02
        if std_v > threshold:
            add_issue(
                findings,
                "warning",
                "high_std",
                f"{dataset}/{method} test_f1 std={std_v:.4f} over {len(nums)} seeds; inspect split sensitivity before using a single-seed claim.",
                dataset=dataset,
                method=method,
                evidence={"mean_f1": mean_v, "std_f1": std_v, "seeds": values},
                code_hint="threshold_or_metric_issue" if std_v > 0.08 else None,
            )
        for seed, value in values:
            if std_v > 0 and abs(value - mean_v) > max(2 * std_v, 0.03):
                add_issue(
                    findings,
                    "warning",
                    "seed_outlier",
                    f"{dataset}/{method} seed {seed} F1={value:.4f}, mean={mean_v:.4f}, std={std_v:.4f}.",
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    evidence={"mean_f1": mean_v, "std_f1": std_v, "seed_f1": value},
                    code_hint="threshold_or_metric_issue",
                )

    # Raw-tabular policy and feature leakage checks.
    raw_violations: list[str] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            for method in TABULAR_METHODS:
                path = metric_path(BASE, dataset, seed, f"tabular/{method}")
                if not path.exists():
                    continue
                metrics = load_json(path)
                if metrics is None:
                    raw_violations.append(f"{rel(path)} unreadable")
                    continue
                data = metrics.get("data", {})
                if data.get("raw_node_features_only") is not True:
                    raw_violations.append(f"{rel(path)} raw_node_features_only={data.get('raw_node_features_only')}")
                original = data.get("original_num_features")
                current = data.get("num_features")
                if original is not None and current is not None and float(current) > float(original):
                    raw_violations.append(f"{rel(path)} num_features={current} > original_num_features={original}")
                feature_cols = [str(x).lower() for x in data.get("feature_columns", [])]
                leaky = [c for c in feature_cols if any(k in c for k in ["label", "split", "target", "fraud", "phish", "ponzi", "illicit", "root"])]
                if leaky:
                    raw_violations.append(f"{rel(path)} possible leaky columns={leaky[:5]}")
    if raw_violations:
        add_issue(
            findings,
            "error",
            "raw_tabular_policy",
            "Raw-tabular baseline policy violation(s) detected.",
            evidence={"violations": raw_violations[:20]},
            code_hint="raw_tabular_policy",
        )
    else:
        add_issue(findings, "ok", "raw_tabular_policy", "All completed tabular baseline metrics satisfy raw-node-feature-only checks.")

    # Baselines must not use evidence, and SOTA metadata must match method/dataset.
    evidence_violations: list[str] = []
    for root, suffixes in [(BASE, GENERIC_GNN_METHODS), (SOTA, SOTA_METHODS)]:
        for dataset in DATASETS:
            for seed in SEEDS:
                for suffix in suffixes:
                    path = metric_path(root, dataset, seed, suffix)
                    if not path.exists():
                        continue
                    metrics = load_json(path)
                    if metrics is None:
                        add_issue(
                            findings,
                            "error",
                            "bad_metrics_json",
                            f"{rel(path)} is unreadable.",
                            dataset=dataset,
                            method=suffix,
                            seed=seed,
                        )
                        continue
                    data = metrics.get("data", {})
                    if data.get("uses_evidence") not in (False, None):
                        evidence_violations.append(rel(path))
                    if root == SOTA:
                        args = metrics.get("args", {})
                        if args.get("method") != suffix:
                            add_issue(
                                findings,
                                "error",
                                "sota_method_mismatch",
                                f"{rel(path)} has args.method={args.get('method')} but table suffix expects {suffix}.",
                                dataset=dataset,
                                method=suffix,
                                seed=seed,
                                code_hint="sota_method_mismatch",
                            )
                        data_dir = str(args.get("data_dir", ""))
                        expected_tokens = {
                            "Meta": ["Meta_hop1"],
                            "ZipZap": ["ZipZap_hop1"],
                            "Illicit-ETH": ["Illicit-ETH"],
                            "EPSD-Ponzi": ["EPSD-Ponzi"],
                        }[dataset]
                        if not any(tok in data_dir for tok in expected_tokens):
                            add_issue(
                                findings,
                                "error",
                                "dataset_path_mismatch",
                                f"{rel(path)} data_dir={data_dir} does not match dataset={dataset}.",
                                dataset=dataset,
                                method=suffix,
                                seed=seed,
                                code_hint="dataset_path_mismatch",
                            )
                        cache_path = data.get("sequence_cache")
                        if cache_path:
                            cache = Path(cache_path)
                            if not cache.exists():
                                add_issue(
                                    findings,
                                    "error",
                                    "sequence_cache_missing",
                                    f"{rel(path)} points to missing sequence cache {cache_path}.",
                                    dataset=dataset,
                                    method=suffix,
                                    seed=seed,
                                    code_hint="cache_mismatch_or_unverifiable",
                                )
                            else:
                                # The original cache schema may not include metadata; flag as code-level audit debt.
                                if data.get("sequence_cache_metadata_verified") is not True:
                                    add_issue(
                                        findings,
                                        "warning",
                                        "sequence_cache_unverified",
                                        f"{rel(path)} does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.",
                                        dataset=dataset,
                                        method=suffix,
                                        seed=seed,
                                        code_hint="cache_mismatch_or_unverifiable",
                                    )
    if evidence_violations:
        add_issue(
            findings,
            "error",
            "baseline_used_evidence",
            "Non-Ours baseline unexpectedly reports evidence usage.",
            evidence={"paths": evidence_violations[:20]},
            code_hint="baseline_used_evidence",
        )
    else:
        add_issue(findings, "ok", "baseline_evidence_policy", "Completed non-Ours baseline metrics do not report evidence usage.")

    # Suspicious identical metrics across unrelated methods.
    identical: dict[tuple[str, str, tuple[str, ...]], list[str]] = defaultdict(list)
    for row in main_rows:
        sig = tuple(row.get(k, "") for k in ["test_f1", "test_auc", "test_ap", "precision", "recall"])
        if all(sig):
            identical[(row["dataset"], row["seed"], sig)].append(row["method"])
    for (dataset, seed, sig), methods in identical.items():
        if len(methods) >= 3:
            add_issue(
                findings,
                "warning",
                "identical_metrics",
                f"{dataset} seed {seed} has identical full metric signatures for {methods}.",
                dataset=dataset,
                seed=int(seed),
                evidence={"metrics_signature": sig, "methods": methods},
                code_hint="implausibly_strong",
            )

    # Implausible strength and threshold consistency.
    for row in main_rows:
        dataset = row["dataset"]
        method = row["method"]
        seed = int(row["seed"])
        f1 = fnum(row.get("test_f1"))
        auc = fnum(row.get("test_auc"))
        recall = fnum(row.get("recall"))
        precision = fnum(row.get("precision"))
        val_f1 = fnum(row.get("val_f1"))
        if f1 is not None and auc is not None and auc > 0.98 and f1 < 0.60:
            add_issue(
                findings,
                "warning",
                "auc_f1_disconnect",
                f"{dataset}/{method}/seed{seed}: AUC={auc:.4f} but F1={f1:.4f}; thresholding or class imbalance may be mishandled.",
                dataset=dataset,
                method=method,
                seed=seed,
                evidence={"test_f1": f1, "test_auc": auc},
                code_hint="threshold_or_metric_issue",
            )
        if precision is not None and recall is not None and max(precision, recall) > 0.95 and min(precision, recall) < 0.35:
            add_issue(
                findings,
                "warning",
                "precision_recall_imbalance",
                f"{dataset}/{method}/seed{seed}: precision={precision:.4f}, recall={recall:.4f}; inspect threshold and split label ratio.",
                dataset=dataset,
                method=method,
                seed=seed,
                evidence={"precision": precision, "recall": recall},
                code_hint="threshold_or_metric_issue",
            )
        if f1 is not None and val_f1 is not None and f1 - val_f1 > 0.08:
            add_issue(
                findings,
                "warning",
                "test_much_higher_than_val",
                f"{dataset}/{method}/seed{seed}: test F1 exceeds validation F1 by {f1 - val_f1:.4f}; inspect split or table path.",
                dataset=dataset,
                method=method,
                seed=seed,
                evidence={"test_f1": f1, "val_f1": val_f1},
                code_hint="implausibly_strong",
            )

    agg_mean = mean_by_method(main_agg)
    for dataset in DATASETS:
        ours = agg_mean.get((dataset, "ours"))
        graphsage = agg_mean.get((dataset, "graphsage"))
        lightgbm = agg_mean.get((dataset, "lightgbm"))
        if ours is not None and graphsage is not None:
            add_issue(
                findings,
                "info",
                "delta_ours_graphsage",
                f"{dataset}: Ours-GraphSAGE delta F1 = {ours - graphsage:+.4f}.",
                dataset=dataset,
                evidence={"ours_f1": ours, "graphsage_f1": graphsage, "delta": ours - graphsage},
            )
        if ours is not None and lightgbm is not None and ours < lightgbm:
            add_issue(
                findings,
                "warning",
                "ours_below_lightgbm",
                f"{dataset}: Ours F1={ours:.4f} is below raw LightGBM F1={lightgbm:.4f}; inspect whether graph/evidence helps this dataset.",
                dataset=dataset,
                evidence={"ours_f1": ours, "lightgbm_f1": lightgbm, "delta": ours - lightgbm},
                code_hint="llm_delta_issue",
            )

    # Core LLM evidence deltas from ablation table.
    ab_mean = {(r["dataset"], r["method"]): fnum(r.get("test_f1_mean")) for r in ab_agg}
    for dataset in DATASETS:
        base = ab_mean.get((dataset, "graphsage"))
        rule = ab_mean.get((dataset, "rule"))
        qwen = ab_mean.get((dataset, "qwen"))
        rq = ab_mean.get((dataset, "rule_qwen"))
        if base is not None and qwen is not None and qwen - base <= 0:
            add_issue(
                findings,
                "warning",
                "qwen_not_improving_base",
                f"{dataset}: Qwen motif minus Base GraphSAGE delta={qwen - base:+.4f}.",
                dataset=dataset,
                evidence={"base": base, "qwen": qwen, "delta": qwen - base},
                code_hint="llm_delta_issue",
            )
        if rule is not None and rq is not None and rq - rule <= 0:
            add_issue(
                findings,
                "warning",
                "rule_qwen_not_improving_rule",
                f"{dataset}: Rule+Qwen minus Rule delta={rq - rule:+.4f}.",
                dataset=dataset,
                evidence={"rule": rule, "rule_qwen": rq, "delta": rq - rule},
                code_hint="llm_delta_issue",
            )

    # Summarize severe code-suspect issues separately.
    code_suspects = [f for f in findings if "likely_code_issue" in f and f["severity"] in {"error", "warning"}]
    summary["code_suspect_count"] = len(code_suspects)
    summary["error_count"] = sum(1 for f in findings if f["severity"] == "error")
    summary["warning_count"] = sum(1 for f in findings if f["severity"] == "warning")

    report_lines = [
        "# ResearchH Experiment Anomaly Report",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        f"Main: {summary['main_completed']}/{summary['main_expected']} complete; missing {summary['main_missing']}.",
        f"Ablation: {summary['ablation_completed']}/{summary['ablation_expected']} complete; missing {summary['ablation_missing']}.",
        f"Warnings: {summary['warning_count']}; Errors: {summary['error_count']}; Code-suspect findings: {summary['code_suspect_count']}.",
        "",
        "## Missing By Method",
    ]
    if summary["missing_by_method"]:
        for method, count in summary["missing_by_method"].items():
            report_lines.append(f"- {method}: {count}")
    else:
        report_lines.append("- none")
    report_lines += ["", "## Findings"]
    for item in findings:
        loc = " ".join(
            str(item[k])
            for k in ["dataset", "method", "seed"]
            if k in item
        )
        prefix = f"- [{item['severity']}] {item['type']}"
        if loc:
            prefix += f" ({loc})"
        report_lines.append(f"{prefix}: {item['message']}")
        if "likely_code_issue" in item:
            hint = item["likely_code_issue"]
            report_lines.append(f"  Likely code issue: {hint['category']}")
            if hint["suspect_files"]:
                report_lines.append("  Suspect files: " + ", ".join(hint["suspect_files"]))
            for check in hint["checks"]:
                report_lines.append(f"  Check: {check}")

    payload = {"summary": summary, "findings": findings, "repairs": repairs}
    (TABLE / "anomaly_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (TABLE / "anomaly_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "summary": summary,
        "top_findings": findings[:12],
        "top_code_suspects": code_suspects[:8],
        "report_md": rel(TABLE / "anomaly_report.md"),
        "report_json": rel(TABLE / "anomaly_report.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

