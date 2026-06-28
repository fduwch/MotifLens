#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_rationale_gnn.data import DERIVED_NODE_FEATURE_COLUMNS, load_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tabular feature baselines on labeled graph nodes.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", default="logreg,rf,mlp,lightgbm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-split-column", action="store_true")
    parser.add_argument("--graph-cache", default=None)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--raw-node-features-only",
        action="store_true",
        help="Use only original numeric node columns for tabular baselines, excluding loader-derived graph statistics.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def softmax_positive(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp[:, 1] / exp.sum(axis=1)


def positive_prob(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(x)
        if prob.ndim == 2 and prob.shape[1] > 1:
            return prob[:, 1].astype(np.float64)
        return prob.reshape(-1).astype(np.float64)
    if hasattr(model, "decision_function"):
        score = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-score))
    pred = model.predict(x)
    return pred.astype(np.float64)


def metrics_np(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
        out["ap"] = float(average_precision_score(y_true, prob)) if len(np.unique(y_true)) > 1 else 0.0
    except Exception:
        out["auc"] = 0.0
        out["ap"] = 0.0
    return out


def best_f1_threshold_np(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, dict[str, float]]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(prob, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    best_threshold = 0.5
    best_metrics = metrics_np(y_true, prob, 0.5)
    for threshold in thresholds:
        item = metrics_np(y_true, prob, float(threshold))
        if item["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = item
    best_metrics["threshold"] = float(best_threshold)
    return best_threshold, best_metrics


def select_tabular_features(loaded, raw_only: bool) -> tuple[np.ndarray, list[str], dict[str, object]]:
    data = loaded.data
    x = data.x.detach().cpu().numpy().astype(np.float32, copy=False)
    feature_columns = list(getattr(loaded, "feature_columns", []) or [])
    if not raw_only:
        return x, feature_columns, {
            "raw_node_features_only": False,
            "original_num_features": int(x.shape[1]),
            "excluded_derived_node_features": [],
        }

    derived = set(DERIVED_NODE_FEATURE_COLUMNS)
    keep_idx = [idx for idx, name in enumerate(feature_columns) if name not in derived]
    selected_columns = [feature_columns[idx] for idx in keep_idx]
    excluded_columns = [name for name in feature_columns if name in derived]
    if not keep_idx:
        raise ValueError(
            "--raw-node-features-only selected zero features. "
            "Check nodes.csv for original numeric node features."
        )
    return x[:, keep_idx].astype(np.float32, copy=False), selected_columns, {
        "raw_node_features_only": True,
        "original_num_features": int(x.shape[1]),
        "excluded_derived_node_features": excluded_columns,
    }


def load_cached_or_graph(data_dir: str, seed: int, ignore_split_column: bool, graph_cache: str | None):
    cache_path = Path(graph_cache) if graph_cache else None
    if cache_path and cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        return SimpleNamespace(
            data=payload["data"],
            feature_columns=payload.get("feature_columns", []),
            edge_feature_columns=payload.get("edge_feature_columns", []),
            node_ids=payload.get("node_ids"),
            cache_used=str(cache_path),
        )
    loaded = load_graph(data_dir, seed=seed, build_edge_attr=False, use_split_column=not ignore_split_column)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "data": loaded.data,
                "feature_columns": loaded.feature_columns,
                "edge_feature_columns": loaded.edge_feature_columns,
                "seed": seed,
                "data_dir": data_dir,
                "use_split_column": not ignore_split_column,
            },
            cache_path,
        )
    return loaded


def class_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (2.0 * counts)
    return weights[y.astype(int)]


def fit_model(name: str, x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace):
    if name == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", n_jobs=args.n_jobs, random_state=args.seed)
        model.fit(x_train, y_train)
        return model
    if name == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=args.n_jobs,
        )
        model.fit(x_train, y_train)
        return model
    if name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            early_stopping=True,
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        kwargs = {}
        if "sample_weight" in inspect.signature(model.fit).parameters:
            kwargs["sample_weight"] = class_sample_weight(y_train)
        model.fit(x_train, y_train, **kwargs)
        return model
    if name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is not installed in this environment.")
        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        return model
    raise ValueError(f"Unknown method: {name}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    started = time.time()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    loaded = load_cached_or_graph(args.data_dir, args.seed, args.ignore_split_column, args.graph_cache)
    data = loaded.data
    x, selected_feature_columns, feature_selection_info = select_tabular_features(loaded, args.raw_node_features_only)
    y = data.y.detach().cpu().numpy().astype(np.int64, copy=False)
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    val_idx = data.val_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
    test_idx = data.test_mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

    print(
        json.dumps(
            {
                "phase": "load_done",
                "data_dir": args.data_dir,
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.num_edges),
                "num_features": int(x.shape[1]),
                "original_num_features": int(data.x.size(-1)),
                "raw_node_features_only": bool(args.raw_node_features_only),
                "train_nodes": int(len(train_idx)),
                "val_nodes": int(len(val_idx)),
                "test_nodes": int(len(test_idx)),
                "elapsed_seconds": round(time.time() - started, 2),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    for method in methods:
        out_dir = out_root / method
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists() and metrics_path.stat().st_size > 0:
            print(json.dumps({"phase": "skip_existing", "method": method, "metrics": str(metrics_path)}, ensure_ascii=False), flush=True)
            continue
        try:
            model_started = time.time()
            model = fit_model(method, x_train, y_train, args)
            train_prob = positive_prob(model, x_train)
            val_prob = positive_prob(model, x_val)
            test_prob = positive_prob(model, x_test)
            threshold, val_best = best_f1_threshold_np(y_val, val_prob)
            metrics = {
                "method": method,
                "best_epoch": None,
                "selection_metric": "val_best_f1",
                "best_selection_score": val_best["f1"],
                "best_val_f1": metrics_np(y_val, val_prob, 0.5)["f1"],
                "best_val_auc": metrics_np(y_val, val_prob, 0.5)["auc"],
                "best_val_ap": metrics_np(y_val, val_prob, 0.5)["ap"],
                "best_val_threshold_f1": val_best["f1"],
                "best_threshold": threshold,
                "best_test": metrics_np(y_test, test_prob, 0.5),
                "best_test_at_val_threshold": {**metrics_np(y_test, test_prob, threshold), "threshold": threshold},
                "train": metrics_np(y_train, train_prob, 0.5),
                "val": metrics_np(y_val, val_prob, 0.5),
                "data": {
                    "num_nodes": int(data.num_nodes),
                    "num_edges": int(data.num_edges),
                    "num_features": int(x.shape[1]),
                    "original_num_features": int(data.x.size(-1)),
                    "train_nodes": int(len(train_idx)),
                    "val_nodes": int(len(val_idx)),
                    "test_nodes": int(len(test_idx)),
                    "model_name": method,
                    "feature_columns": selected_feature_columns,
                    **feature_selection_info,
                },
                "args": vars(args),
                "elapsed_seconds": round(time.time() - model_started, 2),
            }
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "phase": "train_done",
                        "method": method,
                        "best_val_threshold_f1": metrics["best_val_threshold_f1"],
                        "best_test_at_val_threshold": metrics["best_test_at_val_threshold"],
                        "elapsed_seconds": metrics["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            error = {"method": method, "error": repr(exc), "args": vars(args)}
            (out_dir / "error.json").write_text(json.dumps(error, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"phase": "train_error", **error}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
