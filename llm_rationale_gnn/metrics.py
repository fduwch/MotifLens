from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    idx = mask.detach().cpu().numpy().astype(bool)
    if idx.sum() == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0, "ap": 0.0}

    y_true = y.detach().cpu().numpy()[idx]
    logits_np = logits.detach().cpu().numpy()[idx]
    y_pred = logits_np.argmax(axis=1)

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) <= 2 else "macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) <= 2 else "macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) <= 2 else "macro", zero_division=0)),
    }

    try:
        if logits_np.shape[1] == 2 and len(np.unique(y_true)) > 1:
            exp = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
            prob = exp[:, 1] / exp.sum(axis=1)
            out["auc"] = float(roc_auc_score(y_true, prob))
            out["ap"] = float(average_precision_score(y_true, prob))
        else:
            out["auc"] = 0.0
            out["ap"] = 0.0
    except Exception:
        out["auc"] = 0.0
        out["ap"] = 0.0
    return out


def positive_probabilities(logits: torch.Tensor) -> np.ndarray:
    logits_np = logits.detach().cpu().numpy()
    if logits_np.shape[1] == 1:
        return 1.0 / (1.0 + np.exp(-logits_np[:, 0]))
    exp = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
    return exp[:, 1] / exp.sum(axis=1)


def threshold_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, threshold: float) -> Dict[str, float]:
    idx = mask.detach().cpu().numpy().astype(bool)
    if idx.sum() == 0:
        return {"threshold": float(threshold), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    y_true = y.detach().cpu().numpy()[idx]
    prob = positive_probabilities(logits)[idx]
    y_pred = (prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def best_f1_threshold(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> tuple[float, Dict[str, float]]:
    idx = mask.detach().cpu().numpy().astype(bool)
    if idx.sum() == 0:
        return 0.5, threshold_metrics(logits, y, mask, 0.5)
    prob = positive_probabilities(logits)[idx]
    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(prob, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    best_threshold = 0.5
    best_metrics = threshold_metrics(logits, y, mask, best_threshold)
    for threshold in thresholds:
        metrics = threshold_metrics(logits, y, mask, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics
