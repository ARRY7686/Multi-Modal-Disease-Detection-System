"""
Evaluation metrics for multi-label disease classification.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    classification_report,
)


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
    threshold: float = 0.5,
) -> float:
    """
    Compute F1 score.

    Args:
        y_true: (N, C) binary ground-truth labels
        y_pred: (N, C) predicted probabilities (will be thresholded)
        average: "macro", "micro", "weighted", or "samples"
        threshold: probability threshold for binarisation
    """
    y_bin = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_bin, average=average, zero_division=0)


def compute_roc_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Compute ROC-AUC score.  Handles classes that have no positive samples.

    Args:
        y_true: (N, C) binary ground-truth
        y_pred: (N, C) predicted probabilities
    """
    # Only evaluate classes that have both positive and negative samples
    valid_cols = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0 and y_true[:, c].sum() < len(y_true):
            valid_cols.append(c)

    if not valid_cols:
        return 0.0

    return roc_auc_score(
        y_true[:, valid_cols],
        y_pred[:, valid_cols],
        average=average,
    )


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Per-class F1, precision, recall, and ROC-AUC.

    Returns:
        {label_name: {"f1": ..., "precision": ..., "recall": ..., "roc_auc": ...}}
    """
    y_bin = (y_pred >= threshold).astype(int)
    results = {}

    for i, name in enumerate(label_names):
        gt = y_true[:, i]
        pred_prob = y_pred[:, i]
        pred_bin = y_bin[:, i]

        f1 = f1_score(gt, pred_bin, zero_division=0)
        prec = precision_score(gt, pred_bin, zero_division=0)
        rec = recall_score(gt, pred_bin, zero_division=0)

        # ROC-AUC only if both classes present
        if gt.sum() > 0 and gt.sum() < len(gt):
            auc = roc_auc_score(gt, pred_prob)
        else:
            auc = float("nan")

        results[name] = {
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "roc_auc": round(auc, 4) if not np.isnan(auc) else None,
        }

    return results


def compute_average_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Mean Average Precision (mAP)."""
    valid_cols = [
        c for c in range(y_true.shape[1])
        if y_true[:, c].sum() > 0
    ]
    if not valid_cols:
        return 0.0
    return average_precision_score(
        y_true[:, valid_cols],
        y_pred[:, valid_cols],
        average=average,
    )


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    search_range: np.ndarray = np.arange(0.1, 0.9, 0.05),
) -> np.ndarray:
    """
    Find per-class optimal thresholds maximising F1.

    Returns:
        thresholds: (C,) array of optimal thresholds
    """
    num_classes = y_true.shape[1]
    best_thresholds = np.full(num_classes, 0.5)

    for c in range(num_classes):
        if y_true[:, c].sum() == 0:
            continue
        best_f1 = 0.0
        for t in search_range:
            pred_bin = (y_pred[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], pred_bin, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = t

    return best_thresholds
