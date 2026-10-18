"""
Evaluation for the multimodal detector.

Each modality is scored only on its own label range, with the metric its
field reports:

  chest  multi-label (NIH ChestX-ray14) → per-class AUROC, macro AUROC, mAP
  skin   single-label (HAM10000)        → balanced accuracy (ISIC 2018 metric), macro AUROC
  brain  single-label                   → accuracy, macro F1, macro AUROC

Confidence intervals are percentile bootstraps over test images. Temperature
and thresholds are always fitted on validation data, never on test.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from api.config import Config

MODALITIES = ("chest", "skin", "brain")
MULTI_LABEL = {"chest"}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def modality_mask(cfg: Config, modality_idx: np.ndarray) -> np.ndarray:
    """(N, C) mask that is 1 on each row's own modality label range."""
    mask = np.zeros((len(modality_idx), cfg.num_classes), dtype=np.float32)
    for m, name in enumerate(MODALITIES):
        start, end = cfg.modality_index_range(name)
        mask[modality_idx == m, start:end] = 1.0
    return mask


# ── Metric primitives ────────────────────────────────────────────────────────

def macro_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mean one-vs-rest AUROC over classes that have both positives and negatives."""
    aucs = [
        roc_auc_score(labels[:, c], scores[:, c])
        for c in range(labels.shape[1])
        if 0 < labels[:, c].sum() < len(labels)
    ]
    return float(np.mean(aucs)) if aucs else float("nan")


def _single_label(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return labels.argmax(1), scores.argmax(1)


def balanced_accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(balanced_accuracy_score(*_single_label(labels, scores)))


def accuracy(labels: np.ndarray, scores: np.ndarray) -> float:
    y, p = _single_label(labels, scores)
    return float((y == p).mean())


def macro_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    y, p = _single_label(labels, scores)
    return float(f1_score(y, p, average="macro", zero_division=0))


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    labels: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 500,
    seed: int = 0,
) -> Tuple[float, float]:
    """95% percentile bootstrap interval of `metric` over rows."""
    if n_boot <= 0 or len(labels) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        value = metric(labels[idx], scores[idx])
        if not math.isnan(value):
            stats.append(value)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Binary ECE over all (row, class) entries given as flat arrays."""
    if len(probs) == 0:
        return float("nan")
    bins = np.minimum((probs * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(probs[in_bin].mean() - labels[in_bin].mean())
    return float(ece)


# ── Calibration and thresholds (fit on validation only) ──────────────────────

def fit_temperature(logits: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    """Single temperature T minimising masked BCE of logits / T."""
    keep = mask.astype(bool)
    z = torch.tensor(logits[keep], dtype=torch.float64)
    y = torch.tensor(labels[keep], dtype=torch.float64)
    if len(z) == 0:
        return 1.0
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def fit_thresholds(
    cfg: Config,
    probs: np.ndarray,
    labels: np.ndarray,
    modality_idx: np.ndarray,
    default: float = 0.5,
) -> np.ndarray:
    """Per-class threshold maximising F1 on that class's own-modality rows."""
    thresholds = np.full(cfg.num_classes, default, dtype=np.float32)
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    for m, name in enumerate(MODALITIES):
        rows = modality_idx == m
        start, end = cfg.modality_index_range(name)
        for c in range(start, end):
            y = labels[rows, c]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            p = probs[rows, c]
            f1s = [f1_score(y, p >= t, zero_division=0) for t in grid]
            thresholds[c] = grid[int(np.argmax(f1s))]
    return thresholds


# ── Report ───────────────────────────────────────────────────────────────────

def evaluate_mode(
    cfg: Config,
    logits: np.ndarray,
    labels: np.ndarray,
    modality_idx: np.ndarray,
    temperature: float = 1.0,
    n_boot: int = 0,
) -> Dict:
    """
    Metrics for one mode's logits.

    Returns {"auc": mean of per-modality macro AUROC, "ece_raw", "ece",
             "<modality>": {...}} — "auc" is what the API reads to warn
    about weak modes.
    """
    probs_raw = sigmoid(logits)
    probs = sigmoid(logits / temperature)
    mask = modality_mask(cfg, modality_idx).astype(bool)
    result: Dict = {
        "ece_raw": expected_calibration_error(probs_raw[mask], labels[mask]),
        "ece": expected_calibration_error(probs[mask], labels[mask]),
    }

    aucs = []
    for m, name in enumerate(MODALITIES):
        rows = modality_idx == m
        if not rows.any():
            continue
        start, end = cfg.modality_index_range(name)
        y, p = labels[rows, start:end], probs[rows, start:end]
        entry: Dict[str, Optional[float]] = {"n": int(rows.sum()), "macro_auroc": macro_auroc(y, p)}
        entry["macro_auroc_ci"] = bootstrap_ci(macro_auroc, y, p, n_boot)
        if name in MULTI_LABEL:
            entry["per_class_auroc"] = {
                cfg.all_labels[start + c]: (
                    float(roc_auc_score(y[:, c], p[:, c])) if 0 < y[:, c].sum() < len(y) else None
                )
                for c in range(end - start)
            }
            valid = [c for c in range(end - start) if y[:, c].sum() > 0]
            entry["mean_ap"] = (
                float(average_precision_score(y[:, valid], p[:, valid], average="macro")) if valid else None
            )
        else:
            entry["balanced_accuracy"] = balanced_accuracy(y, p)
            entry["balanced_accuracy_ci"] = bootstrap_ci(balanced_accuracy, y, p, n_boot)
            entry["accuracy"] = accuracy(y, p)
            entry["macro_f1"] = macro_f1(y, p)
        result[name] = entry
        if not math.isnan(entry["macro_auroc"]):
            aucs.append(entry["macro_auroc"])

    result["auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return result
