"""Threshold selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdResult:
    """Best threshold and resulting metrics."""

    f1: float
    threshold: float
    precision: float
    recall: float


def threshold_sweep(probs: np.ndarray, y_true: np.ndarray) -> ThresholdResult:
    """Find the best F1 threshold on a 0.05 grid."""
    best = ThresholdResult(f1=0.0, threshold=0.5, precision=0.0, recall=0.0)
    for threshold in np.arange(0.05, 0.96, 0.05):
        pred = (probs >= threshold).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        if f1 > best.f1:
            best = ThresholdResult(
                f1=float(f1),
                threshold=float(threshold),
                precision=float(precision),
                recall=float(recall),
            )
    return best
