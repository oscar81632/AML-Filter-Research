"""Metric calculation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return F1, precision, and recall for binary predictions."""
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def f1_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    """Return binary F1 after thresholding probabilities."""
    return float(f1_score(y_true, (probs >= threshold).astype(int), zero_division=0))
