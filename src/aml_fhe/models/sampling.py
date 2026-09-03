"""Training sample selection helpers."""

from __future__ import annotations

import numpy as np


def train_subsample(
    x: np.ndarray,
    y: np.ndarray,
    normal_rows: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all illicit rows and sample up to normal_rows normal rows."""
    illicit_idx = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    sampled = rng.choice(normal_idx, size=min(normal_rows, len(normal_idx)), replace=False)
    idx = np.concatenate([illicit_idx, sampled])
    rng.shuffle(idx)
    return x[idx], y[idx]


def eval_sample(
    x: np.ndarray,
    y: np.ndarray,
    n_rows: int,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Random evaluation subset preserving natural class distribution in expectation."""
    if n_rows <= 0 or n_rows >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n_rows, replace=False)
    return x[idx], y[idx]


def random_sample(
    x: np.ndarray,
    y: np.ndarray,
    n_rows: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Random sample capped by available rows."""
    if n_rows <= 0 or n_rows >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n_rows, replace=False)
    return x[idx], y[idx]
