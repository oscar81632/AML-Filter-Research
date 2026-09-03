"""FHE simulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from aml_fhe.models.sampling import random_sample


@dataclass(frozen=True)
class FheSimulationResult:
    """FHE simulation comparison against clear predictions."""

    sample_rows: int
    illicit_in_sample: int
    seconds: float
    match_percent: float
    f1_at_default_threshold: float


def simulate_predictions(model, x_test: np.ndarray, y_test: np.ndarray, sample_rows: int) -> FheSimulationResult:
    """Run FHE simulation and compare against clear mode predictions."""
    from sklearn.metrics import f1_score

    x_sample, y_sample = random_sample(x_test, y_test, sample_rows, seed=0)
    start = time.time()
    y_sim = model.predict(x_sample, fhe="simulate")
    elapsed = time.time() - start
    y_clear = model.predict(x_sample, fhe="disable")
    return FheSimulationResult(
        sample_rows=int(len(x_sample)),
        illicit_in_sample=int(y_sample.sum()),
        seconds=float(elapsed),
        match_percent=float((y_sim == y_clear).mean() * 100),
        f1_at_default_threshold=float(f1_score(y_sample, y_sim, zero_division=0)),
    )
