"""Actual FHE execution helpers (tiny-sample sanity check)."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import numpy as np

from aml_fhe.models.sampling import random_sample


@dataclass(frozen=True)
class FheExecuteResult:
    """Result of an actual FHE execution/decryption check on a tiny sample."""

    sample_rows: int
    illicit_in_sample: int
    seconds: float
    match_percent: float
    note: str = "Actual FHE execution/decryption check; tiny sample only."


def execute_predictions(
    model, x_test: np.ndarray, y_test: np.ndarray, execute_rows: int
) -> Optional[FheExecuteResult]:
    """Run actual FHE execution and compare against clear predictions.

    Returns None if execute_rows == 0.
    """
    if execute_rows <= 0:
        return None

    x_sample, y_sample = random_sample(x_test, y_test, execute_rows, seed=1)
    start = time.time()
    y_exec = model.predict(x_sample, fhe="execute")
    elapsed = time.time() - start
    y_clear = model.predict(x_sample, fhe="disable")
    return FheExecuteResult(
        sample_rows=int(len(x_sample)),
        illicit_in_sample=int(y_sample.sum()),
        seconds=float(elapsed),
        match_percent=float((y_exec == y_clear).mean() * 100),
    )
