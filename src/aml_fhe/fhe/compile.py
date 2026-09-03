"""Concrete ML model compilation."""

from __future__ import annotations

import numpy as np


def compile_model(model, x_train: np.ndarray, compile_rows: int) -> None:
    """Compile a Concrete ML model on a capped training sample."""
    model.compile(x_train[:compile_rows])
