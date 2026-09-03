"""Post-parquet preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_encode(
    parquet_path: str | Path,
    label_col: str,
    string_cols: tuple[str, ...],
    encoders: dict[str, LabelEncoder] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, LabelEncoder]]:
    """Load a feature parquet and label-encode configured string columns."""
    df = pd.read_parquet(parquet_path)
    y = df[label_col].astype(int).values
    df = df.drop(columns=[label_col])

    fit_encoders = encoders is None
    if fit_encoders:
        encoders = {}

    for column in string_cols:
        if fit_encoders:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column].astype(str))
            encoders[column] = encoder
        else:
            known = encoders[column].classes_
            df[column] = df[column].astype(str).map(
                lambda value, classes=known: np.searchsorted(classes, value) if value in classes else 0
            )

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = df.to_numpy(dtype=np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return x, y, encoders
