"""Concrete ML XGBoost model path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from concrete.ml.sklearn import XGBClassifier

from aml_fhe.config import FheConfig, ModelConfig


def concrete_xgb_params(model_config: ModelConfig, fhe_config: FheConfig) -> dict:
    """Build Concrete ML XGBoost params matching the reference benchmark."""
    return {
        "max_depth": model_config.max_depth,
        "scale_pos_weight": model_config.scale_pos_weight,
        "learning_rate": model_config.learning_rate,
        "subsample": model_config.subsample,
        "colsample_bytree": model_config.colsample_bytree,
        "n_estimators": model_config.n_estimators,
        "random_state": model_config.random_seed,
        "n_bits": fhe_config.n_bits,
        "n_jobs": 1,
    }


def train_concrete_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    model_config: ModelConfig,
    fhe_config: FheConfig,
) -> XGBClassifier:
    """Train the Concrete ML XGBoost model."""
    model = XGBClassifier(**concrete_xgb_params(model_config, fhe_config))
    model.fit(x_train, y_train)
    return model


def predict_proba_clear(model: XGBClassifier, x: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities with FHE disabled."""
    return model.predict_proba(x, fhe="disable")[:, 1]


def save_concrete_model_best_effort(model: XGBClassifier, path: str | Path) -> tuple[bool, str | None]:
    """Persist Concrete ML model dump when supported by the installed version."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as file:
            model.dump(file)
    except Exception as exc:  # pragma: no cover - artifact support varies by version
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()
        return False, repr(exc)
    return True, None
