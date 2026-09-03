"""Plain XGBoost model path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from aml_fhe.config import ModelConfig


def plain_xgb_params(config: ModelConfig) -> dict:
    """Build the plain XGBoost params used by the reference benchmark."""
    return {
        "max_depth": config.max_depth,
        "scale_pos_weight": config.scale_pos_weight,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "n_estimators": config.n_estimators,
        "random_state": config.random_seed,
        "eval_metric": "aucpr",
        "n_jobs": -1,
    }


def train_plain_xgb(x_train: np.ndarray, y_train: np.ndarray, config: ModelConfig) -> XGBClassifier:
    """Train the plain XGBoost baseline."""
    model = XGBClassifier(**plain_xgb_params(config))
    model.fit(x_train, y_train)
    return model


def predict_proba(model: XGBClassifier, x: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities."""
    return model.predict_proba(x)[:, 1]


def save_plain_model(model: XGBClassifier, path: str | Path) -> None:
    """Persist a plain XGBoost model as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
