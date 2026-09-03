"""Artifact path and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write stable, human-readable JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def model_artifact_paths(artifact_dir: str | Path) -> dict[str, Path]:
    """Return standard model artifact paths."""
    model_dir = Path(artifact_dir) / "models"
    return {
        "model_dir": model_dir,
        "plain_xgb": model_dir / "plain_xgb.json",
        "concrete_ml_xgb": model_dir / "concrete_ml_xgb.joblib",
        "metrics": model_dir / "metrics.json",
    }
