"""Feature build orchestration."""

from __future__ import annotations

import os

from aml_fhe.config import PipelineConfig
from aml_fhe.legacy_runtime import run_module


def build_features(config: PipelineConfig) -> None:
    """Build graph-based feature parquets by running the _legacy_* Spark modules."""
    _set_dataset_env(config.data.dataset)
    run_module("aml_fhe.features._legacy_2_model")


def _set_dataset_env(dataset: str) -> None:
    normalized = dataset.lower().replace("_", "-")
    if normalized != "hi-small":
        raise ValueError(f"Only HI-Small is supported for feature generation, got {dataset!r}")
    os.environ["EXSTRAQT_HIGH_ILLICIT"] = "1"
    os.environ["EXSTRAQT_FILE_SIZE"] = "Small"
