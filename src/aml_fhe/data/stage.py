"""Raw Kaggle IBM AML data staging."""

from __future__ import annotations

import os

from aml_fhe.config import PipelineConfig
from aml_fhe.legacy_runtime import run_module


def stage_data(config: PipelineConfig) -> None:
    """Stage Kaggle IBM AML raw files by running the _legacy_* Spark modules."""
    _set_dataset_env(config.data.dataset)
    run_module("aml_fhe.data._legacy_prepare_input_small")


def _set_dataset_env(dataset: str) -> None:
    normalized = dataset.lower().replace("_", "-")
    if normalized != "hi-small":
        raise ValueError(f"Only HI-Small is supported for staging, got {dataset!r}")
    os.environ["EXSTRAQT_HIGH_ILLICIT"] = "1"
    os.environ["EXSTRAQT_FILE_SIZE"] = "Small"
