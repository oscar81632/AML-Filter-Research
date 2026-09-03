from pathlib import Path

from aml_fhe.config import load_config
from aml_fhe.config import load_pipeline_config


def test_load_config_reads_toml() -> None:
    config = load_config(Path("configs/hi-small.toml"))

    assert config.data["data"]["dataset"] == "HI-Small"


def test_pipeline_config_materializes_defaults() -> None:
    config = load_pipeline_config(Path("configs/fhe-baseline.toml"))

    assert config.paths.train_features == Path("features/hi-small/train_features.parquet")
    assert config.data.label_col == "is_laundering"
    assert config.model.n_estimators == 50
    assert config.fhe.n_bits == 3
