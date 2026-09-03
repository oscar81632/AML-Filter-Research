"""Configuration loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:  # pragma: no cover - local env fallback
        tomllib = None


@dataclass(frozen=True)
class ProjectConfig:
    """Raw TOML-backed project config."""

    path: Path
    data: dict


@dataclass(frozen=True)
class PathConfig:
    """Resolved filesystem paths used by the lifecycle."""

    raw_data_dir: Path = Path("data/raw")
    staged_data_dir: Path = Path("data/staged")
    feature_dir: Path = Path("features")
    artifact_dir: Path = Path("artifacts")
    train_features: Path = Path("features/hi-small/train_features.parquet")
    valid_features: Path = Path("features/hi-small/valid_features.parquet")
    test_features: Path = Path("features/hi-small/test_features.parquet")
    log_dir: Path = Path("fhe_logs")


@dataclass(frozen=True)
class DataConfig:
    """Dataset-level settings."""

    dataset: str = "HI-Small"
    label_col: str = "is_laundering"
    string_cols: tuple[str, ...] = ("target_currency", "format", "source_currency")


@dataclass(frozen=True)
class ModelConfig:
    """Plain XGBoost and Concrete ML model settings."""

    train_normal_rows: int = 100_000
    eval_rows: int = 0
    n_estimators: int = 20
    max_depth: int = 5
    scale_pos_weight: float = 3.0
    learning_rate: float = 0.3
    subsample: float = 0.7
    colsample_bytree: float = 0.5
    random_seed: int = 42


@dataclass(frozen=True)
class FheConfig:
    """Concrete ML quantization and FHE settings."""

    n_bits: int = 3
    compile_rows: int = 1024
    simulate_rows: int = 500
    execute_rows: int = 0
    mode: str = "simulate"


@dataclass(frozen=True)
class PipelineConfig:
    """Typed config used by runnable pipeline commands."""

    raw: ProjectConfig
    paths: PathConfig
    data: DataConfig
    model: ModelConfig
    fhe: FheConfig


def load_config(path: str | Path) -> ProjectConfig:
    """Load a TOML config file."""
    config_path = Path(path)
    if tomllib is not None:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    else:
        data = _load_simple_toml(config_path)
    return ProjectConfig(path=config_path, data=data)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load config and materialize typed lifecycle settings."""
    raw = load_config(path)
    data = raw.data
    paths_section = data.get("paths", {})
    data_section = data.get("data", {})
    model_section = data.get("model", {})
    fhe_section = data.get("fhe", {})
    features_section = data.get("features", {})

    feature_dir = _path(paths_section.get("feature_dir", "features"))
    artifact_dir = _path(paths_section.get("artifact_dir", "artifacts"))

    paths = PathConfig(
        raw_data_dir=_path(paths_section.get("raw_data_dir", "data/raw")),
        staged_data_dir=_path(paths_section.get("staged_data_dir", "data/staged")),
        feature_dir=feature_dir,
        artifact_dir=artifact_dir,
        train_features=_path(features_section.get("train_output", feature_dir / "hi-small/train_features.parquet")),
        valid_features=_path(features_section.get("valid_output", feature_dir / "hi-small/valid_features.parquet")),
        test_features=_path(features_section.get("test_output", feature_dir / "hi-small/test_features.parquet")),
        log_dir=_path(paths_section.get("log_dir", "fhe_logs")),
    )
    data_config = DataConfig(
        dataset=str(data_section.get("dataset", "HI-Small")),
        label_col=str(data_section.get("label_col", "is_laundering")),
        string_cols=tuple(data_section.get("string_cols", ("target_currency", "format", "source_currency"))),
    )
    model = ModelConfig(
        train_normal_rows=int(model_section.get("train_normal_rows", 100_000)),
        eval_rows=int(model_section.get("eval_rows", 0)),
        n_estimators=int(model_section.get("n_estimators", model_section.get("n_est", 20))),
        max_depth=int(model_section.get("max_depth", 5)),
        scale_pos_weight=float(model_section.get("scale_pos_weight", 3.0)),
        learning_rate=float(model_section.get("learning_rate", 0.3)),
        subsample=float(model_section.get("subsample", 0.7)),
        colsample_bytree=float(model_section.get("colsample_bytree", 0.5)),
        random_seed=int(model_section.get("random_seed", 42)),
    )
    fhe = FheConfig(
        n_bits=int(fhe_section.get("n_bits", 3)),
        compile_rows=int(fhe_section.get("compile_rows", 1024)),
        simulate_rows=int(fhe_section.get("simulate_rows", 500)),
        execute_rows=int(fhe_section.get("execute_rows", 0)),
        mode=str(fhe_section.get("mode", "simulate")),
    )
    return PipelineConfig(raw=raw, paths=paths, data=data_config, model=model, fhe=fhe)


def _path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _load_simple_toml(path: Path) -> dict[str, dict[str, Any]]:
    """Small TOML subset parser for local Python 3.10 environments without tomli."""
    parsed: dict[str, dict[str, Any]] = {}
    section: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = parsed.setdefault(line[1:-1].strip(), {})
            continue
        if section is None or "=" not in line:
            raise ValueError(f"Unsupported TOML line in {path}: {raw_line!r}")
        key, raw_value = [part.strip() for part in line.split("=", 1)]
        section[key] = _parse_scalar(raw_value)
    return parsed


def _parse_scalar(raw_value: str) -> Any:
    if raw_value.startswith('"') and raw_value.endswith('"'):
        return raw_value[1:-1]
    if raw_value.lower() in {"true", "false"}:
        return raw_value.lower() == "true"
    try:
        return int(raw_value)
    except ValueError:
        try:
            return float(raw_value)
        except ValueError:
            return raw_value
