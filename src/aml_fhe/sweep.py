"""Hyperparameter sweep orchestration."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import pandas as pd

from aml_fhe.benchmark import run_benchmark
from aml_fhe.config import FheConfig
from aml_fhe.config import ModelConfig
from aml_fhe.config import PipelineConfig
from aml_fhe.config import load_pipeline_config
from aml_fhe.runs.artifacts import write_json


@dataclass(frozen=True)
class HyperparamConfig:
    """A single hyperparameter configuration."""

    n_bits: int
    n_est: int
    train_normal_rows: int
    max_depth: int
    scale_pos_weight: float
    learning_rate: float
    subsample: float
    colsample_bytree: float
    eval_rows: int = 0


@dataclass(frozen=True)
class SweepResult:
    """Result of a single hyperparameter run."""

    config: HyperparamConfig
    plain_xgb_f1: float
    concrete_ml_f1: float
    f1_drop: float
    compile_time_s: float
    wall_time_s: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "results": {
                "plain_xgb_f1": self.plain_xgb_f1,
                "concrete_ml_f1": self.concrete_ml_f1,
                "f1_drop": self.f1_drop,
                "compile_time_s": self.compile_time_s,
                "wall_time_s": self.wall_time_s,
            },
            "error": self.error,
        }


def search_space(strategy: str) -> list[HyperparamConfig]:
    """Return the reference sweep search space."""
    if strategy == "quick":
        base = dict(
            n_bits=3,
            max_depth=5,
            train_normal_rows=100_000,
            subsample=0.7,
            colsample_bytree=0.5,
            eval_rows=100_000,
        )
        return [
            HyperparamConfig(**base, n_est=20, learning_rate=0.30, scale_pos_weight=3),
            HyperparamConfig(**base, n_est=50, learning_rate=0.12, scale_pos_weight=3),
            HyperparamConfig(**base, n_est=20, learning_rate=0.30, scale_pos_weight=5),
            HyperparamConfig(**base, n_est=20, learning_rate=0.30, scale_pos_weight=7),
            HyperparamConfig(**base, n_est=50, learning_rate=0.12, scale_pos_weight=5),
        ]
    if strategy == "smoke":
        return [
            HyperparamConfig(
                n_bits=3,
                n_est=1,
                train_normal_rows=1_000,
                max_depth=2,
                scale_pos_weight=3,
                learning_rate=0.3,
                subsample=0.7,
                colsample_bytree=0.5,
                eval_rows=1_000,
            )
        ]
    raise ValueError(f"Unknown sweep strategy: {strategy}")


def run_sweep(config_path: str | Path, strategy: str = "quick", save_report: bool = False) -> list[SweepResult]:
    """Run a sweep using the provided base config for paths and data settings."""
    base_config = load_pipeline_config(config_path)
    results: list[SweepResult] = []
    started = time.time()
    configs = search_space(strategy)

    for index, hyperparams in enumerate(configs, 1):
        run_config = _config_for_run(base_config, hyperparams, index)
        run_started = time.time()
        try:
            metrics = run_benchmark(run_config)
            plain_f1 = float(metrics["plain_xgb"]["best_f1"])
            concrete_f1 = float(metrics["concrete_ml_clear"]["best_f1"])
            results.append(
                SweepResult(
                    config=hyperparams,
                    plain_xgb_f1=plain_f1,
                    concrete_ml_f1=concrete_f1,
                    f1_drop=plain_f1 - concrete_f1,
                    compile_time_s=float(metrics["compile_seconds"]),
                    wall_time_s=time.time() - run_started,
                )
            )
        except Exception as exc:
            results.append(
                SweepResult(
                    config=hyperparams,
                    plain_xgb_f1=0.0,
                    concrete_ml_f1=0.0,
                    f1_drop=0.0,
                    compile_time_s=0.0,
                    wall_time_s=time.time() - run_started,
                    error=repr(exc),
                )
            )

    if save_report:
        _save_report(base_config.paths.artifact_dir / "sweeps", strategy, started, results)
    return results


def _config_for_run(base: PipelineConfig, hyperparams: HyperparamConfig, index: int) -> PipelineConfig:
    model = ModelConfig(
        train_normal_rows=hyperparams.train_normal_rows,
        eval_rows=hyperparams.eval_rows,
        n_estimators=hyperparams.n_est,
        max_depth=hyperparams.max_depth,
        scale_pos_weight=hyperparams.scale_pos_weight,
        learning_rate=hyperparams.learning_rate,
        subsample=hyperparams.subsample,
        colsample_bytree=hyperparams.colsample_bytree,
        random_seed=base.model.random_seed,
    )
    fhe = FheConfig(
        n_bits=hyperparams.n_bits,
        compile_rows=base.fhe.compile_rows,
        simulate_rows=base.fhe.simulate_rows,
        mode=base.fhe.mode,
    )
    paths = base.paths.__class__(
        raw_data_dir=base.paths.raw_data_dir,
        staged_data_dir=base.paths.staged_data_dir,
        feature_dir=base.paths.feature_dir,
        artifact_dir=base.paths.artifact_dir / f"sweep-{index:02d}",
        train_features=base.paths.train_features,
        valid_features=base.paths.valid_features,
        test_features=base.paths.test_features,
        log_dir=base.paths.log_dir / f"sweep-{index:02d}",
    )
    return PipelineConfig(raw=base.raw, paths=paths, data=base.data, model=model, fhe=fhe)


def _save_report(output_dir: Path, strategy: str, started: float, results: list[SweepResult]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "total_configs": len(results),
        "successful_runs": len([result for result in results if result.error is None]),
        "total_time_s": time.time() - started,
        "results": [result.to_dict() for result in results],
    }
    write_json(output_dir / "hyperparameter_sweep_report.json", report)
    pd.DataFrame(_csv_rows(results)).to_csv(output_dir / "hyperparameter_sweep_comparison.csv", index=False)


def _csv_rows(results: list[SweepResult]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.append(
            {
                "n_bits": result.config.n_bits,
                "n_est": result.config.n_est,
                "train_normal_rows": result.config.train_normal_rows,
                "max_depth": result.config.max_depth,
                "scale_pos_weight": result.config.scale_pos_weight,
                "learning_rate": result.config.learning_rate,
                "subsample": result.config.subsample,
                "colsample_bytree": result.config.colsample_bytree,
                "plain_xgb_f1": result.plain_xgb_f1,
                "concrete_ml_f1": result.concrete_ml_f1,
                "f1_drop": result.f1_drop,
                "compile_time_s": result.compile_time_s,
                "wall_time_s": result.wall_time_s,
                "error": result.error,
            }
        )
    return rows
