"""Post-parquet benchmark and FHE evaluation runner."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
from typing import Any

from sklearn.metrics import f1_score

from aml_fhe.config import PipelineConfig
from aml_fhe.evaluation.metrics import binary_metrics
from aml_fhe.evaluation.thresholds import threshold_sweep
from aml_fhe.fhe.compile import compile_model
from aml_fhe.fhe.execute import execute_predictions
from aml_fhe.fhe.simulate import simulate_predictions
from aml_fhe.models.concrete_xgb import concrete_xgb_params
from aml_fhe.models.concrete_xgb import predict_proba_clear
from aml_fhe.models.concrete_xgb import save_concrete_model_best_effort
from aml_fhe.models.concrete_xgb import train_concrete_xgb
from aml_fhe.models.plain_xgb import plain_xgb_params
from aml_fhe.models.plain_xgb import predict_proba
from aml_fhe.models.plain_xgb import save_plain_model
from aml_fhe.models.plain_xgb import train_plain_xgb
from aml_fhe.models.preprocess import load_and_encode
from aml_fhe.models.sampling import eval_sample
from aml_fhe.models.sampling import train_subsample
from aml_fhe.runs.artifacts import ensure_dir
from aml_fhe.runs.artifacts import model_artifact_paths
from aml_fhe.runs.artifacts import write_json
from aml_fhe.runs.logging import configure_logging


def run_benchmark(config: PipelineConfig) -> dict[str, Any]:
    """Run plain XGBoost, Concrete ML clear-mode, and FHE simulation."""
    logger = configure_logging(config.paths.log_dir, "benchmark.log")
    logger.info("=== AML FHE Benchmark ===")
    logger.info(
        "TRAIN_NORMAL_ROWS=%s  N_BITS=%s  N_EST=%s",
        config.model.train_normal_rows,
        config.fhe.n_bits,
        config.model.n_estimators,
    )
    logger.info(
        "XGB: max_depth=%s  scale_pos_weight=%s  learning_rate=%s  subsample=%s  colsample_bytree=%s",
        config.model.max_depth,
        config.model.scale_pos_weight,
        config.model.learning_rate,
        config.model.subsample,
        config.model.colsample_bytree,
    )

    x_train_full, y_train_full, encoders = _load_split(
        logger,
        "train",
        config.paths.train_features,
        config.data.label_col,
        config.data.string_cols,
    )
    x_test, y_test, _ = _load_split(
        logger,
        "test",
        config.paths.test_features,
        config.data.label_col,
        config.data.string_cols,
        encoders,
    )
    x_valid, y_valid, _ = _load_split(
        logger,
        "valid",
        config.paths.valid_features,
        config.data.label_col,
        config.data.string_cols,
        encoders,
    )

    x_train, y_train = train_subsample(
        x_train_full,
        y_train_full,
        config.model.train_normal_rows,
        seed=config.model.random_seed,
    )
    logger.info(
        "Training subset: %s  (illicit=%s  normal=%s  IR=%.2f%%)",
        f"{len(x_train):,}",
        f"{int((y_train == 1).sum()):,}",
        f"{int((y_train == 0).sum()):,}",
        y_train.mean() * 100,
    )
    del x_train_full, y_train_full

    logger.info("Training plain XGBoost (n_est=%s)...", config.model.n_estimators)
    start = time.time()
    plain = train_plain_xgb(x_train, y_train, config.model)
    logger.info("Plain XGB trained in %.1fs", time.time() - start)

    start = time.time()
    probs_plain_valid = predict_proba(plain, x_valid)
    logger.info("Plain XGB predict_proba (valid) in %.1fs", time.time() - start)
    plain_threshold = threshold_sweep(probs_plain_valid, y_valid)
    logger.info("Plain XGB val-tuned threshold: t=%.2f", plain_threshold.threshold)

    start = time.time()
    probs_plain_test = predict_proba(plain, x_test)
    logger.info("Plain XGB predict_proba (test) in %.1fs", time.time() - start)
    plain_default_f1 = f1_score(y_test, (probs_plain_test >= 0.5).astype(int), zero_division=0)
    plain_best_pred = (probs_plain_test >= plain_threshold.threshold).astype(int)
    plain_best = binary_metrics(y_test, plain_best_pred)
    logger.info(
        "Plain XGB: default-thr F1=%.4f  best F1=%.4f @ t=%.2f  Prec=%.4f  Rec=%.4f",
        plain_default_f1,
        plain_best["f1"],
        plain_threshold.threshold,
        plain_best["precision"],
        plain_best["recall"],
    )

    logger.info("Training Concrete ML XGBClassifier (n_bits=%s)...", config.fhe.n_bits)
    start = time.time()
    concrete = train_concrete_xgb(x_train, y_train, config.model, config.fhe)
    logger.info("Concrete ML trained in %.1fs", time.time() - start)

    x_valid_eval, y_valid_eval = eval_sample(x_valid, y_valid, config.model.eval_rows)
    if config.model.eval_rows > 0:
        logger.info("CML eval subsample (valid): %s rows", f"{len(x_valid_eval):,}")
    start = time.time()
    probs_concrete_valid = predict_proba_clear(concrete, x_valid_eval)
    logger.info("Concrete ML predict_proba (valid, clear) in %.1fs", time.time() - start)
    concrete_threshold = threshold_sweep(probs_concrete_valid, y_valid_eval)
    logger.info("Concrete ML val-tuned threshold: t=%.2f", concrete_threshold.threshold)

    x_test_eval, y_test_eval = eval_sample(x_test, y_test, config.model.eval_rows)
    if config.model.eval_rows > 0:
        logger.info("CML eval subsample (test): %s rows", f"{len(x_test_eval):,}")
    start = time.time()
    probs_concrete_test = predict_proba_clear(concrete, x_test_eval)
    logger.info("Concrete ML predict_proba (test, clear) in %.1fs", time.time() - start)
    concrete_default_f1 = f1_score(
        y_test_eval,
        (probs_concrete_test >= 0.5).astype(int),
        zero_division=0,
    )
    concrete_best_pred = (probs_concrete_test >= concrete_threshold.threshold).astype(int)
    concrete_best = binary_metrics(y_test_eval, concrete_best_pred)
    logger.info(
        "Concrete ML clear: default-thr F1=%.4f  best F1=%.4f @ t=%.2f  Prec=%.4f  Rec=%.4f",
        concrete_default_f1,
        concrete_best["f1"],
        concrete_threshold.threshold,
        concrete_best["precision"],
        concrete_best["recall"],
    )

    logger.info("Compiling FHE circuit...")
    start = time.time()
    compile_model(concrete, x_train, config.fhe.compile_rows)
    compile_seconds = time.time() - start
    logger.info("Compile done in %.1fs", compile_seconds)

    fhe_circuit: dict[str, Any] | None = None
    try:
        circuit = concrete.fhe_circuit_
        fhe_circuit = {
            "size_of_secret_keys": circuit.size_of_secret_keys,
            "size_of_bootstrap_keys": circuit.size_of_bootstrap_keys,
            "size_of_keyswitch_keys": circuit.size_of_keyswitch_keys,
            "size_of_inputs": circuit.size_of_inputs,
            "size_of_outputs": circuit.size_of_outputs,
            "complexity": circuit.complexity,
            "mlir_chars": len(circuit.mlir),
        }
        logger.info(
            "FHE circuit: complexity=%s  mlir_chars=%s  secret_keys=%s  bootstrap_keys=%s",
            fhe_circuit["complexity"],
            fhe_circuit["mlir_chars"],
            fhe_circuit["size_of_secret_keys"],
            fhe_circuit["size_of_bootstrap_keys"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not capture FHE circuit metrics: %s", exc)

    simulation = simulate_predictions(concrete, x_test_eval, y_test_eval, config.fhe.simulate_rows)
    logger.info(
        "FHE simulate (n=%s): t=%.2fs  match=%.1f%%  [%s illicit in sample; F1=%.4f at default thr]",
        simulation.sample_rows,
        simulation.seconds,
        simulation.match_percent,
        simulation.illicit_in_sample,
        simulation.f1_at_default_threshold,
    )

    fhe_execute_result = None
    if config.fhe.execute_rows > 0:
        logger.info("Running actual FHE execution (execute_rows=%s)...", config.fhe.execute_rows)
        fhe_execute_result = execute_predictions(concrete, x_test_eval, y_test_eval, config.fhe.execute_rows)
        logger.info(
            "FHE execute (n=%s): t=%.2fs  match=%.1f%%  [%s illicit in sample]",
            fhe_execute_result.sample_rows,
            fhe_execute_result.seconds,
            fhe_execute_result.match_percent,
            fhe_execute_result.illicit_in_sample,
        )
    else:
        logger.info("FHE execute skipped (execute_rows=0)")

    artifacts = model_artifact_paths(config.paths.artifact_dir)
    ensure_dir(artifacts["model_dir"])
    save_plain_model(plain, artifacts["plain_xgb"])
    concrete_saved, concrete_error = save_concrete_model_best_effort(concrete, artifacts["concrete_ml_xgb"])

    metrics = _metrics_payload(
        config=config,
        plain_params=plain_xgb_params(config.model),
        concrete_params=concrete_xgb_params(config.model, config.fhe),
        training_rows=len(x_train),
        illicit_rows=int((y_train == 1).sum()),
        normal_rows=int((y_train == 0).sum()),
        feature_count=int(x_train.shape[1]),
        plain_default_f1=float(plain_default_f1),
        plain_threshold=plain_threshold,
        plain_best=plain_best,
        concrete_default_f1=float(concrete_default_f1),
        concrete_threshold=concrete_threshold,
        concrete_best=concrete_best,
        concrete_eval_rows=len(x_test_eval),
        simulation=simulation,
        fhe_circuit=fhe_circuit,
        fhe_execute_result=fhe_execute_result,
        compile_seconds=compile_seconds,
        artifact_paths=artifacts,
        concrete_saved=concrete_saved,
        concrete_error=concrete_error,
    )
    write_json(artifacts["metrics"], metrics)
    _log_summary(logger, config, metrics)
    return metrics


def _load_split(logger: logging.Logger, name: str, path, label_col, string_cols, encoders=None):
    logger.info("Loading %s features from %s...", name, path)
    start = time.time()
    x, y, encoders = load_and_encode(path, label_col, string_cols, encoders)
    logger.info(
        "%s loaded in %.1fs  shape=%s  IR=%.4f%%",
        name.title(),
        time.time() - start,
        x.shape,
        y.mean() * 100,
    )
    return x, y, encoders


def _metrics_payload(**kwargs) -> dict[str, Any]:
    config: PipelineConfig = kwargs["config"]
    artifacts = kwargs["artifact_paths"]
    concrete_artifact: dict[str, Any] = {
        "path": str(artifacts["concrete_ml_xgb"]),
        "format": "concrete_ml_json_dump",
        "saved": kwargs["concrete_saved"],
    }
    if kwargs["concrete_error"]:
        concrete_artifact["error"] = kwargs["concrete_error"]

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_features": str(config.paths.train_features),
        "valid_features": str(config.paths.valid_features),
        "test_features": str(config.paths.test_features),
        "label_col": config.data.label_col,
        "string_cols": list(config.data.string_cols),
        "training_subset": {
            "rows": int(kwargs["training_rows"]),
            "normal_rows_requested": int(config.model.train_normal_rows),
            "illicit_rows": int(kwargs["illicit_rows"]),
            "normal_rows": int(kwargs["normal_rows"]),
            "illicit_ratio": float(kwargs["illicit_rows"] / kwargs["training_rows"]),
        },
        "features_used": int(kwargs["feature_count"]),
        "plain_xgb_params": kwargs["plain_params"],
        "concrete_ml_params": kwargs["concrete_params"],
        "plain_xgb": {
            "default_threshold_f1": kwargs["plain_default_f1"],
            "best_f1": kwargs["plain_best"]["f1"],
            "best_threshold": kwargs["plain_threshold"].threshold,
            "threshold_source": "validation",
            "precision_at_best_threshold": kwargs["plain_best"]["precision"],
            "recall_at_best_threshold": kwargs["plain_best"]["recall"],
        },
        "concrete_ml_clear": {
            "default_threshold_f1": kwargs["concrete_default_f1"],
            "best_f1": kwargs["concrete_best"]["f1"],
            "best_threshold": kwargs["concrete_threshold"].threshold,
            "threshold_source": "validation",
            "precision_at_best_threshold": kwargs["concrete_best"]["precision"],
            "recall_at_best_threshold": kwargs["concrete_best"]["recall"],
            "eval_rows": int(kwargs["concrete_eval_rows"]),
            "eval_rows_note": "F1 computed on subsample when eval_rows>0"
            if config.model.eval_rows > 0
            else "full test set",
        },
        "fhe_simulate": {
            "sample_rows": kwargs["simulation"].sample_rows,
            "illicit_in_sample": kwargs["simulation"].illicit_in_sample,
            "seconds": kwargs["simulation"].seconds,
            "match_percent": kwargs["simulation"].match_percent,
            "f1_at_default_thr": kwargs["simulation"].f1_at_default_threshold,
            "note": "500-row scalable consistency check; match_percent is the meaningful metric",
        },
        "fhe_circuit": kwargs["fhe_circuit"],
        "fhe_execute": _fhe_execute_payload(kwargs["fhe_execute_result"]),
        "compile_seconds": float(kwargs["compile_seconds"]),
        "artifacts": {
            "plain_xgb": {
                "path": str(artifacts["plain_xgb"]),
                "format": "xgboost_json",
                "saved": True,
            },
            "concrete_ml_xgb": concrete_artifact,
            "metrics": str(artifacts["metrics"]),
        },
    }


def _fhe_execute_payload(result) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "sample_rows": result.sample_rows,
        "illicit_in_sample": result.illicit_in_sample,
        "seconds": result.seconds,
        "match_percent": result.match_percent,
        "note": result.note,
    }


def _log_summary(logger: logging.Logger, config: PipelineConfig, metrics: dict[str, Any]) -> None:
    logger.info("")
    logger.info("=" * 50)
    logger.info("QUANTIZATION IMPACT SUMMARY")
    logger.info("=" * 50)
    logger.info(
        "Training subset:    %s rows  (%s normal + all illicit)",
        f"{metrics['training_subset']['rows']:,}",
        f"{config.model.train_normal_rows:,}",
    )
    logger.info("Features:           %s", metrics["features_used"])
    logger.info("n_bits:             %s", config.fhe.n_bits)
    logger.info("n_estimators:       %s", config.model.n_estimators)
    logger.info(
        "Plain XGB best F1:  %.4f @ t=%.2f (val-tuned)  (Prec=%.4f Rec=%.4f)",
        metrics["plain_xgb"]["best_f1"],
        metrics["plain_xgb"]["best_threshold"],
        metrics["plain_xgb"]["precision_at_best_threshold"],
        metrics["plain_xgb"]["recall_at_best_threshold"],
    )
    logger.info(
        "CML clear best F1:  %.4f @ t=%.2f (val-tuned)  (Prec=%.4f Rec=%.4f)",
        metrics["concrete_ml_clear"]["best_f1"],
        metrics["concrete_ml_clear"]["best_threshold"],
        metrics["concrete_ml_clear"]["precision_at_best_threshold"],
        metrics["concrete_ml_clear"]["recall_at_best_threshold"],
    )
    logger.info(
        "F1 drop (quant):    %+.4f",
        metrics["plain_xgb"]["best_f1"] - metrics["concrete_ml_clear"]["best_f1"],
    )
    logger.info(
        "FHE simulate match: %.1f%%  (%s illicit in %s-row sample)",
        metrics["fhe_simulate"]["match_percent"],
        metrics["fhe_simulate"]["illicit_in_sample"],
        metrics["fhe_simulate"]["sample_rows"],
    )
    if metrics.get("fhe_circuit"):
        c = metrics["fhe_circuit"]
        logger.info(
            "FHE circuit: complexity=%s  mlir_chars=%s  secret_keys=%s  bootstrap_keys=%s",
            c.get("complexity"),
            c.get("mlir_chars"),
            c.get("size_of_secret_keys"),
            c.get("size_of_bootstrap_keys"),
        )
    if metrics.get("fhe_execute"):
        e = metrics["fhe_execute"]
        logger.info(
            "FHE execute match: %.1f%%  (%.2fs  %s-row sample)",
            e["match_percent"],
            e["seconds"],
            e["sample_rows"],
        )
    else:
        logger.info("FHE execute: not run (execute_rows=0)")
