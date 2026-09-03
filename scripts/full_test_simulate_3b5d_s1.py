"""
Full test-set FHE simulate — 3-bit depth-5 50-tree seed=1 (recommended setting).

Retrains with config.model.random_seed=1 (controls both subsample and XGBoost),
val-tunes the threshold on the validation set, then runs fhe="simulate" on every
row in the test set. Stores y_true, y_pred_sim, y_proba_sim for all 1,014,540
rows. F1 computed at the val-tuned threshold is the provable, reproducible number.

Outputs:
  artifacts/full_test_simulate/3b5d_s1_test_predictions.parquet
  artifacts/full_test_simulate/3b5d_s1_test_predictions.log
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aml_fhe.config import load_pipeline_config
from aml_fhe.fhe.compile import compile_model
from aml_fhe.models.concrete_xgb import train_concrete_xgb
from aml_fhe.models.preprocess import load_and_encode
from aml_fhe.models.sampling import train_subsample

CONFIG_PATH = REPO_ROOT / "configs/sweep/3b5d50t_s1.toml"
OUTPUT_DIR  = REPO_ROOT / "artifacts/full_test_simulate"
OUTPUT_PQ   = OUTPUT_DIR / "3b5d_s1_test_predictions.parquet"
OUTPUT_LOG  = OUTPUT_DIR / "3b5d_s1_test_predictions.log"
CHUNK_SIZE  = 50_000
THRESHOLDS  = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("full_test_simulate_3b5d_s1")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def f1_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> tuple[float, float, float]:
    y_pred = (y_proba >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def simulate_in_chunks(model, x: np.ndarray, chunk_size: int, logger: logging.Logger, label: str) -> np.ndarray:
    n = len(x)
    n_chunks = (n + chunk_size - 1) // chunk_size
    all_proba: list[np.ndarray] = []
    t0 = time.time()
    for i in range(n_chunks):
        lo, hi = i * chunk_size, min((i + 1) * chunk_size, n)
        proba_chunk = model.predict_proba(x[lo:hi], fhe="simulate")[:, 1]
        all_proba.append(proba_chunk)
        elapsed = time.time() - t0
        avg = elapsed / hi
        remaining = avg * (n - hi)
        logger.info("%s chunk %s/%s  rows %s–%s  avg=%.3fs/row  remaining≈%.1fmin",
                    label, i + 1, n_chunks, f"{lo:,}", f"{hi:,}", avg, remaining / 60)
    return np.concatenate(all_proba)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUTPUT_LOG)
    logger.info("=== Full test-set FHE simulate — 3b5d50t seed=1 (recommended) ===")
    logger.info("Config: %s", CONFIG_PATH)

    config = load_pipeline_config(CONFIG_PATH)
    seed = config.model.random_seed  # 1 — controls both subsample and XGBoost
    logger.info("random_seed=%s  (controls subsample + XGBoost model)", seed)

    # --- Train ---
    logger.info("Loading train features...")
    t0 = time.time()
    x_train_full, y_train_full, encoders = load_and_encode(
        config.paths.train_features, config.data.label_col, config.data.string_cols
    )
    logger.info("Train loaded in %.1fs  shape=%s", time.time() - t0, x_train_full.shape)

    x_train, y_train = train_subsample(
        x_train_full, y_train_full, config.model.train_normal_rows, seed=seed
    )
    logger.info("Training subset: %s rows  (%s illicit  %s normal)",
                f"{len(x_train):,}", f"{int((y_train==1).sum()):,}", f"{int((y_train==0).sum()):,}")
    del x_train_full, y_train_full

    logger.info("Training Concrete ML XGBClassifier (n_bits=%s n_est=%s depth=%s seed=%s)...",
                config.fhe.n_bits, config.model.n_estimators, config.model.max_depth, seed)
    t0 = time.time()
    concrete = train_concrete_xgb(x_train, y_train, config.model, config.fhe)
    logger.info("CML trained in %.1fs", time.time() - t0)

    logger.info("Compiling FHE circuit (compile_rows=%s)...", config.fhe.compile_rows)
    t0 = time.time()
    compile_model(concrete, x_train, config.fhe.compile_rows)
    logger.info("Compiled in %.1fs", time.time() - t0)

    # --- Val set: find best threshold ---
    logger.info("Loading validation features...")
    t0 = time.time()
    x_val, y_val, _ = load_and_encode(
        config.paths.valid_features, config.data.label_col, config.data.string_cols, encoders
    )
    logger.info("Val loaded in %.1fs  shape=%s  illicit=%s (%.4f%%)",
                time.time() - t0, x_val.shape, f"{int(y_val.sum()):,}", y_val.mean() * 100)

    logger.info("Simulating validation set to find best threshold...")
    y_val_proba = simulate_in_chunks(concrete, x_val, CHUNK_SIZE, logger, "val")
    del x_val

    best_thr, best_f1 = 0.5, 0.0
    logger.info("Val F1 at each threshold:")
    for thr in THRESHOLDS:
        f1, prec, rec = f1_at_threshold(y_val, y_val_proba, thr)
        marker = " <-- BEST" if f1 > best_f1 else ""
        logger.info("  thr=%.2f  F1=%.4f  Prec=%.4f  Rec=%.4f%s", thr, f1, prec, rec, marker)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    logger.info("Val-tuned threshold: %.2f  (val F1=%.4f)", best_thr, best_f1)

    # --- Test set: full simulate ---
    logger.info("Loading test features...")
    t0 = time.time()
    x_test, y_test, _ = load_and_encode(
        config.paths.test_features, config.data.label_col, config.data.string_cols, encoders
    )
    n_test = len(x_test)
    n_illicit = int(y_test.sum())
    logger.info("Test loaded in %.1fs  shape=%s  illicit=%s (%.4f%%)",
                time.time() - t0, x_test.shape, f"{n_illicit:,}", y_test.mean() * 100)

    logger.info("Simulating full test set (%s rows)...", f"{n_test:,}")
    sim_start = time.time()
    y_proba = simulate_in_chunks(concrete, x_test, CHUNK_SIZE, logger, "test")
    total_sim_seconds = time.time() - sim_start
    logger.info("Simulate complete: %.1fs (%.1f min)  avg=%.4fs/row",
                total_sim_seconds, total_sim_seconds / 60, total_sim_seconds / n_test)

    # --- Compute F1 at val-tuned threshold and key alternatives ---
    logger.info("")
    logger.info("Test F1 at key thresholds:")
    for thr in THRESHOLDS:
        f1, prec, rec = f1_at_threshold(y_test, y_proba, thr)
        marker = " <-- val-tuned" if thr == best_thr else ""
        logger.info("  thr=%.2f  F1=%.4f  Prec=%.4f  Rec=%.4f%s", thr, f1, prec, rec, marker)

    f1_final, prec_final, rec_final = f1_at_threshold(y_test, y_proba, best_thr)
    y_pred = (y_proba >= best_thr).astype(int)

    # --- Save parquet ---
    logger.info("Saving predictions to %s ...", OUTPUT_PQ)
    df = pd.DataFrame({
        "test_idx":       np.arange(n_test),
        "y_true":         y_test.astype(np.int8),
        "y_pred_sim":     y_pred.astype(np.int8),
        "y_proba_sim":    y_proba.astype(np.float32),
        "threshold_used": np.full(n_test, best_thr, dtype=np.float32),
    })
    df.to_parquet(OUTPUT_PQ, index=False)
    logger.info("Saved %s rows  (%.1f MB)", f"{len(df):,}", OUTPUT_PQ.stat().st_size / 1e6)

    logger.info("")
    logger.info("=" * 60)
    logger.info("FINAL RESULT")
    logger.info("=" * 60)
    logger.info("Config:        3b5d50t_s1  (n_bits=3 depth=5 n_est=50 seed=1)")
    logger.info("Test rows:     %s", f"{n_test:,}")
    logger.info("Illicit rows:  %s (%.4f%%)", f"{n_illicit:,}", y_test.mean() * 100)
    logger.info("Val threshold: %.2f", best_thr)
    logger.info("Test F1:       %.4f", f1_final)
    logger.info("Precision:     %.4f", prec_final)
    logger.info("Recall:        %.4f", rec_final)
    logger.info("Simulate time: %.1fs (%.1f min)", total_sim_seconds, total_sim_seconds / 60)
    logger.info("Output:        %s", OUTPUT_PQ)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
