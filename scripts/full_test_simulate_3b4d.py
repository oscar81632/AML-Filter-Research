"""
Full test-set FHE simulate — 3-bit depth-4 50-tree (recommended setting).

Runs fhe="simulate" on every row in the test set and records y_true,
y_pred_sim (binary at 0.5), and y_proba_sim (class-1 soft score).
Saves results to a parquet file. F1 computed from this file is a complete,
reproducible proof of the reported score.

Outputs:
  artifacts/full_test_simulate/3b4d_test_predictions.parquet
  artifacts/full_test_simulate/3b4d_test_predictions.log
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

CONFIG_PATH = REPO_ROOT / "configs/sweep/3b4d50t_s42.toml"
OUTPUT_DIR  = REPO_ROOT / "artifacts/full_test_simulate"
OUTPUT_PQ   = OUTPUT_DIR / "3b4d_test_predictions.parquet"
OUTPUT_LOG  = OUTPUT_DIR / "3b4d_test_predictions.log"
RANDOM_SEED = 42
CHUNK_SIZE  = 50_000  # rows per simulate call


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("full_test_simulate")
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUTPUT_LOG)
    logger.info("=== Full test-set FHE simulate — 3b4d50t (recommended) ===")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("Goal: simulate all %s test rows, record y_true + y_pred + y_proba", "1M+")

    config = load_pipeline_config(CONFIG_PATH)

    # --- Train ---
    logger.info("Loading train features...")
    t0 = time.time()
    x_train_full, y_train_full, encoders = load_and_encode(
        config.paths.train_features, config.data.label_col, config.data.string_cols
    )
    logger.info("Train loaded in %.1fs  shape=%s", time.time() - t0, x_train_full.shape)

    x_train, y_train = train_subsample(
        x_train_full, y_train_full, config.model.train_normal_rows, seed=RANDOM_SEED
    )
    logger.info("Training subset: %s rows  (%s illicit  %s normal)",
                f"{len(x_train):,}", f"{int((y_train==1).sum()):,}", f"{int((y_train==0).sum()):,}")
    del x_train_full, y_train_full

    logger.info("Training Concrete ML XGBClassifier (n_bits=%s n_est=%s depth=%s)...",
                config.fhe.n_bits, config.model.n_estimators, config.model.max_depth)
    t0 = time.time()
    concrete = train_concrete_xgb(x_train, y_train, config.model, config.fhe)
    logger.info("CML trained in %.1fs", time.time() - t0)

    logger.info("Compiling FHE circuit (compile_rows=%s)...", config.fhe.compile_rows)
    t0 = time.time()
    compile_model(concrete, x_train, config.fhe.compile_rows)
    logger.info("Compiled in %.1fs", time.time() - t0)

    # --- Load test set ---
    logger.info("Loading test features...")
    t0 = time.time()
    x_test, y_test, _ = load_and_encode(
        config.paths.test_features, config.data.label_col, config.data.string_cols, encoders
    )
    n_test = len(x_test)
    n_illicit = int(y_test.sum())
    logger.info("Test loaded in %.1fs  shape=%s  illicit=%s (%.4f%%)",
                time.time() - t0, x_test.shape, f"{n_illicit:,}", y_test.mean() * 100)

    # --- Simulate in chunks ---
    logger.info("Running FHE simulate on all %s test rows (chunk_size=%s)...",
                f"{n_test:,}", f"{CHUNK_SIZE:,}")
    n_chunks = (n_test + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info("Chunks: %s", n_chunks)

    all_proba: list[np.ndarray] = []
    sim_start = time.time()

    for i in range(n_chunks):
        lo = i * CHUNK_SIZE
        hi = min(lo + CHUNK_SIZE, n_test)
        t_chunk = time.time()
        proba_chunk = concrete.predict_proba(x_test[lo:hi], fhe="simulate")[:, 1]
        all_proba.append(proba_chunk)
        elapsed = time.time() - sim_start
        avg_per_row = elapsed / hi
        remaining = avg_per_row * (n_test - hi)
        logger.info("Chunk %s/%s  rows %s–%s  chunk_time=%.1fs  avg=%.3fs/row  remaining≈%.1fmin",
                    i + 1, n_chunks, f"{lo:,}", f"{hi:,}",
                    time.time() - t_chunk, avg_per_row, remaining / 60)

    total_sim_seconds = time.time() - sim_start
    logger.info("Simulate complete in %.1fs  (%.2f min)", total_sim_seconds, total_sim_seconds / 60)

    y_proba = np.concatenate(all_proba)
    y_pred  = (y_proba >= 0.5).astype(int)

    # --- F1 at key thresholds ---
    logger.info("")
    logger.info("F1 at key thresholds:")
    for thr in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        f1, prec, rec = f1_at_threshold(y_test, y_proba, thr)
        logger.info("  thr=%.2f  F1=%.4f  Prec=%.4f  Rec=%.4f", thr, f1, prec, rec)

    f1_default, prec_default, rec_default = f1_at_threshold(y_test, y_proba, 0.5)
    logger.info("")
    logger.info("Default threshold (0.5): F1=%.4f  Prec=%.4f  Rec=%.4f",
                f1_default, prec_default, rec_default)

    # --- Save parquet ---
    logger.info("Saving predictions to %s ...", OUTPUT_PQ)
    df = pd.DataFrame({
        "test_idx":    np.arange(n_test),
        "y_true":      y_test.astype(np.int8),
        "y_pred_sim":  y_pred.astype(np.int8),
        "y_proba_sim": y_proba.astype(np.float32),
    })
    df.to_parquet(OUTPUT_PQ, index=False)
    logger.info("Saved %s rows  (%s MB)", f"{len(df):,}",
                round(OUTPUT_PQ.stat().st_size / 1e6, 1))

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Config:          3b4d50t_s42  (n_bits=3 depth=4 n_est=50)")
    logger.info("Test rows:       %s", f"{n_test:,}")
    logger.info("Illicit rows:    %s (%.4f%%)", f"{n_illicit:,}", y_test.mean() * 100)
    logger.info("Simulate time:   %.1fs (%.1f min)", total_sim_seconds, total_sim_seconds / 60)
    logger.info("Avg s/row:       %.4f", total_sim_seconds / n_test)
    logger.info("F1 @ 0.5:        %.4f", f1_default)
    logger.info("Output:          %s", OUTPUT_PQ)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
