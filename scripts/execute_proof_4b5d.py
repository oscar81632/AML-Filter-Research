"""
FHE execute proof — 4-bit depth-5 50-tree 3M recommended setting.

Proves that actual FHE execution matches Concrete ML clear predictions
across ALL illicit rows in the test set plus a sample of normal rows.

Skips CML clear inference on the full 1M test/valid sets to stay within
the 6-hour time budget. F1 metrics are not the goal here — match% is.

Outputs:
  artifacts/execute_proof/execute_proof_4b5d.json  — full per-row results
  artifacts/execute_proof/execute_proof_4b5d.log   — progress log
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from aml_fhe.config import load_pipeline_config
from aml_fhe.fhe.compile import compile_model
from aml_fhe.models.concrete_xgb import train_concrete_xgb
from aml_fhe.models.preprocess import load_and_encode
from aml_fhe.models.sampling import train_subsample

CONFIG_PATH   = REPO_ROOT / "configs/sweep/4b5d50t_s42.toml"
OUTPUT_DIR    = REPO_ROOT / "artifacts/execute_proof"
OUTPUT_JSON   = OUTPUT_DIR / "execute_proof_4b5d.json"
OUTPUT_LOG    = OUTPUT_DIR / "execute_proof_4b5d.log"
N_NORMAL_ROWS = 300  # normal rows to include alongside all illicit rows
RANDOM_SEED   = 42


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("execute_proof")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(OUTPUT_LOG)
    logger.info("=== FHE Execute Proof — 4b5d50t (recommended setting) ===")
    logger.info("Config: %s", CONFIG_PATH)
    logger.info("Goal: execute==clear match on ALL test illicit rows + %s normal rows", N_NORMAL_ROWS)

    config = load_pipeline_config(CONFIG_PATH)

    # --- Load and encode train features (needed for subsample + compile) ---
    logger.info("Loading train features...")
    t0 = time.time()
    x_train_full, y_train_full, encoders = load_and_encode(
        config.paths.train_features, config.data.label_col, config.data.string_cols
    )
    logger.info("Train loaded in %.1fs  shape=%s", time.time() - t0, x_train_full.shape)

    x_train, y_train = train_subsample(
        x_train_full, y_train_full, config.model.train_normal_rows, seed=RANDOM_SEED
    )
    logger.info(
        "Training subset: %s rows  (%s illicit  %s normal)",
        f"{len(x_train):,}", f"{int((y_train==1).sum()):,}", f"{int((y_train==0).sum()):,}",
    )
    del x_train_full, y_train_full

    # --- Load test features ---
    logger.info("Loading test features...")
    t0 = time.time()
    x_test, y_test, _ = load_and_encode(
        config.paths.test_features, config.data.label_col, config.data.string_cols, encoders
    )
    logger.info(
        "Test loaded in %.1fs  shape=%s  illicit=%s (%.4f%%)",
        time.time() - t0, x_test.shape,
        f"{int(y_test.sum()):,}", y_test.mean() * 100,
    )

    # --- Train Concrete ML model ---
    logger.info("Training Concrete ML XGBClassifier (n_bits=%s n_est=%s depth=%s)...",
                config.fhe.n_bits, config.model.n_estimators, config.model.max_depth)
    t0 = time.time()
    concrete = train_concrete_xgb(x_train, y_train, config.model, config.fhe)
    logger.info("CML trained in %.1fs", time.time() - t0)

    # --- Compile ---
    logger.info("Compiling FHE circuit (compile_rows=%s)...", config.fhe.compile_rows)
    t0 = time.time()
    compile_model(concrete, x_train, config.fhe.compile_rows)
    compile_seconds = time.time() - t0
    logger.info("Compiled in %.1fs", compile_seconds)

    circuit = concrete.fhe_circuit_
    circuit_info = {
        "complexity":            circuit.complexity,
        "mlir_chars":            len(circuit.mlir),
        "size_of_secret_keys":   circuit.size_of_secret_keys,
        "size_of_bootstrap_keys": circuit.size_of_bootstrap_keys,
        "size_of_keyswitch_keys": circuit.size_of_keyswitch_keys,
        "size_of_inputs":        circuit.size_of_inputs,
        "size_of_outputs":       circuit.size_of_outputs,
    }
    logger.info(
        "Circuit: complexity=%.3e  bootstrap_keys=%s MB  mlir_chars=%s",
        circuit_info["complexity"],
        round(circuit_info["size_of_bootstrap_keys"] / 1e6, 1),
        circuit_info["mlir_chars"],
    )

    # --- Build stratified sample: ALL illicit + N_NORMAL_ROWS normal ---
    illicit_idx = np.where(y_test == 1)[0]
    normal_idx  = np.where(y_test == 0)[0]
    rng = np.random.default_rng(RANDOM_SEED)
    normal_selected = rng.choice(normal_idx, size=N_NORMAL_ROWS, replace=False)

    selected_idx = np.concatenate([illicit_idx, normal_selected])
    rng.shuffle(selected_idx)

    x_sample = x_test[selected_idx]
    y_sample  = y_test[selected_idx]

    n_illicit_in_sample = int((y_sample == 1).sum())
    n_normal_in_sample  = int((y_sample == 0).sum())
    logger.info(
        "Execute sample: %s rows  (%s illicit = %.1f%%  %s normal)",
        len(x_sample), n_illicit_in_sample,
        n_illicit_in_sample / len(x_sample) * 100,
        n_normal_in_sample,
    )
    logger.info(
        "Illicit coverage: %s / %s test illicit rows (%.1f%%)",
        n_illicit_in_sample, int(y_test.sum()),
        n_illicit_in_sample / y_test.sum() * 100,
    )

    # Estimate time
    est_seconds = len(x_sample) * 9.5
    logger.info(
        "Estimated execute time: %.1f hours (%.1fs/row est × %s rows)",
        est_seconds / 3600, 9.5, len(x_sample),
    )

    # --- Run FHE execute row by row ---
    logger.info("Starting FHE execute...")
    exec_start = time.time()
    results: list[dict] = []
    row_times: list[float] = []

    for i in range(len(x_sample)):
        row_x = x_sample[i : i + 1]
        t_row = time.time()
        y_exec  = int(concrete.predict(row_x, fhe="execute")[0])
        y_clear = int(concrete.predict(row_x, fhe="disable")[0])
        row_elapsed = time.time() - t_row
        row_times.append(row_elapsed)

        results.append({
            "sample_pos":   i,
            "test_idx":     int(selected_idx[i]),
            "y_true":       int(y_sample[i]),
            "y_exec":       y_exec,
            "y_clear":      y_clear,
            "match":        bool(y_exec == y_clear),
            "row_seconds":  round(row_elapsed, 3),
        })

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - exec_start
            avg = elapsed / (i + 1)
            remaining = avg * (len(x_sample) - i - 1)
            matches_so_far = sum(r["match"] for r in results)
            logger.info(
                "[%s/%s]  match=%s/%s (%.1f%%)  avg=%.2fs/row  remaining≈%.1fh",
                i + 1, len(x_sample),
                matches_so_far, i + 1,
                matches_so_far / (i + 1) * 100,
                avg,
                remaining / 3600,
            )

    total_exec_seconds = time.time() - exec_start

    # --- Aggregate results ---
    total_rows    = len(results)
    total_matches = sum(r["match"] for r in results)

    illicit_results = [r for r in results if r["y_true"] == 1]
    normal_results  = [r for r in results if r["y_true"] == 0]
    illicit_matches = sum(r["match"] for r in illicit_results)
    normal_matches  = sum(r["match"] for r in normal_results)

    mismatches = [r for r in results if not r["match"]]

    logger.info("")
    logger.info("=" * 60)
    logger.info("EXECUTE PROOF RESULTS")
    logger.info("=" * 60)
    logger.info("Total rows:          %s", total_rows)
    logger.info("Total match:         %s / %s  (%.4f%%)", total_matches, total_rows,
                total_matches / total_rows * 100)
    logger.info("Illicit match:       %s / %s  (%.4f%%)", illicit_matches, len(illicit_results),
                illicit_matches / len(illicit_results) * 100 if illicit_results else 0)
    logger.info("Normal match:        %s / %s  (%.4f%%)", normal_matches, len(normal_results),
                normal_matches / len(normal_results) * 100 if normal_results else 0)
    logger.info("Mismatches:          %s", len(mismatches))
    logger.info("Total execute time:  %.1fs  (%.2f hours)", total_exec_seconds, total_exec_seconds / 3600)
    logger.info("Avg per row:         %.3fs", total_exec_seconds / total_rows)
    logger.info("=" * 60)

    output = {
        "experiment":          "execute_proof_4b5d",
        "config":              str(CONFIG_PATH),
        "n_bits":              config.fhe.n_bits,
        "max_depth":           config.model.max_depth,
        "n_estimators":        config.model.n_estimators,
        "train_normal_rows":   config.model.train_normal_rows,
        "random_seed":         RANDOM_SEED,
        "compile_seconds":     round(compile_seconds, 3),
        "circuit":             circuit_info,
        "sample": {
            "total_rows":           total_rows,
            "illicit_rows":         n_illicit_in_sample,
            "normal_rows":          n_normal_in_sample,
            "illicit_coverage_pct": round(n_illicit_in_sample / float(y_test.sum()) * 100, 4),
            "total_test_illicit":   int(y_test.sum()),
        },
        "results_summary": {
            "total_match":         total_matches,
            "total_match_pct":     round(total_matches / total_rows * 100, 6),
            "illicit_match":       illicit_matches,
            "illicit_match_pct":   round(illicit_matches / len(illicit_results) * 100, 6) if illicit_results else None,
            "normal_match":        normal_matches,
            "normal_match_pct":    round(normal_matches / len(normal_results) * 100, 6) if normal_results else None,
            "mismatch_count":      len(mismatches),
            "mismatch_test_idxs":  [r["test_idx"] for r in mismatches],
        },
        "timing": {
            "total_exec_seconds":  round(total_exec_seconds, 3),
            "avg_seconds_per_row": round(total_exec_seconds / total_rows, 3),
            "min_seconds_per_row": round(min(row_times), 3),
            "max_seconds_per_row": round(max(row_times), 3),
        },
        "per_row_results": results,
    }

    OUTPUT_JSON.write_text(json.dumps(output, indent=2))
    logger.info("Results written to %s", OUTPUT_JSON)


if __name__ == "__main__":
    main()
