"""
Analysis of 3b5d seed=1 full test predictions.

Computes from artifacts/full_test_simulate/3b5d_s1_test_predictions.parquet:
  - AUC-PR (area under precision-recall curve)
  - Full precision-recall curve (100 threshold points)
  - Bootstrap 95% CI on F1 at the val-tuned threshold (0.55)

Outputs:
  artifacts/full_test_simulate/3b5d_s1_analysis.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET   = Path("artifacts/full_test_simulate/3b5d_s1_test_predictions.parquet")
OUTPUT    = Path("artifacts/full_test_simulate/3b5d_s1_analysis.json")
VAL_THR   = 0.55
N_BOOT    = 2000
BOOT_SEED = 42


def f1_prec_rec(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> tuple[float, float, float]:
    y_pred = (y_proba >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return f1, prec, rec


def main() -> None:
    print(f"Loading {PARQUET} ...")
    df = pd.read_parquet(PARQUET)
    y_true  = df["y_true"].to_numpy().astype(int)
    y_proba = df["y_proba_sim"].to_numpy().astype(float)
    n = len(y_true)
    n_illicit = int(y_true.sum())
    print(f"  {n:,} rows  {n_illicit:,} illicit ({n_illicit/n*100:.4f}%)")

    # --- Precision-recall curve (200 threshold points) ---
    thresholds = np.linspace(0.0, 1.0, 201)
    pr_curve = []
    for thr in thresholds:
        f1, prec, rec = f1_prec_rec(y_true, y_proba, float(thr))
        pr_curve.append({"threshold": round(float(thr), 4), "f1": round(f1, 6),
                         "precision": round(prec, 6), "recall": round(rec, 6)})

    # AUC-PR via trapezoidal integration over recall axis
    recs  = np.array([p["recall"] for p in pr_curve])
    precs = np.array([p["precision"] for p in pr_curve])
    # sort by recall ascending for trapz
    order  = np.argsort(recs)
    auc_pr = float(np.trapz(precs[order], recs[order]))
    print(f"  AUC-PR: {auc_pr:.4f}")

    # --- F1 at val-tuned threshold ---
    f1_val, prec_val, rec_val = f1_prec_rec(y_true, y_proba, VAL_THR)
    print(f"  F1 @ {VAL_THR}: {f1_val:.4f}  prec={prec_val:.4f}  rec={rec_val:.4f}")

    # --- Bootstrap 95% CI on F1 at val-tuned threshold ---
    print(f"  Bootstrap CI ({N_BOOT} iterations, seed={BOOT_SEED}) ...")
    rng = np.random.default_rng(BOOT_SEED)
    boot_f1s = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        f1_b, _, _ = f1_prec_rec(y_true[idx], y_proba[idx], VAL_THR)
        boot_f1s.append(f1_b)
    boot_f1s = np.array(boot_f1s)
    ci_lo = float(np.percentile(boot_f1s, 2.5))
    ci_hi = float(np.percentile(boot_f1s, 97.5))
    ci_std = float(boot_f1s.std())
    print(f"  Bootstrap F1: mean={boot_f1s.mean():.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  std={ci_std:.4f}")

    # --- Best threshold on test set (oracle, for reference only) ---
    best_f1, best_thr = 0.0, VAL_THR
    for p in pr_curve:
        if p["f1"] > best_f1:
            best_f1, best_thr = p["f1"], p["threshold"]
    print(f"  Best test-set F1 (oracle): {best_f1:.4f} @ thr={best_thr}")

    output = {
        "source_parquet":    str(PARQUET),
        "config":            "3b5d50t_s1  (n_bits=3 depth=5 n_est=50 seed=1)",
        "n_test_rows":       n,
        "n_illicit":         n_illicit,
        "natural_ir_pct":    round(n_illicit / n * 100, 4),
        "auc_pr":            round(auc_pr, 6),
        "val_tuned_threshold": VAL_THR,
        "f1_at_val_thr":     round(f1_val, 6),
        "precision_at_val_thr": round(prec_val, 6),
        "recall_at_val_thr": round(rec_val, 6),
        "bootstrap_ci": {
            "n_iterations":  N_BOOT,
            "seed":          BOOT_SEED,
            "f1_mean":       round(float(boot_f1s.mean()), 6),
            "f1_std":        round(ci_std, 6),
            "ci_95_lo":      round(ci_lo, 6),
            "ci_95_hi":      round(ci_hi, 6),
        },
        "best_test_oracle": {
            "f1":        round(best_f1, 6),
            "threshold": best_thr,
            "note":      "Upper bound — threshold selected on test set directly, not valid for reporting",
        },
        "pr_curve": pr_curve,
    }

    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"  Written to {OUTPUT}")


if __name__ == "__main__":
    main()
