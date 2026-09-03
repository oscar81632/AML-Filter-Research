"""Analyze how same_amount_cnt relates to pre-filter scores and retention.

This is a diagnostic script. The operating threshold is still selected on
validation, while the bucket summaries are reported on the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from filter_xgb import chain_aware_buckets
from filter_xgb import build_sender_features
from filter_xgb import choose_threshold
from filter_xgb import evaluate_split
from filter_xgb import make_matrix
from filter_LR import load
from filter_LR_chainsplit import SEED
from filter_LR_chainsplit import build_chain_ids
from filter_LR_chainsplit import parse_patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transactions_csv")
    parser.add_argument("patterns_txt")
    parser.add_argument("--feature-set", default="old_same_amount")
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def bucketize_same_amount(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-0.5, 0.5, 1.5, 4.5, 9.5, np.inf],
        labels=["0", "1", "2-4", "5-9", "10+"],
    )


def summarize_bins(df: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows = []
    for bucket, part in df.groupby("same_amount_bucket", observed=False):
        if part.empty:
            continue
        kept = part["kept"].astype(bool)
        illicit = part["is_laundering"].astype(bool)
        rows.append(
            {
                "same_amount_cnt_bucket": str(bucket),
                "rows": int(len(part)),
                "row_share_pct": float(100 * len(part) / len(df)),
                "illicit": int(illicit.sum()),
                "illicit_rate_pct": float(100 * illicit.mean()),
                "mean_score": float(part["score"].mean()),
                "median_score": float(part["score"].median()),
                "kept_pct": float(100 * kept.mean()),
                "dropped_pct": float(100 * (~kept).mean()),
            }
        )
    return rows


def print_bucket_table(rows: list[dict[str, float | int | str]]) -> None:
    print("\n=== same_amount_cnt bucket analysis on TEST ===")
    print(
        f"{'Bucket':<8}"
        f"{'Rows':>12}"
        f"{'Row share':>12}"
        f"{'Illicit':>10}"
        f"{'Illicit rate':>15}"
        f"{'Mean score':>13}"
        f"{'Median score':>15}"
        f"{'Kept':>10}"
        f"{'Dropped':>10}"
    )
    print("-" * 105)
    for row in rows:
        print(
            f"{row['same_amount_cnt_bucket']:<8}"
            f"{row['rows']:>12,}"
            f"{row['row_share_pct']:>11.2f}%"
            f"{row['illicit']:>10,}"
            f"{row['illicit_rate_pct']:>14.4f}%"
            f"{row['mean_score']:>13.4f}"
            f"{row['median_score']:>15.4f}"
            f"{row['kept_pct']:>9.2f}%"
            f"{row['dropped_pct']:>9.2f}%"
        )


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()

    raw = load(args.transactions_csv)
    key2chain, n_chains = parse_patterns(args.patterns_txt)
    chain_id = build_chain_ids(args.transactions_csv, key2chain, len(raw))[: len(raw)]
    feat_df, feature_cols = build_sender_features(raw, args.feature_set, [], [])
    x = make_matrix(feat_df, feature_cols)
    y = feat_df["Is Laundering"].astype(int).to_numpy()

    bucket, t_train, t_valid = chain_aware_buckets(raw, chain_id, 0.60, 0.20)
    split_bucket = bucket[feat_df["transaction_id"].to_numpy()]
    train_mask = split_bucket == 0
    valid_mask = split_bucket == 1
    test_mask = split_bucket == 2

    scale_pos_weight = float((y[train_mask] == 0).sum() / max((y[train_mask] == 1).sum(), 1))
    model = XGBClassifier(
        max_depth=args.max_depth,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=args.seed,
        eval_metric="logloss",
    )
    model.fit(x[train_mask], y[train_mask])
    proba_valid = model.predict_proba(x[valid_mask])[:, 1]
    proba_test = model.predict_proba(x[test_mask])[:, 1]
    threshold = choose_threshold(y[valid_mask], proba_valid, args.target_recall)
    test_metrics = evaluate_split("test", y[test_mask], proba_test, threshold)

    test_df = feat_df.loc[test_mask, ["same_amount_cnt", "Is Laundering"]].copy()
    test_df["score"] = proba_test
    test_df["kept"] = proba_test >= threshold
    test_df["is_laundering"] = test_df["Is Laundering"].astype(int)
    test_df["same_amount_bucket"] = bucketize_same_amount(test_df["same_amount_cnt"])
    bucket_rows = summarize_bins(test_df)

    print("\n=== Run summary ===")
    print(f"feature set             : {args.feature_set}")
    print(f"features                : {len(feature_cols)}")
    print(f"laundering chains       : {n_chains:,}")
    print(f"train cutoff            : {t_train}")
    print(f"valid cutoff            : {t_valid}")
    print(f"threshold selected valid: {threshold:.6f}")
    print(f"TEST illicit retention  : {100 * test_metrics['illicit_retention']:.2f}%")
    print(f"TEST drop rate          : {100 * test_metrics['drop_rate']:.2f}%")
    print(f"TEST ROC-AUC            : {test_metrics['roc_auc']:.4f}")
    print(f"TEST PR-AUC             : {test_metrics['pr_auc']:.4f}")
    print_bucket_table(bucket_rows)

    elapsed_seconds = time.perf_counter() - started_at
    result = {
        "protocol": {
            "feature_set": args.feature_set,
            "target_recall": args.target_recall,
            "threshold_selected_on": "valid",
            "seed": args.seed,
            "feature_cols": feature_cols,
        },
        "test": test_metrics,
        "same_amount_bucket_table": bucket_rows,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60,
        },
    }
    print("\n=== Runtime ===")
    print(f"Total elapsed time      : {elapsed_seconds:.2f}s ({elapsed_seconds / 60:.2f} min)")

    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON: {path}")


if __name__ == "__main__":
    main()
