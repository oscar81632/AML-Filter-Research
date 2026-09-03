"""
Filter aligned to the main pipeline's 60/20/20 chronological split, with proper
use of the validation set: the operating threshold (95% illicit retention) is
selected on VALID and applied to TEST -- no test-set threshold peeking.

The split is reproduced by the main pipeline's rule (sort by timestamp, 60/20/20 by
count) on the raw CSV, chain-aware (each laundering chain assigned whole to one
bucket by its first transaction's time). Exact row-join to the feature parquets is
not possible (they drop the raw identifiers), so this aligns the split methodology
and the test time-window, not bit-identical row membership.

Usage:
  python scripts/filter_LR_622.py data/HI-Small_Trans.csv data/HI-Small_Patterns.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_LR import load, expand_to_local_observations, engineer_pooled_features, BASE_FEATURES
from filter_LR_chainsplit import parse_patterns, build_chain_ids, SEED
from filter_LR_features_poc import add_local_features, NEW_FEATURES

TARGET_RECALL = 0.95


def buckets_chain_aware(raw, chain_id):
    """Per-transaction bucket 0=train(<60%) 1=valid(60-80%) 2=test(>=80%), chain-aware."""
    ts = raw["Timestamp"].to_numpy()
    t60 = np.datetime64(raw["Timestamp"].quantile(0.60))
    t80 = np.datetime64(raw["Timestamp"].quantile(0.80))
    bucket = np.where(ts < t60, 0, np.where(ts < t80, 1, 2))
    m = chain_id >= 0
    dfc = pd.DataFrame({"cid": chain_id[m], "ts": ts[m]})
    first = dfc.groupby("cid")["ts"].transform("min").to_numpy()
    bucket[m] = np.where(first < t60, 0, np.where(first < t80, 1, 2))
    return bucket, t60, t80


def evaluate(name, model, X, y, tb):
    tr, va, te = tb == 0, tb == 1, tb == 2
    model.fit(X[tr], y[tr])
    pv = model.predict_proba(X[va])[:, 1]
    pt = model.predict_proba(X[te])[:, 1]

    # threshold for 95% retention chosen on VALID
    _, rv, thv = precision_recall_curve(y[va], pv)
    idx = np.where(rv[:-1] >= TARGET_RECALL)[0]
    thr = thv[idx[-1]] if len(idx) else 0.0

    # apply to TEST
    yt = y[te]
    drop = 100 * (pt < thr).mean()
    kept = pt >= thr
    ret_test = ((kept) & (yt == 1)).sum() / max((yt == 1).sum(), 1)
    auc = roc_auc_score(yt, pt)
    ap = average_precision_score(yt, pt)
    print(f"  {name:<24} thr(valid)={thr:.3f}  TEST: drop {drop:5.2f}%  "
          f"retain {100*ret_test:5.1f}%  ROC-AUC {auc:.4f}  PR-AUC {ap:.4f}")


def main():
    csv, patterns = sys.argv[1], sys.argv[2]
    print("Loading + features ...", file=sys.stderr)
    raw = load(csv)
    key2chain, _ = parse_patterns(patterns)
    chain_id = build_chain_ids(csv, key2chain, len(raw))

    feat_df = add_local_features(engineer_pooled_features(expand_to_local_observations(raw)))
    feat_df = feat_df[feat_df["role"].isin(["sender", "internal_debit"])].reset_index(drop=True)
    y = feat_df["Is Laundering"].astype(int).to_numpy()

    old_cols = [c for c in BASE_FEATURES if c in feat_df.columns] + \
               [c for c in feat_df.columns if c.startswith("Payment Format_")]
    new_cols = old_cols + NEW_FEATURES

    bucket, t60, t80 = buckets_chain_aware(raw, chain_id)
    tb = bucket[feat_df["transaction_id"].to_numpy()]
    for b, nm in [(0, "train"), (1, "valid"), (2, "test")]:
        sel = tb == b
        print(f"{nm}: {sel.sum():,} txns, {int(y[sel].sum()):,} illicit "
              f"({100*y[sel].mean():.4f}%)")
    spw = (y[tb == 0] == 0).sum() / max((y[tb == 0] == 1).sum(), 1)

    def X_of(cols):
        return feat_df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()

    def lr():
        return Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))])

    Xo, Xn = X_of(old_cols), X_of(new_cols)
    print("\n=== sender-only, 60/20/20 chain-aware, threshold chosen on VALID ===")
    evaluate("A) LR   + old", lr(), Xo, y, tb)
    evaluate("B) LR   + old+new", lr(), Xn, y, tb)
    evaluate("C) XGBd2+ old+new",
             XGBClassifier(max_depth=2, n_estimators=300, learning_rate=0.1,
                           scale_pos_weight=spw, n_jobs=-1, random_state=SEED,
                           eval_metric="logloss"),
             Xn, y, tb)


if __name__ == "__main__":
    main()
