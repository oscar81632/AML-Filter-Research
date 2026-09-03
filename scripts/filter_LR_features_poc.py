"""
POC: do stronger BANK-LOCAL features (windowed velocity / amount structure) enrich
the top of the ranking and raise drop% at 95% retention?

Compares, on the sender-only + chain-aware setup (the honest/realistic one):
  A) LR      + old features            (baseline, ~ current filter)
  B) LR      + old + new features
  C) XGB(d2) + old + new features      (shallow trees, spec-allowed)

All new features are causal (time-sorted, look only backward) and computable from a
single bank's view of its own account -- no global graph.

Usage:
  python scripts/filter_LR_features_poc.py data/HI-Small_Trans.csv data/HI-Small_Patterns.txt [--nrows N]
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_LR import load, expand_to_local_observations, engineer_pooled_features, BASE_FEATURES
from filter_LR_chainsplit import parse_patterns, build_chain_ids, TRAIN_FRAC, SEED

NEW_FEATURES = ["tx_24h", "new_cp_24h", "is_new_cp", "same_amount_cnt",
                "amt_vs_prevmean", "amount_round1k"]


def add_local_features(feat_df):
    """Windowed velocity + amount-structure features (causal, own-account local)."""
    feat_df = feat_df.copy()
    feat_df["is_new_cp"] = (feat_df["counterparty_prev_count"] == 0).astype(np.int8)
    feat_df["amount_round1k"] = (feat_df["Amount Paid"] % 1000 == 0).astype(np.int8)

    # sort by own account + time; compute, then join back by index
    s = feat_df.sort_values(["own_bank", "own_account", "Timestamp"])
    g = s.groupby(["own_bank", "own_account"], sort=False)

    # rolling 24h transaction burst + new-counterparty burst (fan-in / fan-out)
    s["tx_24h"] = g.rolling("24h", on="Timestamp")["transaction_id"].count().to_numpy()
    s["new_cp_24h"] = g.rolling("24h", on="Timestamp")["is_new_cp"].sum().to_numpy()

    # repeated identical amounts (layering / structuring)
    s["same_amount_cnt"] = s.groupby(
        ["own_bank", "own_account", "Amount Paid"], sort=False).cumcount()

    # current amount vs the account's past-average amount (spike)
    ga = s.groupby(["own_bank", "own_account"], sort=False)["Amount Paid"]
    prev_sum = ga.cumsum() - s["Amount Paid"]
    prev_cnt = ga.cumcount()
    prev_mean = prev_sum / prev_cnt.replace(0, np.nan)
    ratio = (s["Amount Paid"] / prev_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    s["amt_vs_prevmean"] = np.log1p(ratio.clip(lower=0))   # log1p compresses the heavy tail

    return feat_df.join(s[["tx_24h", "new_cp_24h", "same_amount_cnt", "amt_vs_prevmean"]])


def make_X(feat_df, cols):
    X = feat_df[cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    return X.to_numpy()


def evaluate(name, model, X, y, split, tid, feat_df):
    tr = split[feat_df["transaction_id"].to_numpy()]
    te = ~tr
    model.fit(X[tr], y[tr])
    proba = model.predict_proba(X[te])[:, 1]
    yt = y[te]
    auc = roc_auc_score(yt, proba)
    ap = average_precision_score(yt, proba)
    _, recall, thr = precision_recall_curve(yt, proba)
    idx = np.where(recall[:-1] >= 0.95)[0]
    t = thr[idx[-1]] if len(idx) else 0.0
    drop = 100 * (proba < t).mean()
    print(f"  {name:<26} ROC-AUC {auc:.4f}  PR-AUC {ap:.4f}  drop@95% {drop:5.2f}%")
    return drop, auc, ap


def main():
    from xgboost import XGBClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("patterns")
    ap.add_argument("--nrows", type=int, default=None, help="sample first N transactions (quick test)")
    args = ap.parse_args()

    print("Loading + features ...", file=sys.stderr)
    raw = load(args.csv)
    if args.nrows:
        raw = raw.iloc[:args.nrows].copy()
    key2chain, _ = parse_patterns(args.patterns)
    chain_id = build_chain_ids(args.csv, key2chain, len(raw))[:len(raw)]

    feat_df = engineer_pooled_features(expand_to_local_observations(raw))
    feat_df = add_local_features(feat_df)

    # paying-bank-only (one row per txn)
    feat_df = feat_df[feat_df["role"].isin(["sender", "internal_debit"])].reset_index(drop=True)
    y = feat_df["Is Laundering"].astype(int).to_numpy()

    old_cols = [c for c in BASE_FEATURES if c in feat_df.columns] + \
               [c for c in feat_df.columns if c.startswith("Payment Format_")]
    new_cols = old_cols + NEW_FEATURES

    # chain-aware split
    ts = raw["Timestamp"].to_numpy()
    cutoff = raw["Timestamp"].quantile(TRAIN_FRAC)
    split = ts < np.datetime64(cutoff)
    m = chain_id >= 0
    dfc = pd.DataFrame({"cid": chain_id[m], "ts": ts[m]})
    split[m] = dfc.groupby("cid")["ts"].transform("min").to_numpy() < np.datetime64(cutoff)

    tid = feat_df["transaction_id"].to_numpy()
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"test illicit rate, features old={len(old_cols)} new={len(new_cols)}, scale_pos_weight={spw:.0f}")

    X_old = make_X(feat_df, old_cols)
    X_new = make_X(feat_df, new_cols)

    def lr():
        return Pipeline([("s", StandardScaler()),
                         ("c", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))])

    print("\n=== sender-only, chain-aware ===")
    evaluate("A) LR   + old", lr(), X_old, y, split, tid, feat_df)
    evaluate("B) LR   + old+new", lr(), X_new, y, split, tid, feat_df)
    evaluate("C) XGBd2+ old+new",
             XGBClassifier(max_depth=2, n_estimators=300, learning_rate=0.1,
                           scale_pos_weight=spw, n_jobs=-1, random_state=SEED,
                           eval_metric="logloss"),
             X_new, y, split, tid, feat_df)


if __name__ == "__main__":
    main()
