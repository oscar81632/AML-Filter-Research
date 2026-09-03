"""
filter_LR with a chain-aware split, compared head-to-head against the original
timestamp split, to measure the group/campaign leakage from laundering chains
that straddle the train/test cut.

Reuses the baseline filter_LR feature pipeline verbatim; only the split changes.

Usage:
  python scripts/filter_LR_chainsplit.py data/HI-Small_Trans.csv data/HI-Small_Patterns.txt
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix, roc_auc_score, average_precision_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_LR import (
    load, expand_to_local_observations, engineer_pooled_features, BASE_FEATURES,
)

TRAIN_FRAC = 0.7
TARGET_RECALL = 0.95
SEED = 42
KEY_IDX = [0, 1, 2, 3, 4, 7]  # Timestamp, From Bank, From Account, To Bank, To Account, Amount Paid


def parse_patterns(path):
    """key(str) -> chain_id (0..N-1). Key = ts|fromBank|fromAcct|toBank|toAcct|amtPaid."""
    key2chain = {}
    chain = -1
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                chain += 1
            elif line.startswith("END LAUNDERING ATTEMPT"):
                continue
            elif line.strip():
                p = line.split(",")
                key = "|".join(p[i] for i in KEY_IDX)
                key2chain[key] = chain
    return key2chain, chain + 1


def build_chain_ids(csv_path, key2chain, n_rows):
    """Return chain_id per transaction (indexed by transaction_id = row order); -1 if none."""
    s = pd.read_csv(csv_path, dtype=str, header=0, names=list(range(11)), usecols=KEY_IDX)
    key = s[KEY_IDX[0]].str.cat([s[i] for i in KEY_IDX[1:]], sep="|")
    cid = key.map(key2chain)
    return cid.fillna(-1).astype(int).to_numpy()


def eval_split(name, txn_is_train, X, y, feat_df, raw):
    tr = txn_is_train[feat_df["transaction_id"].to_numpy()]
    te = ~tr
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    txn_id_te = feat_df.loc[te, "transaction_id"].to_numpy()

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
    ])
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]

    # transaction-level OR combine (max proba across the 2 local observations)
    tmp = pd.DataFrame({"tid": txn_id_te, "proba": proba, "label": yte.to_numpy()})
    txn = tmp.groupby("tid").agg(proba=("proba", "max"), label=("label", "max")).reset_index()
    yt, pt = txn["label"].to_numpy(), txn["proba"].to_numpy()
    n_txn = len(txn)
    auc = roc_auc_score(yt, pt)
    ap = average_precision_score(yt, pt)   # PR-AUC (baseline = positive rate)
    base_rate = yt.mean()

    precision, recall, thr = precision_recall_curve(yt, pt)
    # drop% at each retention level
    rows = []
    for target in (0.99, 0.98, 0.95, 0.90):
        idx = np.where(recall[:-1] >= target)[0]
        if len(idx) == 0:
            rows.append((target, None, None)); continue
        t = thr[idx[-1]]
        dropped = int((pt < t).sum())
        rows.append((target, dropped, 100 * dropped / n_txn))

    # chosen point @ 95
    idx95 = np.where(recall[:-1] >= TARGET_RECALL)[0]
    t95 = thr[idx95[-1]] if len(idx95) else 0.0
    keep = pt >= t95
    tn, fp, fn, tp = confusion_matrix(yt, keep.astype(int), labels=[0, 1]).ravel()

    print(f"\n================ SPLIT: {name} ================")
    print(f"train txns(expanded rows): {tr.sum():,}   test: {te.sum():,}")
    print(f"test transactions: {n_txn:,}   test illicit: {int(yt.sum()):,}")
    print(f"Transaction-level ROC-AUC: {auc:.4f}   PR-AUC (AP): {ap:.4f}  "
          f"(random baseline PR-AUC = {base_rate:.5f})")
    print(f"{'retain illicit':>14} | {'dropped':>10} | {'drop %':>8}")
    for target, dropped, pct in rows:
        if dropped is None:
            print(f"{target:>14.2f} | {'--':>10} | {'--':>8}")
        else:
            print(f"{target:>14.2f} | {dropped:>10,} | {pct:>7.2f}%")
    print(f"@95% retention -> drop {(~keep).sum():,}/{n_txn:,} ({100*(~keep).mean():.2f}%), "
          f"illicit kept {tp}/{int(yt.sum())}, missed {fn}")

    # precision at each recall (retention) level -> shows how enriched the kept set is
    print(f"  {'recall(retain)':>14} | {'precision':>10} | {'kept':>11} | {'illicit/kept':>16}")
    for target in (0.99, 0.95, 0.90, 0.80, 0.50, 0.20):
        idx = np.where(recall >= target)[0]
        if len(idx) == 0 or idx[-1] >= len(thr):
            continue
        t = thr[idx[-1]]
        keptn = int((pt >= t).sum())
        ill = int(((pt >= t) & (yt == 1)).sum())
        prec = ill / keptn if keptn else 0.0
        print(f"  {target:>14.2f} | {prec:>9.2%} | {keptn:>11,} | {ill:>6,}/{keptn:,}")
    return 100 * (~keep).mean()


def main():
    csv = sys.argv[1]
    patterns = sys.argv[2]

    print("Loading + matching chains ...", file=sys.stderr)
    raw = load(csv)
    key2chain, n_chains = parse_patterns(patterns)
    chain_id = build_chain_ids(csv, key2chain, len(raw))
    n_matched = int((chain_id >= 0).sum())
    print(f"Patterns: {n_chains} chains, {len(key2chain):,} pattern-tx keys; "
          f"matched to {n_matched:,} transactions in the CSV "
          f"({100*n_matched/max(len(key2chain),1):.1f}% of keys)")

    long_raw = expand_to_local_observations(raw)
    feat_df = engineer_pooled_features(long_raw)
    y = feat_df["Is Laundering"].astype(int)
    feature_cols = [c for c in BASE_FEATURES if c in feat_df.columns] + \
                   [c for c in feat_df.columns if c.startswith("Payment Format_")]
    X = feat_df[feature_cols].fillna(0).astype(np.float32)

    ts = raw["Timestamp"].to_numpy()
    cutoff = raw["Timestamp"].quantile(TRAIN_FRAC)
    print(f"cutoff = {cutoff}")

    # --- split A: original timestamp split (per transaction) ---
    split_ts = ts < np.datetime64(cutoff)

    # --- split B: chain-aware. Whole chain -> train/test by its FIRST tx time. ---
    split_chain = split_ts.copy()
    mask = chain_id >= 0
    dfc = pd.DataFrame({"cid": chain_id[mask], "ts": ts[mask]})
    first_ts = dfc.groupby("cid")["ts"].transform("min").to_numpy()
    split_chain[mask] = first_ts < np.datetime64(cutoff)
    n_moved = int((split_chain != split_ts).sum())
    print(f"chain-aware split moved {n_moved:,} transactions vs timestamp split "
          f"(these were the straddling-chain tails)")

    drop_ts = eval_split("timestamp (original)", split_ts, X, y, feat_df, raw)
    drop_chain = eval_split("chain-aware", split_chain, X, y, feat_df, raw)

    print("\n==================== COMPARISON ====================")
    print(f"drop% @95% retention:  timestamp {drop_ts:.2f}%   ->   chain-aware {drop_chain:.2f}%")
    print(f"leakage optimism (gap): {drop_ts - drop_chain:+.2f} percentage points")


if __name__ == "__main__":
    main()
