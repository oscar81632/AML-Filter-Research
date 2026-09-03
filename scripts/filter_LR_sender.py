"""
Sender-only (paying-bank-only) variant: the paying bank filters its OWN outgoing
transactions using only its OWN view. Restrict to the paying observation of each
transaction (role in {sender, internal_debit}) -> exactly one row per transaction,
no cross-bank OR. Reports both the timestamp split and the chain-aware split.

Usage:
  python scripts/filter_LR_sender.py data/HI-Small_Trans.csv data/HI-Small_Patterns.txt
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_LR import load, expand_to_local_observations, engineer_pooled_features, BASE_FEATURES
from filter_LR_chainsplit import parse_patterns, build_chain_ids, eval_split, TRAIN_FRAC


def main():
    csv, patterns = sys.argv[1], sys.argv[2]
    print("Loading + matching chains ...", file=sys.stderr)
    raw = load(csv)
    key2chain, n_chains = parse_patterns(patterns)
    chain_id = build_chain_ids(csv, key2chain, len(raw))

    long_raw = expand_to_local_observations(raw)
    feat_df = engineer_pooled_features(long_raw)

    # --- keep only the paying-bank observation of each transaction ---
    paying = feat_df["role"].isin(["sender", "internal_debit"]).to_numpy()
    feat_df = feat_df[paying].reset_index(drop=True)
    print(f"Paying-side observations (1 per txn): {len(feat_df):,}", file=sys.stderr)

    y = feat_df["Is Laundering"].astype(int)
    feature_cols = [c for c in BASE_FEATURES if c in feat_df.columns] + \
                   [c for c in feat_df.columns if c.startswith("Payment Format_")]
    X = feat_df[feature_cols].fillna(0).astype(np.float32)

    ts = raw["Timestamp"].to_numpy()
    cutoff = raw["Timestamp"].quantile(TRAIN_FRAC)
    print(f"cutoff = {cutoff}")

    split_ts = ts < np.datetime64(cutoff)

    split_chain = split_ts.copy()
    mask = chain_id >= 0
    dfc = pd.DataFrame({"cid": chain_id[mask], "ts": ts[mask]})
    first_ts = dfc.groupby("cid")["ts"].transform("min").to_numpy()
    split_chain[mask] = first_ts < np.datetime64(cutoff)

    d_ts = eval_split("SENDER-ONLY | timestamp", split_ts, X, y, feat_df, raw)
    d_ch = eval_split("SENDER-ONLY | chain-aware", split_chain, X, y, feat_df, raw)

    print("\n==================== SENDER-ONLY COMPARISON ====================")
    print(f"drop% @95% retention:  timestamp {d_ts:.2f}%   ->   chain-aware {d_ch:.2f}%")
    print(f"leakage optimism (gap): {d_ts - d_ch:+.2f} pp")


if __name__ == "__main__":
    main()
