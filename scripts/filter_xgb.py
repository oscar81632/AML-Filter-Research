"""
Train a sender-only XGBoost pre-filter on IBM AML HI-Small.

This script combines the cleaner pieces from the filter_* POCs:

- one row per transaction, using only the paying bank's local view
- old + new causal local features
- chain-aware chronological train/valid/test split
- threshold selected on VALID, then applied once to TEST

Usage:
  python scripts/filter_xgb.py \
    data/HI-Small_Trans.csv data/HI-Small_Patterns.txt

Optional:
  python scripts/filter_xgb.py \
    data/HI-Small_Trans.csv data/HI-Small_Patterns.txt \
    --metrics-json artifacts/filter_xgb_metrics.json \
    --predictions-csv artifacts/filter_xgb_test_predictions.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import deque
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_LR import BASE_FEATURES
from filter_LR import engineer_pooled_features
from filter_LR import expand_to_local_observations
from filter_LR import load
from filter_LR_chainsplit import SEED
from filter_LR_chainsplit import build_chain_ids
from filter_LR_chainsplit import parse_patterns
from filter_LR_features_poc import NEW_FEATURES
from filter_LR_features_poc import add_local_features


DEFAULT_TARGET_RECALL = 0.95

TOP10_COMPACT_FEATURES = [
    "tx_24h",
    "new_cp_24h",
    "is_new_cp",
    "same_amount_cnt",
    "amt_vs_prevmean",
    "counterparty_prev_count",
    "time_since_last_tx_counterparty",
    "edge_tx_count",
    "own_unique_counterparties_prev",
    "own_debit_credit_amount_logratio",
]

COMPACT_GRAPH_FEATURES = [
    "edge_tx_count",
    "bilateral_prev_amount_sum_log",
    "bilateral_prev_amount_mean_log",
    "bilateral_amount_vs_prevmean",
    "bilateral_time_span_hours",
    "own_prev_debit_count",
    "own_prev_credit_count",
    "own_debit_credit_count_ratio",
    "own_prev_debit_amount_sum_log",
    "own_prev_credit_amount_sum_log",
    "own_debit_credit_amount_logratio",
    "own_unique_counterparties_prev",
    "own_activity_total_prev",
    "own_flow_balance_abs_logratio",
]

COMPACT_EDGE_FEATURES = [
    "edge_tx_count",
    "bilateral_prev_amount_sum_log",
    "bilateral_prev_amount_mean_log",
    "bilateral_amount_vs_prevmean",
    "bilateral_time_span_hours",
]

COMPACT_NODE_FLOW_FEATURES = [
    "own_prev_debit_count",
    "own_prev_credit_count",
    "own_debit_credit_count_ratio",
    "own_prev_debit_amount_sum_log",
    "own_prev_credit_amount_sum_log",
    "own_debit_credit_amount_logratio",
    "own_unique_counterparties_prev",
    "own_activity_total_prev",
    "own_flow_balance_abs_logratio",
]

REPEATED_AMOUNT_FEATURES = [
    "same_amount_24h_cnt",
    "same_amount_counterparty_cnt",
    "amount_round10k",
    "amount_round100k",
]

CANDIDATE_WINDOW_AMOUNT_FEATURES = [
    "sender_tx_1h",
    "sender_tx_24h",
    "sender_amount_sum_24h_log",
    "sender_amount_mean_24h_log",
]

CANDIDATE_SAME_AMOUNT_SPREAD_FEATURES = [
    "same_amount_receivers_prev",
    "same_amount_receiver_banks_prev",
]

CANDIDATE_RECEIVER_BANK_BURST_FEATURES = [
    "receiver_bank_count_24h",
]

ACCOUNT_DECAY_FEATURES = [
    "own_decay_count_12h",
]

BEHAVIOR_SHIFT_FEATURES = [
    "sender_tx_1h",
    "sender_tx_24h",
    "sender_amount_sum_24h_log",
    "sender_tx_24h_vs_prev_mean",
    "sender_amount_24h_vs_prev_mean",
    "same_amount_24h_cnt",
    "same_amount_receivers_prev",
    "same_amount_receiver_banks_prev",
    "sender_format_ratio_prev",
    "counterparty_unique_24h_ratio",
]

BEHAVIOR_SHIFT_V2_FEATURES = [
    "sender_tx_1h",
    "sender_tx_24h",
    "sender_amount_sum_24h_log",
    "same_amount_24h_cnt",
    "same_amount_receivers_prev",
    "same_amount_receiver_banks_prev",
]

CANDIDATE_FEATURES = (
    CANDIDATE_WINDOW_AMOUNT_FEATURES
    + CANDIDATE_SAME_AMOUNT_SPREAD_FEATURES
    + CANDIDATE_RECEIVER_BANK_BURST_FEATURES
)

FEATURE_GROUPS = {
    "time": ["hour", "dayofweek"],
    "amount": ["log_amount_paid", "log_amount_received"],
    "currency_format": ["same_currency", "Payment Format_"],
    "internal_self": ["self_account", "is_sender", "internal_transfer"],
    "own_history": ["own_account_prev_count", "time_since_last_tx_own"],
    "counterparty_history": ["counterparty_prev_count", "time_since_last_tx_counterparty"],
    "burst": ["tx_24h", "new_cp_24h", "is_new_cp"],
    "amount_pattern": ["same_amount_cnt", "amt_vs_prevmean", "amount_round1k"],
    "new_all": list(NEW_FEATURES),
    "repeated_extended": REPEATED_AMOUNT_FEATURES,
    "candidate_window_amount": CANDIDATE_WINDOW_AMOUNT_FEATURES,
    "candidate_same_amount_spread": CANDIDATE_SAME_AMOUNT_SPREAD_FEATURES,
    "candidate_receiver_bank_burst": CANDIDATE_RECEIVER_BANK_BURST_FEATURES,
    "account_decay": ACCOUNT_DECAY_FEATURES,
    "behavior_shift": BEHAVIOR_SHIFT_FEATURES,
    "behavior_shift_v2": BEHAVIOR_SHIFT_V2_FEATURES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("transactions_csv", help="Path to HI-Small_Trans.csv.")
    parser.add_argument("patterns_txt", help="Path to HI-Small_Patterns.txt.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Use only the first N raw transactions for a quick smoke run.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.60,
        help="Chronological train fraction before chain adjustment.",
    )
    parser.add_argument(
        "--valid-frac",
        type=float,
        default=0.20,
        help="Chronological validation fraction before chain adjustment.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=DEFAULT_TARGET_RECALL,
        help="Illicit retention target used to choose the threshold on VALID.",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--feature-set",
        choices=[
            "old",
            "old_new",
            "old_same_amount",
            "old_amount_core",
            "old_repeated_extended",
            "compact_core",
            "compact_core_edge",
            "compact_core_node_flow",
            "compact_core_graph",
            "compact_core_window_amount",
            "compact_core_same_amount_spread",
            "compact_core_same_amount_spread_decay12h",
            "compact_core_receiver_bank_burst",
            "compact_core_candidates_all",
            "compact_core_behavior_shift",
            "compact_core_behavior_shift_v2",
            "top10_compact",
            "compact_h",
        ],
        default="old_new",
        help="Feature group to train with.",
    )
    parser.add_argument(
        "--drop-group",
        action="append",
        choices=sorted(FEATURE_GROUPS),
        default=[],
        help="Feature group to remove after selecting --feature-set. Can be used multiple times.",
    )
    parser.add_argument(
        "--drop-feature",
        action="append",
        default=[],
        help="Individual feature to remove after selecting --feature-set. Can be used multiple times.",
    )
    parser.add_argument(
        "--metrics-json",
        default=None,
        help="Optional path to save metrics as JSON.",
    )
    parser.add_argument(
        "--predictions-csv",
        default=None,
        help="Optional path to save test predictions.",
    )
    return parser.parse_args()


def chain_aware_buckets(
    raw: pd.DataFrame,
    chain_id: np.ndarray,
    train_frac: float,
    valid_frac: float,
) -> tuple[np.ndarray, np.datetime64, np.datetime64]:
    """Return per-transaction buckets: 0=train, 1=valid, 2=test."""
    if not (0.0 < train_frac < 1.0):
        raise ValueError("--train-frac must be in (0, 1)")
    if not (0.0 < valid_frac < 1.0):
        raise ValueError("--valid-frac must be in (0, 1)")
    if train_frac + valid_frac >= 1.0:
        raise ValueError("--train-frac + --valid-frac must be < 1")

    ts = raw["Timestamp"].to_numpy()
    t_train = np.datetime64(raw["Timestamp"].quantile(train_frac))
    t_valid = np.datetime64(raw["Timestamp"].quantile(train_frac + valid_frac))
    bucket = np.where(ts < t_train, 0, np.where(ts < t_valid, 1, 2))

    in_chain = chain_id >= 0
    if in_chain.any():
        chain_frame = pd.DataFrame({"chain_id": chain_id[in_chain], "ts": ts[in_chain]})
        chain_first_ts = chain_frame.groupby("chain_id")["ts"].transform("min").to_numpy()
        bucket[in_chain] = np.where(
            chain_first_ts < t_train,
            0,
            np.where(chain_first_ts < t_valid, 1, 2),
        )

    return bucket, t_train, t_valid


def add_compact_graph_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add compact causal local-graph summaries.

    These features summarize the local account and bilateral relationship history
    visible before the current transaction. They intentionally stay much smaller
    than the full Spark graph feature set.
    """
    feat_df = feat_df.copy()
    feat_df["local_amount"] = np.where(
        feat_df["role"].isin(["sender", "internal_debit"]),
        feat_df["Amount Paid"],
        feat_df["Amount Received"],
    )
    feat_df["is_debit_view"] = feat_df["role"].isin(["sender", "internal_debit"]).astype(np.int8)
    feat_df["is_credit_view"] = 1 - feat_df["is_debit_view"]

    s = feat_df.sort_values(["own_bank", "own_account", "Timestamp", "transaction_id"]).copy()
    own_keys = ["own_bank", "own_account"]
    own_group = s.groupby(own_keys, sort=False)

    debit_amount = s["local_amount"] * s["is_debit_view"]
    credit_amount = s["local_amount"] * s["is_credit_view"]
    s["own_prev_debit_count"] = own_group["is_debit_view"].cumsum() - s["is_debit_view"]
    s["own_prev_credit_count"] = own_group["is_credit_view"].cumsum() - s["is_credit_view"]
    s["own_prev_debit_amount_sum"] = debit_amount.groupby([s[k] for k in own_keys], sort=False).cumsum() - debit_amount
    s["own_prev_credit_amount_sum"] = credit_amount.groupby([s[k] for k in own_keys], sort=False).cumsum() - credit_amount
    s["own_debit_credit_count_ratio"] = np.log1p(s["own_prev_debit_count"]) - np.log1p(s["own_prev_credit_count"])
    s["own_prev_debit_amount_sum_log"] = np.log1p(s["own_prev_debit_amount_sum"])
    s["own_prev_credit_amount_sum_log"] = np.log1p(s["own_prev_credit_amount_sum"])
    s["own_debit_credit_amount_logratio"] = (
        s["own_prev_debit_amount_sum_log"] - s["own_prev_credit_amount_sum_log"]
    )
    s["own_activity_total_prev"] = s["own_prev_debit_count"] + s["own_prev_credit_count"]
    s["own_flow_balance_abs_logratio"] = s["own_debit_credit_amount_logratio"].abs()

    bilateral_keys = ["own_bank", "own_account", "counterpart_bank", "counterpart_account"]
    s["edge_tx_count"] = s.groupby(bilateral_keys, sort=False).cumcount()
    bilateral_amount_sum = s.groupby(bilateral_keys, sort=False)["local_amount"].cumsum() - s["local_amount"]
    s["bilateral_prev_amount_sum_log"] = np.log1p(bilateral_amount_sum)
    bilateral_mean = bilateral_amount_sum / s["edge_tx_count"].replace(0, np.nan)
    s["bilateral_prev_amount_mean_log"] = np.log1p(bilateral_mean.fillna(0))
    s["bilateral_amount_vs_prevmean"] = np.log1p(
        (s["local_amount"] / bilateral_mean).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(lower=0)
    )
    first_ts = s.groupby(bilateral_keys, sort=False)["Timestamp"].transform("first")
    s["bilateral_time_span_hours"] = (
        (s["Timestamp"] - first_ts).dt.total_seconds().fillna(0) / 3600.0
    )
    s.loc[s["edge_tx_count"] == 0, "bilateral_time_span_hours"] = 0.0

    first_bilateral = (s["edge_tx_count"] == 0).astype(np.int8)
    s["own_unique_counterparties_prev"] = (
        first_bilateral.groupby([s[k] for k in own_keys], sort=False).cumsum() - first_bilateral
    )

    return feat_df.join(s[COMPACT_GRAPH_FEATURES])


def add_repeated_amount_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add causal repeated-amount variants for structuring-style behavior."""
    feat_df = feat_df.copy()
    feat_df["amount_round10k"] = (feat_df["Amount Paid"] % 10000 == 0).astype(np.int8)
    feat_df["amount_round100k"] = (feat_df["Amount Paid"] % 100000 == 0).astype(np.int8)

    s = feat_df.sort_values(["own_bank", "own_account", "Amount Paid", "Timestamp", "transaction_id"])
    same_amount_keys = ["own_bank", "own_account", "Amount Paid"]
    same_amount_group = s.groupby(same_amount_keys, sort=False)
    s["same_amount_24h_cnt"] = same_amount_group.rolling(
        "24h", on="Timestamp"
    )["transaction_id"].count().to_numpy() - 1

    counterparty_amount_keys = [
        "own_bank",
        "own_account",
        "counterpart_bank",
        "counterpart_account",
        "Amount Paid",
    ]
    s["same_amount_counterparty_cnt"] = s.groupby(counterparty_amount_keys, sort=False).cumcount()

    return feat_df.join(s[["same_amount_24h_cnt", "same_amount_counterparty_cnt"]])


def _previous_unique_count_24h(
    timestamps: pd.Series,
    values: pd.Series,
) -> np.ndarray:
    """Count distinct values in the previous 24h for a time-sorted group."""
    window: deque[tuple[pd.Timestamp, object]] = deque()
    counts: Counter[object] = Counter()
    out = np.zeros(len(timestamps), dtype=np.float32)
    for i, (timestamp, value) in enumerate(zip(timestamps, values)):
        cutoff = timestamp - pd.Timedelta(hours=24)
        while window and window[0][0] < cutoff:
            _, old_value = window.popleft()
            counts[old_value] -= 1
            if counts[old_value] <= 0:
                del counts[old_value]
        out[i] = len(counts)
        window.append((timestamp, value))
        counts[value] += 1
    return out


def add_candidate_sender_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add candidate sender-only causal features not in compact_core."""
    feat_df = feat_df.copy()

    s = feat_df.sort_values(["own_bank", "own_account", "Timestamp", "transaction_id"]).copy()
    own_keys = ["own_bank", "own_account"]
    own_group = s.groupby(own_keys, sort=False)

    rolling_1h = own_group.rolling("1h", on="Timestamp")["transaction_id"].count().to_numpy()
    rolling_24h_count = own_group.rolling("24h", on="Timestamp")["transaction_id"].count().to_numpy()
    rolling_24h_amount = own_group.rolling("24h", on="Timestamp")["Amount Paid"].sum().to_numpy()

    s["sender_tx_1h"] = np.maximum(rolling_1h - 1, 0)
    s["sender_tx_24h"] = np.maximum(rolling_24h_count - 1, 0)
    amount_sum_prev = np.maximum(rolling_24h_amount - s["Amount Paid"].to_numpy(), 0)
    s["sender_amount_sum_24h_log"] = np.log1p(amount_sum_prev)
    amount_mean_prev = amount_sum_prev / pd.Series(s["sender_tx_24h"]).replace(0, np.nan)
    s["sender_amount_mean_24h_log"] = np.log1p(amount_mean_prev.fillna(0).to_numpy())

    amount_keys = ["own_bank", "own_account", "Amount Paid"]
    receiver_first = s.groupby(amount_keys + ["counterpart_account"], sort=False).cumcount() == 0
    receiver_bank_first = s.groupby(amount_keys + ["counterpart_bank"], sort=False).cumcount() == 0
    s["same_amount_receivers_prev"] = (
        receiver_first.astype(np.int8).groupby([s[k] for k in amount_keys], sort=False).cumsum()
        - receiver_first.astype(np.int8)
    )
    s["same_amount_receiver_banks_prev"] = (
        receiver_bank_first.astype(np.int8).groupby([s[k] for k in amount_keys], sort=False).cumsum()
        - receiver_bank_first.astype(np.int8)
    )

    receiver_bank_counts = []
    for _, group in s.groupby(own_keys, sort=False):
        receiver_bank_counts.append(
            pd.Series(
                _previous_unique_count_24h(group["Timestamp"], group["counterpart_bank"]),
                index=group.index,
            )
        )
    s["receiver_bank_count_24h"] = pd.concat(receiver_bank_counts).sort_index()

    return feat_df.join(s[CANDIDATE_FEATURES])


def _previous_exp_decay_count(
    timestamps: np.ndarray,
    half_life_seconds: float,
) -> np.ndarray:
    """Causal exponentially decayed previous-event count for sorted timestamps."""
    n = len(timestamps)
    out = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return out

    decay_rate = np.log(2.0) / half_life_seconds
    dt_seconds = np.diff(timestamps).astype("timedelta64[s]").astype(np.float64)
    decay = np.exp(-decay_rate * dt_seconds)
    for i in range(1, n):
        out[i] = decay[i - 1] * (out[i - 1] + 1.0)
    return out


def add_account_decay_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add causal recency-weighted own-account activity features."""
    s = feat_df.sort_values(["own_bank", "own_account", "Timestamp", "transaction_id"]).copy()
    gb_keys = s[["own_bank", "own_account"]].to_numpy()
    boundaries = np.concatenate(
        (
            [0],
            np.where((gb_keys[1:] != gb_keys[:-1]).any(axis=1))[0] + 1,
            [len(s)],
        )
    )

    timestamps = s["Timestamp"].to_numpy()
    out = np.empty(len(s), dtype=np.float64)
    for st, en in zip(boundaries[:-1], boundaries[1:]):
        out[st:en] = _previous_exp_decay_count(timestamps[st:en], 3600 * 12)

    s["own_decay_count_12h"] = out
    return feat_df.join(s[ACCOUNT_DECAY_FEATURES])


def add_behavior_shift_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Add compact causal features for recent sender behavior shifts."""
    feat_df = feat_df.copy()

    s = feat_df.sort_values(["own_bank", "own_account", "Timestamp", "transaction_id"]).copy()
    own_keys = ["own_bank", "own_account"]
    own_group = s.groupby(own_keys, sort=False)

    rolling_1h = own_group.rolling("1h", on="Timestamp")["transaction_id"].count().to_numpy()
    rolling_24h_count = own_group.rolling("24h", on="Timestamp")["transaction_id"].count().to_numpy()
    rolling_24h_amount = own_group.rolling("24h", on="Timestamp")["Amount Paid"].sum().to_numpy()

    s["sender_tx_1h"] = np.maximum(rolling_1h - 1, 0)
    s["sender_tx_24h"] = np.maximum(rolling_24h_count - 1, 0)
    amount_sum_prev = np.maximum(rolling_24h_amount - s["Amount Paid"].to_numpy(), 0)
    s["sender_amount_sum_24h_log"] = np.log1p(amount_sum_prev)

    first_time = own_group["Timestamp"].transform("first")
    elapsed_days = ((s["Timestamp"] - first_time).dt.total_seconds() / 86400.0).clip(lower=1.0)
    long_run_tx_per_24h = s["own_account_prev_count"].to_numpy() / elapsed_days.to_numpy()
    s["sender_tx_24h_vs_prev_mean"] = s["sender_tx_24h"].to_numpy() / (long_run_tx_per_24h + 1.0)

    prev_amount_sum = own_group["Amount Paid"].cumsum() - s["Amount Paid"]
    prev_amount_mean = prev_amount_sum / s["own_account_prev_count"].replace(0, np.nan)
    s["sender_amount_24h_vs_prev_mean"] = (
        amount_sum_prev / (prev_amount_mean.fillna(0).to_numpy() + 1.0)
    )

    amount_keys = ["own_bank", "own_account", "Amount Paid"]
    amount_group = s.groupby(amount_keys, sort=False)
    s["same_amount_24h_cnt"] = (
        amount_group.rolling("24h", on="Timestamp")["transaction_id"].count().to_numpy() - 1
    )

    receiver_first = s.groupby(amount_keys + ["counterpart_account"], sort=False).cumcount() == 0
    receiver_bank_first = s.groupby(amount_keys + ["counterpart_bank"], sort=False).cumcount() == 0
    s["same_amount_receivers_prev"] = (
        receiver_first.astype(np.int8).groupby([s[k] for k in amount_keys], sort=False).cumsum()
        - receiver_first.astype(np.int8)
    )
    s["same_amount_receiver_banks_prev"] = (
        receiver_bank_first.astype(np.int8).groupby([s[k] for k in amount_keys], sort=False).cumsum()
        - receiver_bank_first.astype(np.int8)
    )

    format_cols = [col for col in s.columns if col.startswith("Payment Format_")]
    if format_cols:
        s["_payment_format_value"] = s[format_cols].idxmax(axis=1)
    else:
        s["_payment_format_value"] = "unknown"
    format_count_prev = s.groupby(own_keys + ["_payment_format_value"], sort=False).cumcount()
    s["sender_format_ratio_prev"] = format_count_prev / (s["own_account_prev_count"] + 1.0)

    counterparty_counts = []
    for _, group in s.groupby(own_keys, sort=False):
        counterparty_counts.append(
            pd.Series(
                _previous_unique_count_24h(group["Timestamp"], group["counterpart_account"]),
                index=group.index,
            )
        )
    s["counterparty_unique_24h_ratio"] = (
        pd.concat(counterparty_counts).sort_index() / (s["sender_tx_24h"] + 1.0)
    )

    return feat_df.join(s[BEHAVIOR_SHIFT_FEATURES])


def choose_feature_columns(
    feat_df: pd.DataFrame,
    feature_set: str,
    drop_groups: list[str],
    drop_features: list[str],
) -> list[str]:
    old_cols = [col for col in BASE_FEATURES if col in feat_df.columns]
    old_cols += [col for col in feat_df.columns if col.startswith("Payment Format_")]
    new_cols = [col for col in NEW_FEATURES if col in feat_df.columns]
    compact_cols = [col for col in COMPACT_GRAPH_FEATURES if col in feat_df.columns]
    compact_edge_cols = [col for col in COMPACT_EDGE_FEATURES if col in feat_df.columns]
    compact_node_flow_cols = [col for col in COMPACT_NODE_FLOW_FEATURES if col in feat_df.columns]
    repeated_cols = [col for col in REPEATED_AMOUNT_FEATURES if col in feat_df.columns]
    candidate_window_amount_cols = [
        col for col in CANDIDATE_WINDOW_AMOUNT_FEATURES if col in feat_df.columns
    ]
    candidate_same_amount_spread_cols = [
        col for col in CANDIDATE_SAME_AMOUNT_SPREAD_FEATURES if col in feat_df.columns
    ]
    account_decay_cols = [col for col in ACCOUNT_DECAY_FEATURES if col in feat_df.columns]
    candidate_receiver_bank_burst_cols = [
        col for col in CANDIDATE_RECEIVER_BANK_BURST_FEATURES if col in feat_df.columns
    ]
    behavior_shift_cols = [col for col in BEHAVIOR_SHIFT_FEATURES if col in feat_df.columns]
    behavior_shift_v2_cols = [
        col for col in BEHAVIOR_SHIFT_V2_FEATURES if col in feat_df.columns
    ]
    candidate_cols = [col for col in CANDIDATE_FEATURES if col in feat_df.columns]
    top10_cols = [col for col in TOP10_COMPACT_FEATURES if col in feat_df.columns]

    compact_core_cols = [
        col
        for col in old_cols + [col for col in ["same_amount_cnt"] if col in feat_df.columns]
        if col not in set(FEATURE_GROUPS["time"] + FEATURE_GROUPS["internal_self"])
    ]

    if feature_set == "old":
        feature_cols = old_cols
    elif feature_set == "old_new":
        feature_cols = old_cols + new_cols
    elif feature_set == "old_same_amount":
        feature_cols = old_cols + [col for col in ["same_amount_cnt"] if col in feat_df.columns]
    elif feature_set == "old_amount_core":
        feature_cols = old_cols + [
            col for col in ["same_amount_cnt", "amount_round1k"] if col in feat_df.columns
        ]
    elif feature_set == "old_repeated_extended":
        feature_cols = (
            old_cols
            + [col for col in ["same_amount_cnt", "amount_round1k"] if col in feat_df.columns]
            + repeated_cols
        )
    elif feature_set == "compact_core":
        feature_cols = compact_core_cols
    elif feature_set == "compact_core_edge":
        feature_cols = compact_core_cols + compact_edge_cols
    elif feature_set == "compact_core_node_flow":
        feature_cols = compact_core_cols + compact_node_flow_cols
    elif feature_set == "compact_core_graph":
        feature_cols = compact_core_cols + compact_edge_cols + compact_node_flow_cols
    elif feature_set == "compact_core_window_amount":
        feature_cols = compact_core_cols + candidate_window_amount_cols
    elif feature_set == "compact_core_same_amount_spread":
        feature_cols = compact_core_cols + candidate_same_amount_spread_cols
    elif feature_set == "compact_core_same_amount_spread_decay12h":
        feature_cols = compact_core_cols + candidate_same_amount_spread_cols + account_decay_cols
    elif feature_set == "compact_core_receiver_bank_burst":
        feature_cols = compact_core_cols + candidate_receiver_bank_burst_cols
    elif feature_set == "compact_core_candidates_all":
        feature_cols = compact_core_cols + candidate_cols
    elif feature_set == "compact_core_behavior_shift":
        feature_cols = compact_core_cols + behavior_shift_cols
    elif feature_set == "compact_core_behavior_shift_v2":
        feature_cols = compact_core_cols + behavior_shift_v2_cols
    elif feature_set == "top10_compact":
        feature_cols = top10_cols
    elif feature_set == "compact_h":
        feature_cols = old_cols + new_cols + compact_cols
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    to_drop = set(drop_features)
    prefix_drops = []
    for group in drop_groups:
        for col in FEATURE_GROUPS[group]:
            if col.endswith("_"):
                prefix_drops.append(col)
            else:
                to_drop.add(col)
    return [
        col
        for col in feature_cols
        if col not in to_drop and not any(col.startswith(prefix) for prefix in prefix_drops)
    ]


def build_sender_features(
    raw: pd.DataFrame,
    feature_set: str,
    drop_groups: list[str],
    drop_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    feat_df = engineer_pooled_features(expand_to_local_observations(raw))
    feat_df = add_local_features(feat_df)
    if feature_set == "old_repeated_extended":
        feat_df = add_repeated_amount_features(feat_df)
    if feature_set in {
        "top10_compact",
        "compact_h",
        "compact_core_edge",
        "compact_core_node_flow",
        "compact_core_graph",
    }:
        feat_df = add_compact_graph_features(feat_df)
    feat_df = feat_df[feat_df["role"].isin(["sender", "internal_debit"])].reset_index(drop=True)
    if feature_set in {
        "compact_core_window_amount",
        "compact_core_same_amount_spread",
        "compact_core_same_amount_spread_decay12h",
        "compact_core_receiver_bank_burst",
        "compact_core_candidates_all",
    }:
        feat_df = add_candidate_sender_features(feat_df)
    if feature_set == "compact_core_same_amount_spread_decay12h":
        feat_df = add_account_decay_features(feat_df)
    if feature_set in {"compact_core_behavior_shift", "compact_core_behavior_shift_v2"}:
        feat_df = add_behavior_shift_features(feat_df)

    feature_cols = choose_feature_columns(feat_df, feature_set, drop_groups, drop_features)
    return feat_df, feature_cols


def make_matrix(feat_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return (
        feat_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype(np.float32)
        .to_numpy()
    )


def choose_threshold(y_valid: np.ndarray, proba_valid: np.ndarray, target_recall: float) -> float:
    _, recall, thresholds = precision_recall_curve(y_valid, proba_valid)
    valid_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.0
    return float(thresholds[valid_idx[-1]])


def build_retention_table(
    y_valid: np.ndarray,
    proba_valid: np.ndarray,
    y_test: np.ndarray,
    proba_test: np.ndarray,
    target_recalls: list[float],
    recommended_recall: float,
) -> list[dict[str, float | int | str]]:
    """Choose each threshold on VALID, then report the resulting TEST trade-off."""
    rows = []
    for target_recall in target_recalls:
        threshold = choose_threshold(y_valid, proba_valid, target_recall)
        kept = proba_test >= threshold
        illicit = y_test == 1
        illicit_kept = int((kept & illicit).sum())
        illicit_total = int(illicit.sum())
        illicit_missed = illicit_total - illicit_kept
        dropped = int((~kept).sum())
        remained = int(kept.sum())
        retained = illicit_kept / max(illicit_total, 1)
        label = f"{100 * target_recall:.1f}%"
        if np.isclose(target_recall, recommended_recall):
            label += " (recommended)"
        rows.append(
            {
                "target_retention": float(target_recall),
                "label": label,
                "threshold": threshold,
                "actual_illicit_retained": float(retained),
                "transactions_dropped_pct": float(100 * dropped / len(y_test)),
                "transactions_remained_count": remained,
                "illicit_missed_count": illicit_missed,
            }
        )
    return rows


def build_fixed_actual_recall_table(
    y_test: np.ndarray,
    proba_test: np.ndarray,
    target_recalls: list[float],
) -> list[dict[str, float | int | str]]:
    """Diagnostic table: choose thresholds on TEST to compare at matched recall."""
    rows = []
    _, recall, thresholds = precision_recall_curve(y_test, proba_test)
    for target_recall in target_recalls:
        valid_idx = np.where(recall[:-1] >= target_recall)[0]
        threshold = float(thresholds[valid_idx[-1]]) if len(valid_idx) else 0.0
        kept = proba_test >= threshold
        illicit = y_test == 1
        illicit_kept = int((kept & illicit).sum())
        illicit_total = int(illicit.sum())
        dropped = int((~kept).sum())
        rows.append(
            {
                "target_actual_retention": float(target_recall),
                "threshold_selected_on": "test",
                "threshold": threshold,
                "actual_illicit_retained": float(illicit_kept / max(illicit_total, 1)),
                "transactions_dropped_pct": float(100 * dropped / len(y_test)),
                "transactions_remained_count": int(kept.sum()),
                "illicit_missed_count": int(illicit_total - illicit_kept),
            }
        )
    return rows


def print_retention_table(rows: list[dict[str, float | int | str]]) -> None:
    print("\n=== TEST pre-filter trade-off table ===")
    print(
        f"{'Target illicit retained (%)':<30}"
        f"{'Actual illicit retained (%)':>32}"
        f"{'Transactions dropped (%)':>28}"
        f"{'Transactions remained (count)':>34}"
        f"{'Illicit missed (count)':>28}"
    )
    print("-" * 152)
    for row in rows:
        print(
            f"{str(row['label']):<30}"
            f"{100 * row['actual_illicit_retained']:>31.2f}%"
            f"{row['transactions_dropped_pct']:>27.2f}%"
            f"{row['transactions_remained_count']:>34,}"
            f"{row['illicit_missed_count']:>28,}"
        )


def print_fixed_actual_recall_table(rows: list[dict[str, float | int | str]]) -> None:
    print("\n=== TEST diagnostic table at fixed actual recall ===")
    print(
        f"{'Target actual retained (%)':<30}"
        f"{'Actual illicit retained (%)':>32}"
        f"{'Transactions dropped (%)':>28}"
        f"{'Transactions remained (count)':>34}"
        f"{'Illicit missed (count)':>28}"
    )
    print("-" * 152)
    for row in rows:
        label = f"{100 * row['target_actual_retention']:.1f}%"
        print(
            f"{label:<30}"
            f"{100 * row['actual_illicit_retained']:>31.2f}%"
            f"{row['transactions_dropped_pct']:>27.2f}%"
            f"{row['transactions_remained_count']:>34,}"
            f"{row['illicit_missed_count']:>28,}"
        )


def evaluate_split(
    split_name: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, float | int | str]:
    keep = proba >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, keep.astype(int), labels=[0, 1]).ravel()
    illicit = int(y_true.sum())
    total = int(len(y_true))
    kept = int(keep.sum())
    dropped = total - kept
    roc_auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, proba))
        pr_auc = float(average_precision_score(y_true, proba))
    return {
        "split": split_name,
        "rows": total,
        "illicit": illicit,
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "dropped": dropped,
        "drop_rate": float(dropped / total),
        "kept": kept,
        "kept_rate": float(kept / total),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "illicit_retention": float(tp / max(illicit, 1)),
        "precision_among_kept": float(tp / max(kept, 1)),
    }


def print_split_summary(name: str, mask: np.ndarray, y: np.ndarray) -> None:
    n = int(mask.sum())
    illicit = int(y[mask].sum())
    print(f"{name:<5}: {n:>10,} rows  illicit={illicit:>7,} ({100 * y[mask].mean():.4f}%)")


def main() -> None:
    started_at = time.perf_counter()
    args = parse_args()

    print(f"Loading {args.transactions_csv} ...", file=sys.stderr)
    raw = load(args.transactions_csv)
    if args.nrows is not None:
        raw = raw.iloc[: args.nrows].copy()

    print(f"Parsing laundering chains from {args.patterns_txt} ...", file=sys.stderr)
    key2chain, n_chains = parse_patterns(args.patterns_txt)
    chain_id = build_chain_ids(args.transactions_csv, key2chain, len(raw))[: len(raw)]
    matched = int((chain_id >= 0).sum())

    print(f"Building sender-only features: {args.feature_set} ...", file=sys.stderr)
    feat_df, feature_cols = build_sender_features(
        raw,
        args.feature_set,
        args.drop_group,
        args.drop_feature,
    )
    X = make_matrix(feat_df, feature_cols)
    y = feat_df["Is Laundering"].astype(int).to_numpy()

    bucket, t_train, t_valid = chain_aware_buckets(
        raw,
        chain_id,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
    )
    tb = bucket[feat_df["transaction_id"].to_numpy()]
    train_mask, valid_mask, test_mask = tb == 0, tb == 1, tb == 2

    print("\n=== Dataset ===")
    print(f"raw transactions       : {len(raw):,}")
    print(f"sender-only rows       : {len(feat_df):,}")
    print(f"matched chain txns     : {matched:,}")
    print(f"laundering chains      : {n_chains:,}")
    print(f"train cutoff           : {t_train}")
    print(f"valid cutoff           : {t_valid}")
    print(f"features               : {len(feature_cols)}")
    print_split_summary("train", train_mask, y)
    print_split_summary("valid", valid_mask, y)
    print_split_summary("test", test_mask, y)

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

    print(f"\nTraining XGBoost sender-only {args.feature_set} pre-filter ...", file=sys.stderr)
    model.fit(X[train_mask], y[train_mask])
    proba_valid = model.predict_proba(X[valid_mask])[:, 1]
    proba_test = model.predict_proba(X[test_mask])[:, 1]
    threshold = choose_threshold(y[valid_mask], proba_valid, args.target_recall)
    retention_targets = [0.999, 0.99, 0.98, 0.95, 0.90, 0.85]
    retention_table = build_retention_table(
        y[valid_mask],
        proba_valid,
        y[test_mask],
        proba_test,
        retention_targets,
        args.target_recall,
    )
    fixed_actual_recall_table = build_fixed_actual_recall_table(
        y[test_mask],
        proba_test,
        retention_targets,
    )

    valid_metrics = evaluate_split("valid", y[valid_mask], proba_valid, threshold)
    test_metrics = evaluate_split("test", y[test_mask], proba_test, threshold)
    metrics = {
        "protocol": {
            "model": "XGBClassifier",
            "view": "sender-only",
            "features": args.feature_set,
            "drop_groups": args.drop_group,
            "drop_features": args.drop_feature,
            "selected_feature_cols": feature_cols,
            "split": "chain-aware chronological train/valid/test",
            "target_recall": args.target_recall,
            "threshold_selected_on": "valid",
            "seed": args.seed,
        },
        "xgboost": {
            "max_depth": args.max_depth,
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
        },
        "data": {
            "transactions_csv": args.transactions_csv,
            "patterns_txt": args.patterns_txt,
            "raw_transactions": int(len(raw)),
            "sender_only_rows": int(len(feat_df)),
            "matched_chain_transactions": matched,
            "laundering_chains": int(n_chains),
            "train_cutoff": str(t_train),
            "valid_cutoff": str(t_valid),
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
        },
        "valid": valid_metrics,
        "test": test_metrics,
        "retention_table": retention_table,
        "test_fixed_actual_recall_table": fixed_actual_recall_table,
    }

    print_retention_table(retention_table)
    print_fixed_actual_recall_table(fixed_actual_recall_table)

    print("\n=== Recommended operating point details ===")
    print(f"Threshold selected on VALID : {threshold:.6f}")
    print(f"TEST ROC-AUC                : {test_metrics['roc_auc']:.4f}")
    print(f"TEST PR-AUC                 : {test_metrics['pr_auc']:.4f}")
    print(f"TEST illicit retention      : {100 * test_metrics['illicit_retention']:.2f}%")
    print(f"TEST precision among kept   : {100 * test_metrics['precision_among_kept']:.2f}%")
    print(
        f"TEST TP/FP/TN/FN            : {test_metrics['true_positive']:,}/"
        f"{test_metrics['false_positive']:,}/{test_metrics['true_negative']:,}/"
        f"{test_metrics['false_negative']:,}"
    )

    if args.predictions_csv:
        path = Path(args.predictions_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = feat_df.loc[test_mask, ["transaction_id", "Timestamp", "Is Laundering"]].copy()
        out["xgb_score"] = proba_test
        out["threshold"] = threshold
        out["kept_for_downstream"] = proba_test >= threshold
        out.to_csv(path, index=False)
        print(f"Saved test predictions CSV: {path}")

    elapsed_seconds = time.perf_counter() - started_at
    metrics["runtime"] = {
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60,
    }
    print("\n=== Runtime ===")
    print(f"Total elapsed time        : {elapsed_seconds:.2f}s ({elapsed_seconds / 60:.2f} min)")

    if args.metrics_json:
        path = Path(args.metrics_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nSaved metrics JSON: {path}")


if __name__ == "__main__":
    main()
