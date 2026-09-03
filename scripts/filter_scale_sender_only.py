"""Memory-lean filter for large datasets (HI-Medium / HI-Large).

The pandas path in filter_LR.py loads the whole CSV with string columns and expands
每 transaction into 2 rows -- fine for HI-Small (5M txns) but far beyond RAM for
HI-Large (~190M txns). This version keeps everything in compact numpy arrays:

  * chunked CSV read, only the needed columns, parsed straight into int/float arrays
  * account keys hashed to uint64 (stable across chunks, no global dict)
  * each transaction is scored once from the paying-side decision point
  * account history can be selected by ``--history-mode``:
    ``bank_local`` includes previous incoming and outgoing transactions visible
    to the bank; ``strict_sender`` uses only previous outgoing sender activity
  * rolling 24h windows done with a composite-key searchsorted, not pandas rolling

By default this script uses the current compact mainline feature set:
``compact_core_same_amount_spread_decay12h``. The older 24-feature scale
baseline remains available through ``--feature-set scale24`` for comparison.

Usage:
  python scripts/filter_scale_sender_only.py data/HI-Medium_Trans.csv data/HI-Medium_Patterns.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_LR import CURRENCY_USD_RATE

CHUNK = 5_000_000
DAY = 86_400
SEED = 42
TARGETS = (0.999, 0.99, 0.98, 0.95, 0.92, 0.91, 0.90, 0.85)
CALIBRATION_TARGETS = (0.975, 0.970, 0.965, 0.960, 0.955, 0.950, 0.945, 0.940)
COLS = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account",
        "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
        "Payment Format", "Is Laundering"]
MAINLINE_FEATURES = [
    "log_amount_paid",
    "log_amount_received",
    "same_currency",
    "own_account_prev_count",
    "time_since_last_tx_own",
    "counterparty_prev_count",
    "time_since_last_tx_counterparty",
    "Payment Format_ACH",
    "Payment Format_Bitcoin",
    "Payment Format_Cash",
    "Payment Format_Cheque",
    "Payment Format_Credit Card",
    "Payment Format_Reinvestment",
    "Payment Format_Wire",
    "same_amount_cnt",
    "same_amount_receivers_prev",
    "same_amount_receiver_banks_prev",
    "own_decay_count_12h",
]
RECEIVER_HISTORY_FEATURES = [
    "receiver_account_prev_count",
    "time_since_last_tx_receiver",
    "receiver_decay_count_12h",
    "same_amount_senders_prev",
    "same_amount_sender_banks_prev",
]
FLOW_AMOUNT_FEATURES = [
    "own_amount_24h_sum",
    "own_amount_24h_mean",
    "counterparty_amount_24h_sum",
    "counterparty_amount_24h_mean",
    "receiver_amount_24h_sum",
    "receiver_amount_24h_mean",
]
PAIR_RECEIVER_FLOW_FEATURES = [
    "counterparty_amount_24h_sum",
    "counterparty_amount_24h_mean",
    "receiver_amount_24h_sum",
    "receiver_amount_24h_mean",
]
RECEIVER_FANIN_FLOW_FEATURES = [
    "receiver_account_prev_count",
    "time_since_last_tx_receiver",
    "receiver_unique_senders_prev",
    "receiver_unique_sender_banks_prev",
    "counterparty_amount_24h_sum",
    "counterparty_amount_24h_mean",
    "receiver_amount_24h_sum",
    "receiver_amount_24h_mean",
]
RECEIVER_BANK_ROLE_FEATURES = [
    "receiver_account_prev_count",
    "time_since_last_tx_receiver",
    "receiver_unique_senders_prev",
    "receiver_fanin_ratio",
    "receiver_bank_prev_count",
    "time_since_last_tx_receiver_bank",
    "receiver_bank_amount_24h_sum",
    "receiver_bank_amount_24h_mean",
    "receiver_bank_unique_senders_prev",
    "receiver_bank_fanin_ratio",
    "pair_to_sender_ratio",
    "same_amount_to_sender_ratio",
]
SCALE24_EXTRA_FEATURES = [
    "hour",
    "dayofweek",
    "internal_transfer",
    "self_account",
    "tx_24h",
    "new_cp_24h",
    "is_new_cp",
    "amt_vs_prevmean",
    "amount_round1k",
]


def _hash_keys(bank: pd.Series, acct: pd.Series) -> np.ndarray:
    return pd.util.hash_pandas_object(bank.astype(str) + "|" + acct.astype(str), index=False).to_numpy()


def _hash_values(values: pd.Series | np.ndarray) -> np.ndarray:
    return pd.util.hash_pandas_object(pd.Series(values), index=False).to_numpy()


def load_compact(path: str):
    """Chunked read -> compact arrays. Returns dict of numpy arrays."""
    parts = {
        k: []
        for k in (
            "ts",
            "src",
            "src_bank",
            "dst",
            "dst_bank",
            "paid",
            "paid_hash",
            "recv",
            "same_cur",
            "pay_cur",
            "recv_cur",
            "fmt",
            "y",
            "internal",
        )
    }
    fmt_codes: dict[str, int] = {}
    cur_codes: dict[str, int] = {}
    t0 = time.time()
    for i, ch in enumerate(pd.read_csv(path, header=0, names=COLS, chunksize=CHUNK,
                                       dtype={"From Bank": str, "From Account": str,
                                              "To Bank": str, "To Account": str})):
        # epoch seconds fit in int32 until 2038 -- halves every timestamp-sized array
        ts = (pd.to_datetime(ch["Timestamp"]).astype("int64").to_numpy() // 10**9).astype(np.int32)
        paid = (ch["Amount Paid"].to_numpy(np.float64)
                * ch["Payment Currency"].map(CURRENCY_USD_RATE).to_numpy(np.float64)).astype(np.float32)
        paid_hash = _hash_values(paid.astype(np.float64))
        recv = (ch["Amount Received"].to_numpy(np.float64)
                * ch["Receiving Currency"].map(CURRENCY_USD_RATE).to_numpy(np.float64)).astype(np.float32)
        for code in ch["Payment Format"].unique():
            fmt_codes.setdefault(code, len(fmt_codes))
        for code in pd.concat([ch["Payment Currency"], ch["Receiving Currency"]]).unique():
            cur_codes.setdefault(code, len(cur_codes))
        parts["ts"].append(ts)
        parts["src"].append(_hash_keys(ch["From Bank"], ch["From Account"]))
        parts["src_bank"].append(_hash_values(ch["From Bank"].astype(str)))
        parts["dst"].append(_hash_keys(ch["To Bank"], ch["To Account"]))
        parts["dst_bank"].append(_hash_values(ch["To Bank"].astype(str)))
        parts["paid"].append(paid)
        parts["paid_hash"].append(paid_hash)
        parts["recv"].append(recv)
        parts["same_cur"].append((ch["Payment Currency"] == ch["Receiving Currency"]).to_numpy(np.int8))
        parts["pay_cur"].append(ch["Payment Currency"].map(cur_codes).to_numpy(np.int8))
        parts["recv_cur"].append(ch["Receiving Currency"].map(cur_codes).to_numpy(np.int8))
        parts["fmt"].append(ch["Payment Format"].map(fmt_codes).to_numpy(np.int8))
        parts["y"].append(ch["Is Laundering"].to_numpy(np.int8))
        parts["internal"].append((ch["From Bank"] == ch["To Bank"]).to_numpy(np.int8))
        print(f"  chunk {i}: {len(ts):,} rows ({time.time()-t0:.0f}s)", flush=True)
    d = {k: np.concatenate(v) for k, v in parts.items()}
    d["fmt_codes"] = fmt_codes
    d["cur_codes"] = cur_codes
    print(f"loaded {len(d['ts']):,} transactions in {time.time()-t0:.0f}s", flush=True)
    return d


def group_stats(key: np.ndarray, ts: np.ndarray, window: int, extra: np.ndarray | None = None):
    """Per-row: prior count, seconds since previous, rolling-window count, rolling sum of `extra`.

    Sorted by (key, ts); windows use a composite key so searchsorted never crosses groups.
    """
    n = len(key)
    # int32 indices where the row count allows -- halves the several index-sized
    # temporaries, which dominate peak RSS on large datasets.
    ix = np.int32 if n < 2**31 - 1 else np.int64
    order = np.lexsort((ts, key)).astype(ix, copy=False)
    t = ts[order]
    # group boundaries without materialising the full sorted key array (saves 8 B/row
    # at peak on large inputs): compare consecutive sorted keys chunk-wise.
    new_grp = np.empty(n, bool)
    new_grp[0] = True
    step = 20_000_000
    for a in range(1, n, step):
        b = min(a + step, n)
        np.not_equal(key[order[a:b]], key[order[a - 1:b - 1]], out=new_grp[a:b])
    grp = (np.cumsum(new_grp) - 1).astype(ix, copy=False)
    idx = np.arange(n, dtype=ix)
    start = np.flatnonzero(new_grp).astype(ix, copy=False)
    prev_count = (idx - start[grp]).astype(np.float32)

    since = np.empty(n, np.float32)
    since[0] = -1
    since[1:] = (t[1:] - t[:-1]).astype(np.float32)
    since[new_grp] = -1
    del new_grp, start

    span = np.int64(t.max() + 1)
    comp = grp.astype(np.int64) * span
    left = np.searchsorted(comp + t, comp + (t - window), side="left").astype(ix, copy=False)
    del comp, grp
    win_count = (idx - left + 1).astype(np.float32)

    win_extra = None
    if extra is not None:
        cs = np.concatenate([[0], np.cumsum(extra[order].astype(np.int64))])
        win_extra = (cs[idx.astype(np.int64) + 1] - cs[left.astype(np.int64)]).astype(np.float32)
        del cs
    del idx, left, t

    inv = np.empty(n, ix)
    inv[order] = np.arange(n, dtype=ix)
    del order
    out = (prev_count[inv], since[inv], win_count[inv])
    del prev_count, since, win_count
    return out + ((win_extra[inv],) if extra is not None else ())


def combine_keys(a: np.ndarray, b: np.ndarray, multiplier: int) -> np.ndarray:
    return a ^ (b.astype(np.uint64, copy=False) * np.uint64(multiplier))


def previous_unique_count(group_key: np.ndarray, value_key: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Per-row count of distinct previous values within each time-sorted group."""
    n = len(group_key)
    ix = np.int32 if n < 2**31 - 1 else np.int64
    row = np.arange(n, dtype=ix)

    pair_order = np.lexsort((row, ts, value_key, group_key)).astype(ix, copy=False)
    first_pair = np.empty(n, bool)
    first_pair[0] = True
    step = 20_000_000
    for a in range(1, n, step):
        b = min(a + step, n)
        same_group = group_key[pair_order[a:b]] == group_key[pair_order[a - 1:b - 1]]
        same_value = value_key[pair_order[a:b]] == value_key[pair_order[a - 1:b - 1]]
        first_pair[a:b] = ~(same_group & same_value)
    is_first_value = np.zeros(n, np.int8)
    is_first_value[pair_order] = first_pair.astype(np.int8)
    del pair_order, first_pair

    group_order = np.lexsort((row, ts, group_key)).astype(ix, copy=False)
    new_group = np.empty(n, bool)
    new_group[0] = True
    for a in range(1, n, step):
        b = min(a + step, n)
        np.not_equal(group_key[group_order[a:b]], group_key[group_order[a - 1:b - 1]], out=new_group[a:b])
    grp = (np.cumsum(new_group) - 1).astype(ix, copy=False)
    idx = np.arange(n, dtype=ix)
    start = np.flatnonzero(new_group).astype(ix, copy=False)
    flags = is_first_value[group_order].astype(np.int32)
    cs = np.concatenate([[0], np.cumsum(flags)])
    out_sorted = (cs[idx.astype(np.int64)] - cs[start[grp].astype(np.int64)]).astype(np.float32)
    out = np.empty(n, np.float32)
    out[group_order] = out_sorted
    return out


def previous_exp_decay_count(key: np.ndarray, ts: np.ndarray, half_life_seconds: float) -> np.ndarray:
    """Causal exponentially decayed previous-event count within each sorted group."""
    n = len(key)
    ix = np.int32 if n < 2**31 - 1 else np.int64
    order = np.lexsort((ts, key)).astype(ix, copy=False)
    t = ts[order]
    new_group = np.empty(n, bool)
    new_group[0] = True
    step = 20_000_000
    for a in range(1, n, step):
        b = min(a + step, n)
        np.not_equal(key[order[a:b]], key[order[a - 1:b - 1]], out=new_group[a:b])

    decay_rate = np.log(2.0) / half_life_seconds
    out_sorted = np.zeros(n, np.float32)
    boundaries = np.concatenate(([0], np.flatnonzero(new_group)[1:], [n]))
    for st, en in zip(boundaries[:-1], boundaries[1:]):
        if en - st <= 1:
            continue
        dt = np.diff(t[st:en]).astype(np.float32)
        decay = np.exp(-decay_rate * dt)
        acc = 0.0
        for i in range(st + 1, en):
            acc = float(decay[i - st - 1]) * (acc + 1.0)
            out_sorted[i] = acc

    out = np.empty(n, np.float32)
    out[order] = out_sorted
    return out


def build_features(d, feature_set: str, history_mode: str):
    """Paying-side feature matrix (one row per transaction).

    The prediction row is paying-side only. With ``bank_local`` history,
    own-account history is computed from incoming and outgoing local account
    observations before keeping the paying-side prediction rows. With
    ``strict_sender`` history, own-account history uses only previous outgoing
    sender activity.
    """
    ts, src, dst = d["ts"], d["src"], d["dst"]
    n = len(ts)
    t0 = time.time()

    if history_mode == "bank_local":
        # Bank-local own-account history over both incoming and outgoing
        # observations, then keep the paying-side rows as prediction rows.
        both_key = np.concatenate([src, dst])
        both_ts = np.concatenate([ts, ts])
        own_prev_all, own_since_all, own_tx24_all = group_stats(both_key, both_ts, DAY)
        own_prev, own_since, own_tx24 = own_prev_all[:n], own_since_all[:n], own_tx24_all[:n]
        del both_key, both_ts, own_prev_all, own_since_all, own_tx24_all
        print(f"  bank-local own-account history {time.time()-t0:.0f}s", flush=True)
    elif history_mode == "strict_sender":
        own_prev, own_since, own_tx24 = group_stats(src, ts, DAY)
        print(f"  strict sender own-account history {time.time()-t0:.0f}s", flush=True)
    else:
        raise ValueError(f"Unsupported history mode: {history_mode}")

    # bilateral (own, counterparty) history, paying side
    t0 = time.time()
    pair = src ^ (dst * np.uint64(0x9E3779B97F4A7C15))
    cp_prev, cp_since, _ = group_stats(pair, ts, DAY)
    is_new_cp = (cp_prev == 0).astype(np.int8)
    del pair
    print(f"  bilateral history {time.time()-t0:.0f}s", flush=True)

    new_cp24 = None
    if feature_set in {"scale24", "mainline_plus_scale24"}:
        # 24h new-counterparty burst (fan-out) on the paying account
        t0 = time.time()
        _, _, _, new_cp24 = group_stats(src, ts, DAY, extra=is_new_cp)
        print(f"  fan-out window {time.time()-t0:.0f}s", flush=True)

    # repeated identical amount, and amount vs the account's past mean
    t0 = time.time()
    amt_key = combine_keys(src, d["paid_hash"], 0xC2B2AE3D27D4EB4F)
    same_amt, _, _ = group_stats(amt_key, ts, DAY)
    amt_vs_mean = None
    if feature_set in {"scale24", "mainline_plus_scale24"}:
        order = np.lexsort((ts, src))
        p = d["paid"][order].astype(np.float64)
        k = src[order]
        new_grp = np.empty(n, bool)
        new_grp[0] = True
        np.not_equal(k[1:], k[:-1], out=new_grp[1:])
        grp = np.cumsum(new_grp) - 1
        idx = np.arange(n, dtype=np.int64)
        start = np.flatnonzero(new_grp)
        cs = np.concatenate([[0.0], np.cumsum(p)])
        prev_sum = cs[idx] - cs[start[grp]]
        prev_cnt = idx - start[grp]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(prev_cnt > 0, p / (prev_sum / np.maximum(prev_cnt, 1)), 1.0)
        ratio = np.log1p(np.clip(np.nan_to_num(ratio, nan=1.0, posinf=1e6), 0, None)).astype(np.float32)
        inv = np.empty(n, np.int64)
        inv[order] = idx
        amt_vs_mean = ratio[inv]
    print(f"  amount features {time.time()-t0:.0f}s", flush=True)

    same_amount_receivers_prev = None
    same_amount_receiver_banks_prev = None
    same_amount_senders_prev = None
    same_amount_sender_banks_prev = None
    own_decay_count_12h = None
    receiver_prev = None
    receiver_since = None
    receiver_decay_count_12h = None
    uses_mainline = feature_set in {
        "mainline_decay12h",
        "mainline_plus_scale24",
        "mainline_receiver_history",
        "mainline_currency",
        "mainline_light_context",
        "mainline_flow_amounts",
        "mainline_pair_receiver_flow",
        "mainline_receiver_fanin_flow",
        "mainline_receiver_bank_role",
    }
    if uses_mainline:
        t0 = time.time()
        same_amount_receivers_prev = previous_unique_count(amt_key, dst, ts)
        same_amount_receiver_banks_prev = previous_unique_count(amt_key, d["dst_bank"], ts)
        print(f"  same-amount receiver spread {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        if history_mode == "bank_local":
            both_key = np.concatenate([src, dst])
            both_ts = np.concatenate([ts, ts])
            decay_all = previous_exp_decay_count(both_key, both_ts, 3600 * 12)
            own_decay_count_12h = decay_all[:n]
            del both_key, both_ts, decay_all
            print(f"  bank-local decay12h {time.time()-t0:.0f}s", flush=True)
        elif history_mode == "strict_sender":
            own_decay_count_12h = previous_exp_decay_count(src, ts, 3600 * 12)
            print(f"  strict sender decay12h {time.time()-t0:.0f}s", flush=True)
    if feature_set == "mainline_receiver_history":
        t0 = time.time()
        receiver_prev, receiver_since, _ = group_stats(dst, ts, DAY)
        print(f"  receiver account history {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        receiver_decay_count_12h = previous_exp_decay_count(dst, ts, 3600 * 12)
        print(f"  receiver decay12h {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        same_amount_senders_prev = previous_unique_count(amt_key, src, ts)
        same_amount_sender_banks_prev = previous_unique_count(amt_key, d["src_bank"], ts)
        print(f"  same-amount sender spread {time.time()-t0:.0f}s", flush=True)
    own_amt24 = None
    pair_amt24 = None
    pair_tx24 = None
    recv_amt24 = None
    recv_tx24 = None
    receiver_unique_senders_prev = None
    receiver_unique_sender_banks_prev = None
    receiver_bank_prev = None
    receiver_bank_since = None
    receiver_bank_tx24 = None
    receiver_bank_amt24 = None
    receiver_bank_unique_senders_prev = None
    if feature_set in {
        "mainline_flow_amounts",
        "mainline_pair_receiver_flow",
        "mainline_receiver_fanin_flow",
        "mainline_receiver_bank_role",
    }:
        t0 = time.time()
        if feature_set == "mainline_flow_amounts":
            if history_mode == "bank_local":
                both_key = np.concatenate([src, dst])
                both_ts = np.concatenate([ts, ts])
                both_paid = np.concatenate([d["paid"], d["paid"]])
                _, _, _, own_amt24_all = group_stats(both_key, both_ts, DAY, extra=both_paid)
                own_amt24 = own_amt24_all[:n]
                del both_key, both_ts, both_paid, own_amt24_all
            else:
                _, _, _, own_amt24 = group_stats(src, ts, DAY, extra=d["paid"])
            print(f"  own amount-flow window {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        pair = src ^ (dst * np.uint64(0x9E3779B97F4A7C15))
        _, _, pair_tx24, pair_amt24 = group_stats(pair, ts, DAY, extra=d["paid"])
        del pair
        print(f"  pair amount-flow window {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        receiver_prev, receiver_since, recv_tx24, recv_amt24 = group_stats(dst, ts, DAY, extra=d["paid"])
        print(f"  receiver amount-flow window {time.time()-t0:.0f}s", flush=True)
    if feature_set in {"mainline_receiver_fanin_flow", "mainline_receiver_bank_role"}:
        t0 = time.time()
        receiver_unique_senders_prev = previous_unique_count(dst, src, ts)
        receiver_unique_sender_banks_prev = previous_unique_count(dst, d["src_bank"], ts)
        print(f"  receiver fan-in unique history {time.time()-t0:.0f}s", flush=True)
    if feature_set == "mainline_receiver_bank_role":
        t0 = time.time()
        receiver_bank_prev, receiver_bank_since, receiver_bank_tx24, receiver_bank_amt24 = group_stats(
            d["dst_bank"], ts, DAY, extra=d["paid"]
        )
        print(f"  receiver-bank amount window {time.time()-t0:.0f}s", flush=True)

        t0 = time.time()
        receiver_bank_unique_senders_prev = previous_unique_count(d["dst_bank"], src, ts)
        print(f"  receiver-bank unique sender history {time.time()-t0:.0f}s", flush=True)
    del amt_key

    cols = {
        "log_amount_paid": np.log1p(d["paid"]),
        "log_amount_received": np.log1p(d["recv"]),
        "same_currency": d["same_cur"].astype(np.float32),
        "own_account_prev_count": own_prev,
        "time_since_last_tx_own": own_since,
        "counterparty_prev_count": cp_prev,
        "time_since_last_tx_counterparty": cp_since,
        "same_amount_cnt": same_amt,
    }
    for name, code in d["fmt_codes"].items():
        cols[f"Payment Format_{name}"] = (d["fmt"] == code).astype(np.float32)
    currency_feature_names = []
    if feature_set in {"mainline_currency", "mainline_light_context"}:
        for name, code in d["cur_codes"].items():
            pay_name = f"Payment Currency_{name}"
            recv_name = f"Receiving Currency_{name}"
            cols[pay_name] = (d["pay_cur"] == code).astype(np.float32)
            cols[recv_name] = (d["recv_cur"] == code).astype(np.float32)
            currency_feature_names.extend([pay_name, recv_name])

    if uses_mainline:
        cols["same_amount_receivers_prev"] = same_amount_receivers_prev
        cols["same_amount_receiver_banks_prev"] = same_amount_receiver_banks_prev
        cols["own_decay_count_12h"] = own_decay_count_12h
        names = [name for name in MAINLINE_FEATURES if name in cols]
    if feature_set == "mainline_receiver_history":
        cols["receiver_account_prev_count"] = receiver_prev
        cols["time_since_last_tx_receiver"] = receiver_since
        cols["receiver_decay_count_12h"] = receiver_decay_count_12h
        cols["same_amount_senders_prev"] = same_amount_senders_prev
        cols["same_amount_sender_banks_prev"] = same_amount_sender_banks_prev
        names = [name for name in MAINLINE_FEATURES + RECEIVER_HISTORY_FEATURES if name in cols]
    if feature_set in {"scale24", "mainline_plus_scale24"}:
        # hour / day-of-week arithmetically from epoch seconds (UTC).
        cols.update(
            {
                "hour": ((ts // 3600) % 24).astype(np.float32),
                "dayofweek": (((ts // 86400) + 3) % 7).astype(np.float32),
                "internal_transfer": d["internal"].astype(np.float32),
                "self_account": ((src == dst) & (d["internal"] == 1)).astype(np.float32),
                "tx_24h": own_tx24,
                "new_cp_24h": new_cp24,
                "is_new_cp": is_new_cp.astype(np.float32),
                "amt_vs_prevmean": amt_vs_mean,
                "amount_round1k": (np.mod(d["paid"], 1000) == 0).astype(np.float32),
            }
        )
    if feature_set == "mainline_plus_scale24":
        names = [name for name in MAINLINE_FEATURES + SCALE24_EXTRA_FEATURES if name in cols]
    elif feature_set in {"mainline_currency", "mainline_light_context"}:
        names = [name for name in MAINLINE_FEATURES + currency_feature_names if name in cols]
        if feature_set == "mainline_light_context":
            cols.update(
                {
                    "hour": ((ts // 3600) % 24).astype(np.float32),
                    "dayofweek": (((ts // 86400) + 3) % 7).astype(np.float32),
                    "internal_transfer": d["internal"].astype(np.float32),
                    "amount_round1k": (np.mod(d["paid"], 1000) == 0).astype(np.float32),
                    "amount_round10k": (np.mod(d["paid"], 10000) == 0).astype(np.float32),
                    "amount_round100k": (np.mod(d["paid"], 100000) == 0).astype(np.float32),
                }
            )
            names += [
                "hour",
                "dayofweek",
                "internal_transfer",
                "amount_round1k",
                "amount_round10k",
                "amount_round100k",
            ]
    elif feature_set == "mainline_flow_amounts":
        cols["own_amount_24h_sum"] = np.log1p(np.maximum(own_amt24, 0))
        cols["own_amount_24h_mean"] = np.log1p(np.maximum(own_amt24, 0) / np.maximum(own_tx24, 1))
        cols["counterparty_amount_24h_sum"] = np.log1p(np.maximum(pair_amt24, 0))
        cols["counterparty_amount_24h_mean"] = np.log1p(np.maximum(pair_amt24, 0) / np.maximum(pair_tx24, 1))
        cols["receiver_amount_24h_sum"] = np.log1p(np.maximum(recv_amt24, 0))
        cols["receiver_amount_24h_mean"] = np.log1p(np.maximum(recv_amt24, 0) / np.maximum(recv_tx24, 1))
        names = [name for name in MAINLINE_FEATURES + FLOW_AMOUNT_FEATURES if name in cols]
    elif feature_set == "mainline_pair_receiver_flow":
        cols["counterparty_amount_24h_sum"] = np.log1p(np.maximum(pair_amt24, 0))
        cols["counterparty_amount_24h_mean"] = np.log1p(np.maximum(pair_amt24, 0) / np.maximum(pair_tx24, 1))
        cols["receiver_amount_24h_sum"] = np.log1p(np.maximum(recv_amt24, 0))
        cols["receiver_amount_24h_mean"] = np.log1p(np.maximum(recv_amt24, 0) / np.maximum(recv_tx24, 1))
        names = [name for name in MAINLINE_FEATURES + PAIR_RECEIVER_FLOW_FEATURES if name in cols]
    elif feature_set == "mainline_receiver_fanin_flow":
        cols["receiver_account_prev_count"] = receiver_prev
        cols["time_since_last_tx_receiver"] = receiver_since
        cols["receiver_unique_senders_prev"] = receiver_unique_senders_prev
        cols["receiver_unique_sender_banks_prev"] = receiver_unique_sender_banks_prev
        cols["counterparty_amount_24h_sum"] = np.log1p(np.maximum(pair_amt24, 0))
        cols["counterparty_amount_24h_mean"] = np.log1p(np.maximum(pair_amt24, 0) / np.maximum(pair_tx24, 1))
        cols["receiver_amount_24h_sum"] = np.log1p(np.maximum(recv_amt24, 0))
        cols["receiver_amount_24h_mean"] = np.log1p(np.maximum(recv_amt24, 0) / np.maximum(recv_tx24, 1))
        names = [name for name in MAINLINE_FEATURES + RECEIVER_FANIN_FLOW_FEATURES if name in cols]
    elif feature_set == "mainline_receiver_bank_role":
        cols["receiver_account_prev_count"] = receiver_prev
        cols["time_since_last_tx_receiver"] = receiver_since
        cols["receiver_unique_senders_prev"] = receiver_unique_senders_prev
        cols["receiver_fanin_ratio"] = receiver_unique_senders_prev / np.maximum(receiver_prev, 1)
        cols["receiver_bank_prev_count"] = receiver_bank_prev
        cols["time_since_last_tx_receiver_bank"] = receiver_bank_since
        cols["receiver_bank_amount_24h_sum"] = np.log1p(np.maximum(receiver_bank_amt24, 0))
        cols["receiver_bank_amount_24h_mean"] = np.log1p(
            np.maximum(receiver_bank_amt24, 0) / np.maximum(receiver_bank_tx24, 1)
        )
        cols["receiver_bank_unique_senders_prev"] = receiver_bank_unique_senders_prev
        cols["receiver_bank_fanin_ratio"] = receiver_bank_unique_senders_prev / np.maximum(receiver_bank_prev, 1)
        cols["pair_to_sender_ratio"] = cp_prev / np.maximum(own_prev, 1)
        cols["same_amount_to_sender_ratio"] = same_amt / np.maximum(own_prev, 1)
        names = [name for name in MAINLINE_FEATURES + RECEIVER_BANK_ROLE_FEATURES if name in cols]
    elif feature_set == "scale24":
        names = list(cols)
    elif feature_set not in {"mainline_decay12h", "mainline_receiver_history"}:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    X = np.empty((n, len(names)), np.float32)
    for j, nm in enumerate(names):
        X[:, j] = cols[nm]
    return X, names


def chain_buckets(ts, y, patterns_path, csv_path):
    """60/20/20 chronological buckets, chain-aware via Patterns.txt line matching."""
    t60, t80 = np.quantile(ts, [0.60, 0.80]).astype(np.int64)
    bucket = np.where(ts < t60, 0, np.where(ts < t80, 1, 2)).astype(np.int8)
    # map pattern lines -> chain id, then to transactions by (ts,from,to,amount) key
    key2chain, chain = {}, -1
    idxs = [0, 1, 2, 3, 4, 7]
    with open(patterns_path) as f:
        for line in f:
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                chain += 1
            elif line.startswith("END") or not line.strip():
                continue
            else:
                p = line.rstrip("\n").split(",")
                key2chain["|".join(p[i] for i in idxs)] = chain
    # Chunked: holding all 6 key columns as Python strings would need ~60 GB on HI-Large.
    cid_parts = []
    for s in pd.read_csv(csv_path, dtype=str, header=0, names=list(range(11)),
                         usecols=idxs, chunksize=CHUNK):
        key = s[idxs[0]].str.cat([s[i] for i in idxs[1:]], sep="|")
        cid_parts.append(key.map(key2chain).fillna(-1).astype(np.int32).to_numpy())
        del s, key
    cid = np.concatenate(cid_parts)
    del cid_parts
    m = cid >= 0
    if m.any():
        dfc = pd.DataFrame({"cid": cid[m], "ts": ts[m]})
        first = dfc.groupby("cid")["ts"].transform("min").to_numpy()
        bucket[m] = np.where(first < t60, 0, np.where(first < t80, 1, 2))
    print(f"  chain-matched {int(m.sum()):,} txns to {chain+1} chains", flush=True)
    return bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transactions_csv")
    parser.add_argument("patterns_txt")
    parser.add_argument(
        "--feature-set",
        choices=[
            "mainline_decay12h",
            "mainline_receiver_history",
            "mainline_currency",
            "mainline_light_context",
            "mainline_flow_amounts",
            "mainline_pair_receiver_flow",
            "mainline_receiver_fanin_flow",
            "mainline_receiver_bank_role",
            "mainline_plus_scale24",
            "scale24",
        ],
        default="mainline_decay12h",
        help="mainline_decay12h matches compact_core_same_amount_spread_decay12h.",
    )
    parser.add_argument(
        "--history-mode",
        choices=["bank_local", "strict_sender"],
        default="bank_local",
        help=(
            "bank_local uses incoming+outgoing account history visible to the bank; "
            "strict_sender uses only previous outgoing sender activity."
        ),
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=20_000_000)
    parser.add_argument("--metrics-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    csv, patterns = args.transactions_csv, args.patterns_txt
    metrics_json = args.metrics_json
    wall = time.time()
    print(f"=== {Path(csv).name} | {args.feature_set} | {args.history_mode} ===", flush=True)
    d = load_compact(csv)
    n = len(d["ts"])
    y = d["y"].astype(np.int8)
    print(f"transactions {n:,}  illicit {int(y.sum()):,} ({100*y.mean():.4f}%)", flush=True)

    ts_all = d["ts"]
    t0 = time.time(); X, names = build_features(d, args.feature_set, args.history_mode); feat_t = time.time() - t0
    for k in (
        "src", "src_bank", "dst", "dst_bank", "paid", "paid_hash", "recv",
        "same_cur", "pay_cur", "recv_cur", "fmt", "internal",
    ):
        d.pop(k, None)
    import gc; gc.collect()
    print(f"features {X.shape} in {feat_t:.0f}s", flush=True)

    b = chain_buckets(ts_all, y, patterns, csv)
    tr, va, te = b == 0, b == 1, b == 2
    print(f"train {tr.sum():,} / valid {va.sum():,} / test {te.sum():,}", flush=True)

    # Cap training rows: keep every illicit row plus a sample of normals. Keeps XGB's
    # memory bounded on large datasets without changing the operating-point behaviour.
    tr_idx = np.flatnonzero(tr)
    if len(tr_idx) > args.max_train_rows:
        ill = tr_idx[y[tr_idx] == 1]
        nor = tr_idx[y[tr_idx] == 0]
        rng = np.random.default_rng(SEED)
        nor = rng.choice(nor, size=args.max_train_rows - len(ill), replace=False)
        tr_idx = np.concatenate([ill, nor])
        print(f"train subsampled to {len(tr_idx):,} rows ({len(ill):,} illicit)", flush=True)
    del tr

    spw = (y[tr_idx] == 0).sum() / max((y[tr_idx] == 1).sum(), 1)
    t0 = time.time()
    m = XGBClassifier(max_depth=args.max_depth, n_estimators=args.n_estimators, learning_rate=args.learning_rate, scale_pos_weight=spw,
                      n_jobs=-1, random_state=SEED, tree_method="hist", eval_metric="logloss")
    m.fit(X[tr_idx], y[tr_idx])
    fit_t = time.time() - t0
    pv, pt = m.predict_proba(X[va])[:, 1], m.predict_proba(X[te])[:, 1]
    yt = y[te]; tot = int(yt.sum()); ntest = len(yt)
    print(f"\nXGB(d{args.max_depth}) {len(names)} feats | fit {fit_t:.0f}s | TEST {ntest:,}/{tot:,} illicit"
          f" | ROC-AUC {roc_auc_score(yt,pt):.4f} | PR-AUC {average_precision_score(yt,pt):.4f}")
    _, rv, thv = precision_recall_curve(y[va], pv)
    rows = []
    print(f"{'target retain':>13} | {'recall(test)':>12} | {'drop %':>8} | {'remained':>12} | {'missed':>8}")
    for target in TARGETS:
        i = np.where(rv[:-1] >= target)[0]
        if len(i) == 0 or i[-1] >= len(thv):
            continue
        t = thv[i[-1]]
        kept_mask = pt >= t
        kept = int(kept_mask.sum())
        illk = int((kept_mask & (yt == 1)).sum())
        row = {
            "target_retention": float(target),
            "threshold": float(t),
            "actual_illicit_retained": float(illk / max(tot, 1)),
            "transactions_dropped_pct": float(100 * (ntest - kept) / ntest),
            "transactions_remained_count": kept,
            "illicit_missed_count": int(tot - illk),
        }
        rows.append(row)
        print(
            f"{target:>12.1%} | {illk/tot:>11.1%} | "
            f"{100*(ntest-kept)/ntest:>7.2f}% | {kept:>12,} | {tot-illk:>8,}"
        )
    calibration_rows = []
    print(f"\n{'valid target':>13} | {'recall(test)':>12} | {'drop %':>8} | {'remained':>12} | {'missed':>8} | {'threshold':>10}")
    for target in CALIBRATION_TARGETS:
        i = np.where(rv[:-1] >= target)[0]
        if len(i) == 0 or i[-1] >= len(thv):
            continue
        t = thv[i[-1]]
        kept_mask = pt >= t
        kept = int(kept_mask.sum())
        illk = int((kept_mask & (yt == 1)).sum())
        row = {
            "valid_retention_target": float(target),
            "threshold": float(t),
            "actual_illicit_retained": float(illk / max(tot, 1)),
            "transactions_dropped_pct": float(100 * (ntest - kept) / ntest),
            "transactions_remained_count": kept,
            "illicit_missed_count": int(tot - illk),
        }
        calibration_rows.append(row)
        print(
            f"{target:>12.1%} | {illk/tot:>11.2%} | "
            f"{100*(ntest-kept)/ntest:>7.2f}% | {kept:>12,} | {tot-illk:>8,} | {t:>10.4g}"
        )
    order = np.argsort(pt)[::-1]
    sorted_y = yt[order]
    csum_ill = np.cumsum(sorted_y == 1)
    fixed_actual_rows = []
    print(f"\n{'actual retain':>13} | {'drop %':>8} | {'remained':>12} | {'missed':>8} | {'threshold':>10}")
    for target in TARGETS:
        need = int(np.ceil(target * max(tot, 1)))
        if need <= 0 or need > tot:
            continue
        pos = int(np.searchsorted(csum_ill, need, side="left"))
        threshold = float(pt[order[pos]])
        kept_mask = pt >= threshold
        kept = int(kept_mask.sum())
        illk = int((kept_mask & (yt == 1)).sum())
        row = {
            "actual_retention_target": float(target),
            "threshold": threshold,
            "actual_illicit_retained": float(illk / max(tot, 1)),
            "transactions_dropped_pct": float(100 * (ntest - kept) / ntest),
            "transactions_remained_count": kept,
            "illicit_missed_count": int(tot - illk),
        }
        fixed_actual_rows.append(row)
        print(
            f"{target:>12.1%} | {100*(ntest-kept)/ntest:>7.2f}% | "
            f"{kept:>12,} | {tot-illk:>8,} | {threshold:>10.4g}"
        )
    total_wall = time.time() - wall
    print(f"\nTOTAL wall {total_wall:.0f}s")
    if metrics_json:
        out = {
            "script": "filter_scale_sender_only.py",
            "feature_set": args.feature_set,
            "history_mode": args.history_mode,
            "transactions_csv": csv,
            "patterns_txt": patterns,
            "features": names,
            "feature_count": len(names),
            "raw_transactions": int(n),
            "illicit_transactions": int(y.sum()),
            "train_rows": int(tr_idx.size),
            "valid_rows": int(va.sum()),
            "test_rows": int(te.sum()),
            "test_illicit": int(tot),
            "xgboost": {
                "max_depth": args.max_depth,
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_train_rows": args.max_train_rows,
            },
            "roc_auc": float(roc_auc_score(yt, pt)),
            "pr_auc": float(average_precision_score(yt, pt)),
            "retention_table": rows,
            "calibration_table": calibration_rows,
            "fixed_actual_recall_table": fixed_actual_rows,
            "feature_seconds": float(feat_t),
            "fit_seconds": float(fit_t),
            "elapsed_seconds": float(total_wall),
        }
        path = Path(metrics_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"saved metrics JSON: {path}")


if __name__ == "__main__":
    main()
