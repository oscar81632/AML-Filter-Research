import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, confusion_matrix, roc_auc_score


COLUMN_NAMES = [
    "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]

# FX rates to USD (from the main pipeline's staging, _legacy_prepare_input_small.py).
# The raw CSV uses full currency names; the pipeline dict uses codes -- mapped here.
# ~1% of transactions are cross-currency; without this, amounts are not comparable
# across currencies (e.g. 1M Yen vs 1M USD), and the filter's amount features would
# be currency-naive (unlike the main pipeline, which converts to USD).
CURRENCY_USD_RATE = {
    "US Dollar": 1.0,
    "Euro": 1.171783425225877,
    "Yuan": 0.14930721887033868,
    "Yen": 0.009487665410827868,
    "Swiss Franc": 1.0928961554056371,
    "Shekel": 0.29612081311363503,
    "Canadian Dollar": 0.7579775434031815,
    "Rupee": 0.013615817231245796,
    "Australian Dollar": 0.7078143121927827,
    "Saudi Riyal": 0.2665884611958837,
    "Ruble": 0.012852809604990688,
    "UK Pound": 1.2916554735187644,
    "Bitcoin": 11879.132698717296,
    "Mexican Peso": 0.047296753463246695,
    "Brazil Real": 0.1771008654705292,
}

BASE_FEATURES = [
    "hour", "dayofweek", "log_amount_paid", "log_amount_received",
    "same_currency", "self_account", "is_sender", "internal_transfer",
    "own_account_prev_count", "time_since_last_tx_own",
    "counterparty_prev_count", "time_since_last_tx_counterparty",
]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = COLUMN_NAMES
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    # Convert amounts to USD so amount-based features are comparable across currencies.
    df["Amount Paid"] = df["Amount Paid"] * df["Payment Currency"].map(CURRENCY_USD_RATE)
    df["Amount Received"] = df["Amount Received"] * df["Receiving Currency"].map(CURRENCY_USD_RATE)
    # transaction_id = the original transaction's row position. Both row
    # expanded from this transaction (sender/ receiver siblings)
    # carries this same id, so train/test splitting can be done by grouping
    # on transaction_id instead of by individual expanded row to prevent leakage.

    df.insert(0, "transaction_id", np.arange(len(df)))
    return df


def expand_to_local_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Turn each transaction into two local observation rows."""

    common_cols = ["transaction_id", "Timestamp", "Amount Paid", "Amount Received",
                    "Payment Currency", "Receiving Currency", "Payment Format", "Is Laundering"]

    internal_mask = (df["From Bank"] == df["To Bank"])
    cross = df.loc[~internal_mask]
    internal = df.loc[internal_mask]

    sender = cross[common_cols].copy()
    sender["role"] = "sender"
    sender["own_bank"] = cross["From Bank"]
    sender["own_account"] = cross["From Account"]
    sender["counterpart_bank"] = cross["To Bank"]
    sender["counterpart_account"] = cross["To Account"]

    receiver = cross[common_cols].copy()
    receiver["role"] = "receiver"
    receiver["own_bank"] = cross["To Bank"]
    receiver["own_account"] = cross["To Account"]
    receiver["counterpart_bank"] = cross["From Bank"]
    receiver["counterpart_account"] = cross["From Account"]

    internal_debit = internal[common_cols].copy()
    internal_debit["role"] = "internal_debit"
    internal_debit["own_bank"] = internal["From Bank"]
    internal_debit["own_account"] = internal["From Account"]
    internal_debit["counterpart_bank"] = internal["To Bank"]
    internal_debit["counterpart_account"] = internal["To Account"]

    internal_credit = internal[common_cols].copy()
    internal_credit["role"] = "internal_credit"
    internal_credit["own_bank"] = internal["To Bank"]
    internal_credit["own_account"] = internal["To Account"]
    internal_credit["counterpart_bank"] = internal["From Bank"]
    internal_credit["counterpart_account"] = internal["From Account"]

    long_df = pd.concat([sender, receiver, internal_debit, internal_credit], ignore_index=True)
    return long_df


def engineer_pooled_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering for the pooled long-format dataframe. Every
    feature is scoped to what the row's OWN bank could see for its OWN
    account, or its OWN bilateral history with the specific counterparty --
    never a global, cross-bank statistic.
    """
    df = long_df.sort_values(["Timestamp", "transaction_id"]).reset_index(drop=True).copy()

    df["hour"] = df["Timestamp"].dt.hour
    df["dayofweek"] = df["Timestamp"].dt.dayofweek

    df["log_amount_paid"] = np.log1p(df["Amount Paid"])
    df["log_amount_received"] = np.log1p(df["Amount Received"])

    df["same_currency"] = (df["Payment Currency"] == df["Receiving Currency"]).astype(int)
    # Treat the paying side (sender or internal_debit) as the "sending" side.
    df["is_sender"] = df["role"].isin(["sender", "internal_debit"]).astype(int)
    df["internal_transfer"] = (df["own_bank"] == df["counterpart_bank"]).astype(int)
    df["self_account"] = (
        (df["own_account"] == df["counterpart_account"]) & df["internal_transfer"].astype(bool)
    ).astype(int)
    # Causal aggregates are scoped by column lists below.

    # Own-account history: what this row's bank can see for its own account.
    df["own_account_prev_count"] = df.groupby(["own_bank", "own_account"]).cumcount()
    df["time_since_last_tx_own"] = (
        df.groupby(["own_bank", "own_account"])["Timestamp"]
        .diff()
        .dt.total_seconds()
        .fillna(-1)
    )

    # Bilateral history: must include both sides so different banks' separate
    # relationships with the same external counterparty are not merged.
    df["counterparty_prev_count"] = df.groupby(
        ["own_bank", "own_account", "counterpart_bank", "counterpart_account"]
    ).cumcount()
    df["time_since_last_tx_counterparty"] = (
        df.groupby(["own_bank", "own_account", "counterpart_bank", "counterpart_account"])[
            "Timestamp"
        ]
        .diff()
        .dt.total_seconds()
        .fillna(-1)
    )

    df = pd.get_dummies(df, columns=["Payment Format"], drop_first=False)
    return df


def report_curve(y_true, proba, n_total, target_recalls, label):
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    print(f"\n=== [{label}] Share filtered out vs. illicit retention rate ===")
    print(f"{'target recall':>14} | {'threshold':>10} | {'dropped':>10} | {'drop %':>8}")
    for target in target_recalls:
        idx = np.where(recall[:-1] >= target)[0]
        if len(idx) == 0:
            continue
        thr = thresholds[idx[-1]]
        dropped = int((proba < thr).sum())
        print(f"{target:>14.3f} | {thr:>10.3f} | {dropped:>10,} | {100*dropped/n_total:>7.2f}%")
    return recall, thresholds


def chosen_operating_point(y_true, proba, thresholds, recall, target_recall, n_total, label):
    valid_idx = np.where(recall[:-1] >= target_recall)[0]
    if len(valid_idx) == 0:
        threshold = 0.0
        print(f"\nWARNING [{label}]: could not reach target recall {target_recall}; falling back to threshold=0.", file=sys.stderr)
    else:
        threshold = thresholds[valid_idx[-1]]

    keep_mask = proba >= threshold
    pred_label = keep_mask.astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred_label, labels=[0, 1]).ravel()
    achieved_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    print(f"\n=== [{label}] Chosen operating point (target recall : {target_recall}) ===")
    print(f"Threshold                                 : {threshold:.3f}")
    print(f"Rows/txns (total)                         : {n_total:,}")
    print(f"Dropped (confident clean)                 : {(~keep_mask).sum():,} ({100*(~keep_mask).mean():.2f}%)")
    print(f"Kept (flagged / mixed)                    : {keep_mask.sum():,}")
    print(f"Illicit wrongly dropped (false negatives) : {fn:,} ")
    print(f"Illicit kept (true positives)             : {tp:,} / {int(np.sum(y_true)):,}")
    return threshold


TRAIN_FRAC = 0.7
TARGET_RECALL = 0.95
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to a *_Trans.csv file")
    args = parser.parse_args()

    print(f"Loading {args.input} ...", file=sys.stderr)
    raw = load(args.input)
    total_rows = len(raw)
    total_illicit = int(raw["Is Laundering"].sum())
    n_internal = int((raw["From Bank"] == raw["To Bank"]).sum())
    n_cross = total_rows - n_internal
    print(f"Original transactions: {total_rows:,} ({n_cross:,} cross-bank, {n_internal:,} internal)  Illicit: {total_illicit:,} ({100*total_illicit/total_rows:.4f}%)")

    print("Expanding: one transaction -> sender-side + receiver-side (2 rows)", file=sys.stderr)
    long_raw = expand_to_local_observations(raw)
    feat_df = engineer_pooled_features(long_raw)
    y = feat_df["Is Laundering"].astype(int)

    feature_cols = [c for c in BASE_FEATURES if c in feat_df.columns] + \
                    [c for c in feat_df.columns if c.startswith("Payment Format_")]
    X = feat_df[feature_cols].fillna(0).astype(np.float32)

    cutoff = raw["Timestamp"].quantile(TRAIN_FRAC)
    txn_is_train = (raw["Timestamp"] < cutoff).to_numpy()  # indexed by transaction_id (0..n-1)
    train_mask = txn_is_train[feat_df["transaction_id"].to_numpy()]
    test_mask = ~train_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    txn_id_test = feat_df.loc[test_mask, "transaction_id"].to_numpy()

    print(f"\n=== Chronological train/test split by transaction_id (train_frac={TRAIN_FRAC}, cutoff={cutoff}) ===")
    print(f"Train: {train_mask.sum():,} expanded rows, {y_train.sum():,} illicit ({100*y_train.mean():.4f}%)")
    print(f"Test : {test_mask.sum():,} expanded rows, {y_test.sum():,} illicit ({100*y_test.mean():.4f}%)")

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED)),
    ])


    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]
    print(f"\n=== Feature contributions (logreg, {len(feature_cols)} features) ===")
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=feature_cols).sort_values(key=abs, ascending=False)
    print(coefs.to_string())

    target_recalls = [0.999, 0.99, 0.98, 0.95, 0.90, 0.80]

    # ---------- row-level Eval ----------
    n_rows = len(proba_test)
    auc_row = roc_auc_score(y_test, proba_test)
    print(f"\nTEST ROC-AUC (row-level)         : {auc_row:.4f}")
    recall, thresholds = report_curve(y_test.to_numpy(), proba_test, n_rows, target_recalls, "row-level")
    chosen_operating_point(y_test.to_numpy(), proba_test, thresholds, recall, TARGET_RECALL, n_rows, "row-level")

    # ---------- transaction-level Eval (OR: max proba across sender and receiver) ----------
    tmp = pd.DataFrame({"transaction_id": txn_id_test, "proba": proba_test, "label": y_test.to_numpy()})
    txn_level = tmp.groupby("transaction_id").agg(proba=("proba", "max"), label=("label", "max")).reset_index()

    n_txns = len(txn_level)
    auc_txn = roc_auc_score(txn_level["label"], txn_level["proba"])
    print(f"\nTEST ROC-AUC (transaction-level) : {auc_txn:.4f}")
    recall2, thresholds2 = report_curve(txn_level["label"].to_numpy(), txn_level["proba"].to_numpy(), n_txns, target_recalls, "transaction-level")
    chosen_operating_point(txn_level["label"].to_numpy(), txn_level["proba"].to_numpy(), thresholds2, recall2, TARGET_RECALL, n_txns, "transaction-level")


if __name__ == "__main__":
    main()
