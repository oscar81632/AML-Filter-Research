"""Produce the per-transaction risk flag a bank ships with its PRISM batch.

This is the first step of the client-side flow: the bank scores its own transactions
with the M2 filter, using only its own plaintext, and emits a table saying which ones
the server should spend FHE inference on. Low-risk transactions are **flagged, not
dropped** -- they still get uploaded, so the graph the composer builds is complete and
the flagged transactions' cross-bank features are unaffected (measured in
`docs/m2-cascade-impact.md`; dropping instead costs 2.3 pp of recall).

Output columns:

  transaction_id   the bank's own reference for the payment
  is_kept          true  -> run FHE inference on this transaction
                   false -> low risk; still upload it (it is a graph edge), but skip FHE
  risk_score       the filter's probability, for auditing and threshold re-tuning

The threshold is chosen on a validation window at a target illicit-retention rate, never
on the window being scored.

Usage:
  # train on a history window and score a batch, in one pass over one CSV
  python scripts/filter_flag_table.py data/HI-Small_Trans.csv \\
      --patterns data/HI-Small_Patterns.txt --retention 0.95 \\
      --output artifacts/flags/hi-small_flags.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from filter_LR import BASE_FEATURES, engineer_pooled_features, expand_to_local_observations, load
from filter_LR_chainsplit import build_chain_ids, parse_patterns
from filter_LR_features_poc import NEW_FEATURES, add_local_features

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transactions", help="the bank's own transaction CSV")
    ap.add_argument("--patterns", default=None,
                    help="Patterns.txt; when given, laundering chains are kept whole "
                         "across the train/valid/score boundaries")
    ap.add_argument("--retention", type=float, default=0.95,
                    help="target share of illicit to keep (threshold picked on valid)")
    ap.add_argument("--output", type=Path,
                    default=REPO_ROOT / "artifacts/flags/flags.parquet")
    ap.add_argument("--id-offset", type=int, default=0,
                    help="added to the row index to form transaction_id; use 1 to match "
                         "the staging pipeline's numbering")
    ap.add_argument("--with-raw-keys", action="store_true",
                    help="also emit the raw (never anonymised) account/timestamp columns "
                         "so the table can be joined to the source data directly")
    args = ap.parse_args()
    from xgboost import XGBClassifier

    t0 = time.time()

    print(f"loading {args.transactions} ...", flush=True)
    raw = load(args.transactions)
    ts = raw["Timestamp"].to_numpy()

    # 60/20/20 by time: train the filter, pick the threshold, then score the newest window
    ts_sorted = np.sort(ts)
    t60 = ts_sorted[int(len(ts_sorted) * 0.60)]
    t80 = ts_sorted[int(len(ts_sorted) * 0.80)]
    bucket = np.where(ts < t60, 0, np.where(ts < t80, 1, 2))
    if args.patterns:
        chain_id = build_chain_ids(args.transactions, parse_patterns(args.patterns)[0], len(raw))
        m = chain_id >= 0
        dfc = pd.DataFrame({"cid": chain_id[m], "ts": ts[m]})
        first = dfc.groupby("cid")["ts"].transform("min").to_numpy()
        bucket[m] = np.where(first < t60, 0, np.where(first < t80, 1, 2))

    print("building bank-local features ...", flush=True)
    feat = add_local_features(engineer_pooled_features(expand_to_local_observations(raw)))
    feat = feat[feat["role"].isin(["sender", "internal_debit"])].reset_index(drop=True)
    tid = feat["transaction_id"].to_numpy()
    y = feat["Is Laundering"].astype(int).to_numpy()
    cols = ([c for c in BASE_FEATURES if c in feat.columns]
            + [c for c in feat.columns if c.startswith("Payment Format_")] + NEW_FEATURES)
    X = feat[cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    tb = bucket[tid]
    tr, va = tb == 0, tb == 1

    print(f"training filter on {tr.sum():,} rows ({len(cols)} bank-local features) ...", flush=True)
    spw = (y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)
    model = XGBClassifier(max_depth=2, n_estimators=300, learning_rate=0.1,
                          scale_pos_weight=spw, n_jobs=8, random_state=SEED,
                          tree_method="hist", eval_metric="logloss")
    model.fit(X[tr], y[tr])

    pv = model.predict_proba(X[va])[:, 1]
    illicit_va = pv[y[va] == 1]
    threshold = float(np.quantile(illicit_va, 1.0 - args.retention)) if len(illicit_va) else 0.0
    score = model.predict_proba(X)[:, 1]
    is_kept = score >= threshold

    table = pd.DataFrame({
        "transaction_id": tid + args.id_offset,
        "is_kept": is_kept,
        "risk_score": score.astype(np.float32),
    })
    if args.with_raw_keys:
        # Nothing in this filter anonymises anything -- OPRF pseudonymisation happens
        # later, in prism. These are the source account identifiers as-is, so the table
        # can be joined straight onto raw transactions or the feature parquets.
        src = raw.iloc[tid]
        table["staging_transaction_id"] = tid + 1  # the feature parquets number from 1
        for col in ("Timestamp", "From Bank", "From Account", "To Bank", "To Account"):
            table[col] = src[col].to_numpy()
    table = table.sort_values("transaction_id", ignore_index=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        table.to_csv(args.output, index=False)
    else:
        table.to_parquet(args.output, index=False)

    kept_illicit = int((is_kept & (y == 1)).sum())
    total_illicit = int((y == 1).sum())
    meta = {
        "rows": int(len(table)),
        "threshold": threshold,
        "target_retention": args.retention,
        "kept": int(is_kept.sum()),
        "kept_share": float(is_kept.mean()),
        "illicit_total": total_illicit,
        "illicit_kept": kept_illicit,
        "illicit_retention": kept_illicit / max(total_illicit, 1),
        "features": cols,
        "id_offset": args.id_offset,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nthreshold {threshold:.4f} (valid @ {args.retention:.0%} retention)")
    print(f"  flagged for FHE : {is_kept.sum():,} / {len(table):,} ({is_kept.mean():.2%})")
    print(f"  illicit retained: {kept_illicit:,} / {total_illicit:,} "
          f"({kept_illicit / max(total_illicit, 1):.2%})")
    print(f"\nwrote {args.output}")
    print(f"      {meta_path}")
    print(table.head(5).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
