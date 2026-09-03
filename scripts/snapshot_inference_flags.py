"""Publish the M2 filter's decisions as a PRISM `inference_flags` artifact.

The v2 snapshot contract has a slot for a client's selection decision and no producer for
it: every flag in the sample handoff is `true` under `no_selection_filter_v0`, which the
reproduce notes describe as "no filter ran" rather than a fabricated threshold. Our filter
is what fills that slot, so this converts its per-transaction table into the four columns
the contract names.

The join is `tx_ref_hash`, and it is the part that needs care. The hash function is
PRISM's (`hash_hex("tx-ref-v1", ...)`, matched byte for byte against
`crates/utils/src/digest.rs`), but the **preimage is bank-authored** -- the sample uses
bank-sim's `batch_id:index`, the composer's tests use `src:dst:start:amount`, and ours
uses the payment's own durable fields. Two banks therefore never share a preimage, and a snapshot
produced by someone else's tooling cannot be keyed by our hashes at all. That is a
property of the contract, not a bug, so this refuses to emit an empty artifact quietly:
below --min-match it reports what it found and stops.

Flags are only ever emitted for edges this bank reports. Decision 8 of the format's design
notes settles that: an edge is built from exactly one report, so the published flag is
that reporter's, and the suppressed side's decision is never sealed. The contract's
`inference_flags == edges` row count therefore applies to the assembled snapshot, after
the composer merges every bank's slice -- not to one bank's contribution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent

SCHEMA = pa.schema([
    ("edge_id", pa.string()),
    ("inference_flag", pa.bool_()),
    ("flag_source_bank", pa.string()),
    ("selection_policy_id", pa.string()),
])

TX_REF_SCHEME = "ibm_aml_hi_small_v1"
TX_REF_FIELDS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Paid",
    "Payment Currency",
    "Amount Received",
    "Receiving Currency",
    "Payment Format",
]


def selection_policy_id(meta: dict | None = None, override: str | None = None) -> str:
    if override:
        return override
    retention = (meta or {}).get("target_retention")
    return f"m2_lr_p{int(round(retention * 100))}_v1" if retention else "m2_lr_v1"


def tx_ref_hash(source_ref: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"tx-ref-v1")
    digest.update(source_ref.encode())
    return f"sha256:{digest.hexdigest()}"


def read_raw(path: str) -> pd.DataFrame:
    cols = [
        "Timestamp",
        "From Bank",
        "From Account",
        "To Bank",
        "To Account",
        "Amount Received",
        "Receiving Currency",
        "Amount Paid",
        "Payment Currency",
        "Payment Format",
        "Is Laundering",
    ]
    raw = pd.read_csv(
        path,
        header=0,
        names=cols,
        dtype={"From Bank": str, "From Account": str, "To Bank": str, "To Account": str},
    )
    raw.insert(0, "transaction_id", range(len(raw)))
    return raw


def payment_references(raw: pd.DataFrame) -> pd.Series:
    fields = raw[TX_REF_FIELDS].astype(str)
    occurrence = fields.groupby(TX_REF_FIELDS, sort=False).cumcount()
    ref = pd.Series(TX_REF_SCHEME, index=raw.index)
    for col in TX_REF_FIELDS:
        ref = ref + "|" + fields[col]
    return ref + "|" + occurrence.astype(str)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True, help="snapshot directory")
    ap.add_argument("--flags", type=Path,
                    default=REPO_ROOT / "artifacts/flags/hi-small_flags.parquet")
    ap.add_argument("--bank", required=True,
                    help="this bank's snapshot code, e.g. bank010 -- flags are emitted "
                         "only for edges it reports")
    ap.add_argument("--policy-id", default=None)
    ap.add_argument("--raw", type=Path, default=REPO_ROOT / "data/HI-Small_Trans.csv",
                    help="source CSV the flag table indexes into; tx_ref_hash preimages "
                         "are built from each payment's own fields, not its row position")
    ap.add_argument("--min-match", type=float, default=0.99,
                    help="fraction of this bank's edges that must join; below it the run "
                         "stops rather than sealing a partial artifact")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    edges = pd.read_parquet(args.snapshot / "edges.parquet",
                            columns=["edge_id", "tx_ref_hash", "reporting_bank"])
    mine = edges[edges.reporting_bank == args.bank]
    print(f"snapshot: {len(edges):,} edges, {len(mine):,} reported by {args.bank}",
          flush=True)
    if mine.empty:
        raise SystemExit(
            f"no edge in this snapshot is reported by {args.bank!r}. Reporting banks "
            f"present: {sorted(edges.reporting_bank.unique())[:10]}"
        )

    flags = pd.read_parquet(args.flags)
    meta_path = args.flags.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    print(f"filter table: {len(flags):,} rows, {int(flags.is_kept.sum()):,} kept "
          f"({flags.is_kept.mean():.2%})", flush=True)

    # The flag table is keyed by row position in the source CSV, but a reference must be
    # a property of the payment, so the CSV is read to recover the payments those
    # positions name.
    refs = payment_references(read_raw(str(args.raw))).map(tx_ref_hash).to_numpy()
    ids = flags.transaction_id.to_numpy()
    if ids.max() >= len(refs):
        raise SystemExit(
            f"flag table references transaction_id {ids.max():,} but {args.raw} has only "
            f"{len(refs):,} rows -- the table and the CSV are not the same extract.")
    flags = flags.assign(tx_ref_hash=refs[ids])
    joined = mine.merge(flags[["tx_ref_hash", "is_kept"]], on="tx_ref_hash", how="left")
    matched = joined.is_kept.notna()
    rate = float(matched.mean())
    print(f"joined on tx_ref_hash: {int(matched.sum()):,}/{len(joined):,} ({rate:.2%})",
          flush=True)

    if rate < args.min_match:
        raise SystemExit(
            f"\nonly {rate:.2%} of {args.bank}'s edges joined, below --min-match "
            f"{args.min_match:.0%}.\n\n"
            "The likely cause is the tx_ref_hash preimage, which each bank authors: this "
            "snapshot was produced by tooling that built the preimage differently (the "
            "sample handoff uses bank-sim's `batch_id:index`; this filter uses the source "
            "row index). Flags cannot be attached to a snapshot whose row locators the "
            "bank did not mint. Regenerate the snapshot from this bank's own uploads, or "
            "align the preimage on both sides before publishing."
        )

    pid = selection_policy_id(meta, args.policy_id)
    out = pd.DataFrame({
        "edge_id": joined.edge_id,
        "inference_flag": joined.is_kept.fillna(False).astype(bool),
        "flag_source_bank": args.bank,
        "selection_policy_id": pid,
    }).sort_values("edge_id", ignore_index=True)

    # the contract's per-file invariants: sorted and unique on the documented key
    assert out.edge_id.is_unique, "edge_id must be unique"
    assert out.edge_id.is_monotonic_increasing, "edge_id must be sorted"

    dest = args.out or (args.snapshot / f"inference_flags_{args.bank}.parquet")
    pq.write_table(pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False), dest)
    print(f"\nwrote {dest}\n  {len(out):,} rows   policy {pid}\n"
          f"  flagged for inference: {int(out.inference_flag.sum()):,} "
          f"({out.inference_flag.mean():.2%})", flush=True)


if __name__ == "__main__":
    main()
