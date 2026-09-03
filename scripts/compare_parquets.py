"""
Lightweight parity check: reference parquets vs newly generated parquets.

Usage:
    python scripts/compare_parquets.py \
        --ref  /path/to/reference/features/hi-small \
        --new  features/hi-small
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


SPLITS = ("train", "valid", "test")
LABEL_COL = "is_laundering"


def load(directory: Path, split: str) -> pd.DataFrame:
    path = directory / f"{split}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def sample_hash(df: pd.DataFrame, n: int = 200, seed: int = 42) -> str:
    sample = df.sample(min(n, len(df)), random_state=seed).sort_index()
    raw = sample.to_csv(index=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def compare(ref_dir: Path, new_dir: Path) -> bool:
    all_ok = True
    rows: list[dict] = []

    for split in SPLITS:
        try:
            ref = load(ref_dir, split)
            new = load(new_dir, split)
        except FileNotFoundError as e:
            print(f"[MISSING] {e}")
            all_ok = False
            continue

        checks = {}

        checks["rows_ref"] = len(ref)
        checks["rows_new"] = len(new)
        checks["rows_match"] = len(ref) == len(new)

        checks["cols_ref"] = len(ref.columns)
        checks["cols_new"] = len(new.columns)
        checks["cols_count_match"] = len(ref.columns) == len(new.columns)

        ref_cols = set(ref.columns)
        new_cols = set(new.columns)
        missing_in_new = sorted(ref_cols - new_cols)
        extra_in_new = sorted(new_cols - ref_cols)
        checks["cols_name_match"] = (missing_in_new == [] and extra_in_new == [])
        if missing_in_new:
            checks["missing_in_new"] = missing_in_new
        if extra_in_new:
            checks["extra_in_new"] = extra_in_new

        if LABEL_COL in ref.columns and LABEL_COL in new.columns:
            checks["label_ref"] = ref[LABEL_COL].value_counts().to_dict()
            checks["label_new"] = new[LABEL_COL].value_counts().to_dict()
            checks["label_match"] = checks["label_ref"] == checks["label_new"]

        checks["nulls_ref"] = int(ref.isnull().sum().sum())
        checks["nulls_new"] = int(new.isnull().sum().sum())

        checks["sample_hash_ref"] = sample_hash(ref)
        checks["sample_hash_new"] = sample_hash(new)
        checks["sample_hash_match"] = checks["sample_hash_ref"] == checks["sample_hash_new"]

        split_ok = all(v for k, v in checks.items() if k.endswith("_match"))
        all_ok = all_ok and split_ok
        rows.append({"split": split, "ok": split_ok, **checks})

    _print_report(rows)
    return all_ok


def _print_report(rows: list[dict]) -> None:
    for r in rows:
        split = r["split"]
        ok_tag = "OK" if r["ok"] else "FAIL"
        print(f"\n=== {split.upper()} [{ok_tag}] ===")
        print(f"  rows       : ref={r.get('rows_ref','?')}  new={r.get('rows_new','?')}  match={r.get('rows_match','?')}")
        print(f"  cols       : ref={r.get('cols_ref','?')}  new={r.get('cols_new','?')}  match={r.get('cols_count_match','?')}")
        print(f"  col names  : match={r.get('cols_name_match','?')}")
        if r.get("missing_in_new"):
            print(f"    missing in new: {r['missing_in_new']}")
        if r.get("extra_in_new"):
            print(f"    extra in new:   {r['extra_in_new']}")
        if "label_ref" in r:
            print(f"  labels     : ref={r['label_ref']}  new={r['label_new']}  match={r.get('label_match','?')}")
        print(f"  nulls      : ref={r.get('nulls_ref','?')}  new={r.get('nulls_new','?')}")
        print(f"  sample hash: ref={r.get('sample_hash_ref','?')}  new={r.get('sample_hash_new','?')}  match={r.get('sample_hash_match','?')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight parquet parity check")
    parser.add_argument("--ref", required=True, help="Reference parquet directory")
    parser.add_argument("--new", default="features/hi-small", help="Newly generated parquet directory")
    args = parser.parse_args()

    ref_dir = Path(args.ref)
    new_dir = Path(args.new)
    print(f"Reference : {ref_dir.resolve()}")
    print(f"New       : {new_dir.resolve()}")

    ok = compare(ref_dir, new_dir)
    print(f"\n{'PARITY PASS' if ok else 'PARITY FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
