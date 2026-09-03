# Inference Flag Table Contract

This repository produces the ML prefilter output only. Bank-sim or PRISM should handle
OPRF, encryption, v5 batch construction, upload, and ingestion.

## Producer

```bash
python scripts/filter_flag_table.py data/HI-Small_Trans.csv \
  --patterns data/HI-Small_Patterns.txt \
  --retention 0.95 \
  --output artifacts/flags/hi-small_flags.parquet \
  --with-raw-keys
```

## Required Output Columns

| Column | Type | Meaning |
| --- | --- | --- |
| `transaction_id` | integer | Row-level transaction reference used by the ML filter. |
| `is_kept` | boolean | `true` means run downstream FHE inference; `false` means low risk and can skip FHE inference. |
| `risk_score` | float | XGBoost probability score used for auditing and threshold tuning. |

## Optional Join Columns

When `--with-raw-keys` is enabled, the output also includes source transaction fields so
the table can be joined back to local bank data or to a snapshot-building pipeline:

| Column |
| --- |
| `staging_transaction_id` |
| `Timestamp` |
| `From Bank` |
| `From Account` |
| `To Bank` |
| `To Account` |

## Companion Metadata

The script writes a sibling JSON file, for example:

```text
artifacts/flags/hi-small_flags.meta.json
```

It records the threshold, target retention, actual retained share, feature names, and
other audit information.

## Example

See:

```text
examples/inference_flags_sample.csv
examples/inference_flags_sample.meta.json
```

These are illustrative contract examples, not benchmark outputs.
