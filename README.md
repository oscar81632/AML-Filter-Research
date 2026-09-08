# AML Filter Research

This repository contains research code for a high-recall AML transaction
prefilter and an FHE-compatible AML inference pipeline. The prefilter is designed
to remove clearly low-risk transactions before they enter the expensive Fully
Homomorphic Encryption (FHE) stage, while retaining nearly all illicit
transactions for downstream encrypted inference.

## Highlights

- **Problem:** FHE inference is privacy-preserving but expensive. A lightweight
  plaintext prefilter can reduce the number of transactions that require FHE
  evaluation.
- **Goal:** Maximize transaction drop rate under a high illicit-retention
  constraint. The main operating point is 95% illicit retention.
- **Method:** Sender-side XGBoost prefiltering with causal transaction-history,
  repeated-amount, bank-local, and time-decay features.
- **Scale:** Experiments cover IBM AML HI-Small, HI-Medium, and HI-Large.
- **Integration:** The repo includes an inference-flag output contract for
  downstream PRISM/Bank-sim ingestion.

## Representative Results

At the 95% illicit-retention operating point:

| Dataset | Feature set | Features | Actual retained | Dropped | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| HI-Small | compact repeated-amount + decay | 18 | 94.7% | 92.2% | 0.988 | 0.437 |
| HI-Medium | compact repeated-amount + decay | 18 | 92.9% | 92.7% | 0.987 | 0.361 |
| HI-Large | mainline decay12h | 18 | 94.8% | 92.5% | 0.989 | 0.475 |
| HI-Large | receiver-bank role context | 30 | 96.4% | 91.2% | 0.990 | 0.520 |

The full result notes, threshold discipline, feature discussion, and command
examples are in
[`docs/prefilter-research-summary.md`](docs/prefilter-research-summary.md).

## Repository Layout

```text
configs/        FHE and model configuration files.
docs/           Design notes, experiment summaries, usage, and integration contract.
examples/       Public-safe sample inference-flag outputs.
scripts/        Prefilter experiments, FHE proof scripts, and export utilities.
src/            AML feature generation, model training, evaluation, and FHE modules.
tests/          Unit tests for configs, sampling, thresholds, and export contract.
```

Raw AML datasets, generated feature matrices, model artifacts, FHE keys, and run
logs are intentionally excluded from Git. They are either too large, regenerated
by the documented commands, or potentially sensitive.

## Install

Python 3.10 is recommended.

```bash
git clone https://github.com/oscar81632/AML-Filter-Research.git
cd AML-Filter-Research
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The dataset download command requires Kaggle credentials for the public IBM AML
dataset.

For development and tests:

```bash
pip install -e ".[dev]"
python -m pytest tests/unit
```

## Quick Start: FHE Pipeline

```bash
make data-download
make data-validate
make stage
make features
make benchmark
```

Pipeline overview:

```text
Kaggle IBM AML raw files
  -> raw data staging
  -> graph-based feature generation
  -> train/valid/test feature tables
  -> plain XGBoost training and evaluation
  -> Concrete ML XGBoost training and evaluation
  -> FHE compile, simulate, and execute checks
```

`make benchmark` trains a new model from scratch. It does not load or reuse an
earlier model.

## Quick Start: Prefilter Experiments

HI-Small or HI-Medium:

```bash
python scripts/filter_xgb.py \
  data/HI-Small_Trans.csv \
  data/HI-Small_Patterns.txt \
  --feature-set compact_core_same_amount_spread_decay12h \
  --metrics-json artifacts/feature_sets/hi_small_compact_decay12h_metrics.json
```

HI-Large memory-lean run:

```bash
python scripts/filter_scale_sender_only.py \
  data/HI-Large_Trans.csv \
  data/HI-Large_Patterns.txt \
  --feature-set mainline_decay12h \
  --history-mode bank_local \
  --metrics-json artifacts/feature_sets/hi_large_mainline_decay12h_metrics.json
```

Inference-flag export:

```bash
python scripts/filter_flag_table.py data/HI-Small_Trans.csv \
  --patterns data/HI-Small_Patterns.txt \
  --retention 0.95 \
  --output artifacts/flags/hi-small_flags.parquet \
  --with-raw-keys
```

The inference-flag format is documented in
[`docs/inference-flag-table.md`](docs/inference-flag-table.md).

## Documentation

- [`docs/prefilter-research-summary.md`](docs/prefilter-research-summary.md):
  main prefilter feature sets and results.
- [`docs/inference-flag-table.md`](docs/inference-flag-table.md): output contract
  for downstream ingestion.
- [`docs/design.md`](docs/design.md): FHE pipeline motivation and design.
- [`docs/architecture.md`](docs/architecture.md): source tree and command flow.
- [`docs/usage.md`](docs/usage.md): full setup, commands, outputs, and schema
  inspection.
- [`docs/experiments.md`](docs/experiments.md): FHE model evaluation and
  configuration trade-offs.

## Notes for Reviewers

- The prefilter is evaluated as a high-recall screening stage, not as a final AML
  decision system.
- Thresholds for deployable results are selected on validation data and then
  applied to test data.
- Sample files under `examples/` illustrate the output contract only; they are
  not benchmark outputs.
