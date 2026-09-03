# FHE-AML-ML

Pipeline for AML (anti-money-laundering) transaction detection using Fully
Homomorphic Encryption (FHE), so an AML model can score encrypted transactions
without ever seeing plaintext data.

This archive also includes the AML plaintext prefilter research code used to
reduce the number of transactions sent into the FHE stage. See
[docs/prefilter-research-summary.md](docs/prefilter-research-summary.md) for the
feature sets, recommended scripts, included files, excluded artifacts, and
representative HI-Small/HI-Medium/HI-Large results.

End-to-end lifecycle:

```text
Kaggle IBM AML raw files
  -> raw data staging (Spark)
  -> graph-based feature generation (Spark + Leiden community detection)
  -> train/valid/test feature parquets
  -> plain XGBoost training/inference
  -> Concrete ML XGBoost training/inference
  -> FHE compile/simulate
  -> metrics and model artifacts
```

## Quick Start

**Requirements:** Python 3.10 and Java 11 (`concrete-ml` requires 3.10; PySpark 3.5.x requires Java 11).

```bash
bash setup.sh      # validates prereqs, creates .venv, installs deps, writes .env
source .env        # loads PYSPARK_PYTHON and other env vars into your shell
```

Then, from a clean checkout:

```bash
make data-download    # downloads the ~7.6 GB Kaggle dataset (credentials required)
make data-validate    # checks the required raw files are present
make stage            # Spark: raw CSVs -> staged parquet (~20-40 min)
make features         # Spark: staged parquet -> graph features -> train/valid/test parquets (~2.5 hr, needs ~62 GB free RAM)
make benchmark        # trains plain XGBoost + Concrete ML from scratch, evaluates, validates FHE correctness
```

Budget about 3 hours end to end for a first run; staging and feature
generation are unattended Spark jobs, not interactive steps.

`make full` runs `data-validate -> stage -> features -> benchmark` in one
shot, but still expects the raw data to already be downloaded
(`make data-download` first).

**`make benchmark` always trains a new model; it never loads or reuses an
earlier one**, despite the name. Running inference against an
already-trained model instead is a separate path, documented in
[docs/usage.md](docs/usage.md#inference-on-an-already-trained-model).

See [docs/usage.md](docs/usage.md) for the full command reference.

## Documentation

- [Design](docs/design.md): why this pipeline exists and why it's shaped this way.
- [Architecture](docs/architecture.md): package structure and command flow.
- [Usage](docs/usage.md): environment setup, commands, configs, tests, and expected outputs.
- [Experiments](docs/experiments.md): how the model was evaluated, key results, and implications.

## Results

Recommended setting: **3-bit quantization, depth 5, 50 trees, seed 1**.

Concrete ML (FHE-compatible) test F1: **0.7356** (precision 0.840, recall 0.655).
FHE execute vs. clear match: **3,000/3,000** rows, zero mismatches.

See [docs/experiments.md](docs/experiments.md) for methodology and full results.

Raw CSV inputs and generated feature parquets are not committed. They are
reproduced locally with the commands above.
