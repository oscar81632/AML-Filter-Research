# Usage

This project is meant to be run from the repo root directory.

`make` targets are the primary interface: each one wraps a fixed
`python -m aml_fhe ...` command with sane defaults, and `make help` lists
them all. Use the raw CLI directly only when you need to override something
a `make` target hardcodes (a different config file, a different sweep
strategy); see [Overriding Defaults](#overriding-defaults) at the end.

## Environment Setup

Run setup to create a Python 3.10 venv, install dependencies, and generate `.env`:

```bash
bash setup.sh
source .env
```

**Hard constraints, read before installing:**

- **Python 3.10 is required.** `concrete-ml` does not publish wheels for 3.11+
  as of the pinned version. Using any other Python version will fail at install
  time or import time.
- **Java 11 is required for the staging step.** PySpark 3.5.x does not support
  Java 17+. Check with `java -version` before running staging or feature
  generation. On Ubuntu: `sudo apt install openjdk-11-jdk`.
- **PySpark workers must use the same Python as the driver.** If your system
  `python3` is a different version than the venv (common when miniconda or
  pyenv is active), Spark workers will fail with a version-mismatch error.
  `source .env` sets `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` for you, so
  `make stage` and `make features` handle this automatically.

### Required deps by lifecycle stage

**Post-parquet benchmark** (no Spark, no graph libs needed):

- `numpy`, `pandas`, `pyarrow`, `scikit-learn`, `xgboost`, `concrete-ml`

**Kaggle-to-parquet generation** (additionally requires):

- `kaggle`, `pyspark==3.5.x`, `igraph`, `leidenalg`, `networkx`
- Java 11 on `PATH`

## Kaggle Credentials

The downloader calls the Kaggle CLI, which reads `~/.kaggle/kaggle.json`.
If that file does not exist, create it:

1. Log in at kaggle.com → Account → API → "Create new token" → downloads `kaggle.json`.
2. Move it into place and lock permissions:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Data Setup

The HI-Small path expects raw Kaggle files directly under `data/`:

```text
data/HI-Small_Trans.csv
data/HI-Small_accounts.csv
data/HI-Small_Patterns.txt
```

Download (~7.6 GB, takes several minutes) and validate:

```bash
make data-download
make data-validate
```

The downloader uses Kaggle dataset `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`:
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

## Build Feature Parquets

```bash
make stage       # ~20-40 min Spark job
make features    # ~2.5 hr Spark job, needs ~62 GB free RAM
```

Staging launches a local Spark cluster. Feature generation needs **~62 GB
of free RAM** for the Spark driver, because graph community detection loads
the full transaction graph in memory.

Expected feature outputs:

```text
features/hi-small/train_features.parquet/
features/hi-small/valid_features.parquet/
features/hi-small/test_features.parquet/
```

## Run Benchmark

**`make benchmark` trains a model. It does not evaluate an existing one.**
Despite the name, every run trains both the plain XGBoost and Concrete ML
models from scratch on whatever is currently in the feature parquets, then
evaluates and FHE-validates what it just trained. There is no flag or
mechanism to point it at an already-trained model instead of training a
new one, and no incremental or warm-start training. See
`src/aml_fhe/benchmark.py` for the full sequence (train, evaluate, compile,
simulate, execute) or [design.md](design.md) for why it's structured that
way.

If you only want to run inference against an already-trained model, that
is a different, separate path: see
[Inference on an Already-Trained Model](#inference-on-an-already-trained-model)
below.

After feature parquets exist:

```bash
make benchmark
```

This defaults to `configs/fhe-baseline.toml`, which currently encodes the
same recommended setting as `configs/hi-small.toml` (3-bit, depth 5, 50
trees, seed 1), so this reproduces the headline result. To run a different
config, see [Overriding Defaults](#overriding-defaults).

Benchmark runs write model files and metrics under the configured
`artifact_dir`, normally:

```text
artifacts/report_runs/3b5d50t_s1/models/plain_xgb.json
artifacts/report_runs/3b5d50t_s1/models/concrete_ml_xgb.joblib
artifacts/report_runs/3b5d50t_s1/models/metrics.json
```

These are outputs, not inputs: nothing in the pipeline reads them back in
on a later run. Logs are written under the configured `log_dir`. Run
outputs (models, metrics, logs) are generated locally and are not
committed to the repo.

See `docs/experiments.md` for reported results and how to interpret them.

## Inference on an Already-Trained Model

Training (`make benchmark`) and inference are two separate code paths.
There is no CLI command for inference; it lives in the demo server
(`demo/server.py --live`, launched via `demo/run_live.sh`).
`DemoState.load_model()` deserializes a previously-trained
`concrete_ml_xgb.joblib` and calls `model.predict(vector, fhe=...)` on one
row at a time. It never calls `.fit()`.

Which model file it loads is not hardcoded in `server.py`. It reads the
path from `demo/data/demo_meta.json`'s `"model_path"` field, which was
baked in when the demo data was built
(`demo/scripts/build_demo_data.py --source-root <path>`, or the
`AML_FHE_SOURCE_ROOT` env var). The committed `demo_meta.json` currently points to a machine-specific
absolute path (`/path/to/project/output/...`) that only resolves on the
machine it was built on. Running `--live` mode
anywhere else fails with `"Recommended model not found"` until this file
is regenerated. To fix
it, retrain a model with `make benchmark`, then rebuild the demo data
pointed at your own output:

```bash
AML_FHE_SOURCE_ROOT=/path/to/your/pipeline/output \
  .venv/bin/python demo/scripts/build_demo_data.py
```

See `demo/README.md` for the rest of the demo setup.

## Inspect Schema

Print a parquet schema without loading the full matrix:

```bash
make schema PATH=features/hi-small/train_features.parquet
```

## Run Sweep

```bash
make sweep
```

Runs the `quick` strategy against `configs/sweep-quick.toml` and saves a
report. To run the `smoke` strategy or a different config, see
[Overriding Defaults](#overriding-defaults).

Sweep reports are written under:

```text
<artifact_dir>/sweeps/hyperparameter_sweep_report.json
<artifact_dir>/sweeps/hyperparameter_sweep_comparison.csv
```

The report sweep configurations are kept in `configs/sweep/`. One-off shell
wrappers used to produce the sweep are not part of the maintained interface.

## Full Lifecycle

```bash
make full
```

Validates raw files, then runs staging, feature generation, and benchmark
evaluation with `configs/hi-small.toml`. Data must already be downloaded
(`make data-download`); this does not download it for you.

## Running Tests

Unit tests cover config loading, sampling, and threshold selection:

```bash
source .env      # or otherwise ensure the venv's python is active
pytest
```

`pyproject.toml` sets `pythonpath = ["src"]` for pytest, so no manual
`PYTHONPATH` export is needed. These tests do not require Spark, Kaggle
credentials, or downloaded data.

## Overriding Defaults

`make` targets are fixed: each one calls a single hardcoded
`python -m aml_fhe` command, so there is no `make`-level way to pass a
different `--config` or `--strategy`. Use the CLI directly for that:

```bash
# Run the benchmark against a specific config instead of the default
python -m aml_fhe benchmark run --config configs/sweep/3b5d50t_s1.toml

# Run a sweep with a different strategy or config
python -m aml_fhe sweep run --config configs/sweep-quick.toml --strategy smoke --save-report

# Run the full lifecycle against a specific config
python -m aml_fhe run all --config configs/hi-small.toml
```

If you run `data stage` or `features build` directly instead of through
`make`, export `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` yourself first
(`source .env` does this for the `make` targets automatically):

```bash
source .venv/bin/activate
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)
python -m aml_fhe data stage
```
