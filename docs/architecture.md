# Architecture

The package is organized around lifecycle boundaries: data, features, models,
evaluation, FHE, and run orchestration.

```text
aml_fhe/
  cli.py                 command routing
  config.py              TOML loading and typed settings
  data/                  Kaggle download and raw data staging
  features/              graph-based feature generation and schema tools
  models/                preprocessing, sampling, plain XGBoost, Concrete ML
  evaluation/            metrics and threshold selection
  fhe/                   Concrete ML compile and simulation helpers
  runs/                  logs and artifact persistence
  benchmark.py           post-parquet benchmark orchestration
  sweep.py               hyperparameter sweep orchestration
  legacy_runtime.py      runs the _legacy_* staging/feature modules as scripts
```

## Command Flow

`python -m aml_fhe data download`

```text
cli.py
  -> data.download.download_kaggle_dataset
  -> Kaggle CLI
  -> data/HI-Small_* files
```

`python -m aml_fhe data stage`

```text
cli.py
  -> data.stage.stage_data
  -> legacy_runtime.run_module
  -> data._legacy_prepare_input_small
  -> data/staged-transactions-hi-small
```

`python -m aml_fhe features build`

```text
cli.py
  -> features.build.build_features
  -> legacy_runtime.run_module
  -> features._legacy_2_model
  -> features._legacy_node_level_features
  -> features._legacy_generate_flow_features
  -> features/hi-small/* parquets
```

`python -m aml_fhe benchmark run`

```text
cli.py
  -> benchmark.run_benchmark
  -> models.preprocess.load_and_encode
  -> models.sampling.train_subsample
  -> models.plain_xgb.train_plain_xgb
  -> models.concrete_xgb.train_concrete_xgb
  -> evaluation.thresholds.threshold_sweep
  -> fhe.compile.compile_model
  -> fhe.simulate.simulate_predictions
  -> runs.artifacts.write_json
```

`python -m aml_fhe sweep run`

```text
cli.py
  -> sweep.run_sweep
  -> benchmark.run_benchmark for each configuration
  -> sweep JSON and CSV reports
```

## Implementation Notes

`data.stage.stage_data` and `features.build.build_features` run their work by
calling `legacy_runtime.run_module`, which executes the `_legacy_*` modules
(`data._legacy_prepare_input_small`, `features._legacy_2_model`, and the
modules those call) as scripts rather than importing them as regular
functions. Four shim files at `src/` level (`common.py`, `communities.py`,
`features.py`, `settings.py`) re-export from the corresponding `_legacy_*`
modules so Spark worker processes can import them under the short names those
modules expect, via `PYTHONPATH=src`.

Everything downstream of the feature parquets (`benchmark.py`, `models/`,
`evaluation/`, `fhe/`) is regular importable Python, organized by
responsibility rather than by script.

## Config Model

Configs are TOML files loaded by `config.py`.

Important sections:

- `[paths]`: raw data, feature, artifact, and log locations
- `[data]`: dataset name and label/string column assumptions (only
  `"HI-Small"` is currently supported; `data stage` and `features build`
  raise an error for any other value, see below)
- `[features]`: train/valid/test feature parquet paths
- `[model]`: XGBoost and sampling parameters
- `[fhe]`: Concrete ML quantization, compile, and simulation settings

`configs/hi-small.toml` is the real full-lifecycle config.

The `[data].dataset` field looks general but is not: `data.stage.stage_data`
and `features.build.build_features` both validate it and raise a `ValueError`
for anything other than `"HI-Small"`. Other IBM AML dataset variants (for
example HI-Medium or LI-Small) are not supported, even though the config
schema has a place for the name.

`configs/sweep-quick.toml` points to local feature parquets for sweep runs. The
sweep search space itself is controlled by the CLI `--strategy` argument.

Report experiment configurations are kept under `configs/sweep/`, with the
recommended setting in `configs/sweep/3b5d50t_s1.toml`. Run outputs (models,
metrics, logs) land under the configured `artifact_dir`/`log_dir`. These are
generated locally, not committed. See `docs/experiments.md` for results and
rationale.

`configs/hi-small.toml`, `configs/fhe-baseline.toml`, and
`configs/sweep/3b5d50t_s1.toml` all currently encode the same recommended
model setting (3-bit, depth 5, 50 trees, seed 1). They exist separately
because each is the default config for a different CLI entrypoint (full
lifecycle, bare benchmark, and the sweep suite respectively), not because
they were meant to diverge. If you change the recommended setting, update
all three.

## Operational Entrypoints

The canonical implementation entrypoint is the Python CLI in `aml_fhe.cli`.

The repo also provides thin convenience wrappers:

- `Makefile`: short local workflow targets
- `scripts/run_full.sh`: full local lifecycle

These wrappers do not contain pipeline logic. They call the same CLI commands,
which keeps operational shortcuts from becoming a second implementation.
