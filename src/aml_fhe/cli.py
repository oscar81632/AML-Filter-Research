"""Command line entrypoint for the AML FHE pipeline."""

from __future__ import annotations

import argparse

from aml_fhe.benchmark import run_benchmark
from aml_fhe.config import load_pipeline_config
from aml_fhe.data.download import download_kaggle_dataset
from aml_fhe.data.download import missing_hi_small_files
from aml_fhe.data.stage import stage_data
from aml_fhe.features.schema import format_schema
from aml_fhe.features.build import build_features
from aml_fhe.sweep import run_sweep


def main() -> None:
    """Run the command line interface."""
    parser = argparse.ArgumentParser(prog="aml-fhe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run post-parquet benchmark commands")
    benchmark_subparsers = benchmark_parser.add_subparsers(dest="benchmark_command", required=True)
    benchmark_run = benchmark_subparsers.add_parser("run", help="Run XGBoost and FHE benchmark")
    benchmark_run.add_argument("--config", default="configs/fhe-baseline.toml")

    data_parser = subparsers.add_parser("data", help="Run data staging commands")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_download = data_subparsers.add_parser("download", help="Download the IBM AML Kaggle dataset")
    data_download.add_argument("--config", default="configs/hi-small.toml")
    data_validate = data_subparsers.add_parser("validate", help="Validate required HI-Small raw files")
    data_validate.add_argument("--config", default="configs/hi-small.toml")
    data_stage = data_subparsers.add_parser("stage", help="Stage raw Kaggle IBM AML files")
    data_stage.add_argument("--config", default="configs/hi-small.toml")

    features_parser = subparsers.add_parser("features", help="Run feature generation commands")
    features_subparsers = features_parser.add_subparsers(dest="features_command", required=True)
    features_build = features_subparsers.add_parser("build", help="Build graph-based feature parquets")
    features_build.add_argument("--config", default="configs/hi-small.toml")
    features_schema = features_subparsers.add_parser("schema", help="Print a feature parquet schema")
    features_schema.add_argument("path")

    run_parser = subparsers.add_parser("run", help="Run lifecycle commands")
    run_subparsers = run_parser.add_subparsers(dest="run_command", required=True)
    run_all = run_subparsers.add_parser("all", help="Run the full lifecycle")
    run_all.add_argument("--config", default="configs/hi-small.toml")

    sweep_parser = subparsers.add_parser("sweep", help="Run benchmark sweeps")
    sweep_subparsers = sweep_parser.add_subparsers(dest="sweep_command", required=True)
    sweep_run = sweep_subparsers.add_parser("run", help="Run a hyperparameter sweep")
    sweep_run.add_argument("--config", default="configs/sweep-quick.toml")
    sweep_run.add_argument("--strategy", choices=["smoke", "quick"], default="quick")
    sweep_run.add_argument("--save-report", action="store_true")

    args = parser.parse_args()

    if args.command == "benchmark" and args.benchmark_command == "run":
        config = load_pipeline_config(args.config)
        run_benchmark(config)
        return
    if args.command == "data" and args.data_command == "stage":
        config = load_pipeline_config(args.config)
        stage_data(config)
        return
    if args.command == "data" and args.data_command == "download":
        config = load_pipeline_config(args.config)
        download_kaggle_dataset(config)
        return
    if args.command == "data" and args.data_command == "validate":
        config = load_pipeline_config(args.config)
        missing = missing_hi_small_files(config.paths.raw_data_dir)
        if missing:
            raise SystemExit(f"Missing required raw files in {config.paths.raw_data_dir}: {', '.join(missing)}")
        print(f"Raw HI-Small files found in {config.paths.raw_data_dir}")
        return
    if args.command == "features" and args.features_command == "build":
        config = load_pipeline_config(args.config)
        build_features(config)
        return
    if args.command == "features" and args.features_command == "schema":
        print(format_schema(args.path))
        return
    if args.command == "sweep" and args.sweep_command == "run":
        results = run_sweep(args.config, strategy=args.strategy, save_report=args.save_report)
        for result in results:
            status = "OK" if result.error is None else f"ERR {result.error}"
            print(
                f"n_bits={result.config.n_bits} n_est={result.config.n_est} "
                f"train_rows={result.config.train_normal_rows} "
                f"plain_f1={result.plain_xgb_f1:.4f} cml_f1={result.concrete_ml_f1:.4f} "
                f"compile_s={result.compile_time_s:.1f} {status}"
            )
        return
    if args.command == "run" and args.run_command == "all":
        config = load_pipeline_config(args.config)
        missing = missing_hi_small_files(config.paths.raw_data_dir)
        if missing:
            raise SystemExit(
                f"Missing required raw files in {config.paths.raw_data_dir}: {', '.join(missing)}. "
                "Run `python -m aml_fhe data download --config configs/hi-small.toml` first."
            )
        stage_data(config)
        build_features(config)
        run_benchmark(config)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
