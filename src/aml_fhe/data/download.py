"""Kaggle dataset download helper."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from aml_fhe.config import PipelineConfig

KAGGLE_DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
REQUIRED_HI_SMALL_FILES = (
    "HI-Small_Trans.csv",
    "HI-Small_accounts.csv",
    "HI-Small_Patterns.txt",
)


def download_kaggle_dataset(config: PipelineConfig) -> None:
    """Download and unzip the IBM AML Kaggle dataset into the configured data dir."""
    output_dir = Path(config.paths.raw_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET,
        "-p",
        str(output_dir),
        "--unzip",
    ]
    subprocess.run(command, check=True)
    missing = missing_hi_small_files(output_dir)
    if missing:
        raise FileNotFoundError(f"Kaggle download completed but required HI-Small files are missing: {missing}")


def missing_hi_small_files(data_dir: str | Path) -> list[str]:
    """Return required HI-Small files that are absent from data_dir."""
    directory = Path(data_dir)
    return [name for name in REQUIRED_HI_SMALL_FILES if not (directory / name).exists()]
