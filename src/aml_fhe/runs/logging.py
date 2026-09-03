"""Run logging helpers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def configure_logging(log_dir: str | Path, filename: str = "benchmark.log") -> logging.Logger:
    """Configure file and stdout logging for a run."""
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s] %(message)s",
        handlers=[
            logging.FileHandler(directory / filename),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("aml_fhe")
