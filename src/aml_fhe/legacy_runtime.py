"""Helpers for running migrated legacy notebook scripts."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import runpy
import sys


def run_script_in_globals(path: str | Path, namespace: dict) -> None:
    """Execute a script path in the caller's global namespace."""
    script_path = Path(path)
    code = compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec")
    namespace["__file__"] = str(script_path)
    exec(code, namespace)


def install_legacy_aliases() -> None:
    """Expose package legacy modules under the import names expected upstream."""
    settings = _fresh_import("settings", "aml_fhe.features._legacy_settings")
    os.environ["EXSTRAQT_DATA_TYPE_FOLDER"] = settings.OUTPUT_POSTFIX.lstrip("-")
    aliases = {
        "common": "aml_fhe.features._legacy_common",
        "communities": "aml_fhe.features._legacy_communities",
        "features": "aml_fhe.features._legacy_features",
    }
    for short_name, module_name in aliases.items():
        _fresh_import(short_name, module_name)


def _fresh_import(short_name: str, module_name: str):
    if short_name in sys.modules:
        del sys.modules[short_name]
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)
    sys.modules[short_name] = module
    return module


def run_module(module_name: str) -> None:
    """Run a legacy module as a script after aliases are installed."""
    install_legacy_aliases()
    runpy.run_module(module_name, run_name="__main__")
