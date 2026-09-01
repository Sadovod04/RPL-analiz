"""Config loader. Single source of truth is ``config/settings.toml``.

Usage::

    from settings import load_settings
    cfg = load_settings()
    x = cfg["target"]["rpl_minutes_threshold"]
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "settings.toml"


@lru_cache(maxsize=8)
def load_settings(path: str | Path | None = None) -> dict:
    """Parse the TOML config. Result is cached per path."""
    p = Path(path) if path is not None else CONFIG_PATH
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def data_raw_dir() -> Path:
    return ROOT / load_settings()["paths"]["data_raw"]


def data_processed_dir() -> Path:
    return ROOT / load_settings()["paths"]["data_processed"]
