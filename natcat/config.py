"""Package configuration: data directory resolution and shared constants.

The data directory is resolved once, at import time, from the environment
variable ``NATCAT_DATA_DIR`` and falls back to ``<repo root>/data``.

Examples
--------
>>> from natcat.config import get_data_dir
>>> get_data_dir().name
'data'
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "DATA_DIR_ENV_VAR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EARTH_RADIUS_KM",
    "EARTH_RADIUS_NM",
    "get_data_dir",
    "get_raw_data_dir",
]

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = PACKAGE_ROOT.parent
DATA_DIR_ENV_VAR: Final[str] = "NATCAT_DATA_DIR"

EARTH_RADIUS_KM: Final[float] = 6371.0
EARTH_RADIUS_NM: Final[float] = 3440.1


def get_data_dir() -> Path:
    """Return the root data directory.

    Resolution order: the ``NATCAT_DATA_DIR`` environment variable, then
    ``<repo root>/data``.

    Returns
    -------
    pathlib.Path
        Absolute path to the data directory. The directory is not created.

    Examples
    --------
    >>> get_data_dir().is_absolute()
    True
    """
    env_value = os.environ.get(DATA_DIR_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return REPO_ROOT / "data"


def get_raw_data_dir() -> Path:
    """Return ``<data dir>/raw``, where ATCF deck files are stored.

    Returns
    -------
    pathlib.Path
        Absolute path to the raw data directory.
    """
    return get_data_dir() / "raw"


#: Convenience constants evaluated at import time.
RAW_DATA_DIR: Final[Path] = get_data_dir() / "raw"
PROCESSED_DATA_DIR: Final[Path] = get_data_dir() / "processed"
