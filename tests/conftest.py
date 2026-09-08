"""Shared pytest fixtures.

Fixtures that need ATCF files under ``data/raw`` skip themselves when the data
directory is absent, so the suite passes on a clean clone and offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from natcat.config import get_data_dir
from natcat.data import read_best_track
from natcat.exposure import synthetic_portfolio
from natcat.tracks import prepare_track

MICHAEL_FILE = "bal142018.dat"


@pytest.fixture(scope="session")
def raw_data_dir() -> Path:
    """Directory holding ATCF B-deck files, skipping the test if absent."""
    directory = get_data_dir() / "raw"
    if not directory.is_dir() or not any(directory.glob("bal*.dat")):
        pytest.skip(f"No ATCF best-track files found in {directory}")
    return directory


@pytest.fixture(scope="session")
def michael_path(raw_data_dir: Path) -> Path:
    """Path to the Hurricane Michael (2018) best-track file."""
    path = raw_data_dir / MICHAEL_FILE
    if not path.exists():
        pytest.skip(f"{path} not available")
    return path


@pytest.fixture(scope="session")
def michael_raw(michael_path: Path) -> pd.DataFrame:
    """Raw, de-duplicated Michael best track."""
    return read_best_track(michael_path)


@pytest.fixture(scope="session")
def michael_track(michael_raw: pd.DataFrame) -> pd.DataFrame:
    """Processed Michael track on a 1-hour grid."""
    return prepare_track(michael_raw, freq="1h")


@pytest.fixture
def tiny_track() -> pd.DataFrame:
    """A four-point synthetic track heading due north at a constant speed."""
    return pd.DataFrame(
        {
            "time": pd.to_datetime(
                ["2020-01-01 00:00", "2020-01-01 06:00", "2020-01-01 12:00", "2020-01-01 18:00"]
            ),
            "latitude": [25.0, 26.0, 27.0, 28.0],
            "longitude": [-80.0, -80.0, -80.0, -80.0],
            "max_wind_speed_kt": [60.0, 80.0, 100.0, 90.0],
            "radius_max_wind_nm": [40.0, 30.0, 25.0, 30.0],
            "storm_type": ["TS", "HU", "HU", "HU"],
        }
    )


@pytest.fixture
def tiny_processed(tiny_track: pd.DataFrame) -> pd.DataFrame:
    """The tiny track processed onto a 1-hour grid."""
    return prepare_track(tiny_track, freq="1h")


@pytest.fixture
def portfolio() -> pd.DataFrame:
    """A small deterministic synthetic portfolio."""
    return synthetic_portfolio(200, seed=7)


@pytest.fixture
def random_coords() -> np.ndarray:
    """A deterministic ``(N, 2)`` block of coordinates over the Gulf coast."""
    rng = np.random.default_rng(11)
    return np.column_stack([rng.uniform(24.0, 34.0, 400), rng.uniform(-90.0, -78.0, 400)])
