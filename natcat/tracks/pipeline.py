"""One-call loading of a processed best track."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import get_data_dir
from ..data.atcf import read_best_track
from .processing import prepare_track

__all__ = ["load_best_track"]

logger = logging.getLogger(__name__)


def load_best_track(
    year: int,
    basin: str,
    storm_number: str | int,
    *,
    freq: str = "1h",
    method: str = "linear",
    data_dir: str | Path | None = None,
    download: bool = True,
) -> pd.DataFrame:
    """Load, download if needed, and process one storm's best track.

    A cached ``<data_dir>/raw/b<basin><nn><year>.dat`` is always preferred; the
    network is only touched when the file is absent and ``download`` is True.

    Parameters
    ----------
    year : int
        Season year, e.g. ``2018``.
    basin : str
        Two-letter basin code, e.g. ``'al'``.
    storm_number : str or int
        ATCF cyclone number, e.g. ``'14'`` or ``14``.
    freq : str, default '1h'
        Time step of the processed track.
    method : {'linear', 'cubic'}, default 'linear'
        Interpolation kind.
    data_dir : str or pathlib.Path, optional
        Root data directory. Defaults to :func:`natcat.config.get_data_dir`.
    download : bool, default True
        Fetch the file from NHC when it is not cached locally.

    Returns
    -------
    pandas.DataFrame
        Processed track (see :func:`natcat.tracks.prepare_track`) with a
        ``storm_id`` column such as ``'AL142018'``.

    Raises
    ------
    FileNotFoundError
        If the file is missing locally and ``download`` is False, or if NHC has
        no such storm.

    Examples
    --------
    >>> load_best_track(2018, "al", "14")  # doctest: +SKIP
    """
    storm_number = f"{int(storm_number):02d}" if not isinstance(storm_number, str) else storm_number
    basin = basin.lower()

    root = Path(data_dir) if data_dir is not None else get_data_dir()
    path = root / "raw" / f"b{basin}{storm_number}{year}.dat"

    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} not found and download=False. "
                "Call natcat.data.download_best_track() or set download=True."
            )
        from ..data.nhc import download_best_track

        logger.info("Best track %s not cached; downloading from NHC.", path.name)
        path = download_best_track(year, basin, storm_number, data_dir=root)

    raw = read_best_track(path)
    track = prepare_track(raw, freq=freq, method=method)
    track["storm_id"] = f"{basin.upper()}{storm_number}{year}"
    return track
