"""Extract past and forecast trajectories from ATCF A-deck data."""

from __future__ import annotations

import logging

import pandas as pd

from .processing import add_heading, add_translation_velocity, fill_missing_rmw, interpolate_track

__all__ = [
    "extract_past_trajectory",
    "extract_future_trajectory",
    "prepare_forecast_track",
]

logger = logging.getLogger(__name__)


def extract_past_trajectory(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    tech: str = "OFCL",
) -> pd.DataFrame:
    """Extract the analysed (tau = 0) trajectory up to a reference time.

    Parameters
    ----------
    df : pandas.DataFrame
        A-deck frame from :func:`natcat.data.read_a_deck`, carrying ``tech``,
        ``tau`` and ``time``. Never mutated.
    timestamp : pandas.Timestamp
        Latest analysis time to include.
    tech : str, default 'OFCL'
        Objective-aid identifier.

    Returns
    -------
    pandas.DataFrame
        One row per analysis time, sorted ascending, with a fresh index.
    """
    out = df[df["tech"] == tech]
    out = out[(out["time"] <= timestamp) & (out["tau"] == 0)]
    out = out.drop_duplicates(subset=["time"], keep="last")
    return out.sort_values("time", kind="stable").reset_index(drop=True)


def extract_future_trajectory(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    tech: str = "OFCL",
) -> pd.DataFrame:
    """Extract the forecast trajectory issued at a reference time.

    Valid times are reconstructed as ``time + tau`` hours.

    Parameters
    ----------
    df : pandas.DataFrame
        A-deck frame from :func:`natcat.data.read_a_deck`. Never mutated.
    timestamp : pandas.Timestamp
        Forecast initialisation time.
    tech : str, default 'OFCL'
        Objective-aid identifier.

    Returns
    -------
    pandas.DataFrame
        One row per forecast valid time, sorted ascending, with a fresh index.
    """
    out = df[(df["tech"] == tech) & (df["time"] == timestamp)].copy()
    out = out.drop_duplicates(subset=["tau"], keep="last")
    out["time"] = out["time"] + pd.to_timedelta(out["tau"], unit="h")
    out = out.drop_duplicates(subset=["time"], keep="last")
    return out.sort_values("time", kind="stable").reset_index(drop=True)


def prepare_forecast_track(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    tech: str = "OFCL",
    *,
    past: bool = True,
    freq: str = "1h",
    method: str | None = None,
) -> pd.DataFrame:
    """Build a processed track from A-deck analysis or forecast data.

    Parameters
    ----------
    df : pandas.DataFrame
        A-deck frame from :func:`natcat.data.read_a_deck`. Never mutated.
    timestamp : pandas.Timestamp
        Reference analysis / initialisation time.
    tech : str, default 'OFCL'
        Objective-aid identifier.
    past : bool, default True
        Use the analysed trajectory (``True``) or the forecast (``False``).
    freq : str, default '1h'
        Time step of the interpolation grid.
    method : {'linear', 'cubic'}, optional
        Interpolation kind. Defaults to ``'linear'`` for the past track (which
        keeps cumulative damage monotone as new fixes arrive) and ``'cubic'``
        for forecasts.

    Returns
    -------
    pandas.DataFrame
        Processed track. Empty input yields an empty frame rather than an error.

    Raises
    ------
    ValueError
        If interpolation fails for the extracted trajectory.
    """
    trajectory = (
        extract_past_trajectory(df, timestamp, tech)
        if past
        else extract_future_trajectory(df, timestamp, tech)
    )
    if trajectory.empty:
        kind = "analysis" if past else "forecast"
        logger.warning("No %s rows for tech=%s at %s", kind, tech, timestamp)
        return trajectory

    if method is None:
        method = "linear" if past else "cubic"

    out = fill_missing_rmw(trajectory)
    out = interpolate_track(out, freq=freq, method=method)
    out = add_translation_velocity(out)
    out = add_heading(out)
    return out
