"""Quality control for track DataFrames."""

from __future__ import annotations

import pandas as pd

__all__ = ["REQUIRED_TRACK_COLUMNS", "validate_track"]

#: Columns every usable track must provide.
REQUIRED_TRACK_COLUMNS: tuple[str, ...] = (
    "time",
    "latitude",
    "longitude",
    "max_wind_speed_kt",
)


def validate_track(df: pd.DataFrame) -> tuple[bool, str]:
    """Check that a track DataFrame is usable by the hazard pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Track to validate. Expected to follow the raw or processed track schema.

    Returns
    -------
    tuple of (bool, str)
        ``(True, "")`` when the track passes, otherwise ``(False, reason)``
        where ``reason`` describes the first failed check.

    Examples
    --------
    >>> import pandas as pd
    >>> validate_track(pd.DataFrame())
    (False, 'DataFrame is empty')
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"

    missing = [col for col in REQUIRED_TRACK_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Missing required column(s): {', '.join(missing)}"

    subset = df[list(REQUIRED_TRACK_COLUMNS)]
    if subset.isna().any().any():
        counts = subset.isna().sum()
        return False, f"Missing values in critical columns: {counts[counts > 0].to_dict()}"

    if not df["latitude"].between(-90, 90).all():
        return False, "Latitude values out of bounds"
    if not df["longitude"].between(-180, 180).all():
        return False, "Longitude values out of bounds"

    winds = df["max_wind_speed_kt"]
    if (winds < 0).any() or (winds > 300).any():
        return False, "Invalid wind speed values (negative or > 300 kt)"

    if not df["time"].is_unique:
        return False, "Duplicate timestamps found"

    return True, ""
