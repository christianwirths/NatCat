"""Track processing: interpolation, kinematics and RMW gap filling.

Every function here is pure: the input DataFrame is copied before any column is
written, so callers can chain transformations without surprises.

The processed-track schema produced by :func:`prepare_track` is

===================== ================ =======
column                dtype            unit
===================== ================ =======
``time``              datetime64[ns]
``latitude``          float            deg N
``longitude``         float            deg E
``max_wind_speed_kt`` float            kt
``radius_max_wind_nm``float            nm
``translation_speed_kt`` float         kt
``heading_deg``       float            deg (0 = N, clockwise)
===================== ================ =======
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..utils.geo import bearing, haversine_distance

__all__ = [
    "INTERPOLATED_COLUMNS",
    "interpolate_track",
    "add_translation_velocity",
    "add_heading",
    "fill_missing_rmw",
    "prepare_track",
]

logger = logging.getLogger(__name__)

#: Numeric columns interpolated onto the new time grid when present.
INTERPOLATED_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "max_wind_speed_kt",
    "radius_max_wind_nm",
    "min_pressure_mb",
)

#: (Vmax upper bound in kt, heuristic RMW in nm) pairs, ascending in Vmax.
_RMW_HEURISTIC: tuple[tuple[float, float], ...] = (
    (35.0, 80.0),
    (64.0, 60.0),
    (96.0, 40.0),
    (137.0, 25.0),
)
#: RMW used for Vmax at or above the last heuristic threshold.
_RMW_DEFAULT: float = 15.0
#: Multiplier applied to RMW for extratropical (``TY == 'EX'``) fixes.
_EX_RMW_FACTOR: float = 1.5

#: Default time step for processed tracks. The hazard footprint is the maximum
#: wind over the *discrete* track positions, so a coarse step under-samples the
#: swath of a compact, fast-moving storm: at 1 h Hurricane Michael (RMW ~10 nm
#: at landfall, 13 kt translation) leaves gaps between successive centres and
#: the modelled loss halves. Five minutes keeps the centre spacing well inside
#: the radius of maximum wind for any realistic storm.
DEFAULT_TRACK_FREQ: str = "5min"


def interpolate_track(
    df: pd.DataFrame,
    freq: str = DEFAULT_TRACK_FREQ,
    method: str = "linear",
) -> pd.DataFrame:
    """Resample a track onto a regular time grid.

    Numeric columns are interpolated with :func:`scipy.interpolate.interp1d` on
    the integer nanosecond representation of ``time``; non-numeric columns
    (``storm_type``, ``storm_id``, ...) are carried forward from the most recent
    original fix.

    Parameters
    ----------
    df : pandas.DataFrame
        Track with a unique, sorted ``time`` column and at least ``latitude``
        and ``longitude``. Never mutated.
    freq : str, default DEFAULT_TRACK_FREQ ('5min')
        Pandas frequency string for the output grid, e.g. ``'30min'``.
    method : {'linear', 'cubic'}, default 'linear'
        Interpolation kind. ``'cubic'`` silently degrades to ``'linear'`` when
        fewer than four fixes are available.

    Returns
    -------
    pandas.DataFrame
        Interpolated track with a fresh ``RangeIndex``. Tracks with fewer than
        two fixes, or whose span is shorter than ``freq``, are returned as a
        copy of the input.

    Raises
    ------
    KeyError
        If ``df`` has no ``time`` column.
    ValueError
        If ``time`` is not unique or not sorted ascending, or if ``method``
        is unknown.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "time": pd.to_datetime(["2018-10-07", "2018-10-08"]),
    ...     "latitude": [20.0, 22.0],
    ...     "longitude": [-86.0, -86.0],
    ... })
    >>> len(interpolate_track(df, freq="12h"))
    3
    """
    from scipy.interpolate import interp1d

    if "time" not in df.columns:
        raise KeyError("interpolate_track requires a 'time' column")
    if method not in ("linear", "cubic"):
        raise ValueError(f"Unknown interpolation method {method!r}; use 'linear' or 'cubic'")

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    if len(out) <= 1:
        return out.reset_index(drop=True)

    if not out["time"].is_unique:
        raise ValueError("interpolate_track requires unique timestamps; de-duplicate first")
    if not out["time"].is_monotonic_increasing:
        raise ValueError("interpolate_track requires 'time' sorted ascending")

    grid = pd.date_range(start=out["time"].iloc[0], end=out["time"].iloc[-1], freq=freq)
    if len(grid) <= 1:
        return out.reset_index(drop=True)

    kind = method
    if kind == "cubic" and len(out) < 4:
        logger.warning("Cubic interpolation needs >= 4 fixes; falling back to linear.")
        kind = "linear"

    x_old = out["time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    x_new = grid.to_numpy(dtype="datetime64[ns]").astype(np.int64)

    data: dict[str, np.ndarray] = {}
    carry: list[str] = []
    for column in out.columns:
        if column == "time":
            continue
        series = out[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = series.to_numpy(dtype=np.float64)
            interpolator = interp1d(
                x_old, values, kind=kind, fill_value="extrapolate", bounds_error=False
            )
            data[column] = interpolator(x_new)
        else:
            carry.append(column)

    result = pd.DataFrame({"time": grid, **data})

    if carry:
        # Step function: take the value of the most recent original fix.
        idx = np.clip(np.searchsorted(x_old, x_new, side="right") - 1, 0, len(x_old) - 1)
        for column in carry:
            result[column] = out[column].to_numpy()[idx]

    return result[["time", *[c for c in out.columns if c != "time"]]].reset_index(drop=True)


def add_translation_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Add the storm translation speed between consecutive fixes.

    The speed of the final fix is forward-filled from the previous step, which
    avoids a spurious drop to zero at the end of the track.

    Parameters
    ----------
    df : pandas.DataFrame
        Track with ``time``, ``latitude`` and ``longitude``. Never mutated.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with a ``translation_speed_kt`` column (knots).
    """
    out = df.copy()
    if len(out) == 0:
        out["translation_speed_kt"] = pd.Series(dtype=float)
        return out
    if len(out) == 1:
        out["translation_speed_kt"] = 0.0
        return out

    lat = out["latitude"].to_numpy(dtype=np.float64)
    lon = out["longitude"].to_numpy(dtype=np.float64)
    distances = haversine_distance(lat[:-1], lon[:-1], lat[1:], lon[1:], unit="nm")

    times = pd.to_datetime(out["time"]).to_numpy(dtype="datetime64[ns]")
    hours = np.diff(times) / np.timedelta64(1, "h")
    with np.errstate(divide="ignore", invalid="ignore"):
        speeds = np.where(hours > 0, distances / hours, 0.0)

    out["translation_speed_kt"] = np.append(speeds, speeds[-1])
    return out


def add_heading(df: pd.DataFrame) -> pd.DataFrame:
    """Add the storm heading between consecutive fixes.

    The heading of the final fix is forward-filled from the previous step.

    Parameters
    ----------
    df : pandas.DataFrame
        Track with ``latitude`` and ``longitude``. Never mutated.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with a ``heading_deg`` column (degrees clockwise from
        true north).
    """
    out = df.copy()
    if len(out) == 0:
        out["heading_deg"] = pd.Series(dtype=float)
        return out
    if len(out) == 1:
        out["heading_deg"] = 0.0
        return out

    lat = out["latitude"].to_numpy(dtype=np.float64)
    lon = out["longitude"].to_numpy(dtype=np.float64)
    headings = bearing(lat[:-1], lon[:-1], lat[1:], lon[1:])

    out["heading_deg"] = np.append(headings, headings[-1])
    return out


def fill_missing_rmw(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing radius-of-maximum-wind values with an intensity heuristic.

    Missing (``NaN`` or non-positive) values are replaced by a Vmax-dependent
    climatological radius: ``< 35`` kt -> 80 nm, ``< 64`` -> 60, ``< 96`` -> 40,
    ``< 137`` -> 25, otherwise 15 nm.  Extratropical fixes (``storm_type ==
    'EX'``) then have their RMW scaled by 1.5, whether filled or observed.

    Parameters
    ----------
    df : pandas.DataFrame
        Track with ``max_wind_speed_kt`` and optionally ``radius_max_wind_nm``
        and ``storm_type``. Never mutated.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with ``radius_max_wind_nm`` filled.

    Raises
    ------
    KeyError
        If ``max_wind_speed_kt`` is absent.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"max_wind_speed_kt": [30.0, 140.0],
    ...                    "radius_max_wind_nm": [float("nan"), float("nan")]})
    >>> fill_missing_rmw(df)["radius_max_wind_nm"].tolist()
    [80.0, 15.0]
    """
    if "max_wind_speed_kt" not in df.columns:
        raise KeyError("fill_missing_rmw requires a 'max_wind_speed_kt' column")

    out = df.copy()
    if "radius_max_wind_nm" not in out.columns:
        out["radius_max_wind_nm"] = np.nan

    winds = out["max_wind_speed_kt"].fillna(0.0).to_numpy(dtype=np.float64)
    conditions = [winds < upper for upper, _ in _RMW_HEURISTIC]
    choices = [value for _, value in _RMW_HEURISTIC]
    heuristic = np.select(conditions, choices, default=_RMW_DEFAULT)

    rmw = out["radius_max_wind_nm"].to_numpy(dtype=np.float64)
    needs_fill = np.isnan(rmw) | (rmw <= 0)
    rmw = np.where(needs_fill, heuristic, rmw)

    if "storm_type" in out.columns:
        is_ex = (out["storm_type"].astype(str).str.strip().str.upper() == "EX").to_numpy()
        rmw = np.where(is_ex, rmw * _EX_RMW_FACTOR, rmw)

    out["radius_max_wind_nm"] = rmw
    return out


def prepare_track(
    df: pd.DataFrame,
    *,
    freq: str = DEFAULT_TRACK_FREQ,
    method: str = "linear",
) -> pd.DataFrame:
    """Turn a raw best track into a processed track.

    Pipeline: de-duplicate timestamps, fill missing RMW, interpolate onto a
    regular grid, then derive translation speed and heading.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw track, e.g. from :func:`natcat.data.read_best_track`. Never mutated.
    freq : str, default DEFAULT_TRACK_FREQ ('5min')
        Time step of the output grid. Keep it fine: the wind footprint is a
        maximum over the discrete track positions, and a step coarser than the
        time the storm needs to travel one radius of maximum wind produces a
        "string of pearls" footprint and under-estimates loss (see
        :data:`DEFAULT_TRACK_FREQ`).
    method : {'linear', 'cubic'}, default 'linear'
        Interpolation kind.

    Returns
    -------
    pandas.DataFrame
        Processed track carrying ``time``, ``latitude``, ``longitude``,
        ``max_wind_speed_kt``, ``radius_max_wind_nm``, ``translation_speed_kt``
        and ``heading_deg``, plus any extra columns of ``df``.

    Examples
    --------
    >>> from natcat.data import read_best_track  # doctest: +SKIP
    >>> prepare_track(read_best_track("data/raw/bal142018.dat"))  # doctest: +SKIP
    """
    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.sort_values("time", kind="stable").drop_duplicates(subset=["time"], keep="first")
    out = out.reset_index(drop=True)

    out = fill_missing_rmw(out)
    out = interpolate_track(out, freq=freq, method=method)
    out = add_translation_velocity(out)
    out = add_heading(out)
    return out
