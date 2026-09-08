"""Parametric tropical-cyclone wind field and footprint computation.

The wind at a point is the sum of a symmetric vortex profile and an asymmetry
term driven by the storm's own motion::

    v(r, theta) = v_sym(r) + factor * translation_speed * sin(theta - heading)

so that the right-hand side of the track (northern hemisphere) is enhanced and
the left-hand side reduced.

Both :func:`max_wind_footprint` and :func:`max_wind_history` broadcast the whole
track against the whole location array in one shot, chunking over locations to
bound peak memory.  The history is a single cumulative pass
(:func:`numpy.maximum.accumulate`) rather than one footprint per timestamp.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ..config import EARTH_RADIUS_NM

__all__ = [
    "DEFAULT_DECAY_EXPONENT",
    "rankine_vortex",
    "motion_asymmetry",
    "max_wind_footprint",
    "max_wind_history",
]

logger = logging.getLogger(__name__)

#: Decay exponent used by default outside the eyewall. ``1`` is the classical
#: Rankine vortex and ``0.4``-``0.6`` the "modified Rankine" common in the TC
#: literature; ``2`` reproduces the historical behaviour of this package.
DEFAULT_DECAY_EXPONENT: float = 2.0

#: Track columns required to evaluate the wind field.
_REQUIRED_TRACK_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "max_wind_speed_kt",
    "radius_max_wind_nm",
    "translation_speed_kt",
    "heading_deg",
)


def rankine_vortex(
    r: ArrayLike,
    vmax: ArrayLike,
    rmw: ArrayLike,
    exponent: float = DEFAULT_DECAY_EXPONENT,
) -> NDArray[np.float64]:
    """Evaluate a (modified) Rankine vortex wind profile.

    Wind grows linearly inside the radius of maximum wind and decays as a power
    law outside it::

        v(r) = vmax * r / rmw            for r <  rmw
        v(r) = vmax * (rmw / r)**exponent for r >= rmw

    Parameters
    ----------
    r : array_like
        Distance from the storm centre, in nautical miles. Broadcasts against
        ``vmax`` and ``rmw``.
    vmax : array_like
        Maximum sustained wind speed at the RMW, in knots.
    rmw : array_like
        Radius of maximum wind, in nautical miles. Must be positive.
    exponent : float, default 2.0
        Decay exponent outside the eyewall. ``1`` is the classical Rankine
        vortex, ``0.5`` the modified Rankine.

    Returns
    -------
    numpy.ndarray
        Wind speed in knots, broadcast to the common shape of the inputs.

    Examples
    --------
    >>> float(rankine_vortex(0.0, 100.0, 20.0))
    0.0
    >>> float(rankine_vortex(20.0, 100.0, 20.0))
    100.0
    >>> float(rankine_vortex(40.0, 100.0, 20.0, exponent=1.0))
    50.0
    """
    radius = np.asarray(r, dtype=np.float64)
    peak = np.asarray(vmax, dtype=np.float64)
    core = np.asarray(rmw, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        inner = peak * (radius / core)
        outer = peak * (core / radius) ** exponent
    return np.where(radius < core, inner, outer)


def motion_asymmetry(
    bearing_to_point: ArrayLike,
    heading: ArrayLike,
    translation_speed: ArrayLike,
    factor: float = 0.5,
) -> NDArray[np.float64]:
    """Wind contribution from the storm's forward motion.

    Parameters
    ----------
    bearing_to_point : array_like
        Bearing from the storm centre to the location, degrees clockwise from
        north.
    heading : array_like
        Storm heading, degrees clockwise from north.
    translation_speed : array_like
        Storm forward speed, in knots.
    factor : float, default 0.5
        Fraction of the translation speed that projects onto the wind field.

    Returns
    -------
    numpy.ndarray
        Signed wind speed contribution in knots: positive to the right of the
        track, negative to the left.

    Examples
    --------
    >>> float(motion_asymmetry(90.0, 0.0, 10.0))   # due east of a northbound storm
    5.0
    >>> float(motion_asymmetry(270.0, 0.0, 10.0))  # due west
    -5.0
    """
    relative = np.asarray(bearing_to_point, dtype=np.float64) - np.asarray(
        heading, dtype=np.float64
    )
    delta = (relative + 180.0) % 360.0 - 180.0
    return factor * np.asarray(translation_speed, dtype=np.float64) * np.sin(np.radians(delta))


def _validate_track(track: pd.DataFrame) -> None:
    """Raise if ``track`` lacks a column the wind field needs."""
    missing = [c for c in _REQUIRED_TRACK_COLUMNS if c not in track.columns]
    if missing:
        raise KeyError(
            f"Track is missing column(s) {missing}. "
            "Run natcat.tracks.prepare_track() to build a processed track."
        )


def _track_arrays(track: pd.DataFrame) -> tuple[NDArray[np.float64], ...]:
    """Extract the wind-field inputs as float columns of shape ``(T, 1)``."""
    _validate_track(track)
    return tuple(
        track[column].to_numpy(dtype=np.float64)[:, None] for column in _REQUIRED_TRACK_COLUMNS
    )


def _as_coords(coords: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split an ``(N, 2)`` coordinate array into latitude and longitude."""
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"coords must have shape (N, 2), got {array.shape}")
    return array[:, 0], array[:, 1]


def _chunk_slices(n_locations: int, n_times: int, chunk_size: int):
    """Yield location slices whose (T x n) block stays under ``chunk_size``."""
    per_chunk = max(1, int(chunk_size) // max(1, n_times))
    for start in range(0, n_locations, per_chunk):
        yield slice(start, min(start + per_chunk, n_locations))


def _wind_block(
    lat_t: NDArray[np.float64],
    lon_t: NDArray[np.float64],
    vmax: NDArray[np.float64],
    rmw: NDArray[np.float64],
    speed: NDArray[np.float64],
    heading: NDArray[np.float64],
    lat_p: NDArray[np.float64],
    lon_p: NDArray[np.float64],
    *,
    vortex: str,
    asymmetry_factor: float,
    exponent: float,
) -> NDArray[np.float64]:
    """Total wind speed for every (track point, location) pair, shape ``(T, n)``.

    The great-circle distance and the along-track angle share their expensive
    trigonometric sub-expressions, and the asymmetry term is evaluated as
    ``sin(bearing - heading)`` in closed form, which removes an ``arctan2``, a
    modulo and a ``sin`` per element relative to the naive composition of
    :func:`~natcat.utils.geo.haversine_distance`,
    :func:`~natcat.utils.geo.bearing` and :func:`motion_asymmetry`.
    """
    if vortex == "lamb-oseen":
        raise NotImplementedError("Lamb-Oseen vortex is not implemented yet.")
    if vortex != "rankine":
        raise ValueError(f"Unknown vortex type: {vortex!r}")

    phi_t = np.radians(lat_t)
    phi_p = np.radians(lat_p)[None, :]
    sin_phi_t, cos_phi_t = np.sin(phi_t), np.cos(phi_t)
    sin_phi_p, cos_phi_p = np.sin(phi_p), np.cos(phi_p)

    half_delta_lambda = 0.5 * np.radians(lon_p[None, :] - lon_t)
    sin_half_lambda = np.sin(half_delta_lambda)
    cos_half_lambda = np.cos(half_delta_lambda)
    sin_half_phi = np.sin(0.5 * (phi_p - phi_t))

    # Haversine great-circle distance, in nautical miles.
    a = sin_half_phi**2 + cos_phi_t * cos_phi_p * sin_half_lambda**2
    distances = EARTH_RADIUS_NM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    with np.errstate(divide="ignore", invalid="ignore"):
        inner = vmax * (distances / rmw)
        outer = vmax * (rmw / distances) ** exponent
    v_sym = np.where(distances < rmw, inner, outer)

    # sin(bearing_to_point - heading) without evaluating the bearing itself:
    # bearing = arctan2(x, y), so sin(b - h) = (x cos h - y sin h) / hypot(x, y).
    sin_lambda = 2.0 * sin_half_lambda * cos_half_lambda
    cos_lambda = 1.0 - 2.0 * sin_half_lambda**2
    x = sin_lambda * cos_phi_p
    y = cos_phi_t * sin_phi_p - sin_phi_t * cos_phi_p * cos_lambda

    heading_rad = np.radians(heading)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin_relative = (x * np.cos(heading_rad) - y * np.sin(heading_rad)) / np.hypot(x, y)
    sin_relative = np.nan_to_num(sin_relative, nan=0.0, posinf=0.0, neginf=0.0)

    return v_sym + asymmetry_factor * speed * sin_relative


def max_wind_footprint(
    track: pd.DataFrame,
    coords: ArrayLike,
    *,
    vortex: str = "rankine",
    asymmetry_factor: float = 0.5,
    exponent: float = DEFAULT_DECAY_EXPONENT,
    chunk_size: int = 200_000,
) -> NDArray[np.float64]:
    """Maximum wind speed experienced at each location over a whole track.

    Parameters
    ----------
    track : pandas.DataFrame
        Processed track carrying ``latitude``, ``longitude``,
        ``max_wind_speed_kt``, ``radius_max_wind_nm``, ``translation_speed_kt``
        and ``heading_deg``.
    coords : array_like
        Shape ``(N, 2)`` array of ``[latitude, longitude]`` pairs, in degrees.
    vortex : {'rankine'}, default 'rankine'
        Symmetric wind profile.
    asymmetry_factor : float, default 0.5
        Fraction of the translation speed added to the wind field.
    exponent : float, default 2.0
        Radial decay exponent outside the eyewall.
    chunk_size : int, default 200_000
        Maximum number of (track point x location) elements held in memory at
        once. Locations are processed in slices of ``chunk_size // len(track)``.

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)`` array of maximum wind speeds in knots, clipped at zero.

    Raises
    ------
    KeyError
        If ``track`` is missing a required column.
    ValueError
        If ``coords`` is not ``(N, 2)`` or ``vortex`` is unknown.

    Examples
    --------
    >>> from natcat.tracks import load_best_track  # doctest: +SKIP
    >>> max_wind_footprint(load_best_track(2018, "al", "14"), coords)  # doctest: +SKIP
    """
    lat_p, lon_p = _as_coords(coords)
    out = np.zeros(lat_p.size, dtype=np.float64)
    if len(track) == 0 or lat_p.size == 0:
        return out

    lat_t, lon_t, vmax, rmw, speed, heading = _track_arrays(track)

    for sl in _chunk_slices(lat_p.size, len(track), chunk_size):
        block = _wind_block(
            lat_t,
            lon_t,
            vmax,
            rmw,
            speed,
            heading,
            lat_p[sl],
            lon_p[sl],
            vortex=vortex,
            asymmetry_factor=asymmetry_factor,
            exponent=exponent,
        )
        np.maximum(out[sl], block.max(axis=0), out=out[sl])

    return out


def max_wind_history(
    track: pd.DataFrame,
    coords: ArrayLike,
    times: ArrayLike,
    *,
    vortex: str = "rankine",
    asymmetry_factor: float = 0.5,
    exponent: float = DEFAULT_DECAY_EXPONENT,
    chunk_size: int = 200_000,
) -> NDArray[np.float64]:
    """Running-maximum wind speed at each location, sampled at given times.

    The cumulative maximum along the track is computed once with
    :func:`numpy.maximum.accumulate` and then sampled, so the cost is the same
    as a single footprint rather than one footprint per requested time.

    Parameters
    ----------
    track : pandas.DataFrame
        Processed track; must also carry a ``time`` column sorted ascending.
    coords : array_like
        Shape ``(N, 2)`` array of ``[latitude, longitude]`` pairs.
    times : array_like
        Shape ``(T,)`` sequence of timestamps at which to report the running
        maximum. Times before the first track fix yield zeros.
    vortex : {'rankine'}, default 'rankine'
        Symmetric wind profile.
    asymmetry_factor : float, default 0.5
        Fraction of the translation speed added to the wind field.
    exponent : float, default 2.0
        Radial decay exponent outside the eyewall.
    chunk_size : int, default 200_000
        Maximum number of (track point x location) elements per chunk.

    Returns
    -------
    numpy.ndarray
        Shape ``(T, N)`` array of wind speeds in knots. Each column is
        non-decreasing in time, and the last row equals
        :func:`max_wind_footprint` when ``times[-1]`` covers the whole track.

    Raises
    ------
    KeyError
        If ``track`` is missing a required column or ``time``.
    ValueError
        If ``coords`` is not ``(N, 2)``.
    """
    if "time" not in track.columns:
        raise KeyError("max_wind_history requires a 'time' column on the track")

    lat_p, lon_p = _as_coords(coords)
    requested = pd.to_datetime(pd.Series(np.asarray(times).ravel())).to_numpy(
        dtype="datetime64[ns]"
    )
    out = np.zeros((requested.size, lat_p.size), dtype=np.float64)
    if len(track) == 0 or lat_p.size == 0 or requested.size == 0:
        return out

    track_times = pd.to_datetime(track["time"]).to_numpy(dtype="datetime64[ns]")
    # Row index of the last track fix at or before each requested time; -1 = none.
    row_index = np.searchsorted(track_times, requested, side="right") - 1
    covered = row_index >= 0
    if not covered.any():
        return out

    lat_t, lon_t, vmax, rmw, speed, heading = _track_arrays(track)

    for sl in _chunk_slices(lat_p.size, len(track), chunk_size):
        block = _wind_block(
            lat_t,
            lon_t,
            vmax,
            rmw,
            speed,
            heading,
            lat_p[sl],
            lon_p[sl],
            vortex=vortex,
            asymmetry_factor=asymmetry_factor,
            exponent=exponent,
        )
        running = np.maximum.accumulate(block, axis=0)
        np.maximum(running, 0.0, out=running)
        out[covered, sl] = running[row_index[covered]]

    return out
