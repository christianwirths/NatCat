"""Geographic helpers: great-circle distance, bearing and forward projection.

All functions are fully vectorised and broadcast their arguments, so they can be
used with scalars, 1-D arrays, or 2-D ``(T, N)`` grids of track points against
locations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..config import EARTH_RADIUS_KM, EARTH_RADIUS_NM

__all__ = ["haversine_distance", "bearing", "destination_point"]


def _earth_radius(unit: str) -> float:
    """Return the Earth radius for ``unit`` (``'km'`` or ``'nm'``)."""
    if unit == "km":
        return EARTH_RADIUS_KM
    if unit == "nm":
        return EARTH_RADIUS_NM
    raise ValueError(f"Invalid unit {unit!r}. Use 'km' or 'nm'.")


def haversine_distance(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
    *,
    unit: str = "nm",
) -> NDArray[np.float64]:
    """Great-circle distance between two sets of points.

    Parameters
    ----------
    lat1, lon1 : array_like
        Latitude and longitude of the first point(s), in degrees.
    lat2, lon2 : array_like
        Latitude and longitude of the second point(s), in degrees.
    unit : {'nm', 'km'}, default 'nm'
        Output unit: nautical miles or kilometres.

    Returns
    -------
    numpy.ndarray
        Distances, broadcast to the common shape of the inputs.

    Raises
    ------
    ValueError
        If ``unit`` is not ``'nm'`` or ``'km'``.

    Examples
    --------
    >>> float(round(haversine_distance(0.0, 0.0, 0.0, 1.0, unit="km"), 1))
    111.2
    """
    radius = _earth_radius(unit)

    phi1 = np.radians(np.asarray(lat1, dtype=np.float64))
    phi2 = np.radians(np.asarray(lat2, dtype=np.float64))
    delta_phi = phi2 - phi1
    delta_lambda = np.radians(
        np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64)
    )

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return radius * c


def bearing(
    lat1: ArrayLike,
    lon1: ArrayLike,
    lat2: ArrayLike,
    lon2: ArrayLike,
) -> NDArray[np.float64]:
    """Initial great-circle bearing from point 1 to point 2.

    Parameters
    ----------
    lat1, lon1 : array_like
        Latitude and longitude of the origin point(s), in degrees.
    lat2, lon2 : array_like
        Latitude and longitude of the target point(s), in degrees.

    Returns
    -------
    numpy.ndarray
        Bearing in degrees clockwise from true north, in ``[0, 360)``.

    Examples
    --------
    >>> float(round(bearing(0.0, 0.0, 1.0, 0.0)))
    0.0
    >>> float(round(bearing(0.0, 0.0, 0.0, 1.0)))
    90.0
    """
    phi1 = np.radians(np.asarray(lat1, dtype=np.float64))
    phi2 = np.radians(np.asarray(lat2, dtype=np.float64))
    delta_lambda = np.radians(
        np.asarray(lon2, dtype=np.float64) - np.asarray(lon1, dtype=np.float64)
    )

    x = np.sin(delta_lambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def destination_point(
    lat: ArrayLike,
    lon: ArrayLike,
    bearing_deg: ArrayLike,
    distance_km: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project a point along a great circle.

    Parameters
    ----------
    lat, lon : array_like
        Starting latitude and longitude, in degrees.
    bearing_deg : array_like
        Travel bearing, in degrees clockwise from true north.
    distance_km : array_like
        Travel distance, in kilometres.

    Returns
    -------
    tuple of numpy.ndarray
        ``(latitude, longitude)`` of the destination point, in degrees.
        Longitude is normalised to ``[-180, 180)``.

    Examples
    --------
    >>> lat, lon = destination_point(0.0, 0.0, 90.0, 111.195)
    >>> float(round(float(lon), 2))
    1.0
    """
    delta = np.asarray(distance_km, dtype=np.float64) / EARTH_RADIUS_KM
    theta = np.radians(np.asarray(bearing_deg, dtype=np.float64))
    phi1 = np.radians(np.asarray(lat, dtype=np.float64))
    lambda1 = np.radians(np.asarray(lon, dtype=np.float64))

    phi2 = np.arcsin(np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta))
    lambda2 = lambda1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(phi1),
        np.cos(delta) - np.sin(phi1) * np.sin(phi2),
    )

    out_lat = np.degrees(phi2)
    out_lon = (np.degrees(lambda2) + 180.0) % 360.0 - 180.0
    return out_lat, out_lon
