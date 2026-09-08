"""Unit conversions and Saffir-Simpson classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "NM_PER_KM",
    "KM_PER_NM",
    "MS_PER_KT",
    "nm2km",
    "km2nm",
    "kt2kmh",
    "kt2ms",
    "kmh2kt",
    "saffir_simpson_category",
]

#: Kilometres in one nautical mile.
KM_PER_NM: float = 1.852
#: Nautical miles in one kilometre.
NM_PER_KM: float = 1.0 / 1.852
#: Metres per second in one knot.
MS_PER_KT: float = 1852.0 / 3600.0

#: Lower Vmax bound (kt) of each Saffir-Simpson class, in ascending order.
_CATEGORY_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (137.0, "C5"),
    (113.0, "C4"),
    (96.0, "C3"),
    (83.0, "C2"),
    (64.0, "C1"),
    (34.0, "TS"),
)


def nm2km(nm: ArrayLike) -> NDArray[np.float64] | float:
    """Convert nautical miles to kilometres.

    Parameters
    ----------
    nm : array_like
        Distance in nautical miles.

    Returns
    -------
    float or numpy.ndarray
        Distance in kilometres.

    Examples
    --------
    >>> nm2km(1.0)
    1.852
    """
    return nm * KM_PER_NM


def km2nm(km: ArrayLike) -> NDArray[np.float64] | float:
    """Convert kilometres to nautical miles.

    Parameters
    ----------
    km : array_like
        Distance in kilometres.

    Returns
    -------
    float or numpy.ndarray
        Distance in nautical miles.
    """
    return km / KM_PER_NM


def kt2kmh(kt: ArrayLike) -> NDArray[np.float64] | float:
    """Convert knots to kilometres per hour.

    Parameters
    ----------
    kt : array_like
        Speed in knots.

    Returns
    -------
    float or numpy.ndarray
        Speed in km/h.

    Examples
    --------
    >>> kt2kmh(10.0)
    18.52
    """
    return kt * KM_PER_NM


def kmh2kt(kmh: ArrayLike) -> NDArray[np.float64] | float:
    """Convert kilometres per hour to knots.

    Parameters
    ----------
    kmh : array_like
        Speed in km/h.

    Returns
    -------
    float or numpy.ndarray
        Speed in knots.
    """
    return kmh / KM_PER_NM


def kt2ms(kt: ArrayLike) -> NDArray[np.float64] | float:
    """Convert knots to metres per second.

    Parameters
    ----------
    kt : array_like
        Speed in knots.

    Returns
    -------
    float or numpy.ndarray
        Speed in m/s.
    """
    return kt * MS_PER_KT


def saffir_simpson_category(max_wind_speed_kt: ArrayLike) -> NDArray[np.str_] | str:
    """Classify a 1-minute sustained wind speed on the Saffir-Simpson scale.

    Parameters
    ----------
    max_wind_speed_kt : array_like
        Maximum sustained wind speed, in knots.

    Returns
    -------
    str or numpy.ndarray of str
        One of ``'TD'``, ``'TS'``, ``'C1'`` ... ``'C5'``. A scalar input
        returns a plain ``str``.

    Examples
    --------
    >>> saffir_simpson_category(140.0)
    'C5'
    >>> saffir_simpson_category(20.0)
    'TD'
    """
    values = np.asarray(max_wind_speed_kt, dtype=np.float64)
    out = np.full(values.shape, "TD", dtype="<U2")
    for threshold, label in reversed(_CATEGORY_THRESHOLDS):
        out = np.where(values >= threshold, label, out)
    if np.ndim(max_wind_speed_kt) == 0:
        return str(out.item())
    return out
