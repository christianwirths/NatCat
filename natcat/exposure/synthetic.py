"""Synthetic exposure portfolios for testing and demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["DEFAULT_BOUNDS", "synthetic_portfolio"]

#: Default bounding box ``(lat_min, lat_max, lon_min, lon_max)``: the Florida
#: Panhandle around Panama City, where Hurricane Michael made landfall.
DEFAULT_BOUNDS: tuple[float, float, float, float] = (29.5, 30.8, -86.0, -84.8)


def synthetic_portfolio(
    n: int = 1000,
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Generate a random property portfolio inside a bounding box.

    Locations are uniform in the box, total insured values are log-normal
    (median about USD 300k with a heavy tail of coastal properties), and
    construction is 70% Frame / 30% Masonry.

    Parameters
    ----------
    n : int, default 1000
        Number of locations.
    bounds : tuple of float, default :data:`DEFAULT_BOUNDS`
        ``(lat_min, lat_max, lon_min, lon_max)`` in degrees.
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`. ``None`` gives a
        non-deterministic portfolio.

    Returns
    -------
    pandas.DataFrame
        Columns ``location_id``, ``latitude``, ``longitude``, ``tiv`` (USD) and
        ``construction``.

    Raises
    ------
    ValueError
        If ``n`` is negative or ``bounds`` is malformed.

    Examples
    --------
    >>> portfolio = synthetic_portfolio(5, seed=0)
    >>> list(portfolio.columns)
    ['location_id', 'latitude', 'longitude', 'tiv', 'construction']
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if len(bounds) != 4:
        raise ValueError("bounds must be (lat_min, lat_max, lon_min, lon_max)")

    lat_min, lat_max, lon_min, lon_max = bounds
    if lat_min > lat_max or lon_min > lon_max:
        raise ValueError(f"Malformed bounds: {bounds}")

    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "location_id": np.arange(1, n + 1),
            "latitude": rng.uniform(lat_min, lat_max, n),
            "longitude": rng.uniform(lon_min, lon_max, n),
            "tiv": rng.lognormal(mean=12.6, sigma=0.5, size=n).round(2),
            "construction": rng.choice(["Frame", "Masonry"], size=n, p=[0.7, 0.3]),
        }
    )
