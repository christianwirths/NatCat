"""Real-world exposure from the CLIMADA LitPop dataset.

CLIMADA is an optional dependency: it is imported inside the functions that
need it, so ``import natcat`` works without it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

__all__ = [
    "load_litpop_exposure",
    "construction_type_heuristic",
    "climada_to_portfolio",
]

logger = logging.getLogger(__name__)

#: TIV (USD) at which a property is equally likely to be Frame or Masonry.
_MASONRY_MIDPOINT: float = 1e6
#: Steepness of the logistic value-to-construction mapping, in 1/USD.
_MASONRY_STEEPNESS: float = 1e-7


def construction_type_heuristic(
    values: np.ndarray,
    seed: int | None = 42,
) -> np.ndarray:
    """Assign construction classes from total insured value.

    More valuable properties are more likely to be masonry, following a
    logistic probability centred on USD 1M.

    Parameters
    ----------
    values : numpy.ndarray
        Total insured values, in USD.
    seed : int, optional
        Seed for :func:`numpy.random.default_rng`.

    Returns
    -------
    numpy.ndarray
        Array of ``'Frame'`` / ``'Masonry'`` strings, same length as ``values``.
    """
    values = np.asarray(values, dtype=np.float64)
    probability = 1.0 / (1.0 + np.exp(-_MASONRY_STEEPNESS * (values - _MASONRY_MIDPOINT)))
    rng = np.random.default_rng(seed)
    return np.where(rng.random(values.size) < probability, "Masonry", "Frame")


def climada_to_portfolio(exposure, *, seed: int | None = 42) -> pd.DataFrame:
    """Convert a CLIMADA ``Exposures`` object to the natcat portfolio schema.

    Parameters
    ----------
    exposure : climada.entity.Exposures
        CLIMADA exposure whose ``gdf`` carries point geometries and ``value``.
    seed : int, optional
        Seed used by :func:`construction_type_heuristic`.

    Returns
    -------
    pandas.DataFrame
        Columns ``location_id``, ``latitude``, ``longitude``, ``tiv``,
        ``construction``.
    """
    gdf = exposure.gdf
    values = np.asarray(gdf["value"].to_numpy(), dtype=np.float64)
    return pd.DataFrame(
        {
            "location_id": np.arange(1, len(gdf) + 1),
            "latitude": gdf.geometry.y.to_numpy(),
            "longitude": gdf.geometry.x.to_numpy(),
            "tiv": values.round(2),
            "construction": construction_type_heuristic(values, seed=seed),
        }
    )


def load_litpop_exposure(
    country: str,
    bounds: tuple[float, float, float, float] | None = None,
    *,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Load a LitPop exposure for one country as a natcat portfolio.

    Parameters
    ----------
    country : str
        Country name or ISO code understood by the CLIMADA data API, e.g.
        ``'USA'``.
    bounds : tuple of float, optional
        ``(lat_min, lat_max, lon_min, lon_max)`` sub-region filter, in degrees.
    seed : int, optional
        Seed used to assign construction classes.

    Returns
    -------
    pandas.DataFrame
        Columns ``location_id``, ``latitude``, ``longitude``, ``tiv``,
        ``construction``.

    Raises
    ------
    ImportError
        If CLIMADA is not installed. Install the ``geo`` extra:
        ``pip install "natcat[geo]"``.

    Examples
    --------
    >>> load_litpop_exposure("USA", (24.5, 32.0, -87.6, -80.0))  # doctest: +SKIP
    """
    try:
        from climada.util.api_client import Client
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "CLIMADA is required for load_litpop_exposure(). "
            'Install it with: pip install "natcat[geo]"'
        ) from exc

    logger.info("Fetching LitPop exposure for %s via the CLIMADA data API.", country)
    exposure = Client().get_litpop(country=country)

    if bounds is not None:
        lat_min, lat_max, lon_min, lon_max = bounds
        gdf = exposure.gdf
        mask = (
            (gdf.geometry.y >= lat_min)
            & (gdf.geometry.y <= lat_max)
            & (gdf.geometry.x >= lon_min)
            & (gdf.geometry.x <= lon_max)
        )
        from climada.entity import Exposures

        exposure = Exposures(gdf[mask])

    exposure.check()
    return climada_to_portfolio(exposure, seed=seed)
