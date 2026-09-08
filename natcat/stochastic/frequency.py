"""Annual storm-frequency model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

__all__ = ["PoissonFrequency"]

logger = logging.getLogger(__name__)


class PoissonFrequency:
    """Poisson model for the number of storms per season.

    Parameters
    ----------
    rate : float, optional
        Pre-set annual rate ``lambda``. Usually left ``None`` and estimated by
        :meth:`fit`.

    Attributes
    ----------
    rate : float or None
        Mean number of storms per year.

    Examples
    --------
    >>> model = PoissonFrequency(rate=10.0)
    >>> model.sample(3, np.random.default_rng(0)).shape
    (3,)
    """

    def __init__(self, rate: float | None = None) -> None:
        self.rate = float(rate) if rate is not None else None

    def fit(self, years: np.ndarray | pd.Series) -> PoissonFrequency:
        """Estimate the annual rate from a sequence of genesis years.

        Parameters
        ----------
        years : numpy.ndarray or pandas.Series
            One entry per historical storm, giving its genesis year.

        Returns
        -------
        PoissonFrequency
            ``self``, for chaining.

        Raises
        ------
        ValueError
            If ``years`` is empty.
        """
        series = pd.Series(np.asarray(years).ravel()).dropna()
        if series.empty:
            raise ValueError("Cannot fit PoissonFrequency on an empty year sequence.")
        self.rate = float(series.value_counts().mean())
        return self

    def sample(self, n_years: int, rng: np.random.Generator) -> np.ndarray:
        """Draw the storm count for each of ``n_years`` synthetic seasons.

        Parameters
        ----------
        n_years : int
            Number of seasons.
        rng : numpy.random.Generator
            Random generator; the only source of randomness used.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_years,)`` array of non-negative integer counts.

        Raises
        ------
        RuntimeError
            If the rate has not been set or fitted.
        """
        if self.rate is None:
            raise RuntimeError("PoissonFrequency is not fitted. Call fit() first.")
        if n_years <= 0:
            return np.zeros(0, dtype=int)
        return rng.poisson(lam=self.rate, size=int(n_years))

    def __repr__(self) -> str:
        rate = "unfitted" if self.rate is None else f"{self.rate:.2f}"
        return f"PoissonFrequency(rate={rate})"
