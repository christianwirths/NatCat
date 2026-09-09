"""Portfolio loss calculation from a hazard and a vulnerability model."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..hazards.base import HazardModel
from ..vulnerability.base import VulnerabilityModel

__all__ = ["LossCalculator"]

logger = logging.getLogger(__name__)

#: Portfolio columns the calculator needs.
_REQUIRED_PORTFOLIO_COLUMNS: tuple[str, ...] = ("latitude", "longitude", "tiv")


class LossCalculator:
    """Combine a hazard footprint with a vulnerability curve to get losses.

    Parameters
    ----------
    hazard : natcat.hazards.HazardModel
        Any hazard model, e.g. :class:`natcat.hazards.TropicalCycloneHazard`.
    vulnerability : natcat.vulnerability.VulnerabilityModel
        Any vulnerability model, e.g. :class:`natcat.vulnerability.WindVulnerability`.

    Attributes
    ----------
    results : pandas.DataFrame or None
        Output of the most recent :meth:`compute` call.

    Examples
    --------
    >>> calculator = LossCalculator(hazard, vulnerability)   # doctest: +SKIP
    >>> results = calculator.compute(portfolio)              # doctest: +SKIP
    >>> calculator.total_loss                                # doctest: +SKIP
    """

    def __init__(self, hazard: HazardModel, vulnerability: VulnerabilityModel) -> None:
        self.hazard = hazard
        self.vulnerability = vulnerability
        self.results: pd.DataFrame | None = None

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _check_portfolio(portfolio: pd.DataFrame) -> None:
        """Raise if the portfolio lacks a required column."""
        missing = [c for c in _REQUIRED_PORTFOLIO_COLUMNS if c not in portfolio.columns]
        if missing:
            raise KeyError(f"Portfolio is missing column(s) {missing}")

    @staticmethod
    def _construction(portfolio: pd.DataFrame) -> np.ndarray | None:
        """Return the per-asset construction classes, if the portfolio has any."""
        if "construction" in portfolio.columns:
            return portfolio["construction"].to_numpy()
        return None

    # -- public API --------------------------------------------------------
    def compute(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """Compute the ground-up loss for every location in the portfolio.

        Parameters
        ----------
        portfolio : pandas.DataFrame
            Exposure with ``latitude``, ``longitude`` and ``tiv`` columns, plus
            an optional ``construction`` column. Never mutated.

        Returns
        -------
        pandas.DataFrame
            Copy of ``portfolio`` with added ``intensity``, ``damage_ratio``
            and ``loss`` columns. Also stored on :attr:`results`.

        Raises
        ------
        KeyError
            If a required portfolio column is missing.
        """
        self._check_portfolio(portfolio)

        coordinates = portfolio[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        intensity = self.hazard.compute_intensity(coordinates)
        tiv = portfolio["tiv"].to_numpy(dtype=np.float64)
        damage_ratio = self.vulnerability.damage_ratio(
            intensity, self._construction(portfolio), tiv=tiv
        )

        out = portfolio.copy()
        out["intensity"] = intensity
        out["damage_ratio"] = damage_ratio
        out["loss"] = damage_ratio * tiv

        self.results = out
        return out

    def compute_history(self, portfolio: pd.DataFrame, times: ArrayLike) -> pd.DataFrame:
        """Compute the cumulative loss at each of a series of timestamps.

        Uses :meth:`~natcat.hazards.HazardModel.compute_intensity_history`, so
        the whole time evolution costs one pass over the track rather than one
        footprint per timestamp.

        Parameters
        ----------
        portfolio : pandas.DataFrame
            Exposure with ``latitude``, ``longitude`` and ``tiv`` columns.
            Never mutated.
        times : array_like
            Timestamps to report. Sorted ascending internally.

        Returns
        -------
        pandas.DataFrame
            Long-format frame: the portfolio columns repeated per timestamp,
            plus ``time``, ``intensity``, ``damage_ratio`` and ``loss``.

        Raises
        ------
        KeyError
            If a required portfolio column is missing.
        ValueError
            If ``times`` is empty.
        """
        self._check_portfolio(portfolio)

        stamps = pd.to_datetime(pd.Series(np.asarray(times).ravel())).sort_values()
        stamps = stamps.reset_index(drop=True)
        if stamps.empty:
            raise ValueError("At least one timestamp must be provided.")

        coordinates = portfolio[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        history = self.hazard.compute_intensity_history(coordinates, stamps.to_numpy())

        construction = self._construction(portfolio)
        tiv = portfolio["tiv"].to_numpy(dtype=np.float64)

        frames = []
        for i, stamp in enumerate(stamps):
            intensity = history[i]
            damage_ratio = self.vulnerability.damage_ratio(intensity, construction, tiv=tiv)
            frame = portfolio.copy()
            frame["time"] = stamp
            frame["intensity"] = intensity
            frame["damage_ratio"] = damage_ratio
            frame["loss"] = damage_ratio * tiv
            frames.append(frame)

        return pd.concat(frames, ignore_index=True)

    @property
    def total_loss(self) -> float:
        """Total ground-up loss over the portfolio.

        Returns
        -------
        float
            Sum of the ``loss`` column of the last :meth:`compute` call.

        Raises
        ------
        RuntimeError
            If :meth:`compute` has not been called yet.
        """
        if self.results is None:
            raise RuntimeError("No results yet. Call compute(portfolio) first.")
        return float(self.results["loss"].sum())

    def calculate_portfolio_loss(self, portfolio: pd.DataFrame) -> pd.DataFrame:
        """Deprecated alias of :meth:`compute`.

        Parameters
        ----------
        portfolio : pandas.DataFrame
            Exposure portfolio.

        Returns
        -------
        pandas.DataFrame
            Same as :meth:`compute`.

        Warns
        -----
        DeprecationWarning
            Always; use :meth:`compute` instead.
        """
        warnings.warn(
            "LossCalculator.calculate_portfolio_loss() is deprecated; use compute().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.compute(portfolio)

    def __repr__(self) -> str:
        state = "computed" if self.results is not None else "not computed"
        return f"LossCalculator(peril={self.hazard.peril_type!r}, status={state})"
