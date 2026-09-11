"""Inland decay of synthetic storms, fitted to historical landfalls.

Over land a tropical cyclone loses its energy source and its wind decays
towards a background value rather than towards zero. Following Kaplan &
DeMaria (1995) the decay is modelled as

.. math::

    v(t) = v_b + (v_0 - v_b)\\, e^{-\\alpha t}

with :math:`v_0` the wind at landfall, :math:`v_b` the background wind and
:math:`\\alpha` the decay rate. Both parameters are estimated by least squares
from every landfall in the historical tracks the catalog is fitted on, so the
synthetic set reproduces the observed hours of hurricane- and storm-force wind
over land instead of relying on a literature value for another basin or era.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .genesis import is_land

__all__ = ["LandDecayModel", "extract_landfall_segments"]


def extract_landfall_segments(tracks: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Collect every over-land run that follows a landfall.

    Parameters
    ----------
    tracks : sequence of pandas.DataFrame
        Processed historical tracks with ``time``, ``latitude``, ``longitude``
        and ``max_wind_speed_kt``.

    Returns
    -------
    pandas.DataFrame
        One row per over-land fix with ``storm`` (index into ``tracks``),
        ``wind_at_landfall_kt`` (last fix over water), ``hours`` since that
        fix, ``max_wind_speed_kt``, ``latitude`` and ``longitude``.
    """
    rows = []
    for index, track in enumerate(tracks):
        if len(track) < 2:
            continue
        lat = track["latitude"].to_numpy(dtype=float)
        lon = track["longitude"].to_numpy(dtype=float)
        wind = track["max_wind_speed_kt"].to_numpy(dtype=float)
        hours = (
            pd.to_datetime(track["time"]) - pd.to_datetime(track["time"]).iloc[0]
        ).dt.total_seconds().to_numpy() / 3600.0
        land = is_land(lat, lon)
        i = 1
        while i < len(track):
            if land[i] and not land[i - 1]:
                wind_0, hour_0 = wind[i - 1], hours[i - 1]
                j = i
                while j < len(track) and land[j]:
                    rows.append((index, wind_0, hours[j] - hour_0, wind[j], lat[j], lon[j]))
                    j += 1
                i = j
            else:
                i += 1
    return pd.DataFrame(
        rows,
        columns=[
            "storm",
            "wind_at_landfall_kt",
            "hours",
            "max_wind_speed_kt",
            "latitude",
            "longitude",
        ],
    )


@dataclass
class LandDecayModel:
    """Exponential decay of wind over land towards a background value.

    Parameters
    ----------
    rate_per_h : float, default 0.069
        Decay rate :math:`\\alpha` (1/h). The default is the fit to all
        Atlantic landfalls 1900-2020 (see :meth:`fit`).
    floor_kt : float, default 25.0
        Background wind :math:`v_b` the decay tends to.
    n_points : int, default 0
        Number of over-land fixes the parameters were fitted on (0 when the
        model was constructed by hand).
    rmse_kt : float, default nan
        Root mean squared residual of the fit.

    Examples
    --------
    >>> model = LandDecayModel(rate_per_h=0.07, floor_kt=25.0)
    >>> round(float(model.step(100.0, 12.0)), 1)
    57.4
    >>> LandDecayModel.from_rate(0.92).floor_kt
    0.0
    """

    rate_per_h: float = 0.069
    floor_kt: float = 25.0
    n_points: int = 0
    rmse_kt: float = float("nan")

    def __post_init__(self) -> None:
        if self.rate_per_h <= 0:
            raise ValueError("rate_per_h must be positive")
        if self.floor_kt < 0:
            raise ValueError("floor_kt must be non-negative")

    @classmethod
    def from_rate(cls, per_hour_factor: float) -> LandDecayModel:
        """Legacy form: multiplicative decay ``factor ** hours`` with no floor."""
        if not 0.0 < per_hour_factor < 1.0:
            raise ValueError("per_hour_factor must be in (0, 1)")
        return cls(rate_per_h=-float(np.log(per_hour_factor)), floor_kt=0.0)

    def step(self, wind_kt, hours):
        """Wind after ``hours`` over land, starting from ``wind_kt``.

        Winds already at or below the floor are left unchanged: the model
        never intensifies a storm over land.
        """
        wind = np.asarray(wind_kt, dtype=float)
        decayed = self.floor_kt + (wind - self.floor_kt) * np.exp(
            -self.rate_per_h * np.asarray(hours)
        )
        return np.minimum(wind, decayed)

    def hours_to(self, wind_0: float, wind_target: float) -> float:
        """Hours over land until ``wind_0`` has decayed to ``wind_target``."""
        if wind_target <= self.floor_kt or wind_0 <= wind_target:
            return float("inf") if wind_0 > wind_target else 0.0
        return float(
            np.log((wind_0 - self.floor_kt) / (wind_target - self.floor_kt)) / self.rate_per_h
        )

    @classmethod
    def fit(
        cls,
        tracks: Sequence[pd.DataFrame],
        *,
        min_landfall_wind_kt: float = 34.0,
        max_hours: float = 48.0,
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.01, 0.0), (0.5, 60.0)),
    ) -> LandDecayModel:
        """Least-squares fit of rate and floor to historical landfall segments.

        Parameters
        ----------
        tracks : sequence of pandas.DataFrame
            Processed historical tracks.
        min_landfall_wind_kt : float, default 34.0
            Only landfalls at or above this wind are used.
        max_hours : float, default 48.0
            Over-land fixes later than this after landfall are ignored.
        bounds : tuple of tuple
            ``((rate_min, floor_min), (rate_max, floor_max))``.

        Returns
        -------
        LandDecayModel

        Raises
        ------
        ValueError
            If fewer than 20 usable over-land fixes are found.
        """
        from scipy.optimize import least_squares

        segments = extract_landfall_segments(tracks)
        usable = segments[
            (segments["wind_at_landfall_kt"] >= min_landfall_wind_kt)
            & (segments["hours"] <= max_hours)
        ]
        if len(usable) < 20:
            raise ValueError(
                f"Only {len(usable)} over-land fixes after landfall; need at least 20 to fit"
            )
        wind_0 = usable["wind_at_landfall_kt"].to_numpy()
        hours = usable["hours"].to_numpy()
        wind = usable["max_wind_speed_kt"].to_numpy()

        def residual(params: np.ndarray) -> np.ndarray:
            rate, floor = params
            return wind - (floor + (wind_0 - floor) * np.exp(-rate * hours))

        outcome = least_squares(residual, x0=[0.09, 25.0], bounds=bounds)
        rate, floor = (float(value) for value in outcome.x)
        return cls(
            rate_per_h=rate,
            floor_kt=floor,
            n_points=int(len(usable)),
            rmse_kt=float(np.sqrt(np.mean(outcome.fun**2))),
        )

    def __repr__(self) -> str:
        fitted = f", n_points={self.n_points}, rmse_kt={self.rmse_kt:.1f}" if self.n_points else ""
        return (
            f"LandDecayModel(rate_per_h={self.rate_per_h:.4f}, "
            f"floor_kt={self.floor_kt:.1f}{fitted})"
        )
