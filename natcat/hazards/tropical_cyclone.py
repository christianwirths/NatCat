"""Tropical cyclone wind hazard model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from .base import Coordinates, HazardModel, IntensityValues
from .wind_field import DEFAULT_DECAY_EXPONENT, max_wind_footprint, max_wind_history

__all__ = ["TropicalCycloneHazard"]

logger = logging.getLogger(__name__)


class TropicalCycloneHazard(HazardModel):
    """Wind hazard from a single tropical cyclone track.

    Parameters
    ----------
    track : pandas.DataFrame
        Processed track (see :func:`natcat.tracks.prepare_track`) carrying
        ``time``, ``latitude``, ``longitude``, ``max_wind_speed_kt``,
        ``radius_max_wind_nm``, ``translation_speed_kt`` and ``heading_deg``.
        Stored by reference; it is never modified.
    vortex : {'rankine'}, default 'rankine'
        Symmetric wind profile.
    asymmetry_factor : float, default 0.5
        Fraction of the storm translation speed added to the wind field.
    decay_exponent : float, default 2.0
        Radial decay exponent outside the eyewall. ``1`` is the classical
        Rankine vortex, ``0.5`` the modified Rankine.
    chunk_size : int, default 200_000
        Maximum number of (track point x location) elements per chunk.

    Attributes
    ----------
    track : pandas.DataFrame
        The track this hazard was built from.

    Examples
    --------
    >>> from natcat.tracks import load_best_track      # doctest: +SKIP
    >>> hazard = TropicalCycloneHazard(load_best_track(2018, "al", "14"))  # doctest: +SKIP
    >>> hazard.peril_type                                             # doctest: +SKIP
    'TC'
    """

    def __init__(
        self,
        track: pd.DataFrame,
        *,
        vortex: str = "rankine",
        asymmetry_factor: float = 0.5,
        decay_exponent: float = DEFAULT_DECAY_EXPONENT,
        chunk_size: int = 200_000,
    ) -> None:
        self.track = track
        self.vortex = vortex
        self.asymmetry_factor = asymmetry_factor
        self.decay_exponent = decay_exponent
        self.chunk_size = chunk_size

    @property
    def peril_type(self) -> str:
        """Peril identifier for tropical cyclone: ``'TC'``."""
        return "TC"

    @property
    def times(self) -> NDArray[np.datetime64]:
        """Timestamps of the underlying track."""
        return pd.to_datetime(self.track["time"]).to_numpy(dtype="datetime64[ns]")

    def compute_intensity(self, coordinates: Coordinates) -> IntensityValues:
        """Maximum wind speed reached at each location over the whole track.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Shape ``(N, 2)`` array of ``[latitude, longitude]`` pairs.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)`` array of wind speeds in knots.
        """
        return max_wind_footprint(
            self.track,
            coordinates,
            vortex=self.vortex,
            asymmetry_factor=self.asymmetry_factor,
            exponent=self.decay_exponent,
            chunk_size=self.chunk_size,
        )

    def compute_intensity_history(
        self,
        coordinates: Coordinates,
        times: ArrayLike,
    ) -> NDArray[np.float64]:
        """Running-maximum wind speed at each location, sampled at ``times``.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Shape ``(N, 2)`` array of ``[latitude, longitude]`` pairs.
        times : array_like
            Shape ``(T,)`` sequence of timestamps.

        Returns
        -------
        numpy.ndarray
            Shape ``(T, N)`` array of wind speeds in knots, non-decreasing
            down each column.
        """
        return max_wind_history(
            self.track,
            coordinates,
            times,
            vortex=self.vortex,
            asymmetry_factor=self.asymmetry_factor,
            exponent=self.decay_exponent,
            chunk_size=self.chunk_size,
        )

    def __repr__(self) -> str:
        return (
            f"TropicalCycloneHazard(n_points={len(self.track)}, vortex={self.vortex!r}, "
            f"asymmetry_factor={self.asymmetry_factor}, decay_exponent={self.decay_exponent})"
        )
