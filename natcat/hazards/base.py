"""Peril-agnostic hazard model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

__all__ = ["Coordinates", "IntensityValues", "HazardModel"]

#: Shape ``(N, 2)`` array of ``[latitude, longitude]`` pairs, in degrees.
Coordinates = NDArray[np.float64]
#: Shape ``(N,)`` array of hazard intensities, one per location.
IntensityValues = NDArray[np.float64]


class HazardModel(ABC):
    """Abstract base class for natural-catastrophe hazard models.

    A hazard model maps geographic locations to a peril-specific intensity:
    TC = 1-minute sustained wind (kt), EQ = peak ground acceleration (g),
    FL = inundation depth (m).
    """

    @property
    @abstractmethod
    def peril_type(self) -> str:
        """Peril identifier, e.g. ``'TC'``, ``'EQ'`` or ``'FL'``."""

    @abstractmethod
    def compute_intensity(self, coordinates: Coordinates) -> IntensityValues:
        """Compute the event footprint intensity at each location.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Shape ``(N, 2)`` array of ``[latitude, longitude]`` pairs.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)`` array of intensities.
        """

    @abstractmethod
    def compute_intensity_history(
        self,
        coordinates: Coordinates,
        times: NDArray[np.datetime64],
    ) -> NDArray[np.float64]:
        """Compute the cumulative intensity reached at each requested time.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Shape ``(N, 2)`` array of ``[latitude, longitude]`` pairs.
        times : array_like of datetime64
            Shape ``(T,)`` sequence of times to report.

        Returns
        -------
        numpy.ndarray
            Shape ``(T, N)`` array; row ``t`` holds the running maximum
            intensity accumulated up to and including ``times[t]``.
        """
