"""
Base classes for hazard modeling.

HazardModel defines the interface for natural catastrophe hazard models.
Implementations compute intensity at locations, which feeds into vulnerability
functions for loss estimation.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


Coordinates = NDArray[np.float64]      # Shape (N, 2): [latitude, longitude] pairs
IntensityValues = NDArray[np.float64]  # Shape (N,): intensity per location


class HazardModel(ABC):
    """
    Abstract base class for NatCat hazard models.
    
    Computes hazard intensity at geographic locations.
    Intensity units: TC = wind (kt), EQ = PGA (g), FL = depth (m).
    """

    @abstractmethod
    def compute_intensity(self, coordinates: Coordinates) -> IntensityValues:
        """
        Compute hazard intensity at given locations.
        
        Args:
            coordinates: Shape (N, 2) array of [lat, lon] pairs.
        
        Returns:
            Shape (N,) array of intensity values.
        """
        pass

    @property
    @abstractmethod
    def peril_type(self) -> str:
        """Return peril identifier: 'TC', 'EQ', 'FL', etc."""
        pass