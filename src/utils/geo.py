"""
Utility functions for geographical calculations.
Includes functions for calculating distances and bearings between geographical points.
"""

from typing import Final 
import numpy as np 
from numpy.typing import NDArray

EARTH_RADIUS_KM: Final[float] = 6371.0  # Earth's radius in kilometers
EARTH_RADIUS_NM: Final[float] = 3440.1  # Earth's radius in nautical miles


def haversine_distance(
        lat1: float,
        lon1: float,
        lat2_array: NDArray[np.float_],
        lon2_array: NDArray[np.float_],
        *,
        unit: str = 'nm') -> NDArray[np.float_]:
    """
    Calculate the great circle distance between a single storm point (lat1, lon1) 
    and many vectorized object (e.g. house) points (lat2_array, lon2_array).
    Returns distance in Nautical Miles.
    """

    if unit == 'km':
        R = EARTH_RADIUS_KM
    elif unit == 'nm':
        R = EARTH_RADIUS_NM
    else:
        raise ValueError("Invalid unit. Use 'km' for kilometers or 'nm' for nautical miles.")

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_array)
    
    delta_phi = np.radians(lat2_array - lat1)
    delta_lambda = np.radians(lon2_array - lon1)
    
    a = np.sin(delta_phi / 2)**2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2)**2
        
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def bearing(
        lat1: float,
        lon1: float, 
        lat2: float, 
        lon2: float) -> float:
    """Calculate the bearing from point 1 (lat1,lon1) to point 2 (lat2,lon2)."""

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    diffLong = np.radians(lon2 - lon1)

    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diffLong))

    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)

    return (initial_bearing + 360) % 360