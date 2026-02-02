"""
Utility functions for natural catastrophe modeling.

Submodules:
- geo: Geographic calculations (distance, bearing)
- units: Unit conversions (nautical miles, knots, etc.)
- track: Tropical cyclone track processing
"""

from .geo import haversine_distance, bearing
from .units import nm2km, kt2kmh
from .track import (
    track_interpolation,
    extract_past_trajectory,
    extract_future_trajectory,
    cyclone_velocity,
    cyclone_bearing,
    prepare_track_data,
)

__all__ = [
    # Geographic utilities
    'haversine_distance',
    'bearing',
    # Unit conversions
    'nm2km',
    'kt2kmh',
    # Track processing
    'track_interpolation',
    'extract_past_trajectory',
    'extract_future_trajectory',
    'cyclone_velocity',
    'cyclone_bearing',
    'prepare_track_data',
]
