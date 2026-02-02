"""
Tropical Cyclone hazard model.

Includes wind field physics (Rankine vortex) and hazard intensity calculations.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .base import HazardModel, Coordinates, IntensityValues
from utils.geo import haversine_distance, bearing


# -------------------------------------------------------------------------
# Wind Field Physics Models
# -------------------------------------------------------------------------

def _rankine_vortex(r: NDArray[np.float64], max_wind_speed: float, 
                    radius_max_wind: float, exponent: float = 2) -> NDArray[np.float64]:
    """
    Calculate wind speed using Rankine vortex model.
    
    The Rankine vortex is a simple parametric model where wind speed increases
    linearly inside the eye and decays as a power law outside.
    
    Reference: https://en.wikipedia.org/wiki/Rankine_vortex
    
    Args:
        r: Distance from storm center (nautical miles).
        max_wind_speed: Maximum wind speed at RMW (knots).
        radius_max_wind: Radius of maximum wind (nautical miles).
        exponent: Decay exponent outside eye (typically 2).
    
    Returns:
        Wind speed at distance r (knots).
    """
    inside_eye = r < radius_max_wind
    outside_eye = r >= radius_max_wind
    
    v_at_r = np.zeros_like(r, dtype=float)
    v_at_r[inside_eye] = max_wind_speed * (r[inside_eye] / radius_max_wind)  
    v_at_r[outside_eye] = max_wind_speed * (radius_max_wind / r[outside_eye]) ** exponent
    
    return v_at_r


def _get_heuristic_rmw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply heuristic estimates for radius of maximum wind (RMW) when data is missing.
    
    Uses wind speed-based heuristics:
    - < 35 kt: 80 nm (Tropical Depression)
    - < 64 kt: 60 nm (Tropical Storm)
    - < 96 kt: 40 nm (Category 1-2)
    - < 137 kt: 25 nm (Category 3-4)
    - >= 137 kt: 15 nm (Category 5)
    
    Also applies 1.5x scaling for extratropical cyclones.
    
    Args:
        df: DataFrame with 'max_wind_speed_kt' and 'radius_max_wind_nm' columns.
        
    Returns:
        DataFrame with filled RMW values.
    """
    df_fixed = df.copy()
    winds = df_fixed['max_wind_speed_kt'].fillna(0)
    
    conditions = [winds < 35, winds < 64, winds < 96, winds < 137]
    choices = [80.0, 60.0, 40.0, 25.0]
    
    heuristic_rmw = np.select(conditions, choices, default=15.0)
    needs_fix_mask = (df_fixed['radius_max_wind_nm'].isna()) | (df_fixed['radius_max_wind_nm'] <= 0)
    df_fixed.loc[needs_fix_mask, 'radius_max_wind_nm'] = heuristic_rmw[needs_fix_mask]
    
    # Apply 1.5x scaling for extratropical cyclones
    if 'storm_type' in df_fixed.columns:
        ex_mask = df_fixed['storm_type'] == 'EX'
        df_fixed.loc[ex_mask, 'radius_max_wind_nm'] *= 1.5
        
    return df_fixed


def _max_wind_speeds_at_locations(
    df: pd.DataFrame, 
    points: list[tuple[float, float]], 
    vortex: str = "rankine", 
    asymmetry_factor: float = 0.5
) -> NDArray[np.float64]:
    """
    Calculate maximum wind speeds at given locations over storm track.
    
    For each location, computes wind speed at every track point and returns
    the maximum value encountered as the storm passes.
    
    Args:
        df: Storm track DataFrame with columns 'latitude', 'longitude',
            'max_wind_speed_kt', 'radius_max_wind_nm', 'velocity_kt', 'bearing_deg'.
        points: List of (latitude, longitude) tuples for locations of interest.
        vortex: Vortex model to use ('rankine' or 'lamb-oseen').
        asymmetry_factor: Fraction of translation speed contributing to asymmetry (0.5-1.0).
        
    Returns:
        Maximum wind speeds (knots) at each location.
    """
    v_max = np.zeros(len(points), dtype=float)
    latitudes = np.array([point[0] for point in points])
    longitudes = np.array([point[1] for point in points])

    for _, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        
        # Distance from storm center to each location
        distances = haversine_distance(lat, lon, latitudes, longitudes, unit='nm')
        
        max_wind_speed = row['max_wind_speed_kt']
        radius_max_wind = row['radius_max_wind_nm']
        
        # Calculate symmetric wind field
        if vortex == "rankine":
            v_symmetric = _rankine_vortex(distances, max_wind_speed, radius_max_wind)
        elif vortex == "lamb-oseen":
            raise NotImplementedError("Lamb-Oseen vortex not implemented yet.")
        else:
            raise ValueError(f"Unknown vortex type: {vortex}")
        
        # Add asymmetry due to storm motion
        angles_to_points = bearing(lat, lon, latitudes, longitudes)
        
        # Angle difference: positive = right of track, negative = left
        angle_diff = (angles_to_points - row["bearing_deg"] + 180) % 360 - 180
        
        # Right side gets wind boost, left side gets reduction
        v_asymmetry = asymmetry_factor * row["velocity_kt"] * np.sin(np.radians(angle_diff))
        v_total = v_symmetric + v_asymmetry
        
        v_max = np.maximum(v_max, v_total)
    
    return v_max


# -------------------------------------------------------------------------
# Hazard Model
# -------------------------------------------------------------------------

class TropicalCycloneHazard(HazardModel):
    """
    Tropical Cyclone hazard model.
    
    Computes wind speed at geographic locations based on storm track data
    and a parametric wind field model.
    """

    def __init__(self, track_data: pd.DataFrame, vortex_model: str = "rankine"):
        """
        Initialize with storm track data and intensity model.
        
        Args:
            track_data: DataFrame with columns ['time', 'latitude', 'longitude',
                        'max_wind_speed_kt', 'radius_max_wind_nm'].
            vortex_model: Which vortex model to use (e.g., RankineVortex).
        """
        self.track_data = track_data
        self.vortex_model = vortex_model

    def compute_intensity(self, coordinates: Coordinates) -> IntensityValues:
        """
        Compute wind speed at given locations.
        
        Args:
            coordinates: Shape (N, 2) array of [lat, lon] pairs.
        
        Returns:
            Shape (N,) array of wind speed values in knots.
        """
        points = [(lat, lon) for lat, lon in coordinates]
        wind_speeds = _max_wind_speeds_at_locations(
            self.track_data,
            points,
            vortex=self.vortex_model
        )
        return wind_speeds

    @property
    def peril_type(self) -> str:
        """Return peril identifier: 'TC' for Tropical Cyclone."""
        return 'TC'
    
