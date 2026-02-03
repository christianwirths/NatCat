"""
Tropical cyclone track data processing utilities.

Functions for extracting, interpolating, and enriching storm track data
with derived fields like velocity and bearing.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .geo import haversine_distance, bearing


def track_interpolation(df: pd.DataFrame, time_step: str = '5min', kind: str = 'linear') -> pd.DataFrame:
    """
    Interpolate track data to finer time intervals.
    
    Args:
        df: DataFrame with 'timestamp', 'latitude', 'longitude', 'max_wind_speed_kt',
            'radius_max_wind_nm' columns.
        time_step: Pandas frequency string for interpolation (e.g., '5min', '1H').
        kind: Interpolation method ('linear', 'cubic', etc.).
    
    Returns:
        DataFrame with interpolated track points at specified time step.
    """
    # Convert time to datetime
    df['time'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    # If only one data point, return as-is (no interpolation needed)
    if len(df) <= 1:
        return df

    # Create a new time index with specified frequency
    new_time_index = pd.date_range(start=df['time'].min(), end=df['time'].max(), freq=time_step)

    # If start and end times are identical, return original data
    if len(new_time_index) <= 1:
        return df

    # Interpolate latitude, longitude, wind speed, and RMW
    lat_interp = interp1d(df['time'].astype(int), df['latitude'], 
                         kind=kind, fill_value="extrapolate", bounds_error=False)
    lon_interp = interp1d(df['time'].astype(int), df['longitude'], 
                         kind=kind, fill_value="extrapolate", bounds_error=False)
    wind_interp = interp1d(df['time'].astype(int), df['max_wind_speed_kt'], 
                          kind=kind, fill_value="extrapolate", bounds_error=False)
    radius_max_wind_interp = interp1d(df['time'].astype(int), df['radius_max_wind_nm'], 
                                     kind=kind, fill_value="extrapolate", bounds_error=False)

    # Create new DataFrame with interpolated values
    interp_df = pd.DataFrame({
        'time': new_time_index,
        'latitude': lat_interp(new_time_index.astype(int)),
        'longitude': lon_interp(new_time_index.astype(int)),
        'max_wind_speed_kt': wind_interp(new_time_index.astype(int)),
        'radius_max_wind_nm': radius_max_wind_interp(new_time_index.astype(int))
    })

    return interp_df


def extract_past_trajectory(df: pd.DataFrame, timestamp: pd.Timestamp, tech: str = 'OFCL') -> pd.DataFrame:
    """
    Extract past trajectory from A-deck forecast data.
    
    Args:
        df: A-deck DataFrame with 'TECH', 'timestamp', and 'tau' columns.
        timestamp: Reference timestamp.
        tech: Model/technology identifier (e.g., 'OFCL', 'HWRF').
    
    Returns:
        DataFrame with past trajectory points (tau=0, timestamp <= reference).
    """
    df_ptrac = df[df['TECH'] == tech].copy()
    df_ptrac = df_ptrac[df_ptrac['timestamp'] <= timestamp]
    df_ptrac = df_ptrac[df_ptrac['tau'] == 0]

    # Remove duplicates
    df_ptrac = df_ptrac.drop_duplicates(subset=['timestamp'], keep='last')
    
    return df_ptrac


def extract_future_trajectory(df: pd.DataFrame, timestamp: pd.Timestamp, tech: str = 'OFCL') -> pd.DataFrame:
    """
    Extract future forecast trajectory from A-deck data.
    
    Args:
        df: A-deck DataFrame with 'TECH', 'timestamp', and 'tau' columns.
        timestamp: Reference timestamp for forecast initialization.
        tech: Model/technology identifier (e.g., 'OFCL', 'HWRF').
    
    Returns:
        DataFrame with future trajectory points from the forecast.
    """
    df_ftrac = df[df['TECH'] == tech].copy()
    df_ftrac = df_ftrac[df_ftrac['timestamp'] == timestamp]

    # Remove duplicates by tau
    df_ftrac = df_ftrac.drop_duplicates(subset=['tau'], keep='last')

    # Add tau to timestamp to get future timestamps
    df_ftrac['timestamp'] = df_ftrac['timestamp'] + pd.to_timedelta(df_ftrac['tau'], unit='h')
    
    # Remove any duplicate timestamps that may have been created
    df_ftrac = df_ftrac.drop_duplicates(subset=['timestamp'], keep='last')
    
    return df_ftrac


def cyclone_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cyclone translation velocity between consecutive track points.
    
    Args:
        df: DataFrame with 'time', 'latitude', 'longitude' columns.
    
    Returns:
        DataFrame with added 'velocity_kt' column (velocity in knots).
    """
    # Calculate distance between consecutive points
    distances = haversine_distance(
        df['latitude'].iloc[:-1].values,
        df['longitude'].iloc[:-1].values,
        df['latitude'].iloc[1:].values,
        df['longitude'].iloc[1:].values,
        unit='nm'
    )
    
    # Calculate time differences in hours
    time_diffs = (df['time'].iloc[1:].values - df['time'].iloc[:-1].values) / np.timedelta64(1, 'h')
    
    # Calculate velocity in knots (nautical miles per hour)
    velocities = distances / time_diffs
    
    # Forward-fill last point with previous velocity (NOT 0, which causes artifacts)
    if len(velocities) > 0:
        velocities = np.append(velocities, velocities[-1])
    else:
        velocities = np.array([0.0])
    
    df['velocity_kt'] = velocities
    return df


def cyclone_bearing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cyclone heading/bearing between consecutive track points.
    
    Args:
        df: DataFrame with 'latitude', 'longitude' columns.
    
    Returns:
        DataFrame with added 'bearing_deg' column (bearing in degrees).
    """
    bearings = bearing(
        df['latitude'].iloc[:-1].values,
        df['longitude'].iloc[:-1].values,
        df['latitude'].iloc[1:].values,
        df['longitude'].iloc[1:].values
    )
    
    # Forward-fill last point with previous bearing (NOT 0, which causes artifacts)
    if len(bearings) > 0:
        bearings = np.append(bearings, bearings[-1])
    else:
        bearings = np.array([0.0])
    
    df['bearing_deg'] = bearings
    return df


def prepare_track_data(
    df: pd.DataFrame, 
    timestamp: pd.Timestamp, 
    model: str, 
    avail: int = 1, 
    past: bool = True
) -> tuple[pd.DataFrame, int]:
    """
    Prepare track data for loss estimation.
    
    Extracts trajectory, fills missing RMW using heuristics, interpolates to fine
    time resolution, and calculates velocity/bearing.
    
    Args:
        df: A-deck DataFrame containing forecast data.
        timestamp: Reference timestamp for extracting trajectories.
        model: Model name/technology identifier (e.g., 'OFCL', 'HWRF').
        avail: Availability flag (1 = available, 0 = unavailable).
        past: If True, extract past trajectory; if False, extract future forecast.
        
    Returns:
        df_out: Processed track DataFrame with interpolated points and derived fields.
        avail: Updated availability flag (0 if processing failed).
    """
    # Import here to avoid circular dependency
    from hazards.tropical_cyclone import get_heuristic_rmw
    
    if past:
        df_out = extract_past_trajectory(df, timestamp, tech=model)
    else: 
        df_out = extract_future_trajectory(df, timestamp, tech=model)

    df_out = get_heuristic_rmw(df_out)
    
    try:
        # Use linear interpolation for past data to ensure monotonic damages
        # Cubic interpolation can change when new points are added
        interp_kind = 'linear' if past else 'cubic'
        df_out = track_interpolation(df_out, time_step='5min', kind=interp_kind)
    except:
        avail = 0
        return df_out, avail

    # Add cyclone bearing and velocity
    df_out = cyclone_velocity(df_out)
    df_out = cyclone_bearing(df_out)

    return df_out, avail
