import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# -------------------------------------------------------------------------
# SECTION: Track Data Utilities
# -------------------------------------------------------------------------
def track_interpolation(df, time_step='30min',kind='linear'):
    """Interpolate the track data to finer time intervals (e.g., hourly)."""

    # Convert time to datetime
    df['time'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')

    # Create a new time index with hourly frequency
    new_time_index = pd.date_range(start=df['time'].min(), end=df['time'].max(), freq=time_step)

    # Interpolate latitude and longitude
    lat_interp = interp1d(df['time'].astype(int), df['latitude'], kind=kind, fill_value="extrapolate")
    lon_interp = interp1d(df['time'].astype(int), df['longitude'], kind=kind, fill_value="extrapolate")
    wind_interp = interp1d(df['time'].astype(int), df['max_wind_speed_kt'], kind=kind, fill_value="extrapolate")
    radius_max_wind_interp = interp1d(df['time'].astype(int), df['radius_max_wind_nm'], kind=kind, fill_value="extrapolate")

    # Create new DataFrame with interpolated values
    interp_df = pd.DataFrame({
        'time': new_time_index,
        'latitude': lat_interp(new_time_index.astype(int)),
        'longitude': lon_interp(new_time_index.astype(int)),
        'max_wind_speed_kt': wind_interp(new_time_index.astype(int)),
        'radius_max_wind_nm': radius_max_wind_interp(new_time_index.astype(int))
    })

    return interp_df

def extract_past_trajectory(df,timestamp, tech='OFCL'):
    df_ptrac = df[df['TECH'] == tech].copy()
    df_ptrac = df_ptrac[df_ptrac['timestamp'] <= timestamp]
    df_ptrac = df_ptrac[df_ptrac['tau'] == 0]

    # Remove duplicates
    df_ptrac = df_ptrac.drop_duplicates(subset=['timestamp'], keep='last')
    
    return df_ptrac

def extract_future_trajectory(df,timestamp, tech='OFCL'):
    df_ftrac = df[df['TECH'] == tech].copy()
    df_ftrac = df_ftrac[df_ftrac['timestamp'] == timestamp]
    #df_ftrac = df_ftrac[df_ftrac['tau'] != 0]

    # Remove duplicates
    df_ftrac = df_ftrac.drop_duplicates(subset=['tau'], keep='last')

    # Add tau to timestamp to get future timestamps
    df_ftrac['timestamp'] = df_ftrac['timestamp'] + pd.to_timedelta(df_ftrac['tau'], unit='h')
    
    return df_ftrac

# -------------------------------------------------------------------------
# SECTION: Geospatial Utilities
# -------------------------------------------------------------------------

def haversine_vectorized(lat1, lon1, lat2_array, lon2_array):
    """
    Calculate the great circle distance between a single storm point (lat1, lon1) 
    and many vectorized object (e.g. house) points (lat2_array, lon2_array).
    Returns distance in Nautical Miles.
    """
    R = 3440.065  # Earth radius in Nautical Miles
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_array)
    
    delta_phi = np.radians(lat2_array - lat1)
    delta_lambda = np.radians(lon2_array - lon1)
    
    a = np.sin(delta_phi / 2)**2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2)**2
        
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c
# -------------------------------------------------------------------------
# SECTION: Unit Conversion Utilities
# -------------------------------------------------------------------------
def nm2km(nm):
    """
    Convert Nautical Miles to Kilometers.
    """
    return nm * 1.852

def kt2kmh(kt):
    """
    Convert Knots to Kilometers per Hour.
    """
    return kt * 1.852

# -------------------------------------------------------------------------
# SECTION: Cyclone Motion Utilities
# -------------------------------------------------------------------------
def cyclone_velocity(df):
    """Calculate the cyclone velocity between consecutive track points."""
    # Calculate distance between consecutive points
    distances = haversine_vectorized(
        df['latitude'].iloc[:-1].values,
        df['longitude'].iloc[:-1].values,
        df['latitude'].iloc[1:].values,
        df['longitude'].iloc[1:].values
    )
    
    # Calculate time differences in hours (use .values to avoid index alignment issues)
    time_diffs = (df['time'].iloc[1:].values - df['time'].iloc[:-1].values) / np.timedelta64(1, 'h')
    
    # Calculate velocity in km/h
    velocities = distances / time_diffs
    
    # Append NaN for the last point to maintain the same length
    velocities = np.append(velocities, 0)
    
    df['velocity_kt'] = velocities
    return df

def beraring(lat1, lon1, lat2, lon2):
    """Calculate the bearing from point 1 to point 2."""
    import numpy as np

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    diffLong = np.radians(lon2 - lon1)

    x = np.sin(diffLong) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diffLong))

    initial_bearing = np.arctan2(x, y)

    # Now we have the initial bearing but math.atan2 return values from -π to +π
    # so we need to normalize the result to 0° ... 360°
    initial_bearing = np.degrees(initial_bearing)

    return (initial_bearing + 360) % 360

def cyclone_bearing(df):
    """Calculate the cyclone bearing between consecutive track points."""
    bearings = beraring(
        df['latitude'].iloc[:-1].values,
        df['longitude'].iloc[:-1].values,
        df['latitude'].iloc[1:].values,
        df['longitude'].iloc[1:].values
    )
    
    # Append NaN for the last point to maintain the same length
    bearings = np.append(bearings, 0)
    
    df['bearing_deg'] = bearings
    return df
