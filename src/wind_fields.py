import numpy as np
from utils import haversine_vectorized, beraring

# Wind velocity at arbitrary latitude and longitude we calculate by using a Rankine vortex
# https://en.wikipedia.org/wiki/Rankine_vortex

def rankine_vortex(r,max_wind_speed, radius_max_wind,exponent=2):
    """Args
    r: distance from the center of the storm
    max_wind_speed: maximum wind speed at the radius of maximum wind
    radius_max_wind: radius at which the maximum wind speed occurs
    exponent: exponent for the decay of wind speed with distance #TODO: default is 2, but maybee other values might fit better

    Returns the wind speed at distance r from the center of the storm. 
    """
    
    #Masking such it can be used vectorized: 
    inside_eye = r < radius_max_wind
    outside_eye = r >= radius_max_wind

    v_at_r = np.zeros_like(r, dtype=float)

    
    v_at_r[inside_eye] = max_wind_speed * (r[inside_eye] / radius_max_wind)  
    v_at_r[outside_eye] = max_wind_speed * (radius_max_wind / r[outside_eye]) ** exponent

    return v_at_r

def get_heuristic_rmw(df):
    """Apply heuristic estimates for radius of maximum wind (RMW) when data is missing.
    
    Uses wind speed-based heuristics to estimate RMW:
    - < 35 kt: 80 nm
    - < 64 kt: 60 nm  
    - < 96 kt: 40 nm
    - < 137 kt: 25 nm
    - >= 137 kt: 15 nm
    
    Also applies 1.5x scaling for extratropical cyclones.
    
    Args:
        df: DataFrame with 'max_wind_speed_kt', 'radius_max_wind_nm', and 'storm_type' columns
        
    Returns:
        DataFrame with filled RMW values
    """
    df_fixed = df.copy()
    winds = df_fixed['max_wind_speed_kt'].fillna(0)
    
    # Wind speed to radius mapping
    conditions = [
        winds < 35,   
        winds < 64,   
        winds < 96,   
        winds < 137   
    ]
    
    choices = [
        80.0, 
        60.0,  
        40.0,  
        25.0   
    ]
    
    heuristic_rmw = np.select(conditions, choices, default=15.0)
    needs_fix_mask = (df_fixed['radius_max_wind_nm'].isna()) | (df_fixed['radius_max_wind_nm'] <= 0)
    
    df_fixed.loc[needs_fix_mask, 'radius_max_wind_nm'] = heuristic_rmw[needs_fix_mask]
    
    # Apply 1.5x scaling for extratropical cyclones
    ex_mask = df_fixed['storm_type'] == 'EX'
    df_fixed.loc[ex_mask, 'radius_max_wind_nm'] *= 1.5
        
    return df_fixed

    
#TODO: Lamb-Oseen Vortex 
# https://en.wikipedia.org/wiki/Lamb–Oseen_vortex


# Calculate max velocity for all points 
def max_wind_speeds_at_locations(df, points, vortex="rankine", asymmetry_factor=0.5):
    """ Calculate the maximum wind speeds at given locations over the storm track.
        Args:
        df (pd.DataFrame): DataFrame containing storm track data with 
            columns 'latitude', 'longitude', 'max_wind_speed_kt', 'radius_max_wind_nm',
            'velocity_kt' (storm translation speed), and 'bearing_deg' (storm heading).
        points (list of tuples): List of (latitude, longitude) tuples representing the locations of interest.
        vortex (str): Type of vortex model to use ('rankine' or 'lamb-oseen').
        asymmetry_factor (float): Fraction of storm translation speed that contributes to asymmetry (typically 0.5-1.0).
        
        Returns:
        np.ndarray: Maximum wind speeds (in knots) at each location over the storm's passage."""
    
    v_rtex_max = np.zeros(len(points), dtype=float)
    latitudes = np.array([point[0] for point in points])
    longitudes = np.array([point[1] for point in points])

    for index, row in df.iterrows():

        lat = row['latitude']
        lon = row['longitude']

        distances = haversine_vectorized(lat, lon, latitudes, longitudes)

        max_wind_speed = row['max_wind_speed_kt']
        radius_max_wind = row['radius_max_wind_nm']

        if vortex == "rankine":
            v_rtex = rankine_vortex(distances, max_wind_speed, radius_max_wind)
        elif vortex == "lamb-oseen":
            raise NotImplementedError("Lamb-Oseen vortex not implemented yet.")
        else:
            raise ValueError("Unknown vortex type. Use 'rankine' or 'lamb-oseen'.")
        
        # Calculate angle from storm center to each point
        angles_to_points = beraring(lat, lon, latitudes, longitudes)
        
        # Angle difference: how far clockwise the point is from storm's heading
        # Normalize to [-180, 180]: positive = right of track, negative = left of track
        angle_diff = (angles_to_points - row["bearing_deg"] + 180) % 360 - 180
        
        # Add asymmetry: right side (positive angle_diff) gets wind boost, left side gets reduction
        # Using sin because max effect is perpendicular to storm motion (90° = right side)
        v_rtex += asymmetry_factor * row["velocity_kt"] * np.sin(np.radians(angle_diff))

        v_rtex_max = np.maximum(v_rtex_max, v_rtex)
    
        
    return v_rtex_max