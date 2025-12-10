import numpy as np
from utils import haversine_vectorized

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

    
#TODO: Lamb-Oseen Vortex 
# https://en.wikipedia.org/wiki/Lamb–Oseen_vortex



# Calculate max velocity for all points 
def max_wind_speeds_at_locations(df, points,vortex="rankine"):
    """ Calculate the maximum wind speeds at given locations over the storm track.
        Args:
        df (pd.DataFrame): DataFrame containing storm track data with 
        columns 'latitude', 'longitude', 'max_wind_speed_kt', and 'radius_max_wind_nm'.
        points (list of tuples): List of (latitude, longitude) tuples representing the locations of interest."""
    
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
        v_rtex_max = np.maximum(v_rtex_max, v_rtex)
    
        
    return v_rtex_max