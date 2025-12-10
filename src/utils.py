import numpy as np

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