import pandas as pd
import numpy as np
import os

def generate_synthetic_portfolio(num_locs=1000):
    """
    Generates a synthetic portfolio of properties around Panama City, FL.
    Target Box: Lat 29.5 to 30.5, Lon -86.0 to -85.0
    """
    np.random.seed(42)
    
    # 1. Generate Locations (Uniform distribution in the box)
    lats = np.random.uniform(29.5, 30.8, num_locs)
    lons = np.random.uniform(-86.0, -84.8, num_locs)
    
    # 2. Generate Values (Log-normal distribution to simulate real estate prices)
    # Mean of 300k, some expensive beach houses
    values = np.random.lognormal(mean=12.6, sigma=0.5, size=num_locs)
    
    # 3. Assign Construction Type
    # 70% Frame (Wood), 30% Masonry (Brick/Concrete)
    constructions = np.random.choice(['Frame', 'Masonry'], num_locs, p=[0.7, 0.3])
    
    df = pd.DataFrame({
        'location_id': range(1, num_locs + 1),
        'latitude': lats,
        'longitude': lons,
        'tiv': values.round(2), # Total Insured Value
        'construction': constructions
    })
    
    return df