import pandas as pd
import numpy as np
import os
from climada.util.api_client import Client
from climada.entity import Exposures

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


# Exctract LitPop 
def get_country_exposure_via_api(country_name, subregion_bounds=None):
    print("Initializing CLIMADA Data Client...")
    client = Client()
    
    # Fetch LitPop data from CLIMADA API
    print(f"Downloading/Loading {country_name} exposure data (this may take a moment)...")
    exp_country = client.get_litpop(country=country_name)
    
    # Subregion filtering
    if subregion_bounds is not None:
        lat_min, lat_max, lon_min, lon_max = subregion_bounds
        print(f"Filtering for subregion: {subregion_bounds}...")
        
        # geometry.y for Latitude and geometry.x for Longitude
        subset_gdf = exp_country.gdf[
            (exp_country.gdf.geometry.y >= lat_min) & 
            (exp_country.gdf.geometry.y <= lat_max) & 
            (exp_country.gdf.geometry.x >= lon_min) & 
            (exp_country.gdf.geometry.x <= lon_max)
        ]
        
        # Create a new Exposures object from the subset
        exp_country = Exposures(subset_gdf)

    exp_country.check()
    
    return exp_country


# Exposure heuristics
def construction_type_heuristic(values, seed=42):
    # Using a sigmoid function that assigns a Masory probability based on value 
    threshold = 1e6  
    steepness = 1e-7
    prob_masonry = 1 / (1 + np.exp(-steepness * (values - threshold)))

    np.random.seed(seed)
    rand_vals = np.random.rand(len(values))
    construction_types = np.where(rand_vals < prob_masonry, 'Masonry', 'Frame') 

    return construction_types


    

def convert_to_loss_model_format(climada_exposure):
    # Extract the raw GeoDataFrame from CLIMADA
    gdf = climada_exposure.gdf
    
    num_locs = len(gdf)
    lats = gdf.geometry.y.values
    lons = gdf.geometry.x.values
    
    values = gdf['value'].values
    
    # Add construction types based on heuristic
    construction_types = construction_type_heuristic(values)
    constructions = construction_types

    # Create DataFrame
    df = pd.DataFrame({
        'location_id': range(1, num_locs + 1),
        'latitude': lats,
        'longitude': lons,
        'tiv': values.round(2), # Total Insured Value
        'construction': constructions
    })
    
    return df