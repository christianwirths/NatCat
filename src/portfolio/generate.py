"""
Functions to generate different types of test portfolios
"""



from climada.util.api_client import Client
from climada.entity import Exposures
import matplotlib.pyplot as plt

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