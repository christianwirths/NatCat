
import sys
import numpy as np
#sys.path.append('../src')
from preprocess_TC_data import clean_track_data
from utils import nm2km, kt2kmh, haversine_vectorized, track_interpolation, beraring, cyclone_velocity, cyclone_bearing
from wind_fields import max_wind_speeds_at_locations
from generate_exposure import generate_synthetic_portfolio
from vulnerbility import get_damage_ratio




def calculate_tc_impacts(tc_track_file, housing_portfolio,full_report=False):
    """Function to calculate tropical cyclone impacts on a housing portfolio.
    Args:
        tc_track_file (str): Path to the tropical cyclone track file.
        housing_portfolio (pd.DataFrame): DataFrame containing the housing portfolio with 'latitude', 'longitude', and 'tiv' columns.
        full_report (bool): If True, prints a detailed report of the impacts.
        Returns: 
        total_damage (float): Estimated total damage to the portfolio.
        portfolio_damages (np.ndarray): Array of damages for each location in the portfolio.
    """
    
    # Clean and load the tropical cyclone track data
    tc_data = clean_track_data(tc_track_file, is_best_track=True)
    tc_data = track_interpolation(tc_data, time_step='5min')
    
    # Drop all rows with wind speed smaller than 40 knots due to minor threshold 
    tc_data = tc_data[tc_data['max_wind_speed_kt'] >= 40].reset_index(drop=True)
    
    #Add TC velocity and def beraring:

    tc_data = cyclone_velocity(tc_data)
    tc_data = cyclone_bearing(tc_data)

    # Get lon lat touples for portfolio
    points = list(zip(housing_portfolio['latitude'], housing_portfolio['longitude']))
    max_wind_speeds = max_wind_speeds_at_locations(tc_data, points, asymmetry_factor=1.0)
    
    
    # Compute damage ratios based on the maximum wind speeds
    damage_ratios = get_damage_ratio(max_wind_speeds)

    portfolio_damages = damage_ratios * housing_portfolio['tiv']
    total_damage = np.sum(portfolio_damages)
    print(f"Estimated total damage to portfolio: ${total_damage:,.2f}")


    if full_report:
        # Mean damage ratio
        mean_damage_ratio = np.mean(damage_ratios)
        print(f"Mean Damage Ratio: {mean_damage_ratio:.4f}")
        # Max damage ratio
        max_damage_ratio = np.max(damage_ratios)
        print(f"Max Damage Ratio: {max_damage_ratio:.4f}")
        # Locations with damage ratio > 0.5
        high_damage_locs = housing_portfolio[damage_ratios > 0.5]
        print(f"Number of locations with damage ratio > 0.5: {len(high_damage_locs)}")
        
        if len(high_damage_locs) > 0:
        #Mean damage ratio for high damage locations
            mean_high_damage_ratio = np.mean(damage_ratios[damage_ratios > 0.5])
            print(f"Mean Damage Ratio for locations with damage ratio > 0.5: {mean_high_damage_ratio:.4f}")

        


    return total_damage, portfolio_damages


if __name__ == "__main__":
    import pandas as pd

    # Example housing portfolio
    housing_portfolio = generate_synthetic_portfolio(num_locs=1000)


    _,_ = calculate_tc_impacts("../data/raw/bal142018.dat", housing_portfolio, full_report=True)