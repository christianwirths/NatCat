
import sys
import numpy as np
import pandas as pd
#sys.path.append('../src')
from preprocess_TC_data import clean_track_data
from utils import nm2km, kt2kmh, haversine_vectorized, track_interpolation, beraring, cyclone_velocity, cyclone_bearing, prepare_track_data
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


def portfolio_damage_model(df_tc_track, housing_portfolio, asymmetry_factor=1.0):
    """Calculate damage to a housing portfolio given a tropical cyclone track.
    
    Args:
        df_tc_track: DataFrame with cyclone track data (interpolated with velocity/bearing)
        housing_portfolio: DataFrame with 'latitude', 'longitude', and 'tiv' columns
        asymmetry_factor: Wind field asymmetry factor (default 1.0)
        
    Returns:
        DataFrame with original portfolio data plus 'damage_ratio' and 'damage' columns
    """
    # Get location tuples for portfolio
    points = list(zip(housing_portfolio['latitude'], housing_portfolio['longitude']))
    max_wind_speeds = max_wind_speeds_at_locations(df_tc_track, points, asymmetry_factor=asymmetry_factor)
    
    # Compute damage ratios based on the maximum wind speeds
    damage_ratios = get_damage_ratio(max_wind_speeds)
    portfolio_damages = damage_ratios * housing_portfolio['tiv']
    
    # Append portfolio damages to housing_portfolio dataframe
    housing_portfolio = housing_portfolio.copy()
    housing_portfolio['damage_ratio'] = damage_ratios
    housing_portfolio['damage'] = portfolio_damages

    return housing_portfolio


def impact_tc_A_deck(df_tc_A_deck, housing_portfolio, timestamp=None, 
                      trusted_models=['OFCL', 'AVNO', 'EMXI', 'HWRF', 'CTCX', 'UKX'], 
                      full_report=False):
    """Calculate impact of a tropical cyclone given its A-deck data and a housing portfolio.
    
    This function processes forecast data from multiple models, computing both historical
    (past) damage and forecasted (future) damage estimates.
    
    Args:
        df_tc_A_deck: DataFrame containing the A-deck forecast data (from clean_track_data2)
        housing_portfolio: DataFrame with 'latitude', 'longitude', and 'tiv' columns
        timestamp: Reference timestamp (if None, uses latest timestamp in data)
        trusted_models: List of forecast model names to process
        full_report: If True, return detailed intermediate results
        
    Returns:
        past_damages: List of portfolio DataFrames with past damage for each available model
        future_damages: List of portfolio DataFrames with future damage for each available model
        history_avail: Array indicating which models had valid historical data (1=yes, 0=no)
        future_avail: Array indicating which models had valid forecast data (1=yes, 0=no)
    """
    if timestamp is None:
        # Use latest timestamp as reference
        timestamp = df_tc_A_deck['timestamp'].max()
    
    df_past_list = []
    df_future_list = []

    history_avail = np.ones(len(trusted_models))
    future_avail = np.ones(len(trusted_models))
    
    # Process each model's track data
    for i, model in enumerate(trusted_models):
        # Past trajectory (initialization data)
        df_past, history_avail[i] = prepare_track_data(
            df_tc_A_deck, timestamp, model, avail=history_avail[i], past=True
        )
        df_past_list.append(df_past)

        # Future trajectory (forecasted data)
        df_future, future_avail[i] = prepare_track_data(
            df_tc_A_deck, timestamp, model, avail=future_avail[i], past=False
        )
        df_future_list.append(df_future)

    # Calculate damages for available models
    past_damages = []
    future_damages = []
    
    for idx, model in enumerate(trusted_models):
        if history_avail[idx] == 1:
            df_past = df_past_list[idx]
            portfolio_damage = portfolio_damage_model(df_past, housing_portfolio)
            past_damages.append(portfolio_damage)

    for idx, model in enumerate(trusted_models):
        if future_avail[idx] == 1:
            df_future = df_future_list[idx]
            portfolio_damage = portfolio_damage_model(df_future, housing_portfolio)
            future_damages.append(portfolio_damage)
    
    return past_damages, future_damages, history_avail, future_avail


if __name__ == "__main__":
    import pandas as pd

    # Example housing portfolio
    housing_portfolio = generate_synthetic_portfolio(num_locs=1000)


    _,_ = calculate_tc_impacts("../data/raw/bal142018.dat", housing_portfolio, full_report=True)