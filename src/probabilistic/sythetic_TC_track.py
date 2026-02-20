"""
Synthetic tropical cyclone track generation module.
"""


import glob
import os
import pandas as pd
from scipy import stats
import numpy as np
from global_land_mask import globe

from preprocessing.tropical_cyclone import clean_track_data, check_a_deck_quality
from utils.track import cyclone_velocity, cyclone_bearing, track_interpolation
from hazards.tropical_cyclone import get_heuristic_rmw

def load_historical_tracks(folder_path: str, basin: str = "al") -> tuple[list, list]:
    """
    Load historical tropical cyclone tracks from a specified folder path.
    Currently only supports loading from ATCF B-deck files. Future versions may include support for A-deck and other formats.
    
    Args:
        folder_path (str): Path to the folder containing historical track data files.
        basin (str): Basin identifier (e.g., 'al' for Atlantic, 'ep' for East Pacific).

    Returns:
        tuple:
            - list: DataFrames containing successfully loaded historical track data.
            - list: File paths that failed to load, each as a (path, error) tuple.
    """

    track_files = glob.glob(os.path.join(folder_path, f"b*{basin}*.dat"))

    track_data = []
    failed_files = []
    for file in track_files:
        try:
            df_track = clean_track_data(file, is_best_track=True)

            passed, reason = check_a_deck_quality(df_track)
            if not passed:
                failed_files.append((file, ValueError(f"Quality check failed: {reason}")))
                continue

            df_track['time'] = df_track['timestamp']
            df_track = track_interpolation(df_track, time_step='3h')
            df_track = get_heuristic_rmw(df_track)
            df_track = cyclone_velocity(df_track)
            df_track = cyclone_bearing(df_track)

            track_data.append(df_track)
        except Exception as e:
            failed_files.append((file, e))

    return track_data, failed_files




def get_tc_origin(track_data: list, num_syntetic_origins: int) -> pd.DataFrame:
    """
    Generates new syntetic storm origins based on the hisotical distribution using gaussian_kde 

    Args:
        track_data (list): List of pandas Dataframe holding the track data of individual cyclones
        num_syntetic_origins (int): Number of synthetic origins to generarte
    
    Returns: 
        synthetic_origins (pd.Dataframe): Contains num_syntetic_origins points of syntehtically generated strom origin points. 
        df_history (pf.DataFrame): Contains the historical points of origin.
        kde (class):  the gaussian_kde   
    """
    
    lat = []
    lon = []
    time= []

    for track in track_data: 
        first = track.iloc[0]
        lat.append(first['latitude'])
        lon.append(first['longitude'])
        time.append(first['time'])

    #Transform to pd.Dataframe 
    df_history = pd.DataFrame({'latitude': lat, 'longitude': lon, 'time': time})

    # Drop any rows with NaN/inf in lat/lon (can occur from interpolation edge cases)
    df_history = df_history.replace([np.inf, -np.inf], np.nan).dropna(subset=['latitude', 'longitude'])

    # Transform time into day of the year
    df_history['day_of_year'] = df_history['time'].dt.dayofyear

    historic_origin = np.vstack([df_history['latitude'].values,df_history['longitude'].values])
    #TODO: Check for day of the year stability of also adding day of year sampling

    #Fit a kde to historical data 
    kde = stats.gaussian_kde(historic_origin)

    #Generate synthetic 
    valid_batches = []
    points_collected = 0
    
    # Smart Buffer: Ask for 20% more than we need to minimize loop iterations
    to_sample = int(num_syntetic_origins * 1.2)

    while points_collected < num_syntetic_origins:
        synthetic_origins = kde.resample(to_sample)
    
        synthetic_lats = synthetic_origins[0, :]
        synthetic_lons = synthetic_origins[1, :]

        #Reject spawns over land
        is_on_land = globe.is_land(synthetic_lats, synthetic_lons)
        ocean_mask = ~is_on_land

        valid_batch = np.column_stack((
            synthetic_lats[ocean_mask], 
            synthetic_lons[ocean_mask], 
        ))
        
        valid_batches.append(valid_batch)
        points_collected += len(valid_batch)

        if points_collected < num_syntetic_origins:
            deficit = num_syntetic_origins - points_collected
            to_sample = int(deficit * 1.2)

    # Combine and crop to num_syntetic_origins
    final_origin = np.vstack(valid_batches)[:num_syntetic_origins]

    synthetic_origins_df = pd.DataFrame({
        'storm_id': [f"SYN_{i}" for i in range(num_syntetic_origins)],
        'latitude': final_origin[:,0],
        'longitude': final_origin[:,1]
    })
    
    return synthetic_origins_df, df_history, kde


def prepare_mcmc_data(track_data: list) -> pd.DataFrame:
    """
    Prepares our track data to build an emperical MC 
    """


    print(f"Stacking {len(track_data)} storm tracks...")
    
    processed_dfs = []

    # Create strom ID: 
    for i, df in enumerate(track_data):
        temp_df = df.copy()
        temp_df['storm_id'] = f"HIST_{i}" 
        processed_dfs.append(temp_df)

    master_df = pd.concat(processed_dfs, ignore_index=True)

    #Add day_of_year
    master_df['day_of_year'] = pd.to_datetime(master_df['date']).dt.dayofyear

    # Get variable deltas 
    master_df['delta_vmax'] = master_df.groupby('storm_id')['wind_speed'].diff()
    master_df['delta_radius'] = master_df.groupby('storm_id')['radius'].diff()

    # Drop rows with NaNs (the first hour of every storm)
    master_df = master_df.dropna(subset=['delta_vmax', 'bearing', 'moving_velocity'])

    return master_df