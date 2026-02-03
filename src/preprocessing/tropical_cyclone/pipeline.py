"""
Full preprocessing pipeline for tropical cyclone data in A and B deck formats.
"""

from .download import download_storm_forecast, extract_gzip
from .track import clean_track_data
from utils.track import cyclone_velocity, cyclone_bearing, track_interpolation
from hazards.tropical_cyclone import get_heuristic_rmw
import pandas as pd
import numpy as np


#----------------------------------------------------------------------------
# Preprocess B-deck data
#----------------------------------------------------------------------------

def preprocess_B_deck(year: int, basin: str, storm_number: str, verbose: int = 0) -> 'pd.DataFrame':
    """
    Preprocess A-deck data for a specified tropical cyclone.
    
    Args:
        year: Year of the storm (e.g., 2018)
        basin: Basin code (e.g., 'al' for Atlantic)
        storm_number: Storm number as a string (e.g., '14')
    """

    # Check fromatting of inputs
    if len(storm_number) != 2 or not storm_number.isdigit():
        raise ValueError("storm_number must be a 2-digit string, e.g., '01', '14'")
    
    if len(basin) != 2 or not basin.isalpha():
        raise ValueError("basin must be a 2-letter string, e.g., 'al', 'ep'")
    
    if type(year) is not int or year < 1800 or year > 2100:
        raise ValueError("year must be a valid integer year, e.g., 2018")


    # Download the B-deck (best track) data for more reliable testing
    if verbose > 0:
        print(f"Downloading B-deck data for storm {basin}{storm_number} in year {year}...")
    gz_file = download_storm_forecast(year=year, basin=basin, storm_id=storm_number, deck='b')

    # Extract the compressed file
    if gz_file:
        track_file = extract_gzip(gz_file)
        if verbose > 0:
            print(f"Extracted track file: {track_file}")
    else:
        raise FileNotFoundError("Failed to download storm data")
    
    # Load and clean track data (B-deck best track)
    df_track = clean_track_data(track_file, is_best_track=True)

    # Create 'time' column (track_interpolation/cyclone_velocity/bearing expect 'time', not 'timestamp')
    # For B deck we need to create time from timestamp
    df_track['time'] = df_track['timestamp']

    # Interpolate track to higher time resolution
    df_track = track_interpolation(df_track, time_step='5min')

    # Fill missing RMW values using heuristics
    df_track = get_heuristic_rmw(df_track)

    # Calculate storm velocity and bearing
    df_track = cyclone_velocity(df_track)
    df_track = cyclone_bearing(df_track)

    if verbose > 1:
        print(f"Track data loaded: {len(df_track)} points")
        print(f"Time range: {df_track['time'].min()} to {df_track['time'].max()}")
        print(f"\nFirst few points:")
        df_track.head()


    return df_track

#----------------------------------------------------------------------------
# Preprocess A-deck data
#----------------------------------------------------------------------------