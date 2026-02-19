"""
Synthetic tropical cyclone track generation module.
"""


import glob
import os

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
