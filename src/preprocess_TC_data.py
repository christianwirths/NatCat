import pandas as pd
import os


A_DECK_COLS = [
    "BASIN", "CY", "YYYYMMDDHH", "TECH", "TAU", "LAT", "LON", 
    "VMAX", "MSLP", "TY", "RAD", "WINDCODE", "RAD1", "RAD2", 
    "RAD3", "RAD4", "POUTER", "ROUTER", "RMW", "GUSTS", "EYE", 
    "SUBREGION", "MAXSEAS", "INITIALS", "DIR", "SPEED", "STORMNAME", 
    "DEPTH", "SEAS", "SEASCODE", "SEAS1", "SEAS2", "SEAS3", "SEAS4"
]

B_DECK_COLS = [
    "BASIN", "CY", "YYYYMMDDHH", "MIN", "TECH", "TAU", "LAT", "LON", 
    "VMAX", "MSLP", "TY", "RAD", "WINDCODE", "RAD1", "RAD2", 
    "RAD3", "RAD4", "POUTER", "ROUTER", "RMW", "GUSTS", "EYE", 
    "SUBREGION", "MAXSEAS", "INITIALS", "DIR", "SPEED", "STORMNAME", 
    "DEPTH", "SEAS", "SEASCODE", "SEAS1", "SEAS2", "SEAS3", "SEAS4","A","B","C","D","E"
]

def parse_lat_lon(coord_str):
    if pd.isna(coord_str): return None
    coord_str = str(coord_str).strip()
    val = float(coord_str[:-1]) / 10.0
    if coord_str[-1] in ['S', 'W']: val *= -1
    return val

def clean_track_data(file_path, is_best_track=True):
    ATCF_COLS = B_DECK_COLS if is_best_track else A_DECK_COLS
    df = pd.read_csv(file_path, names=ATCF_COLS, sep=",", skipinitialspace=True, on_bad_lines='warn')

    # Filter Logic
    if is_best_track:
        # Best tracks are marked as 'BEST' in the TECH column
        df = df[df['TECH'] == 'BEST'].copy()
    else:
        # Existing logic for forecasts
        df = df[df['TECH'] == 'OFCL'].copy()
    
    # Standard cleaning
    df['latitude'] = df['LAT'].apply(parse_lat_lon)
    df['longitude'] = df['LON'].apply(parse_lat_lon)
    df['timestamp'] = pd.to_datetime(df['YYYYMMDDHH'], format='%Y%m%d%H')
    
    # Rename columns for clarity
    final_df = df[[
        'timestamp', 'latitude', 'longitude', 
        'VMAX','RMW', 'MSLP', 'TY' # TY = Storm Type (Hurricane, TD, TS)
    ]].rename(columns={
        'VMAX': 'max_wind_speed_kt',
        'RMW': 'radius_max_wind_nm',
        'MSLP': 'min_pressure_mb',
        'TY': 'storm_type'
    })
    
    return final_df

if __name__ == "__main__":
    # Test on the Best Track file
    input_file = "data/raw/bal142018.dat"
    output_file = "data/processed/michael_2018_best_track.csv"
    
    if os.path.exists(input_file):
        df = clean_track_data(input_file, is_best_track=True)
        df.to_csv(output_file, index=False)
        print(f"Saved Best Track: {len(df)} rows.")
        print(df.head())