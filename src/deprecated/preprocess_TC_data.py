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


def clean_track_data2(file_path, is_best_track=True):
    # Models we want to use
    trusted_models = ['OFCL', 'AVNO', 'EMXI', 'HWRF', 'CTCX', 'UKX']
    # 1. Load without headers (header=None)
    # This prevents pandas from forcing data into the wrong named slots
    # Column names to handel differet row lenghts 
    col_names = list(range(50))

    df = pd.read_csv(file_path, header=None, names=col_names, sep=",", skipinitialspace=True, on_bad_lines='warn')

    # 2. Find the 'TECH' Column dynamically
    # We look for the column that contains 'BEST' or 'OFCL'
    target_tech = 'BEST' if is_best_track else 'OFCL'
    print(f"Looking for TECH column with value '{target_tech}'")
    # Locate the column index that contains the target string
    # We check the first valid row (e.g. row 0) to find where our tech string sits
    # Note: Sometimes A-decks have different techs in the same file, so we check the whole column
    
    tech_col_idx = None
    for col in df.columns:
        if df[col].astype(str).str.contains(target_tech).any():
            tech_col_idx = col
            break
            
    if tech_col_idx is None:
        print(f"Warning: Could not find '{target_tech}' in {file_path}")
        return pd.DataFrame()

    # 3. Define Relative Offsets (Standard ATCF)
    # In standard ATCF, the columns relative to TECH are usually fixed:
    # Date (YYYYMMDDHH) is usually 2 columns BEFORE Tech
    # Tau is usually 1 column AFTER Tech
    # Lat is 2 cols after, Lon is 3 cols after, Vmax is 4 cols after
    
    # # Map the found index to our desired names
    col_mapping = {
        tech_col_idx - 2: 'timestamp_str', # Date
        tech_col_idx:     'TECH',          # Model Name
        tech_col_idx + 1: 'tau',           # Forecast Hour
        tech_col_idx + 2: 'lat_str',
        tech_col_idx + 3: 'lon_str',
        tech_col_idx + 4: 'max_wind_speed_kt',
        tech_col_idx + 5: 'min_pressure_mb',
        tech_col_idx + 6: 'storm_type',
        tech_col_idx + 15: 'radius_max_wind_nm' # RMW is usually far out (index 19 in 0-base, so +16 approx) or 17 not clear 
        # Note: RMW index can be unstable. Using a fixed offset is risky for RMW.
    }
    #col_mapping = {tech_col_idx:     'TECH' }      # Model Name
    # RMW is tricky because it's far down the line. 
    # A safer check for RMW is often index 19 (if 0-indexed) or just grabbing it if we can.
    # For this simplified version, let's stick to the core columns which are stable.

    # Rename columns that exist in our mapping
    df = df.rename(columns=col_mapping)
    
    # 4. Filter
    df = df[df['TECH'].isin(trusted_models)].copy()
    
    # 5. Parse Data
    df['latitude'] = df['lat_str'].apply(parse_lat_lon)
    df['longitude'] = df['lon_str'].apply(parse_lat_lon)
    
    # Convert Timestamp
    # Sometimes there is a generic "Minutes" column in between, handled by dynamic mapping
    df['timestamp'] = pd.to_datetime(df['timestamp_str'].astype(str).str.strip(), format='%Y%m%d%H', errors='coerce')
    
    # Select final columns
    final_cols = ['TECH', 'timestamp', 'tau', 'latitude', 'longitude', 'max_wind_speed_kt','radius_max_wind_nm', 'min_pressure_mb','storm_type']
            
    return df[final_cols].dropna(subset=['latitude', 'longitude'])

if __name__ == "__main__":
    # Test on the Best Track file
    input_file = "data/raw/bal142018.dat"
    output_file = "data/processed/michael_2018_best_track.csv"
    
    if os.path.exists(input_file):
        df = clean_track_data(input_file, is_best_track=True)
        df.to_csv(output_file, index=False)
        print(f"Saved Best Track: {len(df)} rows.")
        print(df.head())