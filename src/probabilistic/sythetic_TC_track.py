"""
Synthetic tropical cyclone track generation module.
"""


import glob
import os
import pandas as pd
from scipy import stats
import numpy as np
from global_land_mask import globe

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from preprocessing.tropical_cyclone import clean_track_data, check_a_deck_quality
from utils.track import cyclone_velocity, cyclone_bearing, track_interpolation
from hazards.tropical_cyclone import get_heuristic_rmw
from utils import kt2kmh, nm2km



#----------------------------------------------------------------------
# Data Preparation for Synthetic Track Generation
#----------------------------------------------------------------------

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


def prepare_mcmc_data(track_data: list, grid_size: float =2.0, verbose: bool = True) -> pd.DataFrame:
    """
    Prepares our track data to build an emperical MC model.
    We create a grid of lat/lon boxes (e.g. 2 degree by 2 degree) and assign a state_id to each point in the track data based on which box it falls into.
    We then calculate the transition deltas (delta_vmax, delta_radius) and the velocity and bearing for each point in the track.
    
    Input:
        track_data: List of DataFrames containing historical track data.
        grid_size: Size of the lat/lon grid boxes in degrees (default is 2.0 degrees).
        verbose: If True, print progress messages.
    Returns:
        mcmc_df: A DataFrame containing the state_id, velocity, bearing, delta_vmax, and delta_radius for each point in the track data.
    """


    if verbose: print(f"Stacking {len(track_data)} storm tracks...")
    
    processed_dfs = []

    # Create strom ID: 
    for i, df in enumerate(track_data):
        temp_df = df.copy()
        temp_df['storm_id'] = f"HIST_{i}" 
        processed_dfs.append(temp_df)

    master_df = pd.concat(processed_dfs, ignore_index=True)

    #Add day_of_year
    master_df['day_of_year'] = pd.to_datetime(master_df['time']).dt.dayofyear

    # Get variable deltas 
    master_df['delta_vmax'] = master_df.groupby('storm_id')['max_wind_speed_kt'].diff()
    master_df['delta_radius'] = master_df.groupby('storm_id')['radius_max_wind_nm'].diff()

    # Drop rows with NaNs (the first hour of every storm)
    master_df = master_df.dropna(subset=['delta_vmax', 'delta_radius', 'velocity_kt', 'bearing_deg'])


    # Create grid boxes and assign state_id
    master_df['lat_bin'] = (master_df['latitude'] // grid_size) * grid_size
    master_df['lon_bin'] = (master_df['longitude'] // grid_size) * grid_size
    master_df['state_id'] = master_df['lat_bin'].astype(str) + "_" + master_df['lon_bin'].astype(str)

    return master_df


#----------------------------------------------------------------------
# Cyclone genesis point generation
#----------------------------------------------------------------------

def generate_synthetic_tc_origin(track_data: list, num_syntetic_origins: int) -> pd.DataFrame:
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
    max_wind_speed_kt = []
    radius_max_wind_nm =[]

    for track in track_data: 
        first = track.iloc[0]
        lat.append(first['latitude'])
        lon.append(first['longitude'])
        time.append(first['time'])
        max_wind_speed_kt.append(first["max_wind_speed_kt"])
        radius_max_wind_nm.append(first["radius_max_wind_nm"])

    #Transform to pd.Dataframe 
    df_history = pd.DataFrame({'latitude': lat, 'longitude': lon, 'time': time, 'max_wind_speed_kt': max_wind_speed_kt, 'radius_max_wind_nm': radius_max_wind_nm})

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


def assign_genesis_intensity(synthetic_origins: pd.DataFrame, historical_origins: pd.DataFrame):
    """  
    Assignes wind speed and a storm eye radius to the synthetic storms using KNN

    Input: 
        synthetic_origins: pandas Dataframe containing lat and lon for all syntetic cyclone origins 
        historical_origins: pandas Dataframe containing lat,lon, wind speed and radius such that we can sample from using KNN

    Returns: 
        synthetic_origins. with wind_speed and radius_max_wind_nm columns added.
    """

    # Lon lat features for KNN
    X_train = historical_origins[['latitude', 'longitude']].values
    X_synthetic = synthetic_origins[['latitude', 'longitude']].values

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_synthetic_scaled = scaler.transform(X_synthetic)

    # Fit KNN
    knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
    knn.fit(X_train_scaled)
    distances, indices = knn.kneighbors(X_synthetic_scaled)

    # Sample a random neighbor for each synthetic point and assign its wind speed and radius
    random_neighbor_choice = np.random.randint(0, 5, size=len(synthetic_origins))
    selected_hist_indices = indices[np.arange(len(synthetic_origins)), random_neighbor_choice]

    synthetic_origins['max_wind_speed_kt'] = historical_origins.iloc[selected_hist_indices]['max_wind_speed_kt'].values
    synthetic_origins['radius_max_wind_nm'] = historical_origins.iloc[selected_hist_indices]['radius_max_wind_nm'].values

    return synthetic_origins


#----------------------------------------------------------------------
# Emperical MCMC to generate tracks 
#----------------------------------------------------------------------

def calculate_next_point(lat, lon, velocity_kt, bearing_deg, time_step_hours=3):
    """
    Calculate the next point in a track given the current position, velocity, and bearing.
    This is a simple kinematic calculation that assumes constant velocity and bearing over the time step.

    Args:
        lat (float): Current latitude in degrees.
        lon (float): Current longitude in degrees.
        velocity_kt (float): Velocity in knots.
        bearing_deg (float): Bearing in degrees from north.
        time_step_hours (int): Time step in hours (default is 3 hours).
    Returns:
        next_lat (float): Next latitude in degrees.
        next_lon (float): Next longitude in degrees.
    """
    
    velocity_kmh = kt2kmh(velocity_kt)
    distance_km = velocity_kmh * (time_step_hours / 1.0)

    delta = distance_km / 6371.0  # Earth radius in km

    bearing_rad = np.radians(bearing_deg)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    next_lat_rad = np.arcsin(np.sin(lat_rad) * np.cos(delta) +
                             np.cos(lat_rad) * np.sin(delta) * np.cos(bearing_rad))
    next_lon_rad = lon_rad + np.arctan2(np.sin(bearing_rad) * np.sin(delta) * np.cos(lat_rad),
                                        np.cos(delta) - np.sin(lat_rad) * np.sin(next_lat_rad))

    next_lat = np.degrees(next_lat_rad)
    next_lon = np.degrees(next_lon_rad)

    return next_lat, next_lon


def build_transition_dictionary(mcmc_df, verbose: bool = True):
    """
    Converts the massive historical DataFrame into a fast dictionary.
    Format: { '24.0_-80.0': [list of historical transitions] }

    Input: 
        mcmc_df: DataFrame with columns ['state_id', 'velocity_kt', 'bearing_deg', 'delta_vmax', 'delta_radius']
                 --> is generated by prepare_mcmc_data() function. 
        verbose: If True, print progress messages.
    Returns:
        transition_dict: Dictionary where keys are state_id (e.g. '24.0_-80.0') and values are arrays of possible transitions (velocity_kt, bearing_deg, delta_vmax, delta_radius) observed in the historical data for that state_id.
    """
    if verbose: print("Building fast transition lookup dictionary...")
    transition_dict = {}
    
    # Group by the state_id, thus it is already in a (2 x 2 degree; default) grid format.
    for state_id, group in mcmc_df.groupby('state_id'):
        # Select whats needed for the transition: velocity, bearing, delta_vmax, delta_radius
        transitions = group[['velocity_kt', 'bearing_deg', 'delta_vmax', 'delta_radius']].values
        transition_dict[state_id] = transitions
        
    return transition_dict


def generate_synthetic_track(start_point: dict, transition_dict: dict, grid_size: float=2.0, max_hours: int=360, time_step: float=3.0):
    """
    Walks a single storm forward until it dies.

    Input:
        start_point: A dictionary containing the initial state of the storm with keys ['latitude', 'longitude', 'wind_speed', 'radius'].
        transition_dict: A dictionary where keys are state_id (e.g. '24.0_-80.0') and values are arrays of possible transitions (velocity_kt, bearing_deg, delta_vmax, delta_radius) observed in the historical data for that state_id.
        grid_size: Size of the lat/lon grid boxes in degrees (default is 2.0 degrees).
        max_hours: Maximum number of hours to simulate (default is 360 hours or 15 days).
    Returns:
        track_df: A DataFrame containing the simulated track of the storm with columns 
    """
    # Initialize track and current position
    track = [start_point.copy()]
    current_state = start_point.copy()

    # Initialise columns required by TropicalCycloneHazard for the genesis record
    current_state.setdefault('velocity_kt', 0.0)
    current_state.setdefault('bearing_deg', 0.0)
    track[0].setdefault('velocity_kt', 0.0)
    track[0].setdefault('bearing_deg', 0.0)
    
    for hour in range(1, max_hours + 1): 
        
        # Find the current grid box/state_id
        grid_lat = np.floor(current_state['latitude'] / grid_size) * grid_size
        grid_lon = np.floor(current_state['longitude'] / grid_size) * grid_size
        state_id = f"{grid_lat}_{grid_lon}"
        
        available_transitions = transition_dict.get(state_id)
        
        # kill the storm if there has never been a storm 
        if available_transitions is None or len(available_transitions) == 0:
            break 
            
        # Pick a random transition 
        draw_idx = np.random.randint(0, len(available_transitions))
        transition = available_transitions[draw_idx]
        
        tmp_vel, tmp_bearing, tmp_dvmax, tmp_dradius = transition

        # Store translation velocity and bearing so the track DataFrame has all columns
        # needed by TropicalCycloneHazard (e.g. for wind field asymmetry calculation)
        current_state['velocity_kt'] = tmp_vel
        current_state['bearing_deg'] = tmp_bearing
        
        # Move 
        next_lat, next_lon = calculate_next_point(
            current_state['latitude'], 
            current_state['longitude'], 
            tmp_vel, 
            tmp_bearing,
            time_step_hours=time_step
        )
        
        # Adjust radius and vmax 
        # check if over land 
        is_over_land = globe.is_land(next_lat,next_lon)

        if is_over_land:
            next_vmax = current_state['max_wind_speed_kt'] * (0.92**time_step)
            next_radius = current_state['radius_max_wind_nm'] * (1.02**time_step)

        else:     
            next_vmax = current_state['max_wind_speed_kt'] + tmp_dvmax
            next_radius = current_state['radius_max_wind_nm'] + tmp_dradius
        
        # Ensure physical plausibility
        next_vmax = max(0.0, next_vmax)  # Wind speed cannot be negative
        next_radius = max(5.0, next_radius)  # We set a minimum radius of 5 nm to avoid unphysical eye sizes
        
        # update state
        current_state['latitude'] = next_lat
        current_state['longitude'] = next_lon
        current_state['max_wind_speed_kt'] = next_vmax
        current_state['radius_max_wind_nm'] = next_radius
        current_state['hour'] = hour
        
        


        track.append(current_state.copy())
        
        # Terminate slow storms
        if next_vmax < 15.0:
            break

    track_df = pd.DataFrame(track)

    # Add synthetic timestamps so downstream tools (track_interpolation, cyclone_velocity, etc.) work out of the box
    ref_time = pd.Timestamp('1900-01-01 00:00:00')
    track_df['time'] = track_df['hour'].apply(lambda h: ref_time + pd.Timedelta(hours=h * time_step))

    return track_df


def storm_per_year(historic_origin: pd.DataFrame, num_years: int=1) -> np.array:
    """
    Fits the number of storms that are spawned per year to a possion distribution and than samples num_years samples from it. 

    
    """

    historic_origin['year'] = historic_origin['time'].dt.year
    storms_per_year = historic_origin.groupby('year').size()

    lambda_p = storms_per_year.mean()

    storms_per_synthetic_year = np.random.poisson(lam=lambda_p, size=num_years)

    return storms_per_synthetic_year


#----------------------------------------------------------------------
# Synthetic TC Catalog Class 
#----------------------------------------------------------------------

class SyntheticTCCatalog:
    """
    Bundles the full synthetic tropical cyclone generation pipeline into a single object.
    Handles loading historical data, fitting the empirical MCMC model, and generating synthetic storm catalogs.

    Usage:
        model = SyntheticTCCatalog(basin='al', grid_size=2.0)
        model.fit("/path/to/data/raw")
        catalog = model.generate(n_storms=10000)
    """

    def __init__(
        self,
        basin: str = "al",
        grid_size: float = 2.0,
        max_hours: int = 360,
        time_step: float = 3.0,
        n_neighbors: int = 5,
        min_wind_kt: float = 15.0,
        land_decay_rate: float = 0.92,
        seed: int = None,
        verbose: bool = True
    ):
        """
        Input:
            basin: Basin identifier (e.g., 'al' for Atlantic, 'ep' for East Pacific).
            grid_size: Size of the lat/lon grid boxes in degrees for the MCMC state space.
            max_hours: Maximum number of hours to simulate per storm (360 = 15 days).
            time_step: Time step in hours for track propagation.
            n_neighbors: Number of KNN neighbors used when assigning genesis intensity.
            min_wind_kt: Wind speed threshold (kt) below which a storm is terminated.
            land_decay_rate: Per-timestep multiplicative decay factor for wind speed over land.
            seed: Random seed for reproducibility. None for non-deterministic.
            verbose: If True, print progress messages. Set to False when using tqdm externally.
        """
        # Config
        self.basin = basin
        self.grid_size = grid_size
        self.max_hours = max_hours
        self.time_step = time_step
        self.n_neighbors = n_neighbors
        self.min_wind_kt = min_wind_kt
        self.land_decay_rate = land_decay_rate
        self.seed = seed
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        # Fitted state (populated by .fit())
        self.track_data = None
        self.failed_files = None
        self.historic_origins = None
        self.kde = None
        self.mcmc_data = None
        self.transition_dict = None
        self._is_fitted = False

    def fit(self, data_path: str):
        """
        Runs the full calibration pipeline:
        load historical tracks -> prepare MCMC data -> build transition dictionary + fit genesis KDE.

        Input:
            data_path: Path to folder containing historical ATCF B-deck files.
        Returns:
            self (for chaining)
        """
        # Load historical tracks
        self.track_data, self.failed_files = load_historical_tracks(data_path, basin=self.basin)

        if self.verbose:
            if self.failed_files:
                print(f"{len(self.failed_files)} file(s) failed to load.")
            print(f"Loaded {len(self.track_data)} tracks.")

        # Extract historic genesis points and fit KDE
        _, self.historic_origins, self.kde = generate_synthetic_tc_origin(self.track_data, num_syntetic_origins=1)
        # We generated 1 dummy point just to fit the KDE and extract historic origins.
        # Actual synthetic origins are generated in .generate()

        # Prepare MCMC transition data
        self.mcmc_data = prepare_mcmc_data(self.track_data, grid_size=self.grid_size, verbose=self.verbose)
        self.transition_dict = build_transition_dictionary(self.mcmc_data, verbose=self.verbose)

        self._is_fitted = True
        if self.verbose: print("Model fitted. Ready to generate synthetic storms.")
        return self

    def generate(self, n_storms: int = None, n_years: int = None) -> pd.DataFrame:
        """
        Generates a synthetic tropical cyclone catalog.
        Provide either n_storms directly, or n_years to sample the number of storms per year from a Poisson distribution.

        Input:
            n_storms: Total number of synthetic storms to generate. 
            n_years: Number of synthetic years. If provided, n_storms is sampled from a Poisson distribution fitted to the historical record.
                     Overrides n_storms if both are provided.
        Returns:
            catalog: A DataFrame containing the full synthetic storm catalog with columns 
                     [storm_id, latitude, longitude, max_wind_speed_kt, radius_max_wind_nm, hour].
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

        if n_years is not None:
            storms_per_yr = storm_per_year(self.historic_origins.copy(), num_years=n_years)
            n_storms = int(storms_per_yr.sum())
            if self.verbose: print(f"Poisson sampling: {n_storms} storms across {n_years} synthetic years.")
        elif n_storms is None:
            raise ValueError("Provide either n_storms or n_years.")

        # Generate genesis points
        synthetic_origins, _, _ = generate_synthetic_tc_origin(self.track_data, num_syntetic_origins=n_storms)

        # Assign genesis intensity via KNN
        synthetic_origins = assign_genesis_intensity(synthetic_origins, self.historic_origins)

        # Walk each storm forward
        synthetic_catalog = []
        if self.verbose: print(f"Generating tracks for {len(synthetic_origins)} synthetic origins...")

        for _, start_point in synthetic_origins.iterrows():
            start_dict = start_point.to_dict()
            start_dict['hour'] = 0

            track_df = generate_synthetic_track(
                start_dict, 
                self.transition_dict, 
                grid_size=self.grid_size,
                max_hours=self.max_hours,
                time_step=self.time_step
            )
            synthetic_catalog.append(track_df)

        catalog = pd.concat(synthetic_catalog, ignore_index=True)
        self.synthetic_catalog = catalog
        if self.verbose: print(f"Simulation complete! Generated {len(synthetic_origins)} synthetic storms ({len(catalog)} track points).")

        return catalog

    def __repr__(self):
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"SyntheticTCCatalog(basin='{self.basin}', grid_size={self.grid_size}, "
            f"max_hours={self.max_hours}, time_step={self.time_step}, status={status})"
        )
    

    def plot_tracks(self, color: str = 'lightcoral', alpha: float = 0.4, linewidth: float = 1.2):
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(14, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
                
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
        ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

        ax.set_extent([-110, -10, 5, 50], crs=ccrs.PlateCarree())

        unique_storms = self.synthetic_catalog['storm_id'].unique()[:]

        for storm in unique_storms:
            single_track = self.synthetic_catalog[self.synthetic_catalog['storm_id'] == storm]
            
            # Plot it as a continuous line
            ax.plot(single_track['longitude'], single_track['latitude'], 
                    color=color,           
                    alpha=alpha,             
                    linewidth=linewidth,         
                    transform=ccrs.PlateCarree())

        plt.title("Synthetic Tropical Cyclone Tracks")
        plt.show()