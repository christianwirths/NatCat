"""
Helper functions for visualizing tropical cyclone tracks and losses.
"""

#Imports 
from IPython.display import Image, display
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#--------------------------------------------------------------------------
# Tropical cyclone track visualization
#--------------------------------------------------------------------------

# Currently only tested for OFCL B-deck data 
# TODO: Add A-deck support 
def plot_trajectory(df: pd.DataFrame, filestring: str = 'trajectory_plot.png', title: str = 'Tropical Cyclone Trajectory', ax_extend: list = [-90, -75, 15, 38]) -> plt.Figure: 
    """
    Plots the trajectory of a tropical cyclone from B-deck data.
    
    Args:
        df: DataFrame containing B-deck track data with 'time', 'latitude', and 'longitude' columns.
        filestring: Filename to save the plot.
        title: Title of the plot.
        ax_extend: List of [lon_min, lon_max, lat_min, lat_max] to set the map extent. Default is set to cover the Caribbean/SE-US region.
    Returns:
        Matplotlib figure object.
    """

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
        
    # Add Map Features
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax.set_extent(ax_extend, crs=ccrs.PlateCarree())

    # Plot the track
    ax.plot(df['longitude'], df['latitude'], marker='o', color='blue', label='OFCL Track')

    # Annotate timestamps 
    for i in range(0, len(df), max(1, len(df) // 12)):
        timestamp = df['time'].iloc[i].strftime('%Y-%m-%d %H:%M')
        #Check if timestamp is not out of bounds of the map with some margin to avoid text going out of the map
        if (ax_extend[0] + 2 < df['longitude'].iloc[i] < ax_extend[1] - 2) and (ax_extend[2] + 2 < df['latitude'].iloc[i] < ax_extend[3] - 2):
            ax.annotate(timestamp, xy=(df['longitude'].iloc[i], df['latitude'].iloc[i]), xytext=(10,0), textcoords='offset points', fontsize=10, transform=ccrs.PlateCarree(), ha='left', va='top')

    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid()
    ax.legend()
    
    # Save the plot
    plt.savefig(filestring)
    
    return fig

#--------------------------------------------------------------------------
# Cyclone track and damage evolution visualization
#--------------------------------------------------------------------------

def spatial_loss(df_loss: pd.DataFrame, portfolio: pd.DataFrame =None, df_track: pd.DataFrame =None, damage_ratio: bool = False, timestamp: str = None, filestring: str = 'spatial_loss.png', title: str = 'Spatial Loss Distribution', ax_extend: list = [-90, -75, 15, 38]) -> plt.Figure:
    """
    Plots the spatial distribution of losses from a tropical cyclone.
    
    Args:
        df_loss: DataFrame containing loss data with 'latitude', 'longitude', and 'loss' or "damage_ratio" columns.
        portfolio: Optional DataFrame containing portfolio locations with 'latitude', 'longitude', and 'tiv' columns.
        df_track: DataFrame containing B-deck track data with 'time', 'latitude', and 'longitude' columns. Default is None.
        damage_ratio: Boolean indicating whether to plot damage ratio instead of absolute loss. Default is False.
        timestamp: Specific timestamp to filter the loss data. Default is None. 
        filestring: Filename to save the plot.
        title: Title of the plot.
        ax_extend: List of [lon_min, lon_max, lat_min, lat_max] to set the map extent. Default is set to cover the Caribbean/SE-US region.
    Returns:
        Matplotlib figure object.
    """

    #Check if required columns are present in df_loss
    if damage_ratio and 'damage_ratio' not in df_loss.columns:
        raise ValueError("DataFrame must contain 'damage_ratio' column when damage_ratio=True.")
    elif not damage_ratio and 'loss' not in df_loss.columns:
        raise ValueError("DataFrame must contain 'loss' column when damage_ratio=False.")


    # "timestep" checks
    if timestamp is not None and 'timestamp' not in df_loss.columns:
        raise ValueError("DataFrame must contain 'timestamp' column to filter by timestamp.")

    if timestamp is not None:
        df_loss = df_loss[df_loss['timestamp'] == timestamp]
    else:
        if 'timestamp' not in df_loss.columns:
            print("Warning: 'timestamp' column not found in df_loss. Plotting all data without timestamp filtering.")
        else:
            df_loss = df_loss[df_loss['timestamp'] == df_loss['timestamp'].max()]   




    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
        
    # Add Map Features
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax.set_extent(ax_extend, crs=ccrs.PlateCarree())

    # Plot portfolio if provided
    if portfolio is not None:
        nonzero_mask = portfolio['tiv'] > 1e8
        ax.scatter(portfolio['longitude'][nonzero_mask], portfolio['latitude'][nonzero_mask], c=portfolio['tiv'][nonzero_mask], label='Portfolio', s=0.5, transform=ccrs.PlateCarree(), zorder=4,cmap='Blues', alpha=0.9)
        pcbar = plt.colorbar(ax.collections[-1], ax=ax, orientation='vertical', label='TIV ($)')
    # Plot the spatial loss distribution
    if damage_ratio:
        nonzero_mask = df_loss['damage_ratio'] > 0
        scatter = ax.scatter(df_loss['longitude'][nonzero_mask], df_loss['latitude'][nonzero_mask], c=df_loss['damage_ratio'][nonzero_mask], cmap='Reds', s=0.5, alpha=0.8, zorder=5, vmax=1, vmin=0, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', label='Damage Ratio')
    else:
        nonzero_mask = df_loss['loss'] > 0
        scatter = ax.scatter(df_loss['longitude'][nonzero_mask], df_loss['latitude'][nonzero_mask], c=df_loss['loss'][nonzero_mask], cmap='Reds', s=0.5, alpha=0.8, zorder=5, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', label='Loss ($)')
    
    

    # Optionally plot the track
    if df_track is not None:
        ax.plot(df_track['longitude'], df_track['latitude'], marker='o', color='blue', label='OFCL Track', markersize=1, transform=ccrs.PlateCarree())

    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid()
    if df_track is not None:
        ax.legend()
    
    # Save the plot
    plt.savefig(filestring)
    
    return fig

#---------------------------------------------------
# Cyclone track and spatial loss animation 
#--------------------------------------------------

def animate_spatial_loss(df_loss: pd.DataFrame, portfolio: pd.DataFrame = None, df_track: pd.DataFrame = None, damage_ratio: bool = False, timestep_hours: int = 6, filestring: str = 'spatial_loss_animation.gif', title: str = 'Tropical Cyclone Loss Evolution', ax_extend: list = [-90, -75, 15, 38], fps: int = 2) -> animation.FuncAnimation:
    """
    Creates an animation of spatial loss evolution over time.
    
    Args:
        df_loss: DataFrame with 'timestamp', 'latitude', 'longitude', and 'loss' or 'damage_ratio' columns.
        portfolio: Optional DataFrame with portfolio locations ('latitude', 'longitude', 'tiv').
        df_track: Optional DataFrame with track data ('time', 'latitude', 'longitude').
        damage_ratio: Whether to plot damage ratio instead of absolute loss. Default is False.
        timestep_hours: Hours between animation frames. Default is 6.
        filestring: Filename to save animation (supports .gif, .mp4).
        title: Title of the plot.
        ax_extend: Map extent as [lon_min, lon_max, lat_min, lat_max].
        fps: Frames per second for the animation. Default is 2.
    Returns:
        Animation object.
    """
    
    # Check required columns
    if 'timestamp' not in df_loss.columns:
        raise ValueError("df_loss must contain 'timestamp' column.")
    if damage_ratio and 'damage_ratio' not in df_loss.columns:
        raise ValueError("df_loss must contain 'damage_ratio' column when damage_ratio=True.")
    elif not damage_ratio and 'loss' not in df_loss.columns:
        raise ValueError("df_loss must contain 'loss' column when damage_ratio=False.")
    
    # Get unique timestamps and resample to desired timestep
    df_loss['timestamp'] = pd.to_datetime(df_loss['timestamp'])
    timestamps = sorted(df_loss['timestamp'].unique())
    
    # Filter timestamps by timestep_hours
    if len(timestamps) > 1:
        time_delta = pd.Timedelta(hours=timestep_hours)
        filtered_timestamps = [timestamps[0]]
        for ts in timestamps[1:]:
            if ts - filtered_timestamps[-1] >= time_delta:
                filtered_timestamps.append(ts)
        timestamps = filtered_timestamps
    
    # Setup figure and axis
    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add base map features
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax.set_extent(ax_extend, crs=ccrs.PlateCarree())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid()
    plt.title(title)
    # Plot portfolio once (static)
    if portfolio is not None:
        nonzero_mask = portfolio['tiv'] > 1e8
        ax.scatter(portfolio['longitude'][nonzero_mask], portfolio['latitude'][nonzero_mask], 
                  c=portfolio['tiv'][nonzero_mask], s=0.5, transform=ccrs.PlateCarree(), 
                  zorder=4, cmap='Blues', alpha=0.5, label='Portfolio')
    
    # Initialize empty plots for dynamic elements
    loss_scatter = ax.scatter([], [], c=[], cmap='Reds', s=0.5, alpha=0.8, zorder=5, 
                             vmax=1 if damage_ratio else None, vmin=0 if damage_ratio else None,
                             transform=ccrs.PlateCarree())
    track_line, = ax.plot([], [], 'o-', color='blue', markersize=2, linewidth=1.5, 
                         label='Track', transform=ccrs.PlateCarree(), zorder=6)
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add colorbar
    cbar_label = 'Damage Ratio' if damage_ratio else 'Loss ($)'
    cbar = plt.colorbar(loss_scatter, ax=ax, orientation='vertical', label=cbar_label)
    
    ax.legend(loc='upper right')
    
    def init():
        """Initialize animation."""
        loss_scatter.set_offsets(np.empty((0, 2)))
        loss_scatter.set_array(np.array([]))
        track_line.set_data([], [])
        time_text.set_text('')
        return loss_scatter, track_line, time_text
    
    def update(frame_idx):
        """Update function for each frame."""
        current_time = timestamps[frame_idx]
        
        # Get cumulative loss data up to current time
        df_current = df_loss[df_loss['timestamp'] <= current_time]
        
        # For spatial plotting, take maximum loss/damage at each location
        if damage_ratio:
            df_plot = df_current.groupby(['latitude', 'longitude'])['damage_ratio'].max().reset_index()
            nonzero_mask = df_plot['damage_ratio'] > 0
            colors = df_plot['damage_ratio'][nonzero_mask]
        else:
            df_plot = df_current.groupby(['latitude', 'longitude'])['loss'].max().reset_index()
            nonzero_mask = df_plot['loss'] > 0
            colors = df_plot['loss'][nonzero_mask]
        
        # Update loss scatter
        if nonzero_mask.sum() > 0:
            offsets = np.c_[df_plot['longitude'][nonzero_mask], df_plot['latitude'][nonzero_mask]]
            loss_scatter.set_offsets(offsets)
            loss_scatter.set_array(colors)
        
        # Update track up to current time
        if df_track is not None:
            df_track_current = df_track[df_track['time'] <= current_time]
            if len(df_track_current) > 0:
                track_line.set_data(df_track_current['longitude'], df_track_current['latitude'])
        
        # Update timestamp text
        time_text.set_text(f'Time: {current_time.strftime("%Y-%m-%d %H:%M")}')
        
        return loss_scatter, track_line, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(timestamps), 
                                  interval=1000/fps, blit=True, repeat=True)
    
    # Save animation
    if filestring.endswith('.gif'):
        anim.save(filestring, writer='pillow', fps=fps)
    elif filestring.endswith('.mp4'):
        anim.save(filestring, writer='ffmpeg', fps=fps)
    else:
        print(f"Warning: Unknown file format for {filestring}. Supported formats: .gif, .mp4")
    
    plt.close()
    return anim