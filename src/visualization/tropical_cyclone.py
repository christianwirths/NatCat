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

def plot_trajectory_B_deck(df: pd.DataFrame, filestring: str = 'trajectory_plot.png', title: str = 'Tropical Cyclone Trajectory', ax_extend: list = [-90, -75, 15, 38]) -> plt.Figure: 
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

