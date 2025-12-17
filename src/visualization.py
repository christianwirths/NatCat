from IPython.display import Image, display
import numpy as np
import pandas as pd
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from utils import extract_past_trajectory, extract_future_trajectory, track_interpolation


def plot_trajectory(df, labels, timestamp):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
        
    # 3. Add Map Features
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax.set_extent([-90, -75, 15, 38], crs=ccrs.PlateCarree())

    markers = ['o', 'x', '^', 's', 'D', '*']
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    trusted_models = ['OFCL', 'AVNO', 'EMXI', 'HWRF', 'CTCX', 'UKX']
    df_past_list = [extract_past_trajectory(df, timestamp, tech=model) for model in trusted_models]
    df_future_list = [extract_future_trajectory(df, timestamp, tech=model) for model in trusted_models]

    
    for i, (df_past, df_future) in enumerate(zip(df_past_list, df_future_list)):
        try:
            df_past = track_interpolation(df_past, kind="cubic")
            df_future = track_interpolation(df_future, kind="cubic")
        except:
            print(f"Warning: Interpolation failed for {trusted_models[i]}. Using original data.")
            print(f"The data has {len(df_past)} past points and {len(df_future)} future points.")
        ax.scatter(df_past['longitude'], df_past['latitude'], marker=markers[i], color=colors[i], label=labels[i])
        ax.scatter(df_future['longitude'], df_future['latitude'], marker=markers[i], facecolors='none', edgecolors='black', s=100)

    ax.set_title(f'Past Trajectory up to {timestamp}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid()
    ax.legend()
    plt.show()

    return fig

def make_trajectory_animation(df_forcast, df_OFCL,
                              trusted_models=None,
                              markers=None, colors=None,
                              out_path='trajectory_evolution.gif',
                              interval=800, fps=2,
                              figsize=(14,10),
                              extent=[-90, -75, 15, 38]):
    if trusted_models is None:
        trusted_models = ['OFCL', 'AVNO', 'EMXI', 'HWRF', 'CTCX', 'UKX']
    if markers is None:
        markers = ['o', 'x', '^', 's', 'D', '*']
    if colors is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    timestamps = sorted(df_OFCL.timestamp.unique())

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    past_scatters = []
    future_scatters = []
    for i in range(len(trusted_models)):
        ps = ax.scatter([], [], marker=markers[i], color=colors[i], label=trusted_models[i])
        fs = ax.scatter([], [], marker=markers[i], facecolors='none', edgecolors='black', s=100)
        past_scatters.append(ps)
        future_scatters.append(fs)

    title = ax.set_title('')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid()
    ax.legend(loc='upper left')

    def _set_scatter(sc, lons, lats):
        if len(lons) == 0:
            sc.set_offsets(np.empty((0, 2)))
        else:
            sc.set_offsets(np.column_stack([lons, lats]))

    def init():
        for ps, fs in zip(past_scatters, future_scatters):
            _set_scatter(ps, [], [])
            _set_scatter(fs, [], [])
        title.set_text('')
        return past_scatters + future_scatters + [title]

    def update(i):
        ts = timestamps[i]
        for idx, model in enumerate(trusted_models):
            df_past = extract_past_trajectory(df_forcast, ts, tech=model)
            df_future = extract_future_trajectory(df_forcast, ts, tech=model)
            try:
                if not df_past.empty:
                    df_past = track_interpolation(df_past, kind="cubic")
            except Exception:
                pass
            try:
                if not df_future.empty:
                    df_future = track_interpolation(df_future, kind="cubic")
            except Exception:
                pass
            _set_scatter(past_scatters[idx],
                         df_past['longitude'].values if not df_past.empty else [],
                         df_past['latitude'].values if not df_past.empty else [])
            _set_scatter(future_scatters[idx],
                         df_future['longitude'].values if not df_future.empty else [],
                         df_future['latitude'].values if not df_future.empty else [])
        title.set_text(f'Past Trajectory up to {pd.to_datetime(ts)}')
        return past_scatters + future_scatters + [title]

    anim = animation.FuncAnimation(fig, update, frames=len(timestamps),
                                   init_func=init, interval=interval, blit=True)

    anim.save(out_path, writer='pillow', fps=fps)
    plt.close(fig)
    display(Image(out_path))
    return anim

# Example usage:
# anim = make_trajectory_animation(df_forcast, df_OFCL)