"""Animated footprint evolution (requires cartopy + Pillow, imported lazily).

Notes
-----
``matplotlib.animation`` itself is lightweight, but this module draws on a
cartopy base map, so cartopy is imported lazily inside ``animate_footprint``
(via ``natcat.plotting.maps``) to keep module import cheap and optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from natcat.plotting.maps import _category_from_kt, _require_cartopy, base_map
from natcat.plotting.style import (
    PALETTE,
    SAFFIR_SIMPSON_COLORS,
    SAFFIR_SIMPSON_ORDER,
    style_context,
)

__all__ = ["animate_footprint"]

_VALUE_CMAPS = {"damage_ratio": "YlOrRd", "loss": "magma_r", "intensity": "YlOrRd"}
_VALUE_LABELS = {
    "damage_ratio": "Damage ratio",
    "loss": "Loss (USD)",
    "intensity": "Max wind speed (kt)",
}


def _subsample_times(times: list[pd.Timestamp], step: str) -> list[pd.Timestamp]:
    """Keep the first timestamp and subsequent ones at least ``step`` apart."""
    if not times:
        return []
    step_td = pd.Timedelta(step)
    kept = [times[0]]
    for t in times[1:]:
        if t - kept[-1] >= step_td:
            kept.append(t)
    return kept


def animate_footprint(
    history: pd.DataFrame,
    *,
    track: pd.DataFrame | None = None,
    portfolio: pd.DataFrame | None = None,
    extent,
    value: Literal["damage_ratio", "loss", "intensity"] = "damage_ratio",
    step: str = "6h",
    fps: int = 2,
    path: str | Path = "footprint.gif",
    title: str | None = None,
    dpi: int = 110,
) -> Path:
    """Animate the growth of a hazard/loss footprint alongside the storm track.

    Parameters
    ----------
    history : pandas.DataFrame
        Long-format loss-history DataFrame with a ``time`` column plus
        portfolio columns and ``intensity``/``damage_ratio``/``loss``
        (as produced by ``LossCalculator.compute_history``).
    track : pandas.DataFrame, optional
        Storm track, drawn incrementally up to the current frame time, with
        a storm marker at the current position sized by category.
    portfolio : pandas.DataFrame, optional
        Full exposure drawn once as a static light-grey background layer.
    extent : sequence of float
        Map extent ``(lon_min, lon_max, lat_min, lat_max)``.
    value : {"damage_ratio", "loss", "intensity"}, default "damage_ratio"
        Which column to animate.
    step : str, default "6h"
        Minimum spacing between animation frames (pandas frequency string).
    fps : int, default 2
        Frames per second of the saved GIF.
    path : str or pathlib.Path, default "footprint.gif"
        Output path; parent directories are created as needed.
    title : str, optional
        Left-aligned bold title.
    dpi : int, default 110
        Render resolution; kept modest to bound file size (~8 MB target).

    Returns
    -------
    pathlib.Path
        The path the GIF was saved to.

    Examples
    --------
    >>> animate_footprint(history, track=track, extent=(-90, -78, 24, 34))  # doctest: +SKIP
    """
    ccrs, _ = _require_cartopy()
    from matplotlib.animation import FuncAnimation, PillowWriter

    if value not in _VALUE_CMAPS:
        raise ValueError(f"value must be one of {sorted(_VALUE_CMAPS)}, got {value!r}")

    with style_context():
        import matplotlib.pyplot as plt

        history = history.copy()
        history["time"] = pd.to_datetime(history["time"])
        times = _subsample_times(sorted(history["time"].unique()), step)
        if track is not None:
            track = track.copy()
            track["time"] = pd.to_datetime(track["time"])

        fig, ax = base_map(extent, figsize=(8.0, 7.0), states=True)
        fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.17)

        if portfolio is not None and len(portfolio):
            tiv = portfolio["tiv"].clip(lower=1.0)
            log_tiv = np.log10(tiv)
            lo, hi = log_tiv.min(), log_tiv.max()
            sizes = (
                np.full(len(portfolio), 3.0)
                if hi <= lo
                else np.interp(log_tiv, (lo, hi), (1.0, 10.0))
            )
            ax.scatter(
                portfolio["longitude"],
                portfolio["latitude"],
                s=sizes,
                color=PALETTE["muted"],
                alpha=0.3,
                linewidth=0,
                transform=ccrs.PlateCarree(),
                zorder=2,
            )

        vmax = (
            1.0 if value == "damage_ratio" else float(np.nanmax(history[value].to_numpy()) or 1.0)
        )
        damage_scatter = ax.scatter(
            [],
            [],
            s=8,
            cmap=_VALUE_CMAPS[value],
            vmin=0.0,
            vmax=vmax,
            linewidth=0,
            alpha=0.9,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )
        # An empty scatter ignores the cmap keyword (no colour data yet), so pin it here.
        damage_scatter.set_cmap(_VALUE_CMAPS[value])
        damage_scatter.set_clim(0.0, vmax)
        (track_line,) = ax.plot(
            [], [], color=PALETTE["ink"], linewidth=1.3, transform=ccrs.PlateCarree(), zorder=5
        )
        storm_marker = ax.scatter(
            [],
            [],
            s=60,
            color=PALETTE["red"],
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
            transform=ccrs.PlateCarree(),
        )
        info_box = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": PALETTE["coast"],
                "lw": 0.5,
                "alpha": 0.9,
            },
            zorder=10,
        )

        cax = ax.inset_axes([0.12, -0.13, 0.76, 0.03])
        cbar = fig.colorbar(damage_scatter, cax=cax, orientation="horizontal")
        cbar.set_label(_VALUE_LABELS[value], fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        category_handles = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=SAFFIR_SIMPSON_COLORS[c],
                markeredgecolor="none",
                markersize=6,
                label=c,
            )
            for c in SAFFIR_SIMPSON_ORDER
        ]
        ax.legend(
            handles=category_handles,
            loc="lower right",
            ncol=1,
            fontsize=7,
            handlelength=0.8,
            frameon=True,
            framealpha=0.9,
            edgecolor="none",
            title="Category",
            title_fontsize=7,
        )

        if title:
            ax.set_title(title, loc="left", fontweight="bold")

        def _update(frame_index: int):
            current_time = times[frame_index]
            snapshot = history[history["time"] <= current_time]
            latest = snapshot.groupby(["latitude", "longitude"], as_index=False)[value].max()
            damaged = latest[latest[value] > 0]
            if len(damaged):
                damage_scatter.set_offsets(np.c_[damaged["longitude"], damaged["latitude"]])
                damage_scatter.set_array(damaged[value].to_numpy())

            total_loss = np.nan
            if "loss" in snapshot.columns:
                total_loss = snapshot.groupby(["latitude", "longitude"])["loss"].max().sum()

            if track is not None:
                track_now = track[track["time"] <= current_time]
                if len(track_now):
                    track_line.set_data(track_now["longitude"], track_now["latitude"])
                    last = track_now.iloc[-1]
                    category = _category_from_kt([last["max_wind_speed_kt"]])[0]
                    storm_marker.set_offsets([[last["longitude"], last["latitude"]]])
                    storm_marker.set_color(SAFFIR_SIMPSON_COLORS[category])
                    storm_marker.set_sizes([40 + 15 * SAFFIR_SIMPSON_ORDER.index(category)])

            label = f"{current_time:%Y-%m-%d %HZ}"
            if np.isfinite(total_loss):
                label += f"\nCumulative loss: ${total_loss / 1e9:.2f}B"
            info_box.set_text(label)
            return damage_scatter, track_line, storm_marker, info_box

        anim = FuncAnimation(fig, _update, frames=len(times), blit=False)
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)

    return out_path
