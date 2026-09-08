"""Map-based figures: base maps, tracks, footprints, genesis points, catalogs.

Cartopy is a heavy optional dependency, so it is imported lazily inside each
function body (via ``_require_cartopy``) rather than at module import time.
This keeps ``import natcat`` and ``import natcat.plotting`` working even when
cartopy is not installed; only calling one of these functions requires it.

Notes
-----
All functions accept an existing ``ax`` (a cartopy ``GeoAxes``) and always
return ``(fig, ax)`` (or ``(fig, (ax1, ax2))`` for the two-panel comparison),
never call ``plt.show()``, and never save a figure unless the caller does so
explicitly (see ``natcat.plotting.style.save``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D

from natcat.plotting.style import (
    PALETTE,
    SAFFIR_SIMPSON,
    SAFFIR_SIMPSON_COLORS,
    SAFFIR_SIMPSON_ORDER,
    category_colormap,
    style_context,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "map_figsize",
    "base_map",
    "plot_catalog",
    "plot_catalog_comparison",
    "plot_footprint",
    "plot_genesis",
    "plot_track",
]

_CARTOPY_HINT = (
    "cartopy is required for natcat.plotting.maps. Install it with "
    "`conda install -c conda-forge cartopy` (recommended) or `pip install cartopy`."
)


def _require_cartopy():
    """Import cartopy lazily, raising a clear error if it is unavailable."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:  # pragma: no cover - exercised only without cartopy
        raise ImportError(_CARTOPY_HINT) from exc
    return ccrs, cfeature


def _category_from_kt(wind_kt: Sequence[float] | np.ndarray) -> np.ndarray:
    """Map wind speeds (kt) to Saffir-Simpson category labels, vectorised."""
    wind = np.asarray(wind_kt, dtype=float)
    labels = np.full(wind.shape, SAFFIR_SIMPSON_ORDER[-1], dtype=object)
    for category in SAFFIR_SIMPSON_ORDER:
        lo, hi = SAFFIR_SIMPSON[category]
        labels[(wind >= lo) & (wind < hi)] = category
    return labels


def _category_legend_handles(linewidth: float = 3.0) -> list[Line2D]:
    return [
        Line2D([0], [0], color=SAFFIR_SIMPSON_COLORS[c], lw=linewidth, label=c)
        for c in SAFFIR_SIMPSON_ORDER
    ]


def map_figsize(
    extent: Sequence[float],
    *,
    max_width: float = 9.0,
    max_height: float = 7.0,
    n_panels: int = 1,
) -> tuple[float, float]:
    """Figure size (inches) matching the aspect ratio of a PlateCarree extent.

    Parameters
    ----------
    extent : sequence of float
        ``(lon_min, lon_max, lat_min, lat_max)``.
    max_width : float, default 9.0
        Upper bound on the total figure width.
    max_height : float, default 7.0
        Upper bound on the figure height.
    n_panels : int, default 1
        Number of side-by-side map panels sharing the extent.

    Returns
    -------
    tuple[float, float]
        ``(width, height)`` in inches, including a margin for labels.
    """
    lon_span = abs(float(extent[1]) - float(extent[0]))
    lat_span = abs(float(extent[3]) - float(extent[2]))
    aspect = lat_span / max(lon_span, 1e-6)
    panel_width = max_width / n_panels
    height = panel_width * aspect
    if height > max_height:
        height = max_height
        panel_width = height / aspect
    return (panel_width * n_panels + 0.6, height + 0.9)


def _horizontal_colorbar(fig: Figure, ax: Axes, mappable, label: str):
    """Colorbar aligned with the visible map width, placed below the axes."""
    cax = ax.inset_axes([0.12, -0.13, 0.76, 0.03])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_edgecolor(PALETTE["coast"])
    return cbar


def base_map(
    extent: Sequence[float],
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    states: bool = True,
    gridlines: bool = True,
) -> tuple[Figure, Axes]:
    """Create (or decorate) a PlateCarree map with house-style land/ocean/coast.

    Parameters
    ----------
    extent : sequence of float
        ``(lon_min, lon_max, lat_min, lat_max)``.
    ax : matplotlib.axes.Axes, optional
        Existing cartopy ``GeoAxes`` to draw on. A new figure/axes is created
        when omitted.
    figsize : tuple[float, float], optional
        Figure size when ``ax`` is not given. Derived from the extent's aspect
        ratio (see ``map_figsize``) when omitted.
    states : bool, default True
        Whether to draw US state borders.
    gridlines : bool, default True
        Whether to draw thin gray gridlines with labels on the left/bottom only.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and the (possibly newly created) map axes. No title is set.

    Examples
    --------
    >>> fig, ax = base_map((-90, -70, 15, 35))  # doctest: +SKIP
    """
    ccrs, cfeature = _require_cartopy()
    with style_context():
        if ax is None:
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=figsize or map_figsize(extent))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        else:
            fig = ax.figure

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=PALETTE["ocean"], zorder=0)
        ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=PALETTE["land"], zorder=0)
        ax.add_feature(
            cfeature.COASTLINE.with_scale("50m"),
            edgecolor=PALETTE["coast"],
            linewidth=0.6,
            zorder=1,
        )
        ax.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            edgecolor=PALETTE["coast"],
            linewidth=0.5,
            zorder=1,
        )
        if states:
            ax.add_feature(
                cfeature.STATES.with_scale("50m"),
                edgecolor=PALETTE["coast"],
                linewidth=0.4,
                zorder=1,
            )
        if gridlines:
            gl = ax.gridlines(
                draw_labels=True,
                linewidth=0.4,
                color=PALETTE["coast"],
                alpha=0.6,
                linestyle="--",
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 8, "color": PALETTE["muted"]}
            gl.ylabel_style = {"size": 8, "color": PALETTE["muted"]}
        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["coast"])
            spine.set_linewidth(0.8)
    return fig, ax


def _add_rmw_circles(ax: Axes, track: pd.DataFrame, ccrs, every: int = 4) -> None:
    """Overlay approximate geodesic RMW circles at a subsample of track fixes."""
    from cartopy import geodesic

    geod = geodesic.Geodesic()
    for _, row in track.iloc[::every].iterrows():
        rmw_nm = row.get("radius_max_wind_nm", np.nan)
        if pd.isna(rmw_nm) or rmw_nm <= 0:
            continue
        radius_m = float(rmw_nm) * 1852.0
        circle = geod.circle(
            lon=row["longitude"], lat=row["latitude"], radius=radius_m, n_samples=60
        )
        ax.plot(
            circle[:, 0],
            circle[:, 1],
            color=PALETTE["muted"],
            linewidth=0.5,
            alpha=0.6,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )


def plot_track(
    track: pd.DataFrame,
    *,
    ax: Axes | None = None,
    extent: Sequence[float] | None = None,
    color_by: Literal["category", "none"] = "category",
    annotate_every: Literal["24h"] | None = "24h",
    show_rmw: bool = False,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single storm track, coloured by Saffir-Simpson category.

    Parameters
    ----------
    track : pandas.DataFrame
        Processed track with ``time``, ``latitude``, ``longitude`` and
        ``max_wind_speed_kt`` columns (see the natcat track data model).
    ax : matplotlib.axes.Axes, optional
        Existing cartopy ``GeoAxes``. A new base map is created when omitted.
    extent : sequence of float, optional
        Map extent; defaults to the track bounding box padded by 3 degrees.
    color_by : {"category", "none"}, default "category"
        Colour track segments by Saffir-Simpson category (with a discrete
        legend) or draw a single ink-coloured line.
    annotate_every : {"24h"} or None, default "24h"
        Label 00Z fixes with small rounded date boxes, or disable with None.
    show_rmw : bool, default False
        Overlay approximate geodesic RMW circles at a subsample of fixes.
    title : str, optional
        Left-aligned bold title.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Examples
    --------
    >>> fig, ax = plot_track(michael_track)  # doctest: +SKIP
    """
    ccrs, _ = _require_cartopy()
    with style_context():
        if extent is None:
            pad = 3.0
            extent = (
                track["longitude"].min() - pad,
                track["longitude"].max() + pad,
                track["latitude"].min() - pad,
                track["latitude"].max() + pad,
            )
        if ax is None:
            fig, ax = base_map(extent, states=True)
        else:
            fig = ax.figure
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        lon = track["longitude"].to_numpy()
        lat = track["latitude"].to_numpy()
        time = pd.to_datetime(track["time"])
        six_hourly = (time.dt.hour % 6 == 0).to_numpy()

        if color_by == "category":
            wind = track["max_wind_speed_kt"].to_numpy()
            categories = _category_from_kt(wind)
            for i in range(len(lon) - 1):
                ax.plot(
                    lon[i : i + 2],
                    lat[i : i + 2],
                    color=SAFFIR_SIMPSON_COLORS[categories[i + 1]],
                    linewidth=2.4,
                    solid_capstyle="round",
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
            dot_colors = [SAFFIR_SIMPSON_COLORS[c] for c in categories[six_hourly]]
            ax.scatter(
                lon[six_hourly],
                lat[six_hourly],
                s=16,
                color=dot_colors,
                edgecolor=PALETTE["ink"],
                linewidth=0.4,
                zorder=6,
                transform=ccrs.PlateCarree(),
            )
            ax.legend(
                handles=_category_legend_handles(),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.06),
                ncol=len(SAFFIR_SIMPSON_ORDER),
                fontsize=7.5,
                handlelength=1.2,
                columnspacing=0.9,
            )
        elif color_by == "none":
            ax.plot(
                lon,
                lat,
                color=PALETTE["ink"],
                linewidth=1.8,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            ax.scatter(
                lon[six_hourly],
                lat[six_hourly],
                s=12,
                color=PALETTE["ink"],
                zorder=6,
                transform=ccrs.PlateCarree(),
            )
        else:
            raise ValueError(f"color_by must be 'category' or 'none', got {color_by!r}")

        if annotate_every == "24h":
            midnight = (time.dt.hour == 0).to_numpy()
            lon_min, lon_max, lat_min, lat_max = extent
            for lo_, la_, stamp in zip(lon[midnight], lat[midnight], time[midnight], strict=True):
                # Skip (rather than merely clip) fixes outside the visible extent: an
                # unclipped Text artist far outside the axes would otherwise force
                # savefig(bbox_inches="tight") to balloon the canvas out to include it.
                if not (lon_min <= lo_ <= lon_max and lat_min <= la_ <= lat_max):
                    continue
                ax.annotate(
                    stamp.strftime("%b %d"),
                    xy=(lo_, la_),
                    xytext=(6, 6),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=PALETTE["ink"],
                    bbox={
                        "boxstyle": "round,pad=0.2",
                        "fc": "white",
                        "ec": PALETTE["coast"],
                        "lw": 0.5,
                        "alpha": 0.9,
                    },
                    transform=ccrs.PlateCarree(),
                    clip_on=True,
                    annotation_clip=True,
                    zorder=7,
                )
        elif annotate_every is not None:
            raise ValueError(f"annotate_every must be '24h' or None, got {annotate_every!r}")

        if show_rmw:
            _add_rmw_circles(ax, track, ccrs)

        if title:
            ax.set_title(title, loc="left", fontweight="bold")
    return fig, ax


def plot_footprint(
    results: pd.DataFrame,
    *,
    ax: Axes | None = None,
    extent: Sequence[float],
    value: Literal["damage_ratio", "loss", "intensity"] = "damage_ratio",
    portfolio: pd.DataFrame | None = None,
    track: pd.DataFrame | None = None,
    min_tiv: float = 0.0,
    title: str | None = None,
    cbar_label: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot a spatial hazard/loss footprint over exposure.

    Parameters
    ----------
    results : pandas.DataFrame
        Loss-result DataFrame (portfolio columns plus ``intensity``,
        ``damage_ratio``, ``loss``; see the natcat data model).
    ax : matplotlib.axes.Axes, optional
        Existing cartopy ``GeoAxes``. A new base map is created when omitted.
    extent : sequence of float
        Map extent ``(lon_min, lon_max, lat_min, lat_max)``.
    value : {"damage_ratio", "loss", "intensity"}, default "damage_ratio"
        Which column to colour damaged locations by.
    portfolio : pandas.DataFrame, optional
        Full exposure to draw as a light grey background layer, sized and
        given alpha by log(TIV).
    track : pandas.DataFrame, optional
        Storm track to overlay thin and dark.
    min_tiv : float, default 0.0
        Drop portfolio locations at or below this TIV before plotting.
    title : str, optional
        Left-aligned bold title.
    cbar_label : str, optional
        Colorbar label; defaults to a sensible label for ``value``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Raises
    ------
    ValueError
        If ``value`` is not one of the supported options.

    Examples
    --------
    >>> fig, ax = plot_footprint(results, extent=(-90, -78, 24, 34))  # doctest: +SKIP
    """
    ccrs, _ = _require_cartopy()
    with style_context():
        if ax is None:
            fig, ax = base_map(extent, states=True)
        else:
            fig = ax.figure
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        if portfolio is not None:
            exposure = portfolio[portfolio["tiv"] > min_tiv] if min_tiv else portfolio
            if len(exposure):
                log_tiv = np.log10(exposure["tiv"].clip(lower=1.0))
                lo, hi = log_tiv.min(), log_tiv.max()
                dense = len(exposure) > 20_000
                size_range = (0.6, 4.0) if dense else (1.5, 12.0)
                sizes = (
                    np.full(len(exposure), np.mean(size_range))
                    if hi <= lo
                    else np.interp(log_tiv, (lo, hi), size_range)
                )
                ax.scatter(
                    exposure["longitude"],
                    exposure["latitude"],
                    s=sizes,
                    color="#B9C0CA",
                    alpha=0.45 if dense else 0.4,
                    linewidth=0,
                    transform=ccrs.PlateCarree(),
                    zorder=3,
                    label="Exposure (TIV)",
                )

        if value == "damage_ratio":
            damaged = results[results["damage_ratio"] > 0]
            mappable = ax.scatter(
                damaged["longitude"],
                damaged["latitude"],
                c=damaged["damage_ratio"],
                cmap="YlOrRd",
                norm=Normalize(vmin=0.0, vmax=1.0),
                s=6,
                alpha=0.85,
                linewidth=0,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            default_label = "Damage ratio"
        elif value == "loss":
            damaged = results[results["loss"] > 0]
            losses = damaged["loss"].clip(lower=1.0)
            norm = LogNorm(vmin=losses.min(), vmax=losses.max()) if len(losses) else None
            mappable = ax.scatter(
                damaged["longitude"],
                damaged["latitude"],
                c=losses,
                cmap="magma_r",
                norm=norm,
                s=6,
                alpha=0.85,
                linewidth=0,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            default_label = "Loss (USD)"
        elif value == "intensity":
            damaged = results[results["intensity"] > 0]
            cmap, norm = category_colormap()
            mappable = ax.scatter(
                damaged["longitude"],
                damaged["latitude"],
                c=damaged["intensity"],
                cmap=cmap,
                norm=norm,
                s=6,
                alpha=0.9,
                linewidth=0,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            default_label = "Max wind speed (kt)"
        else:
            raise ValueError(f"value must be 'damage_ratio', 'loss' or 'intensity', got {value!r}")

        if track is not None:
            ax.plot(
                track["longitude"],
                track["latitude"],
                color=PALETTE["ink"],
                linewidth=1.2,
                transform=ccrs.PlateCarree(),
                zorder=6,
                label="Storm track",
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                handles,
                labels,
                loc="lower left",
                fontsize=8,
                markerscale=3.0,
                frameon=True,
                framealpha=0.9,
                edgecolor="none",
            )
            legend.set_zorder(8)

        _horizontal_colorbar(fig, ax, mappable, cbar_label or default_label)

        if title:
            ax.set_title(title, loc="left", fontweight="bold")
    return fig, ax


def plot_genesis(
    historical: pd.DataFrame,
    synthetic: pd.DataFrame,
    *,
    ax: Axes | None = None,
    extent: Sequence[float] = (-100, -10, 0, 50),
    kde=None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Compare historical vs. synthetic storm genesis locations.

    Parameters
    ----------
    historical : pandas.DataFrame
        Historical genesis points with ``latitude``/``longitude`` columns.
    synthetic : pandas.DataFrame
        Sampled synthetic genesis points, same columns.
    ax : matplotlib.axes.Axes, optional
        Existing cartopy ``GeoAxes``. A new base map is created when omitted.
    extent : sequence of float, default (-100, -10, 0, 50)
        Map extent.
    kde : callable, optional
        Fitted density (e.g. ``scipy.stats.gaussian_kde``) evaluated on a
        ``(2, N)`` array of stacked ``[latitude, longitude]`` grid points, to
        draw as a contour overlay. Omitted when None.
    title : str, optional
        Axes title (left-aligned, bold).

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Examples
    --------
    >>> fig, ax = plot_genesis(historical, synthetic)  # doctest: +SKIP
    """
    ccrs, _ = _require_cartopy()
    with style_context():
        if ax is None:
            fig, ax = base_map(extent, states=False)
        else:
            fig = ax.figure
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        ax.scatter(
            historical["longitude"],
            historical["latitude"],
            s=7,
            color=PALETTE["muted"],
            alpha=0.45,
            linewidth=0,
            transform=ccrs.PlateCarree(),
            zorder=4,
            label=f"Historical genesis (n={len(historical)})",
        )
        ax.scatter(
            synthetic["longitude"],
            synthetic["latitude"],
            s=16,
            color=PALETTE["accent"],
            alpha=0.9,
            edgecolor="white",
            linewidth=0.4,
            transform=ccrs.PlateCarree(),
            zorder=5,
            label=f"Synthetic genesis (n={len(synthetic)})",
        )

        if kde is not None:
            lon_grid = np.linspace(extent[0], extent[1], 150)
            lat_grid = np.linspace(extent[2], extent[3], 150)
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            positions = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()])
            density = np.reshape(kde(positions), lon_mesh.shape)
            ax.contour(
                lon_mesh,
                lat_mesh,
                density,
                levels=6,
                colors=PALETTE["muted"],
                linewidths=0.5,
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8, markerscale=1.6
        )
        if title:
            ax.set_title(title, loc="left", fontweight="bold")
    return fig, ax


def plot_catalog(
    catalog: pd.DataFrame,
    *,
    ax: Axes | None = None,
    extent: Sequence[float],
    max_storms: int | None = None,
    color_by: Literal["category", "none"] = "category",
    historical: list[pd.DataFrame] | None = None,
) -> tuple[Figure, Axes]:
    """Plot many synthetic storm tracks, thin and low-alpha.

    Parameters
    ----------
    catalog : pandas.DataFrame
        Synthetic catalog in long format with ``storm_id``, ``latitude``,
        ``longitude`` and ``max_wind_speed_kt`` columns.
    ax : matplotlib.axes.Axes, optional
        Existing cartopy ``GeoAxes``. A new base map is created when omitted.
    extent : sequence of float
        Map extent.
    max_storms : int, optional
        Randomly subsample to this many storms for readability/speed.
    color_by : {"category", "none"}, default "category"
        Colour each track by its lifetime-maximum category (with legend), or
        draw all tracks in a single accent colour.
    historical : list[pandas.DataFrame], optional
        Optional historical tracks to overlay in ink for context.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    Examples
    --------
    >>> fig, ax = plot_catalog(catalog, extent=(-100, -10, 0, 50))  # doctest: +SKIP
    """
    ccrs, _ = _require_cartopy()
    with style_context():
        if ax is None:
            fig, ax = base_map(extent, states=True)
        else:
            fig = ax.figure
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        storm_ids = catalog["storm_id"].unique()
        if max_storms is not None and len(storm_ids) > max_storms:
            rng = np.random.default_rng(0)
            storm_ids = rng.choice(storm_ids, size=max_storms, replace=False)

        for storm_id in storm_ids:
            track = catalog[catalog["storm_id"] == storm_id]
            if color_by == "category":
                category = _category_from_kt([track["max_wind_speed_kt"].max()])[0]
                color = SAFFIR_SIMPSON_COLORS[category]
            elif color_by == "none":
                color = PALETTE["accent"]
            else:
                raise ValueError(f"color_by must be 'category' or 'none', got {color_by!r}")
            ax.plot(
                track["longitude"],
                track["latitude"],
                color=color,
                linewidth=0.6,
                alpha=0.45,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

        if color_by == "category":
            ax.legend(
                handles=_category_legend_handles(linewidth=2.0),
                loc="upper center",
                bbox_to_anchor=(0.5, -0.06),
                ncol=len(SAFFIR_SIMPSON_ORDER),
                fontsize=7.5,
                handlelength=1.2,
            )

        if historical is not None:
            for track in historical:
                ax.plot(
                    track["longitude"],
                    track["latitude"],
                    color=PALETTE["ink"],
                    linewidth=0.7,
                    alpha=0.5,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
    return fig, ax


def plot_catalog_comparison(
    historical_tracks: list[pd.DataFrame],
    catalog: pd.DataFrame,
    extent: Sequence[float],
    *,
    max_storms: int | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    seed: int | None = 0,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Side-by-side historical vs. synthetic catalog comparison.

    This is the figure to use in documentation: two panels with identical
    map styling, historical tracks on the left and synthetic tracks on the
    right, so genesis regions and track density can be compared visually.

    Parameters
    ----------
    historical_tracks : list[pandas.DataFrame]
        Historical tracks, each with ``latitude``/``longitude`` columns.
    catalog : pandas.DataFrame
        Synthetic catalog in long format (see ``plot_catalog``).
    extent : sequence of float
        Shared map extent for both panels.
    max_storms : int, optional
        Randomly subsample the synthetic catalog to this many storms.
    figsize : tuple[float, float], optional
        Overall figure size; derived from the extent when omitted.
    title : str, optional
        Figure-level title placed above both panels.
    seed : int, optional
        Seed for the synthetic subsample when ``max_storms`` is set.

    Returns
    -------
    tuple[matplotlib.figure.Figure, tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]]

    Examples
    --------
    >>> fig, (ax1, ax2) = plot_catalog_comparison(
    ...     historical_tracks, catalog, extent=(-100, -10, 0, 50)
    ... )  # doctest: +SKIP
    """
    ccrs, _ = _require_cartopy()
    with style_context():
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize or map_figsize(extent, max_width=12.5, n_panels=2))
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        base_map(extent, ax=ax1, states=True)
        base_map(extent, ax=ax2, states=True)

        for track in historical_tracks:
            ax1.plot(
                track["longitude"],
                track["latitude"],
                color=PALETTE["ink"],
                linewidth=0.7,
                alpha=0.5,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
        ax1.set_title(f"Historical (n={len(historical_tracks)})", loc="left", fontweight="bold")

        storm_ids = catalog["storm_id"].unique()
        if max_storms is not None and len(storm_ids) > max_storms:
            rng = np.random.default_rng(seed)
            keep = rng.choice(storm_ids, size=max_storms, replace=False)
            catalog = catalog[catalog["storm_id"].isin(keep)]
        plot_catalog(catalog, ax=ax2, extent=extent, color_by="none")
        ax2.set_title(
            f"Synthetic (n={catalog['storm_id'].nunique()})", loc="left", fontweight="bold"
        )
        if title:
            # Aspect-locked map axes settle their final position only after a draw;
            # anchor the title just above the left panel so no gap opens up.
            fig.canvas.draw()
            pos = ax1.get_position()
            fig.text(
                pos.x0,
                pos.y1 + 0.07,
                title,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=13,
            )
    return fig, (ax1, ax2)
