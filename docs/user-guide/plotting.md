# Plotting

`natcat.plotting` provides a consistent house style and a small set of purpose-built figure
functions for tracks, footprints, curves and animations.

## House style

`apply_style()` mutates `matplotlib.rcParams` for the remainder of the process: white background,
no top/right spines, grid alpha 0.25, a Helvetica Neue &rarr; Arial &rarr; DejaVu Sans font
fallback stack, base font size 10 with left-aligned bold 12pt titles, 200 dpi for saved PNGs,
tight bounding boxes, and default figure sizes of `(7.5, 4.5)` for single-axes plots (maps pass
their own, wider figure size explicitly). Every `natcat.plotting` function applies the style
internally &#8212; scoped to its own call via `style_context()` &#8212; and so does `save()`, so
you never *need* to call `apply_style()` yourself to get correctly styled output files.

Call it once, though, at the top of a notebook if you want figures that are simply displayed
inline (not saved via `natcat.plotting.save`) to also pick up the house style, since matplotlib's
inline renderer uses the current global `rcParams`:

```python
from natcat.plotting import apply_style

apply_style()
```

For custom, non-`natcat.plotting` figures where you don't want to mutate global state, use the
context-manager form instead:

```python
from natcat.plotting import style_context

with style_context():
    ...  # any matplotlib code
```

## Maps

```python
from natcat import plotting

extent = (-87.5, -83.5, 28.0, 31.5)  # (lon_min, lon_max, lat_min, lat_max)

plotting.base_map(extent)                                          # requires cartopy (geo extra)
plotting.plot_track(track)                                          # colored by Saffir-Simpson category
plotting.plot_footprint(result, extent=extent, portfolio=portfolio, track=track)  # damage on exposure; min_value hides trace damage
plotting.plot_genesis(historical_genesis, synthetic_genesis)        # historical vs. KDE genesis density
plotting.plot_catalog(catalog, extent=(-100, -10, 0, 50), max_storms=200)  # sample of synthetic tracks
```

Map functions take an `extent` in `(lon_min, lon_max, lat_min, lat_max)` order (the cartopy
convention &#8212; note this is *not* the same axis order as the `bounds` used by
`synthetic_portfolio`/`load_litpop_exposure`, which is `(lat_min, lat_max, lon_min, lon_max)`).
When `figsize` is omitted, map functions size the figure automatically from the extent's aspect
ratio (internally via `natcat.plotting.maps.map_figsize`), so you rarely need to pass one
yourself.

!!! note "cartopy is optional"
    Everything in `natcat.plotting.maps` imports `cartopy` lazily inside the function body, so
    `import natcat.plotting` never requires the `geo` extra &#8212; only calling a map function does.

## Curves

```python
plotting.plot_vulnerability(vulnerability)                       # damage ratio vs. wind speed, by construction
plotting.plot_wind_profile(vmax=120.0, rmw=25.0)                  # radial wind profile for a set of decay exponents
plotting.plot_ep_curve(ep, kinds=("aep",))                        # AEP and/or OEP, log-log
plotting.plot_annual_loss_distribution(results.annual_losses)     # simulated annual loss distribution
plotting.plot_land_decay(catalog.land_decay, segments)            # observed inland decay vs. the fitted model
```

## Animation

```python
plotting.animate_footprint(history, track=track, portfolio=portfolio, extent=extent, path="damage_evolution.gif")
```

Renders one frame per timestamp in a `compute_history` result and writes an animated GIF of the
damage ratio sweeping across the portfolio as the storm makes landfall &#8212; see
[Single-event loss](single-event-loss.md#time-evolving-loss) for the Hurricane Michael example.

## Color conventions

| Use | Palette |
|-----|---------|
| Saffir&#8211;Simpson category (TD, TS, C1&#8211;C5) | `#5EBAFF`, `#00FAF4`, `#FFF3A0`, `#FFE775`, `#FFC140`, `#FF8F20`, `#FF6060` (NHC-style) |
| Damage ratio / loss (sequential) | `YlOrRd` (damage ratio) or `magma_r` (loss, log-normalized) |
| TIV (exposure markers) | flat gray `#B9C0CA`, marker size log-scaled by TIV |
| Structural elements | ink `#1F2933`, muted `#6B7280`, land `#F3F4F6`, ocean `#E8F1F8`, coastline `#9CA3AF` |
| Accents | blue `#2563EB`, orange `#F59E0B`, red `#DC2626`, green `#059669`, purple `#7C3AED` |

Every figure produced by `natcat.plotting` labels its axes with units, keeps the legend outside
the data area, and adds a small gray source note (e.g. "Source: NHC ATCF best track; LitPop
exposure") where applicable.
