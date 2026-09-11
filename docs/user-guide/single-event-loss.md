# Single-event loss

Compute the ground-up loss a single historical or forecast track causes on a portfolio, including
its full spatial and temporal evolution.

## Load and prepare a track

```python
from natcat.tracks import load_best_track

track = load_best_track(2018, "al", "14")
```

`load_best_track` downloads (if needed), reads, and runs the full `prepare_track` pipeline:
de-duplicate &rarr; `truncate_after_hurricane` &rarr; `fill_missing_rmw` &rarr; `interpolate_track`
&rarr; `add_translation_velocity`
&rarr; `add_heading`. The result matches the [processed track](data-model.md#processed-track)
schema. `freq` controls the interpolation step (default `"5min"`). Keep it fine: the footprint is
a maximum over the *discrete* track positions, so a step coarser than the time the storm needs to
travel one radius of maximum wind leaves gaps between successive centres, produces a
"string of pearls" footprint and under-estimates loss. For Hurricane Michael (RMW about 10 nm at
landfall, 13 kt translation) a 1-hour step halves the modelled portfolio loss; see
[Wind field: temporal sampling](../methodology/wind-field.md#temporal-sampling).

`truncate_after_hurricane` (default `True`) cuts the track after its last fix classified as a
hurricane (ATCF `storm_type == "HU"`, or `max_wind_speed_kt >= 64` when no type is available).
After landfall a decaying storm is recorded with a very large radius of maximum wind (Michael:
120&#8211;180 nm as a tropical/extratropical storm over the Carolinas), which a Rankine vortex turns
into a broad ring of near-threshold winds and trace damage far from the track. Pass
`truncate_after_hurricane=False` to keep the full track, e.g. for tropical-storm-only events.

`rmw_method` (default `"willoughby"`) controls how `prepare_track` fills fixes with a missing or
non-positive radius of maximum wind &#8212; the norm for pre-2005 best tracks. The default uses the
Willoughby, Darling & Rahn (2006) intensity/latitude relation; `rmw_method="step"` falls back to the
package's earlier, coarser intensity-class table. See
[Wind field: radius of maximum wind](../methodology/wind-field.md#radius-of-maximum-wind-rmw).

![Hurricane Michael (2018) best track approaching the Florida Panhandle](../assets/figures/michael_track.png){ width="100%" }
*Figure: OFCL best track for Hurricane Michael (2018), colored by Saffir&#8211;Simpson category.*

## Build the hazard

```python
from natcat.hazards import TropicalCycloneHazard

hazard = TropicalCycloneHazard(track)   # decay_exponent=0.5, asymmetry_factor=0.3 by default
```

`compute_intensity(coords)` returns the maximum wind speed (kt) experienced at each coordinate
over the entire track &#8212; the standard footprint used for a single ground-up loss estimate.

```python
import numpy as np

coords = np.column_stack([portfolio["latitude"], portfolio["longitude"]])
footprint = hazard.compute_intensity(coords)   # shape (N,)
```

![Maximum sustained wind footprint over the exposed area](../assets/figures/michael_footprint.png){ width="100%" }
*Figure: gridded maximum wind footprint (kt) from the Rankine vortex wind field.*

!!! tip "Decay exponent and asymmetry factor are calibrated defaults"
    `TropicalCycloneHazard(..., decay_exponent=..., asymmetry_factor=...)` controls how fast wind
    speed decays outside the radius of maximum wind (RMW) and how much of the storm's own motion
    projects onto the wind field. The defaults, 0.5 and 0.3, are not arbitrary: they are the
    hazard-grid point that produced the best calibrated fit against 19 observed US landfalls (see
    [Calibration](../methodology/calibration.md#hazard-parameter-grid)). See
    [Wind field](../methodology/wind-field.md) for the physical interpretation of both parameters.

## Apply vulnerability and compute loss

```python
from natcat.vulnerability import WindVulnerability
from natcat.loss import LossCalculator

vulnerability = WindVulnerability("Frame")   # or pass construction_types per-asset to compute()
calculator = LossCalculator(hazard, vulnerability)

result = calculator.compute(portfolio)
print(calculator.total_loss)
```

`result` adds `intensity`, `damage_ratio` and `loss` to the portfolio columns (see
[Data model](data-model.md#loss-result)). `calculator.total_loss` sums `loss` across the
portfolio; it raises `RuntimeError` if `compute` has not been called yet.

## Time-evolving loss

To animate or inspect how damage accumulates as the storm passes &#8212; e.g. the Hurricane Michael
example on the [project README](https://github.com/christianwirths/NatCat) &#8212; use
`compute_history` with a series of timestamps:

```python
import pandas as pd

times = pd.date_range(track["time"].min(), track["time"].max(), freq="30min")
history = calculator.compute_history(portfolio, times)
```

`history` is long-format: portfolio columns + `intensity`, `damage_ratio`, `loss`, `time`, with one
block of rows per timestamp. Internally this uses
`TropicalCycloneHazard.compute_intensity_history`, which computes the wind field **once** as a
cumulative running maximum over time (`numpy.maximum.accumulate`) rather than recomputing the
whole footprint at every timestamp &#8212; an important performance fix over the original
implementation, which was quadratic in the number of track points.

```python
from natcat import plotting

extent = (-87.5, -83.5, 28.0, 31.5)  # (lon_min, lon_max, lat_min, lat_max)
plotting.animate_footprint(history, track=track, portfolio=portfolio, extent=extent)
```

![Damage evolution during Hurricane Michael's landfall](../assets/animations/michael_damage_evolution.gif){ width="100%" }
*Figure: spatial damage-ratio evolution as the best-track wind field sweeps across the portfolio.*

## Reference

| Signature | Returns |
|-----------|---------|
| `TropicalCycloneHazard(track, *, vortex="rankine", asymmetry_factor=0.3, decay_exponent=0.5)` | hazard instance |
| `hazard.compute_intensity(coords)` | `(N,)` max wind speed (kt) |
| `hazard.compute_intensity_history(coords, times)` | `(T, N)` cumulative running max at each time |
| `LossCalculator(hazard, vulnerability)` | calculator instance |
| `calculator.compute(portfolio)` | loss DataFrame; sets `.results` |
| `calculator.compute_history(portfolio, times)` | long-format loss DataFrame with `time` |
| `calculator.total_loss` | `float`; raises `RuntimeError` before `compute()` |

See the full [API reference](../api/hazards.md) for parameter and type details.
