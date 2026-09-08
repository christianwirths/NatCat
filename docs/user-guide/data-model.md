# Data model

NatCat passes plain pandas DataFrames between components rather than custom container classes;
this page is the authoritative column reference used throughout the package, notebooks and docs.

## Processed track

One row per time step, sorted by `time`. This is the shape returned by `load_best_track`,
`prepare_track`, and every stochastic-catalog track, and the shape `TropicalCycloneHazard`
expects.

| Column | Dtype | Unit | Notes |
|--------|-------|------|-------|
| `time` | `datetime64[ns]` | | Unique, sorted ascending |
| `latitude` | `float` | deg N | |
| `longitude` | `float` | deg E | |
| `max_wind_speed_kt` | `float` | kt | 1-minute sustained wind |
| `radius_max_wind_nm` | `float` | nm | Filled by `fill_missing_rmw` if missing |
| `min_pressure_mb` | `float`, optional | hPa | |
| `storm_type` | `str`, optional | ATCF `TY` code | e.g. `HU`, `TS`, `EX` |
| `translation_speed_kt` | `float` | kt | Storm motion speed; added by `add_translation_velocity` |
| `heading_deg` | `float` | deg, 0=N clockwise | Storm motion direction; added by `add_heading` |
| `storm_id` | `str`, optional | | Present on stochastic-catalog and multi-storm tracks |

!!! note "Edge behavior"
    `translation_speed_kt` and `heading_deg` for the last track point are forward-filled from the
    previous point, since motion at the final fix cannot be computed from a following point. This
    is intentional and documented here rather than treated as missing data.

## Raw best track

`natcat.data.atcf.read_best_track` returns the ATCF B-deck fields before track processing:

`time`, `latitude`, `longitude`, `max_wind_speed_kt`, `radius_max_wind_nm`, `min_pressure_mb`,
`storm_type`, `storm_name`, `basin`, `storm_number`.

B-deck files repeat each fix once per reported wind-radius threshold (`RAD` = 34/50/64 kt).
**`read_best_track` de-duplicates on `time`, keeping the first row per timestamp** &#8212; a
correctness fix over naive parsing, which otherwise produces duplicate x-values that break
downstream interpolation. `radius_max_wind_nm` stays `NaN` (or 0) until `fill_missing_rmw` is
applied by `prepare_track`.

## Exposure / portfolio

| Column | Dtype | Unit | Notes |
|--------|-------|------|-------|
| `location_id` | `str`/`int` | | Unique per row |
| `latitude` | `float` | deg N | |
| `longitude` | `float` | deg E | |
| `tiv` | `float` | USD | Total insured value |
| `construction` | `str`, optional | | e.g. `Frame`, `Masonry`; required if no default is set on `WindVulnerability` |

Produced by `natcat.exposure.synthetic_portfolio` or `natcat.exposure.load_litpop_exposure`.

## Loss result

Portfolio columns plus the per-location hazard/vulnerability/loss outputs, produced by
`LossCalculator.compute`:

| Column | Dtype | Unit |
|--------|-------|------|
| *(all portfolio columns)* | | |
| `intensity` | `float` | kt |
| `damage_ratio` | `float` | 0&#8211;1 |
| `loss` | `float` | USD |

`LossCalculator.compute_history` returns the same shape in long format with an added `time`
column, one block of rows per requested timestamp (used for time-evolution figures like the
Hurricane Michael damage-evolution animation).

## Coordinates array

Functions that operate purely on geometry (`max_wind_footprint`, `rankine_vortex`,
`motion_asymmetry`) take/return a coordinates array of shape `(N, 2)`, `float64`, ordered
`[latitude, longitude]` &#8212; not `[longitude, latitude]` as some GIS conventions use.

```python
import numpy as np

coords = np.column_stack([portfolio["latitude"], portfolio["longitude"]])  # (N, 2)
```
