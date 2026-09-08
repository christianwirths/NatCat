# Quickstart

An end-to-end run from a historical best track to a mapped loss footprint, in about 30 lines.

```python
import natcat
from natcat.tracks import load_best_track
from natcat.hazards import TropicalCycloneHazard
from natcat.vulnerability import WindVulnerability
from natcat.exposure import synthetic_portfolio
from natcat.loss import LossCalculator
from natcat import plotting

# 1. Load and process a historical best track.
#    (year, basin, storm_number) -> NHC ATCF identifier, e.g. Hurricane Michael (2018) = AL142018.
#    Downloads the B-deck file into NATCAT_DATA_DIR if not already cached.
track = load_best_track(2018, "al", "14")

# 2. Build the hazard: a parametric wind field driven by the processed track.
hazard = TropicalCycloneHazard(track)

# 3. Choose a vulnerability function for the exposure's construction type.
vulnerability = WindVulnerability(construction_type="Masonry")

# 4. Build (or load) an exposure portfolio.
#    synthetic_portfolio scatters `n` locations with random TIV inside a lat/lon bounding box.
portfolio = synthetic_portfolio(
    n=500,
    bounds=(28.0, 31.5, -87.5, -83.5),  # Florida Panhandle
    seed=42,
)

# 5. Compute ground-up loss: intensity -> damage ratio -> loss, per location.
calculator = LossCalculator(hazard, vulnerability)
result = calculator.compute(portfolio)

print(f"Total insured value:  ${portfolio['tiv'].sum():,.0f}")
print(f"Ground-up loss:       ${calculator.total_loss:,.0f}")
print(f"Mean damage ratio:    {result['damage_ratio'].mean():.3f}")

# 6. Plot the wind footprint and the portfolio loss.
plotting.plot_footprint(
    result, extent=(-87.5, -83.5, 28.0, 31.5), portfolio=portfolio, track=track
)
```

`result` is a DataFrame with the portfolio columns (`location_id`, `latitude`, `longitude`, `tiv`,
`construction`) plus `intensity` (kt), `damage_ratio` (0&#8211;1) and `loss` (USD). See
[Data model](../user-guide/data-model.md) for the full column reference.

## Next steps

- Work through **[Single-event loss](../user-guide/single-event-loss.md)** for the full track
  DataFrame contract, time-evolving loss (`compute_history`), and figure options.
- Move to **[Stochastic event set](../user-guide/stochastic-event-set.md)** to generate a
  synthetic catalog and simulate thousands of years with `LossSimulator`.
- Combine simulated years into **[Financial metrics](../user-guide/financial-metrics.md)**: AEP,
  OEP, return periods and AAL.
- See the **[Examples](../notebooks/01_single_event_loss.ipynb)** notebooks for complete,
  runnable walkthroughs.
