# Migration guide: `src/` to `natcat`

This page maps the first-generation code (the flat `src/` tree imported with
`sys.path.append('../src')`) to the `natcat` package, and lists every change that alters results,
defaults or error behaviour. Everything not listed under *changed behaviour* is the original logic
at a new address.

## Changes that alter results or defaults

A run on the package will not reproduce a run on the old tree because of the following.

| Where | Old | New | Why |
|---|---|---|---|
| `tracks.processing.prepare_track` | `preprocess_B_deck` interpolated first, then filled RMW | Fill RMW first, then interpolate | A raw RMW of 0 was linearly blended with real neighbours before the heuristic saw it. The forecast pipeline (`prepare_track_data`) already used the new order. |
| `data.atcf.read_best_track` | No de-duplication | Duplicate fixes per timestamp dropped; RMW 0 read as `NaN` | B-decks repeat each fix once per wind-radius record (Michael: 85 rows for 38 fixes). |
| `tracks.processing.truncate_after_hurricane` | Did not exist | Track cut after the last `HU` fix, default on | Post-landfall fixes carry radii of maximum wind of 100 nm and more, which the Rankine vortex turned into trace damage far from the track. |
| `loss.simulation.LossSimulator` | `interp_time_step='1h'` | `freq='5min'` | The footprint is a maximum over discrete track positions; 1 h under-sampled compact storms by about a quarter of AAL. |
| `stochastic.catalog.SyntheticTCCatalog` | `max_hours=360` ran 360 steps of 3 h, i.e. 1080 h | `max_hours=720` means 720 h | The parameter now means what its name says. |
| `stochastic.transitions.step_track` | Floors only (wind ≥ 0, RMW ≥ 5 nm) | Also caps: wind ≤ 185 kt, RMW ≤ 150 nm | The unbounded random walk produced radii above 200 nm. |
| `SyntheticTCCatalog(seed=...)` | `self.rng` created but never used; all draws from global NumPy state | Every draw uses `self.rng` | The seed previously had no effect. |
| `SyntheticTCCatalog(n_neighbors=...)` | Ignored; KNN hard-coded to 5 | Honoured, default still 5 | |
| `financial.ep.ExceedanceProbability` | `genpareto.fit(excesses)` with free location | `floc=0` pinned at the threshold | The tail now starts at the threshold, as the two-regime design intends. |
| `ExceedanceProbability.oep` | Independent OEP and AEP tails could cross | OEP clipped to AEP, warning logged | Consistency between the curves. |
| `exposure.synthetic.synthetic_portfolio` | `np.random.seed(42)` on global state | Local `default_rng(seed)` | Same seed value, different draws, no global side effect. |
| `vulnerability.wind.WindVulnerability` | Unknown construction type silently used `v_50=110` | Raises `ValueError` | The silent fallback hid typos. |
| `tracks.processing.interpolate_track` | Mutated its input and returned only four columns | Never mutates; interpolates every numeric column, carries text columns forward | `storm_type` and pressure were being dropped. |
| `LossSimulator.run` | Generated the catalog one year at a time | One `generate(n_years=...)` call, grouped by year | The random sequence differs even with the same seed. |
| `tracks.forecast.prepare_forecast_track` | `time_step='5min'` hard-coded | `freq='1h'` default | Forecast (A-deck) pipeline only; best tracks stay at 5 min. |
| `tracks.processing.fill_missing_rmw` | `rmw_method="step"` (intensity-class table, `prepare_track` default) | `rmw_method="willoughby"` default; `method="step"` still available | Pre-2005 best tracks carry no observed RMW at all; the step table gave every Category 4+ storm a flat 25 nm, when Hurricane Michael's observed RMW at landfall was 6&#8211;10 nm. `willoughby_rmw` (Willoughby, Darling & Rahn 2006) conditions on intensity and latitude instead. |
| `hazards.wind_field.DEFAULT_DECAY_EXPONENT` / `DEFAULT_ASYMMETRY_FACTOR` | `2.0` / `0.5` | `0.5` / `0.3` | Chosen by the hazard-parameter grid in the loss calibration against 19 US landfalls (see [Calibration](methodology/calibration.md)); typical calibrated factor error improved from 4.35x to 1.77x. |
| `stochastic.catalog.SyntheticTCCatalog` (fitting) | Historical tracks truncated after the last hurricane-strength fix before fitting the genesis/transition models | Full track (including post-landfall decay) used when fitting; `truncate_after_hurricane` truncation is still applied by `LossSimulator` at loss-evaluation time | The transition model needs the storm's full life cycle to learn realistic state-to-state transitions, not just the hurricane phase. |

## Same logic, new address

The formulas are unchanged: the Rankine profile and the motion-asymmetry term
`factor · translation speed · sin(angle)` (defaults for both changed after this migration &#8212;
see the table above), the RMW step heuristic (80/60/40/25/15 nm, 1.5x for `EX`, still available as
`method="step"`), the logistic vulnerability with the 40 kt threshold and the Frame/Masonry
parameters, KDE genesis with land rejection, the 2&deg; grid Markov transitions, land decay 0.92
and RMW growth 1.02 per hour, the Poisson frequency, the 95th-percentile tail split, haversine
distance and bearing.

| Old | New | Notes |
|---|---|---|
| `hazards/tropical_cyclone.py::rankine_vortex` | `hazards/wind_field.py::rankine_vortex` | unchanged |
| `hazards/tropical_cyclone.py::max_wind_speeds_at_locations` | `hazards/wind_field.py::max_wind_footprint` | `iterrows` loop replaced by a chunked `(T, N)` broadcast; equal to floating-point precision |
| `TropicalCycloneHazard.compute_intensity_at_timestep` | `TropicalCycloneHazard.compute_intensity_history` | one cumulative pass for all timestamps |
| `hazards/tropical_cyclone.py::get_heuristic_rmw` | `tracks/processing.py::fill_missing_rmw` | step table kept as `method="step"`; default is now `method="willoughby"` (see the table above) |
| `loss_model.py::LossCalculator.calculate_portfolio_loss` | `loss/calculator.py::LossCalculator.compute` | old name kept as a deprecated alias |
| `LossCalculator.calculate_portfolio_loss_over_time` | `LossCalculator.compute_history` | |
| `utils/track.py::track_interpolation` | `tracks/processing.py::interpolate_track` | see table above |
| `utils/track.py::cyclone_velocity` / `cyclone_bearing` | `tracks/processing.py::add_translation_velocity` / `add_heading` | columns `velocity_kt` → `translation_speed_kt`, `bearing_deg` → `heading_deg`, `timestamp` → `time` |
| `utils/track.py::extract_past_trajectory` / `extract_future_trajectory` / `prepare_track_data` | `tracks/forecast.py` (same names; `prepare_forecast_track`) | |
| `preprocessing/tropical_cyclone/pipeline.py::preprocess_B_deck` | `tracks/pipeline.py::load_best_track` + `tracks/processing.py::prepare_track` | download/read and processing separated |
| `preprocessing/tropical_cyclone/track.py::clean_track_data` | `data/atcf.py::read_best_track` / `read_a_deck` | positional ATCF field map instead of fixed column lists |
| `preprocessing/tropical_cyclone/track.py::check_a_deck_quality` | `data/quality.py::validate_track` | duplicate-timestamp check added |
| `preprocessing/tropical_cyclone/download.py` | `data/nhc.py::NHCClient.download_deck` / `download_best_track` | 30 s timeout; raises instead of returning `None` |
| `preprocessing/tropical_cyclone/data_aggregator.py` | `data/nhc.py::NHCClient.download_archive` | cache keyed on the extracted `.dat` |
| `preprocessing/tropical_cyclone/exposure.py`, `portfolio/generate.py` (duplicates) | `exposure/synthetic.py::synthetic_portfolio`, `exposure/litpop.py::load_litpop_exposure` | CLIMADA imported lazily |
| `probabilistic/sythetic_TC_track.py` | `stochastic/{genesis,transitions,catalog,frequency}.py` | `calculate_next_point` → `utils/geo.py::destination_point` |
| `reinsurance/ep.py::ExceedenceProbabilityCalculator` | `financial/ep.py::ExceedanceProbability` (`EPCurve` alias) | `calculate_eOEP/eAEP` → `oep_empirical/aep_empirical`, `calculate_OEP/AEP` → `oep/aep`, `EP_curve` → `curve`, `calculate_AAL` → `aal`; `loss_at_return_period` is new |
| `reinsurance/base.py::EventResults` / `PortfolioResults` | `financial/results.py::EventResult` / `YearResult` | |
| `visualization/tropical_cyclone.py::plot_trajectory` / `spatial_loss` / `animate_spatial_loss` | `plotting/maps.py::plot_track` / `plot_footprint`, `plotting/animation.py::animate_footprint` | functions no longer save files; call `plotting.save` |
| `utils/geo.py`, `utils/units.py` | `utils/geo.py`, `utils/units.py` | unchanged; constants in `config.py` |

## Not carried over

- `TropicalCycloneHazard.compute_intensity_over_time` (single timestamp or window without a running
  maximum).
- `clean_track_data2`, the dynamic-offset A-deck parser.
- `SyntheticTCCatalog.plot_tracks` and `plot_EP_curve` as methods; use `plotting.plot_catalog` and
  `plotting.plot_ep_curve`.
- MP4 export from the animation (GIF only).
- The `recovery_ratio` and `layer_exhausted` result fields. No reinsurance layer logic existed in
  either version.
- The hand-written year/basin/storm-number format checks in `preprocess_B_deck`; a bad id now
  surfaces as `FileNotFoundError` from the NHC client.

The `src/` tree described above is preserved in git history (last commit before the merge:
`f2e3556`). The even older first-generation modules and A-deck notebooks live under `legacy/`.
