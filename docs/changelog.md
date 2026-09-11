# Changelog

All notable changes to NatCat are documented on this page.

## 0.1.0 &#8212; first packaged release

The flat, notebook-oriented `src/` collection (imported via `sys.path.append('../src')`) has been
refactored into an installable package, `natcat` (`pip install -e .`), with a clean, stable public
API, a documentation site, and a test suite.

### Added

- `natcat.calibration`: `load_observed_losses`/`normalise_losses` for observed storm-loss tables
  (the bundled NHC US-landfall CSV, GDP-normalised to an exposure reference year by default),
  `prepare_inputs`/`cases_from_inputs`/`save_inputs`/`load_inputs` for cacheable, hazard-independent
  per-storm inputs (`CaseInput`: track + regional exposure), `build_cases`/`save_cases`/`load_cases`
  as a footprint-included shorthand (`StormCase`), and `Calibrator`/`CalibrationResult` to fit a
  vulnerability model's parameters against them with `scipy.optimize.differential_evolution` plus a
  local polish. See `scripts/calibrate.py`, [Calibration](user-guide/calibration.md) and
  [Calibration methodology](methodology/calibration.md).
- `natcat.calibration.calibrate_hazard_grid`/`HazardGridResult`: an outer loop over the Rankine
  decay exponent and motion-asymmetry factor that recomputes every storm's footprint and refits the
  vulnerability at each grid point, so the hazard shape itself can be chosen by calibrated fit
  quality rather than assumed. See [Calibration methodology](methodology/calibration.md#hazard-parameter-grid).
- `ValueDependentVulnerability`: a logistic wind vulnerability curve whose half-damage wind speed
  varies with the log of exposure tile value, the model `natcat.calibration` is designed to fit.
  `ValueDependentVulnerability.calibrated()`/`CALIBRATED_PARAMS` ship the parameters fitted against
  the bundled 19-storm set (typical factor error 1.89x) as a ready-to-use model, no calibration run
  required.
- `VulnerabilityModel.damage_ratio` gained a `tiv` keyword (ignored by `WindVulnerability`, used by
  `ValueDependentVulnerability`); `WindVulnerability`/`ValueDependentVulnerability` gained
  `params`/`with_params` to expose and update their calibratable scalars, the interface
  `Calibrator` optimises over.
- `natcat.tracks.processing.willoughby_rmw`: the Willoughby, Darling & Rahn (2006) radius-of-
  maximum-wind relation (intensity + latitude), now the default for `fill_missing_rmw`/
  `prepare_track` (`rmw_method="willoughby"`); the previous intensity-class step table survives as
  `method="step"`.
- `plotting.plot_calibration` (modelled vs. observed loss, before/after),
  `plotting.plot_value_dependent_curves` (calibrated damage-ratio curves by tile value), and
  `plotting.plot_hazard_grid` (calibrated factor error heat map over the hazard-parameter grid).
- Installable package layout: `natcat/{data, tracks, hazards, vulnerability, exposure, loss,
  stochastic, financial, plotting, utils}`, each module with a docstring and explicit `__all__`.
- Top-level re-exports from `natcat/__init__.py`: `TropicalCycloneHazard`, `WindVulnerability`,
  `LossCalculator`, `LossSimulator`, `SyntheticTCCatalog`, `ExceedanceProbability`,
  `load_best_track`, `synthetic_portfolio`.
- Test suite (`tests/`) covering geo/unit utilities, ATCF parsing, track processing, the wind
  field, hazard/vulnerability models, loss calculation, the stochastic model, and financial
  metrics; offline-safe via `pytest.skip` when historical fixture data is absent.
- This documentation site (MkDocs Material, `mkdocstrings` API reference, worked example
  notebooks).
- `NATCAT_DATA_DIR` environment variable for explicit data-directory configuration.

### Changed

- **Calibrated hazard defaults.** `DEFAULT_DECAY_EXPONENT` (2.0 &rarr; 0.5) and
  `DEFAULT_ASYMMETRY_FACTOR` (0.5 &rarr; 0.3) in `natcat.hazards.wind_field` are now the winning
  point of the hazard-parameter grid, calibrated against 19 US landfalls (typical factor error
  improved from 4.35x to 1.89x). The pre-calibration values remain available by passing them
  explicitly.
- **RMW fill defaults to the Willoughby relation.** `fill_missing_rmw`/`prepare_track` default to
  `method`/`rmw_method="willoughby"` instead of the intensity-class step table, which gave every
  Category 4+ storm a flat 25 nm RMW when observed values (e.g. Hurricane Michael, 6&#8211;10 nm)
  were often far smaller &#8212; a real problem for the many pre-2005 best tracks that carry no
  observed RMW at all.
- **Catalog fitting no longer truncates historical tracks.** `SyntheticTCCatalog` fits its genesis
  and transition models on each historical track's full life cycle, including the post-landfall
  decay phase; `truncate_after_hurricane` truncation is still applied by `LossSimulator` when
  losses are evaluated. The transition model needs the full life cycle to learn realistic
  state-to-state transitions.
- **`ValueDependentVulnerability.DEFAULT_BOUNDS` are now physical**, not merely wide: `threshold_kt`
  34&#8211;55 kt, `v50_ref` 80&#8211;180, `v50_slope` &#8722;10..30, `k` 0.05&#8211;0.30, `scale`
  0.05&#8211;1, so the optimiser cannot compensate a biased hazard footprint with an implausible
  damage curve.
- **Tracks are truncated after the last hurricane-strength fix** (`truncate_after_hurricane`,
  default on in `prepare_track`, `load_best_track` and `LossSimulator`). The decaying
  post-landfall phase carries observed radii of maximum wind of 100 nm and more, which the
  Rankine vortex turned into trace damage far from the track (for Michael: coastal Georgia from a
  45 kt tropical storm over South Carolina).
- **Track interpolation defaults to 5 minutes** (`DEFAULT_TRACK_FREQ`), restoring the pre-refactor
  behaviour of the single-event pipeline. The hazard footprint is a maximum over discrete track
  positions, so the 1-hour default introduced during the refactor under-sampled compact,
  fast-moving storms (Hurricane Michael's loss halved and the footprint broke up into rings).
- **Footprint maps redrawn.** `plot_footprint` and `animate_footprint` size markers to the
  exposure grid (`tiling_marker_size`), draw weak locations first, hide damage ratios at or below
  0.1 % (`min_value`) and use a pale-to-red colour scale, so trace damage in the tropical-storm
  fringe no longer appears as a solid yellow field.
- **Vectorised wind field.** `max_wind_footprint`/`max_wind_history` replace a Python
  `DataFrame.iterrows()` loop with `(T, N)` broadcasting in bounded chunks; footprint computation
  no longer scales with a per-timestamp Python loop.
- **Fixed time-evolution complexity.** `TropicalCycloneHazard.compute_intensity_history` computes
  the wind field once as a cumulative running maximum (`numpy.maximum.accumulate`) instead of
  recomputing the full footprint at every timestamp (previously \(O(T^2 N)\)).
- **Rewritten exceedance-probability engine.** `ExceedanceProbability` replaces
  `ExceedenceProbabilityCalculator`: fixes a duplicate method definition, vectorises `aep`/`oep`
  over an array of thresholds, removes a list-minus-numpy-scalar bug in the GPD excess
  calculation, and no longer reads a simulation object's private `_results`.
- **Seeded, reproducible stochastic model.** `SyntheticTCCatalog` threads a single
  `numpy.random.Generator` through genesis sampling, KNN neighbor choice, transition draws and
  Poisson sampling; a fixed `seed` now actually reproduces a run (previously silently ignored in
  favor of global `numpy.random` calls).
- **De-duplicated best-track parsing.** `read_best_track` drops duplicate rows per timestamp
  (B-deck repeats each fix once per wind-radius threshold), fixing interpolation failures on
  duplicate x-values.
- **Pure track processing.** `interpolate_track` and related functions no longer mutate the input
  DataFrame or rely on `SettingWithCopy`-prone assignment.
- Renamed for clarity: `velocity_kt` &rarr; `translation_speed_kt`, `bearing_deg` &rarr;
  `heading_deg`, `check_a_deck_quality` &rarr; `validate_track`, "exceedence"/"emperical" spelling
  fixed to "exceedance"/"empirical" throughout.
- `LossCalculator.total_loss` raises `RuntimeError` (not `AttributeError`) when accessed before
  `compute()`.
- `WindVulnerability` raises `ValueError` (listing known construction types) for an unknown
  construction type, instead of silently falling back to default parameters.
- Single implementation of LitPop exposure loading (`natcat.exposure.litpop`), removing the
  duplicate previously split across `portfolio/generate.py` and
  `preprocessing/tropical_cyclone/exposure.py`; `climada` is imported lazily.
- **OEP clipped to AEP.** `ExceedanceProbability.oep`/`loss_at_return_period(kind="oep")` clip the
  occurrence curve to the annual curve (never returning an OEP loss greater than the AEP loss at
  the same return period) for consistency between the two, logging a warning whenever the clip
  binds &#8212; an artefact of fitting independent GPD tails to each.
- **`max_hours` is now in hours**, not days; `SyntheticTCCatalog(..., max_hours=720)` is the new
  default (previously a day-count parameter with an implicit unit mismatch).
- `SyntheticTCCatalog`/`step_track` gained `max_wind_kt=185.0` and `max_rmw_nm=150.0`, physical
  caps applied at every Markov step so the unbounded random walk cannot drift past the strongest
  recorded storm or grow an implausible radius of maximum wind; genesis values are copied directly
  from the historical record and are not affected by these caps.
- All documentation and README figures regenerated with the `natcat.plotting` house style.

### Fixed

- Empty-input handling: zero synthetic genesis points, zero storms in a Poisson-sampled year, and
  empty per-year event concatenation no longer raise.
- Network requests (`download_best_track`, `download_a_deck`, `download_archive`) now set an
  explicit `timeout`, call `raise_for_status()`, and raise `FileNotFoundError`/`RuntimeError` with
  the failing URL instead of silently returning `None`.
- Removed a bare `except:` in track preparation.
- `np.float_` (removed in NumPy 2) replaced with `np.float64` throughout.
- `spatial_loss`'s hard-coded `tiv > 1e8` filter is now a `min_tiv` parameter.
- `storm_per_year` no longer mutates the DataFrame passed to it.

### Removed

- `src/` as an import path; superseded modules preserved verbatim under `legacy/` for reference.
- Unused imports and the unused `DateTime` export from `hazards.base`.
- `TropicalCycloneHazard.compute_intensity_over_time` (incorrect docstring, unused, superseded by
  `compute_intensity_history`).
