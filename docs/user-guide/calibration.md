# Calibration

`natcat.calibration` fits a vulnerability model's parameters against observed storm losses:
load an observed-loss table, prepare one track and regional exposure per storm, evaluate a wind
footprint against a chosen hazard, and minimise a log-ratio objective with `Calibrator`. The result
is a `VulnerabilityModel` you can drop straight into `LossCalculator` or `LossSimulator` in place of
`WindVulnerability` &#8212; or just use the already-calibrated defaults shipped with the package,
`ValueDependentVulnerability.calibrated()`, and skip this page entirely.

```python
from natcat.calibration import Calibrator, prepare_inputs, cases_from_inputs, load_observed_losses, normalise_losses
from natcat.vulnerability import ValueDependentVulnerability

observed = normalise_losses(load_observed_losses(), reference_year=2018)  # GDP-normalised by default
inputs = prepare_inputs(observed)                  # track + regional exposure per storm, once
cases = cases_from_inputs(inputs)                  # wind footprint at the default hazard
result = Calibrator(cases, ValueDependentVulnerability()).fit(seed=0)
print(result.summary())
```

See [Calibration methodology](../methodology/calibration.md) for the value-dependent vulnerability
model, the objective function, the hazard-parameter grid and, importantly, the caveats around the
bundled observed-loss data.

## Observed losses

`load_observed_losses` reads the bundled NHC US-landfall table by default, or a CSV of your own
via `path=`. Every table needs `REQUIRED_COLUMNS`:

`storm_id`, `name`, `year`, `basin`, `storm_number`, `observed_loss_usd`, `loss_year`

and accepts these optional columns:

| Column | Meaning |
|---|---|
| `damage_driver` | `wind` / `mixed` / `surge` / `flood` &#8212; informational, not used by the loader |
| `include` | Drop the row when `included_only=True` (the default). Storms whose damage is dominated by surge or rainfall flooding are marked `False` in the bundled table because the model is wind-only. |
| `normalisation_factor` | Overrides the factor `normalise_losses` would otherwise compute |
| `weight` | Objective weight for this storm (default 1); see [the objective](../methodology/calibration.md#objective-function) |
| `source`, `notes` | Free text, carried through unused |

```python
from natcat.calibration import load_observed_losses, normalise_losses

observed = load_observed_losses()                    # bundled NHC US-landfall table
observed = normalise_losses(observed, reference_year=2018)   # + normalisation_factor, observed_loss_ref_usd
```

`normalise_losses` brings every storm's `observed_loss_usd` (nominal USD of `loss_year`) to the
year `reference_year` refers to &#8212; the year LitPop's exposure values refer to &#8212; by scaling
with the ratio of US nominal GDP between the loss year and the reference year (`method="gdp"`, the
default: a Pielke-style price &times; wealth &times; population proxy, see
[Calibration methodology](../methodology/calibration.md)), the ratio of US CPI-U annual averages
(`method="cpi"`, prices only), or leaving the loss unchanged (`method="none"`). It adds
`normalisation_factor` and `observed_loss_ref_usd`, and never mutates its input. Passing your own
`path` to `load_observed_losses` lets you calibrate against insured losses, a different country, or
a different storm set, as long as the CSV carries the required columns.

!!! warning "The bundled table is a starting point, not an authoritative loss database"
    Check the cited NHC Tropical Cyclone Reports before using a calibration built on it for
    anything beyond exploration, and prefer your own (e.g. insured) losses where available.

## Preparing inputs and building cases

A storm's wind footprint depends on the hazard parameters (decay exponent, asymmetry factor) but its
track and regional exposure do not, and those are the slow part &#8212; downloading best tracks and
building LitPop exposure. `prepare_inputs` does that slow, hazard-independent work once per storm
and returns a `CaseInput` per row of the observed table; `cases_from_inputs` then evaluates a wind
footprint against a chosen hazard to produce the `StormCase`s the calibrator actually fits against:

```python
inputs = prepare_inputs(observed)
cases = cases_from_inputs(inputs)          # default hazard: TropicalCycloneHazard defaults
```

For each row, `prepare_inputs`:

1. Loads the best track (`natcat.tracks.load_best_track`).
2. Derives an exposure box around the part of the track over land (`storm_region`, `margin_deg=2.0`
   padding, clipped to `CONUS_BOUNDS` by default).
3. Loads the regional exposure for that box &#8212; LitPop USA by default (`load_litpop_exposure`),
   overridable via `exposure_loader=lambda bounds: ...` for a different country or your own
   portfolio.

Storms whose track never reaches the land bounds, or whose exposure box is empty, are skipped with
a logged warning rather than raising. `cases_from_inputs(inputs, hazard_factory=TropicalCycloneHazard)`
then evaluates the wind footprint on the exposure locations and stores `intensity`, `tiv`,
`construction` (if present) and the storm's `observed_loss_ref_usd` on each `StormCase`; pass a
`functools.partial(TropicalCycloneHazard, decay_exponent=..., asymmetry_factor=...)` to evaluate a
different hazard without reloading anything. `StormCase.loss(vulnerability)` re-evaluates the
portfolio loss for any `VulnerabilityModel` in one vectorised pass, which is what makes an optimiser
with hundreds of evaluations run in seconds once the footprint is fixed.

Cache the prepared inputs so repeated calibration runs (including different hazard parameters) skip
the download/exposure step entirely:

```python
from pathlib import Path
from natcat.calibration import load_inputs, save_inputs

inputs_path = Path(".cache/calibration_inputs.pkl")
if inputs_path.exists():
    inputs = load_inputs(inputs_path)
else:
    inputs = prepare_inputs(observed)
    save_inputs(inputs, inputs_path)
```

`save_inputs`/`load_inputs` are a plain pickle round-trip; delete the file to force a rebuild (e.g.
after changing `margin_deg`, the exposure loader, or the observed-loss table). `build_cases(observed,
*, hazard_factory=TropicalCycloneHazard, inputs=None, **kwargs)` is a shorthand for
`cases_from_inputs(prepare_inputs(observed, **kwargs), hazard_factory)` when you don't need to reuse
the inputs across hazard parameters; `save_cases`/`load_cases` similarly pickle the finished
`StormCase`s if you only ever calibrate at one hazard setting.

## Fitting with `Calibrator`

```python
from natcat.calibration import Calibrator
from natcat.vulnerability import ValueDependentVulnerability

calibrator = Calibrator(cases, ValueDependentVulnerability())
result = calibrator.fit(method="global", maxiter=200, seed=0)
```

`Calibrator(cases, model, *, param_names=None, bounds=None)` optimises `model.params` &#8212; any
`VulnerabilityModel` that implements `params` (a flat `dict[str, float]`) and `with_params(**updates)`
works, not just `ValueDependentVulnerability`. `param_names` restricts the fit to a subset of
`model.params`, keeping the rest fixed at their starting value; `bounds` overrides the per-parameter
`(lo, hi)` range, which otherwise defaults to the model's `DEFAULT_BOUNDS` where available (both
`ValueDependentVulnerability` and `WindVulnerability` define one) or `(0.1x, 10x)` of the starting
value.

`fit(*, method="global", maxiter=200, seed=None, polish=True)`:

- `method="global"` runs `scipy.optimize.differential_evolution` within `bounds`, then (if
  `polish=True`) a bounded Nelder-Mead polish from the global result &#8212; the recommended default,
  since the objective is not convex in the logistic parameters.
- `method="local"` runs bounded Nelder-Mead directly from the model's current parameter values;
  useful for a quick re-fit after nudging one parameter by hand, or for fitting a subset of
  `param_names` around an already-reasonable starting point.

The optimiser never returns a result worse than the starting model: if it fails to improve on the
initial objective, `fit` keeps the starting parameters and logs a warning.

## `CalibrationResult`

| Attribute | Type | Description |
|---|---|---|
| `model` | `VulnerabilityModel` | Calibrated model |
| `initial_model` | `VulnerabilityModel` | Model the calibration started from |
| `params` / `initial_params` | `dict[str, float]` | Parameter values after / before |
| `objective` / `initial_objective` | `float` | Weighted mean squared log10 ratio after / before |
| `table` | `pd.DataFrame` | Per-storm `observed`, `modelled_initial`, `modelled` (USD) and `log_ratio_initial` / `log_ratio` |
| `n_evaluations` | `int` | Objective evaluations used by the optimiser |
| `method`, `message` | `str` | Optimiser used, optimiser status message |
| `rmse_log10` | `float` (property) | `sqrt(objective)`; typical factor error is `10 ** rmse_log10` |

```python
print(result.summary())
result.table
print(f"typical factor error: {10 ** result.rmse_log10:.2f}x")
```

`summary()` prints a multi-line before/after report of every calibrated parameter plus the
objective and its implied "typical factor error" (`10 ** sqrt(objective)`) &#8212; a $25 B storm
modelled at $50 B and a $1 B storm modelled at $2 B both contribute a factor of 2 to this number.

## Hazard parameter grid

The Rankine decay exponent and motion-asymmetry factor change the wind footprint itself, so they
sit outside `Calibrator` as an outer loop: `calibrate_hazard_grid` recomputes the footprint of every
input and refits the vulnerability at each grid point, then reports the combination with the lowest
calibrated objective (see [Calibration methodology](../methodology/calibration.md#hazard-parameter-grid)):

```python
from natcat.calibration import calibrate_hazard_grid
from natcat.vulnerability import ValueDependentVulnerability

grid = calibrate_hazard_grid(
    inputs, ValueDependentVulnerability(),
    decay_exponents=(0.5, 0.75, 1.0, 1.5, 2.0), asymmetry_factors=(0.3, 0.5, 0.7),
    maxiter=200, seed=0,
)
print(grid.summary())
result = grid.best                 # CalibrationResult at the winning hazard
hazard_factory = grid.hazard_factory()   # partial(TropicalCycloneHazard, **grid.best_hazard)
```

`HazardGridResult.table` holds one row per grid point (`decay_exponent`, `asymmetry_factor`,
`objective`, `factor_error`, calibrated parameters); `results` maps each `(exponent, asymmetry)`
pair to its full `CalibrationResult`; `best_hazard` is the winning hazard parameters as a `dict`,
and `hazard_factory()` turns them into a ready-to-use `functools.partial(TropicalCycloneHazard, ...)`.
Running the bundled 19-storm set gives `decay_exponent=0.5, asymmetry_factor=0.3` &#8212; now the
package defaults, so `TropicalCycloneHazard()` with no arguments already uses them.

## Plotting

```python
from natcat import plotting
from natcat.vulnerability import WindVulnerability

plotting.apply_style()
fig, ax = plotting.plot_calibration(result, title="Modelled vs observed storm loss")
fig, ax = plotting.plot_value_dependent_curves(
    result.model, reference=WindVulnerability("Frame"), title="Calibrated vulnerability by tile value"
)
fig, ax = plotting.plot_hazard_grid(grid, title="Calibrated error over hazard parameters")
```

`plot_calibration` draws modelled versus observed loss per storm on a log-log scatter, with the 1:1
line, a shaded `band_factor`-wide band around it, the pre-calibration points in grey
(`show_initial=True`) and per-storm name labels (`label_storms=True`).

![Modelled vs observed storm loss, before and after calibration](../assets/figures/calibration_scatter.png){ width="100%" }
*Figure: scatter of modelled against observed loss for the calibration set, on a log-log scale.*

`plot_value_dependent_curves` draws the damage-ratio curve of a `ValueDependentVulnerability` for
several tile values (`tivs`, default `1e5` to `1e9`), optionally against a dashed `reference` curve
such as the uncalibrated `WindVulnerability("Frame")`.

![Calibrated damage-ratio curves by tile value](../assets/figures/calibration_curves.png){ width="100%" }
*Figure: mean damage ratio vs. wind speed for several LitPop tile values, calibrated model in
colour, uncalibrated `WindVulnerability("Frame")` dashed.*

`plot_hazard_grid` draws the calibrated factor error as a heat map over the (decay exponent,
asymmetry factor) grid, with the winning cell circled.

![Calibrated error over the hazard-parameter grid](../assets/figures/calibration_hazard_grid.png){ width="100%" }
*Figure: typical factor error after calibration at every hazard grid point.*

## Using the calibrated model

`result.model` is a plain `VulnerabilityModel`, so it drops into `LossCalculator` and
`LossSimulator` exactly like `WindVulnerability`. If you don't need to re-run the calibration at
all, `ValueDependentVulnerability.calibrated()` gives you the same result directly, using the
parameters shipped in `CALIBRATED_PARAMS`:

```python
from natcat.exposure import load_litpop_exposure
from natcat.hazards import TropicalCycloneHazard
from natcat.loss import LossCalculator
from natcat.tracks import load_best_track
from natcat.vulnerability import ValueDependentVulnerability

track = load_best_track(2018, "al", "14")
portfolio = load_litpop_exposure("USA", bounds=(24.5, 32.0, -87.6, -80.0))

model = ValueDependentVulnerability.calibrated()   # or result.model from a fresh calibration
calc = LossCalculator(TropicalCycloneHazard(track), model)   # calibrated hazard defaults too
calc.compute(portfolio)
print(f"Calibrated ground-up loss: ${calc.total_loss:,.0f}")
```

```python
from natcat.loss import LossSimulator

simulator = LossSimulator(catalog, portfolio, model, seed=42)
results = simulator.run(n_years=1000)
```

Because `ValueDependentVulnerability.damage_ratio` reads the `tiv` keyword, `LossCalculator` and
`LossSimulator` (which always pass `tiv=` through) automatically vary the damage ratio by tile
value &#8212; no other change to the pipeline is needed. `TropicalCycloneHazard`'s own defaults
(`decay_exponent=0.5`, `asymmetry_factor=0.3`) already match the hazard `.calibrated()` was fit
against, so passing the track alone is enough; only override them if you re-ran the hazard grid
with a different result.

## Reference

| Signature | Returns |
|-----------|---------|
| `load_observed_losses(path=None, *, included_only=True)` | `pd.DataFrame` |
| `normalise_losses(df, *, reference_year=2018, method="gdp")` | `pd.DataFrame` (`method` also takes `"cpi"` or `"none"`) |
| `prepare_inputs(observed, *, exposure_loader=None, margin_deg=2.0, loss_column="observed_loss_ref_usd", data_dir=None, download=True, progress=True)` | `list[CaseInput]` |
| `cases_from_inputs(inputs, hazard_factory=TropicalCycloneHazard)` | `list[StormCase]` |
| `save_inputs(inputs, path)` / `load_inputs(path)` | `None` / `list[CaseInput]` (pickle round-trip) |
| `build_cases(observed, *, hazard_factory=TropicalCycloneHazard, inputs=None, **kwargs)` | `list[StormCase]` (shorthand for `prepare_inputs` + `cases_from_inputs`) |
| `save_cases(cases, path)` / `load_cases(path)` | `None` / `list[StormCase]` (pickle round-trip) |
| `storm_region(track, *, margin_deg=2.0, land_bounds=CONUS_BOUNDS)` | `(lat_min, lat_max, lon_min, lon_max)` |
| `Calibrator(cases, model, *, param_names=None, bounds=None)` | calibrator instance |
| `calibrator.fit(*, method="global", maxiter=200, seed=None, polish=True)` | `CalibrationResult` |
| `calibrator.modelled_losses(model=None)` | `NDArray`, per-case USD |
| `calibrator.table(model)` | `pd.DataFrame` |
| `calibrate_hazard_grid(inputs, model, *, decay_exponents=(0.5,0.75,1.0,1.5,2.0), asymmetry_factors=(0.3,0.5,0.7), param_names=None, bounds=None, method="global", maxiter=200, seed=None, progress=True)` | `HazardGridResult` |
| `hazard_grid_result.hazard_factory()` / `.summary()` | `partial(TropicalCycloneHazard, ...)` / `str` |
| `ValueDependentVulnerability(*, threshold_kt=40.0, v50_ref=105.0, v50_slope=0.0, k=0.12, scale=1.0, tiv_ref=1e7, v50_bounds=(50.0, 250.0))` | model instance |
| `ValueDependentVulnerability.calibrated(**overrides)` | model instance with `CALIBRATED_PARAMS` |
| `plotting.plot_calibration(result, *, ax=None, show_initial=True, label_storms=True, band_factor=2.0, title=None)` | `(fig, ax)` |
| `plotting.plot_value_dependent_curves(model, *, tivs=(1e5,...,1e9), reference=None, ax=None, title=None)` | `(fig, ax)` |
| `plotting.plot_hazard_grid(result, *, ax=None, title=None)` | `(fig, ax)` |

See the full [API reference](../api/calibration.md) for parameter and type details.
