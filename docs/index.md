# NatCat

**NatCat** is a Python framework for tropical-cyclone catastrophe modelling: it turns a storm
track into ground-up loss, and a historical track archive into a stochastic event set with
exceedance-probability curves and average annual loss (AAL).

<div class="grid cards" markdown>

<div markdown>

**Hazard**

Vectorised parametric wind field (Rankine vortex + motion asymmetry) computes maximum sustained
wind speed at any set of coordinates from a processed storm track.

</div>

<div markdown>

**Vulnerability**

Sigmoid wind-damage curves convert intensity into a mean damage ratio, parameterised by
construction type.

</div>

<div markdown>

**Stochastic event set**

An empirical Markov track model, calibrated on 1900&#8211;2020 Atlantic best-track data, generates
seeded, reproducible synthetic storm catalogs for thousands of simulated years.

</div>

<div markdown>

**Financial metrics**

Annual and occurrence exceedance probability (AEP/OEP) curves with a generalized Pareto tail,
return periods, and AAL &#8212; the standard outputs of a cat model.

</div>

</div>

## Modelling chain

![NatCat pipeline overview](assets/figures/pipeline_overview.svg){ width="100%" }
*Figure: from ATCF best track through hazard, vulnerability and exposure to portfolio loss and
financial metrics; the stochastic path calibrates on historical tracks to generate an event set.*

## Minimal example

```python
from natcat.tracks import load_best_track
from natcat.hazards import TropicalCycloneHazard
from natcat.vulnerability import WindVulnerability
from natcat.exposure import synthetic_portfolio
from natcat.loss import LossCalculator
from natcat import plotting

# Hurricane Michael (2018), Atlantic basin, NHC storm number 14
track = load_best_track(2018, "al", "14")

hazard = TropicalCycloneHazard(track)
vulnerability = WindVulnerability("Masonry")
# Or use the observed-loss-calibrated model: ValueDependentVulnerability.calibrated()
portfolio = synthetic_portfolio(n=500, bounds=(28.0, 31.5, -87.5, -83.5), seed=42)

result = LossCalculator(hazard, vulnerability).compute(portfolio)
print(f"Ground-up loss: ${result['loss'].sum():,.0f}")

plotting.plot_footprint(
    result, extent=(-87.5, -83.5, 28.0, 31.5), portfolio=portfolio, track=track
)
```

## Where to go next

- **[Getting started](getting-started/installation.md)** &#8212; install the package and run the
  quickstart.
- **[User guide](user-guide/concepts.md)** &#8212; the data model and the four core workflows
  (single-event loss, stochastic event set, financial metrics, plotting).
- **[Methodology](methodology/wind-field.md)** &#8212; the physics and statistics behind the
  hazard, vulnerability, stochastic track generator and exceedance-probability engine, with
  references and a candid limitations page.
- **[Examples](notebooks/01_single_event_loss.ipynb)** &#8212; runnable end-to-end notebooks.
- **[API reference](api/data.md)** &#8212; full signatures for every public class and function.

## Scope

Peril scope is tropical cyclone (TC) only, Atlantic basin. Base classes (`HazardModel`,
`VulnerabilityModel`) are peril-agnostic, so the framework can be extended to other perils without
touching the public API shape.
