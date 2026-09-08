# Concepts & workflow

NatCat follows the standard catastrophe-model decomposition: hazard &times; vulnerability &times;
exposure = loss, extended with a stochastic event set and financial (exceedance-probability)
metrics.

## The four components

| Component | Question it answers | NatCat class/function |
|-----------|----------------------|------------------------|
| Hazard | How intense is the peril at this location? | `natcat.hazards.TropicalCycloneHazard` |
| Vulnerability | Given intensity, what fraction of value is damaged? | `natcat.vulnerability.WindVulnerability` |
| Exposure | What value is at risk, and where? | `natcat.exposure.synthetic_portfolio`, `load_litpop_exposure` |
| Loss | Combine the three into ground-up loss | `natcat.loss.LossCalculator` |

A single historical or forecast track gives one **event**; NatCat's stochastic model
(`natcat.stochastic.SyntheticTCCatalog`) generates thousands of physically plausible synthetic
events to build an **event set**, and `natcat.loss.LossSimulator` runs the loss engine over every
event in every simulated year. `natcat.financial.ExceedanceProbability` turns that simulated loss
distribution into the risk metrics used to price and structure catastrophe risk.

## Two workflows

```text
Single historical/forecast track -> TropicalCycloneHazard -> LossCalculator
                                                                    |
                                                                    v
                                                     Ground-up loss per location

Historical track archive -> SyntheticTCCatalog -> LossSimulator -> SimulationResults
                                                                          |
                                                                          v
                                                              ExceedanceProbability
                                                                          |
                                                                          v
                                                    AEP / OEP / return period / AAL
```

1. **Single-event loss** &#8212; load one best track or forecast, compute a wind footprint, apply
   vulnerability, and get a loss per location (and, optionally, a full time-evolution of loss as
   the storm makes landfall). See [Single-event loss](single-event-loss.md).
2. **Portfolio risk metrics** &#8212; calibrate the stochastic track generator on the historical
   archive, simulate many years of synthetic storms against a portfolio, and summarize the
   resulting annual loss distribution with exceedance-probability curves. See
   [Stochastic event set](stochastic-event-set.md) and
   [Financial metrics](financial-metrics.md).

## Design principles

- **Peril-agnostic base classes.** `HazardModel` and `VulnerabilityModel` are abstract base
  classes; `TropicalCycloneHazard` and `WindVulnerability` are the TC implementations. The public
  API is shaped so that a second peril could be added without changing how `LossCalculator` is
  used.
- **Pure, vectorised functions.** Track processing and wind-field functions never mutate their
  input DataFrame and operate on full arrays rather than looping over rows &#8212; important once a
  simulation calls them for thousands of synthetic storms.
- **Deterministic when seeded.** Every stochastic component (`SyntheticTCCatalog`,
  `LossSimulator`) takes a `seed` and threads a single `numpy.random.Generator` through every
  sampling step, so a full run is reproducible end to end.
