# Limitations & validation

A candid list of what NatCat does not model, so results are interpreted with the right caveats.
This is a personal/educational cat-modelling framework, not a vendor or regulatory model, and has
not been calibrated or validated against observed claims data.

## Hazard

- **Wind only.** No storm surge, no rainfall/inland flooding, no tornado spawning &#8212; all of
  which are major contributors to total tropical cyclone loss (surge in particular, for coastal
  exposure).
- **Simplistic asymmetry, not a full boundary-layer model.** The motion-asymmetry correction is a
  first-order sinusoidal term, not a physically resolved boundary-layer or friction model; the
  wind field is close to axisymmetric plus a single correction rather than the more complex,
  often asymmetric structure of real storms (eyewall replacement cycles, convective bursts, wind
  shear-induced tilt).
- **Hazard defaults rest on 19 storms.** `DEFAULT_DECAY_EXPONENT = 0.5` and
  `DEFAULT_ASYMMETRY_FACTOR = 0.3` were chosen by the hazard-parameter grid in
  [Calibration](calibration.md), which minimises the calibrated total-economic-damage error against
  NHC Tropical Cyclone Report losses for 19 US landfalls (1989&#8211;2020, wind-dominated storms
  only). Nineteen storms, one basin, one exposure model (LitPop) and one loss basis (total economic
  damage, not insured loss) is a narrow evidence base for a global default; the grid itself is also
  fairly flat (1.77x&#8211;2.08x typical factor error across it), so the "best" point is not sharply
  identified.
- **RMW relation is a climatological fit, not storm-specific structure.** `willoughby_rmw` gives the
  same radius to every storm of a given intensity and latitude; it carries no information about a
  particular storm's eyewall replacement cycles, environmental shear, or measured RMW trend, and
  systematically misses storms that are unusually compact or broad for their strength.
- **Footprints are sampled in time, not integrated.** The maximum wind at a location is the
  maximum over the discrete, interpolated track positions. The default 5-minute step keeps the
  sampling error small for any realistic storm, but a coarser `freq` (as used for quick runs)
  under-samples compact, fast-moving storms and biases losses low &#8212; by about a quarter of the
  AAL at 1 hour (see [Wind field](wind-field.md#temporal-sampling)).
- **Post-hurricane decay is cut off.** Tracks are truncated after the last hurricane-strength fix
  (`truncate_after_hurricane`), because the Rankine profile cannot represent the broad, weak
  circulation of a decaying storm (radius of maximum wind of 100 nm and more) and would spread
  trace damage far from the track. The price is that inland tropical-storm-force wind damage
  after the downgrade is not modelled, while storms that never reach hurricane strength are
  modelled in full &#8212; a deliberate asymmetry.
- **No gust factors.** All wind speeds are 1-minute sustained wind; no conversion to 3-second gust
  is applied, which some vulnerability curves in the literature expect as their input basis.

## Vulnerability

- **Two construction classes, uncalibrated.** `WindVulnerability` ships two illustrative
  parameterizations (Frame, Masonry); real vulnerability curves used in industry models are
  calibrated against claims data, engineering testing, or detailed structural modelling, none of
  which underlies these curves.
- **No building-code era, height, roof shape, or opening protection.** All of these materially
  change real-world wind vulnerability and are not represented.
- **No demand surge.** Repair costs typically increase after a major landfall due to labor/material
  scarcity; NatCat's damage ratio is a fixed function of intensity regardless of event size or
  concurrent regional demand.
- **No secondary uncertainty.** The damage ratio is a deterministic function of intensity (a mean
  damage ratio curve); no distribution around that mean is sampled per asset, which understates
  loss variance for a given portfolio and event.

## Stochastic event set

- **Atlantic basin only**, calibrated on ATCF best-track data from 1900&#8211;2020. Genesis rates,
  transition statistics and climatological structure are all specific to this record and this
  basin; no basin transfer or climate-change adjustment is applied.
- **No explicit seasonal or interannual structure.** Genesis and transition sampling do not
  condition on day-of-year, ENSO phase, or any other climate driver &#8212; every synthetic year is
  drawn from the same stationary empirical distribution.
- **Markov, not physically dynamical.** Track propagation depends only on the storm's current grid
  cell, not on its history, environment (SST, wind shear, steering flow) or the physical state of
  the atmosphere &#8212; a deliberate simplification, in contrast to more physically forced
  statistical-deterministic approaches (e.g. Emanuel et al. 2006).
- **No storm-count or intensity trend.** The Poisson frequency model uses the full historical mean
  rate; it does not represent any long-term trend in frequency or intensity (e.g. due to
  sea-surface temperature warming).
- **Inland decay is a single fitted curve, not terrain- or size-aware.** `LandDecayModel` fits one
  exponential decay-to-background-wind relation (rate 0.069/h, background wind 25.3 kt) to every
  historical landfall pooled together; the residual RMSE is 8.7 kt, but the model carries no
  dependence on terrain, storm size, translation speed, or track angle at the coast &#8212; factors
  known to modulate real inland decay rates.

- **Radius of maximum wind is an unconstrained random walk.** Most historical fixes carry no
  observed RMW, so the transition deltas are dominated by heuristic class-to-class jumps. Synthetic
  storms therefore drift to large radii (the catalog median is about 60 nm against 15&#8211;40 nm
  for real hurricanes) and combine them with high intensity far more often than observed. The
  `max_wind_kt`/`max_rmw_nm` caps bound the worst cases but do not remove the bias, which makes the
  extreme tail of the loss distribution too heavy. An intensity-conditioned RMW relation is the
  first item on the roadmap.

## Exposure

- **LitPop is a proxy, not an actual insured-value dataset.** LitPop disaggregates asset value from
  nightlight intensity and population, which correlates with but does not equal actual insured
  property values, occupancy mix, or policy terms/limits.
- **Synthetic portfolios are illustrative.** `synthetic_portfolio` scatters uniform-random TIV
  across a bounding box; it is useful for demonstrating the pipeline, not for representing any
  real book of business.

## Financial metrics

- **No correlation across perils or lines.** The event set and loss simulation model tropical
  cyclone in isolation; no correlation with other perils, other regions, or a broader
  multi-peril/multi-line portfolio is represented.
- **Tail extrapolation depends on GPD fit stability.** The generalized Pareto tail fit above the
  95th percentile is sensitive to the number of simulated years and the threshold choice; very
  extreme return periods (e.g. 1-in-10,000-year) extrapolate well beyond the empirical range and
  should be treated as indicative, not precise.
- **No parameter uncertainty propagation.** Point estimates (fitted GPD parameters, Poisson
  \(\lambda\), KDE bandwidth) are used as given; no ensemble or Bayesian treatment of calibration
  uncertainty feeds into the reported metrics.

## What this means in practice

NatCat is well suited to illustrating the *shape* of a catastrophe-modelling pipeline &#8212;
hazard, vulnerability, exposure, event set, financial metrics &#8212; and to experimenting with each
component's methodology. It is **not** suited to production pricing, regulatory capital, or
real-world risk transfer decisions without substantially more validation, calibration against
observed losses, and the additional perils/uncertainty sources listed above.

## References

- Holland, G. J. (1980). *An Analytic Model of the Wind and Pressure Profiles in Hurricanes.*
  Monthly Weather Review, 108(8), 1212&#8211;1218.
- Emanuel, K., Ravela, S., Vivant, E., & Risi, C. (2006). *A Statistical Deterministic Approach to
  Hurricane Risk Assessment.* Bulletin of the American Meteorological Society, 87(3), 299&#8211;314.
- Vickery, P. J., Skerlj, P. F., & Twisdale, L. A. (2000). *Simulation of Hurricane Risk in the
  U.S. Using Empirical Track Model.* Journal of Structural Engineering, 126(10), 1222&#8211;1237.
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values.* Springer Series in
  Statistics.
- Grossi, P., & Kunreuther, H. (Eds.). (2005). *Catastrophe Modeling: A New Approach to Managing
  Risk.* Springer.
- National Hurricane Center. *Automated Tropical Cyclone Forecasting (ATCF) System Documentation.*
  <https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt>
- Eberenz, S., Stocker, D., Röösli, T., & Bresch, D. N. (2020). *Asset exposure data for global
  physical risk assessment.* Earth System Science Data, 12(2), 817&#8211;833 (LitPop).
