# Wind field

NatCat's hazard model is a parametric Rankine vortex with a storm-motion asymmetry correction,
evaluated over the full track to produce a maximum-wind footprint.

## Rankine vortex

The classical Rankine vortex (Rankine, in the fluid-dynamics literature; applied to tropical
cyclones since at least the mid-20th century) models the tangential wind speed \(v\) at radius
\(r\) from the storm center as solid-body rotation inside the radius of maximum wind (RMW,
\(R\)) and a power-law decay outside it:

\[
v(r) =
\begin{cases}
v_{\max} \cdot \dfrac{r}{R}, & r < R \\[4pt]
v_{\max} \cdot \left(\dfrac{R}{r}\right)^{n}, & r \ge R
\end{cases}
\]

where \(v_{\max}\) is the maximum sustained wind speed (kt) and \(n\) is the **decay exponent**.

| Exponent \(n\) | Interpretation |
|----------------|-----------------|
| \(n = 1\) | Classical Rankine vortex |
| \(n \approx 0.4\text{&#8211;}0.6\) | "Modified Rankine" &#8212; commonly fitted to observed TC wind profiles (e.g. Holland 1980 and subsequent parametric-profile literature) |
| \(n = 2\) (**NatCat default**) | Steeper-than-physical decay; kept as the package default for reproducibility with prior results |

```python
from natcat.hazards.wind_field import rankine_vortex

v = rankine_vortex(r, vmax=100.0, rmw=25.0, exponent=2.0)
```

!!! warning "Exponent 2 is not the physically standard Rankine profile"
    `exponent=2.0` decays much faster with distance than a real TC wind field (see Holland 1980 for
    a more physically grounded profile). It is kept as NatCat's default purely for continuity with
    earlier results in this project; pass `exponent=1.0` for the classical Rankine profile or
    `exponent=0.5` for a typical "modified Rankine" fit. `exponent` is exposed on
    `rankine_vortex`, `max_wind_footprint`, and as `TropicalCycloneHazard(..., decay_exponent=...)`.

![Radial wind profile for varying decay exponents](../assets/figures/wind_profile.png){ width="100%" }
*Figure: tangential wind speed vs. radius for `exponent` &isin; {0.5, 1.0, 2.0}, same \(v_{\max}\), \(R\).*

## Motion asymmetry

A moving storm is not azimuthally symmetric: the side of the storm where the vortex rotation and
the storm's translation add constructively ("right of track" in the Northern Hemisphere, for a
cyclonic/counterclockwise vortex) is stronger than the side where they oppose. NatCat adds a
first-order correction:

\[
v_{\text{total}} = v_{\text{sym}} + f \cdot v_{t} \cdot \sin(\Delta\theta)
\]

\[
\Delta\theta = \big[(\theta_{\text{point}} - \theta_{\text{heading}} + 180) \bmod 360\big] - 180
\]

where \(v_{\text{sym}}\) is the symmetric Rankine wind speed, \(v_t\) is the translation speed,
\(\theta_{\text{point}}\) is the bearing from the storm center to the location of interest,
\(\theta_{\text{heading}}\) is the storm's heading, and \(f\) is the **asymmetry factor**
(default 0.5). \(\Delta\theta\) is wrapped to \((-180^\circ, 180^\circ]\) so that positive values
are to the right of the track (positive contribution) and negative values to the left.

```python
from natcat.hazards.wind_field import motion_asymmetry

v_total = v_sym + motion_asymmetry(bearing_to_point, heading, translation_speed, factor=0.5)
```

## Radius of maximum wind (RMW) heuristic

When RMW is missing or non-positive in the raw ATCF record, `fill_missing_rmw` substitutes a
heuristic value based on the intensity class, with a widening factor for extratropical storms:

| \(v_{\max}\) (kt) | RMW (nm) |
|----|----|
| < 35 | 80 |
| < 64 | 60 |
| < 96 | 40 |
| < 137 | 25 |
| &ge; 137 | 15 |

If `storm_type == "EX"` (extratropical), the resulting RMW is multiplied by 1.5 &#8212;
extratropical transition broadens the wind field even as peak winds weaken.

![RMW heuristic by intensity class](../assets/figures/rmw_heuristic.png){ width="100%" }
*Figure: heuristic RMW (nm) as a step function of maximum sustained wind speed (kt).*

## Footprint computation

`max_wind_footprint(track, coords, *, vortex="rankine", asymmetry_factor=0.5, chunk_size=200_000)`
evaluates the wind field at every combination of track point and target coordinate and reduces to
the running maximum per location:

\[
\text{footprint}_j = \max_{i \in \text{track}} v_{\text{total}}(i, j)
\]

This is computed as a single vectorised `(T, N)` broadcast in `chunk_size`-bounded blocks (never a
Python loop over track points), and `max_wind_history` extends this to a full `(T, N)` array of
cumulative running maxima via `numpy.maximum.accumulate` &#8212; one pass over the track regardless
of how many timestamps are later sampled from it. `TropicalCycloneHazard.compute_intensity_history`
uses exactly this to support time-evolving loss without recomputing the footprint at every step.

## References

- Rankine vortex &#8212; classical fluid-dynamics idealization, applied to tropical cyclone wind
  fields throughout the parametric-model literature.
- Holland, G. J. (1980). *An Analytic Model of the Wind and Pressure Profiles in Hurricanes.*
  Monthly Weather Review, 108(8), 1212&#8211;1218.
