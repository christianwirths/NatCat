# Exceedance probability

Financial metrics summarize a simulated loss distribution &#8212; from `LossSimulator` /
`SimulationResults` &#8212; into the exceedance-probability curves, return periods and average
annual loss (AAL) used across the catastrophe risk industry to price and structure risk.

## AEP vs. OEP

- **AEP (Annual Exceedance Probability)**: \(P(\text{annual total loss} > x)\), built from
  `annual_losses` &#8212; the sum of every event's loss within each simulated year.
- **OEP (Occurrence Exceedance Probability)**: \(P(\text{largest single event in a year} > x)\),
  built from `max_event_losses` &#8212; the single largest event loss within each simulated year
  (0 for years with no loss-causing event).

AEP is the natural metric for aggregate/whole-portfolio risk; OEP is the natural metric for
per-occurrence risk (e.g. sizing a single reinsurance layer).

## Empirical estimator

For a sample of \(N\) simulated years with losses \(\{L_1, \dots, L_N\}\) (either the annual series
for AEP or the per-year maxima for OEP), the empirical exceedance probability at threshold \(x\) is
the fraction of years exceeding it:

\[
\hat{P}(L > x) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[L_i > x]
\]

This is exact and unbiased where the sample is dense, but noisy and range-limited in the tail: with
\(N\) simulated years, the empirical estimator cannot resolve exceedance probabilities finer than
about \(1/N\) (e.g. a 1,000-year simulation cannot directly estimate a 1-in-2,000-year loss).

## Generalized Pareto tail

Above a high threshold \(u\) (the `tail_quantile`, default the 95th percentile of the sample),
NatCat fits a **generalized Pareto distribution (GPD)** to the excesses \(L_i - u\) for
\(L_i > u\) (`scipy.stats.genpareto`), and extrapolates the tail via the Pickands&#8211;Balkema&#8211;de
Haan theorem's central result: threshold exceedances of a well-behaved distribution converge to a
GPD as the threshold increases.

\[
P(L > x) = P(L > u) \cdot \left(1 + \xi \frac{x - u}{\sigma}\right)^{-1/\xi}, \qquad x > u
\]

where \(\xi\) (shape) and \(\sigma\) (scale) are the fitted GPD parameters, and \(P(L > u)\) is the
empirical exceedance probability at the threshold itself &#8212; so the tail model is anchored to
match the empirical curve exactly at \(u\) and smoothly extrapolates beyond it. Below \(u\), NatCat
uses the empirical estimator directly; `ExceedanceProbability.aep`/`.oep` dispatch between the two
regimes automatically and are fully vectorised over an array of thresholds.

![AEP and OEP curves, empirical below the tail threshold and GPD above it](../assets/figures/ep_curves.png){ width="100%" }
*Figure: exceedance probability vs. loss on a log-log scale; the kink at the 95th percentile marks
the empirical/GPD transition.*

## OEP is clipped to AEP

Because a year's largest single event can never exceed that year's total loss, \(\text{OEP}(x)
\le \text{AEP}(x)\) must hold at every loss level. The empirical curves satisfy this by
construction, but the AEP and OEP tails are fitted as independent GPDs and can cross far beyond
the sample. NatCat clips the OEP curve (and, equivalently, `loss_at_return_period(kind="oep")`) to
the AEP curve wherever this happens, and logs a warning when the clip binds.

## Return period

The return period at a given loss threshold is simply the reciprocal of its exceedance
probability:

\[
\text{RP}(x) = \frac{1}{P(L > x)}
\]

`return_period(loss, kind="aep"|"oep")` returns `np.inf` where the exceedance probability is zero.
The inverse operation, `loss_at_return_period(rp, kind=...)`, interpolates the fitted curve to
answer "what loss level corresponds to a 1-in-`rp`-year event?" &#8212; the standard way to quote a
reinsurance attachment or exhaustion point.

## Average annual loss (AAL)

\[
\text{AAL} = \mathbb{E}[\text{annual total loss}] = \frac{1}{N}\sum_{i=1}^{N} L_i
\]

the simple mean of `annual_losses` across all simulated years &#8212; the technical (pure) premium
for the modelled peril before loadings.

```python
from natcat.financial import ExceedanceProbability

ep = ExceedanceProbability.from_simulation(results, tail_quantile=0.95)
print(f"AAL: ${ep.aal:,.0f}")
print(f"1-in-100-year OEP loss: ${ep.loss_at_return_period(100, kind='oep'):,.0f}")
```

## References

- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values.* Springer Series in
  Statistics.
- Grossi, P., & Kunreuther, H. (Eds.). (2005). *Catastrophe Modeling: A New Approach to Managing
  Risk.* Springer.
