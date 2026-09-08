"""Exceedance probability curves, return periods and average annual loss.

Two curves are produced from a simulated loss table:

* **AEP** (aggregate exceedance probability) -- the probability that the
  *total* loss of a year exceeds a threshold;
* **OEP** (occurrence exceedance probability) -- the probability that the
  *largest single event* of a year exceeds a threshold.

Below the tail threshold (the ``tail_quantile`` of the sample) the empirical
distribution is used directly.  Above it a generalised Pareto distribution is
fitted to the excesses with the location pinned at zero, and scaled by the
empirical exceedance probability at the threshold, so the curve is continuous
where the two regimes meet.

Because a year's largest event can never exceed that year's total, ``OEP(x) <=
AEP(x)`` must hold at every loss level.  The empirical curves satisfy this by
construction, but the two tails are fitted independently and can cross far
beyond the sample.  The OEP curve is therefore clipped to the AEP curve (and,
equivalently, the OEP loss at a return period to the AEP loss at the same return
period); a warning is logged whenever the clip binds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:  # pragma: no cover
    from ..loss.simulation import SimulationResults

__all__ = ["ExceedanceProbability", "EPCurve"]

logger = logging.getLogger(__name__)

#: Fitted tail parameters: shape, location, scale, threshold, P(X > threshold).
_TailFit = tuple[float, float, float, float, float]

#: Relative slack allowed before an OEP-above-AEP crossing counts as real.
_CLIP_TOLERANCE: float = 1e-12


class ExceedanceProbability:
    """Empirical exceedance probability curves with a fitted GPD tail.

    Parameters
    ----------
    annual_losses : array_like
        Aggregate loss of each simulated year, shape ``(n_years,)``.
    max_event_losses : array_like, optional
        Largest single-event loss of each simulated year, shape
        ``(n_years,)``. Required for the OEP curve.
    tail_quantile : float, default 0.95
        Sample quantile above which the GPD tail replaces the empirical curve.

    Attributes
    ----------
    annual_losses : numpy.ndarray
        The AEP sample.
    max_event_losses : numpy.ndarray or None
        The OEP sample.

    Raises
    ------
    ValueError
        If ``annual_losses`` is empty, the two samples differ in length, or
        ``tail_quantile`` is outside ``(0, 1)``.

    Examples
    --------
    >>> ep = ExceedanceProbability([0.0, 1.0, 2.0, 3.0])
    >>> float(ep.aep(0.0))
    1.0
    >>> float(ep.aal)
    1.5
    """

    def __init__(
        self,
        annual_losses: ArrayLike,
        max_event_losses: ArrayLike | None = None,
        *,
        tail_quantile: float = 0.95,
    ) -> None:
        self.annual_losses = np.asarray(annual_losses, dtype=np.float64).ravel()
        if self.annual_losses.size == 0:
            raise ValueError("annual_losses must not be empty.")
        if not 0.0 < tail_quantile < 1.0:
            raise ValueError(f"tail_quantile must be in (0, 1), got {tail_quantile}")

        if max_event_losses is None:
            self.max_event_losses: NDArray[np.float64] | None = None
        else:
            self.max_event_losses = np.asarray(max_event_losses, dtype=np.float64).ravel()
            if self.max_event_losses.size != self.annual_losses.size:
                raise ValueError(
                    "annual_losses and max_event_losses must have the same length "
                    f"({self.annual_losses.size} != {self.max_event_losses.size})"
                )

        self.tail_quantile = float(tail_quantile)
        self._sorted: dict[str, NDArray[np.float64]] = {}
        self._fits: dict[str, _TailFit | None] = {}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_simulation(cls, results: SimulationResults, **kwargs) -> ExceedanceProbability:
        """Build a curve from a :class:`~natcat.loss.SimulationResults`.

        Parameters
        ----------
        results : natcat.loss.SimulationResults
            Output of :meth:`natcat.loss.LossSimulator.run`.
        **kwargs
            Forwarded to the constructor, e.g. ``tail_quantile``.

        Returns
        -------
        ExceedanceProbability
            Curve over the simulated annual and per-event losses.
        """
        return cls(results.annual_losses, results.max_event_losses, **kwargs)

    # -- internals ---------------------------------------------------------
    def _sample(self, kind: str) -> NDArray[np.float64]:
        """Return the sorted loss sample backing ``kind``."""
        kind = kind.lower()
        if kind not in ("aep", "oep"):
            raise ValueError(f"kind must be 'aep' or 'oep', got {kind!r}")
        if kind not in self._sorted:
            if kind == "aep":
                values = self.annual_losses
            else:
                if self.max_event_losses is None:
                    raise ValueError("OEP requires max_event_losses; none were provided.")
                values = self.max_event_losses
            self._sorted[kind] = np.sort(values)
        return self._sorted[kind]

    @staticmethod
    def _empirical(sorted_values: NDArray[np.float64], loss: NDArray[np.float64]):
        """Empirical ``P(X >= loss)`` for a sorted sample, vectorised."""
        n = sorted_values.size
        exceedances = n - np.searchsorted(sorted_values, loss, side="left")
        return exceedances / n

    def _fit(self, kind: str) -> _TailFit | None:
        """Fit (and cache) the GPD tail for ``kind``; ``None`` if not possible."""
        if kind in self._fits:
            return self._fits[kind]

        from scipy.stats import genpareto

        values = self._sample(kind)
        threshold = float(np.quantile(values, self.tail_quantile))
        excesses = values[values > threshold] - threshold

        fit: _TailFit | None
        if excesses.size < 2:
            logger.warning(
                "Only %d excess(es) above the %.0f%% threshold; %s tail stays empirical.",
                excesses.size,
                100 * self.tail_quantile,
                kind.upper(),
            )
            fit = None
        else:
            # floc=0 pins the GPD at the threshold, keeping the curve continuous.
            shape, loc, scale = genpareto.fit(excesses, floc=0.0)
            p_threshold = float(self._empirical(values, np.asarray(threshold)))
            fit = (float(shape), float(loc), float(scale), threshold, p_threshold)

        self._fits[kind] = fit
        return fit

    def _raw_exceedance(self, loss: ArrayLike, kind: str) -> NDArray[np.float64]:
        """Blended empirical / GPD exceedance probability, vectorised."""
        values = self._sample(kind)
        query = np.atleast_1d(np.asarray(loss, dtype=np.float64))
        out = self._empirical(values, query).astype(np.float64)

        fit = self._fit(kind)
        if fit is not None:
            from scipy.stats import genpareto

            shape, loc, scale, threshold, p_threshold = fit
            tail = query > threshold
            if tail.any():
                out[tail] = p_threshold * genpareto.sf(
                    query[tail] - threshold, shape, loc=loc, scale=scale
                )
        return np.clip(out, 0.0, 1.0)

    def _exceedance(self, loss: ArrayLike, kind: str) -> NDArray[np.float64]:
        """Exceedance probability with the OEP curve capped by the AEP curve."""
        out = self._raw_exceedance(loss, kind)
        if kind.lower() != "oep":
            return out

        ceiling = self._raw_exceedance(loss, "aep")
        crossing = out > ceiling + _CLIP_TOLERANCE
        if crossing.any():
            logger.warning(
                "OEP exceeded AEP at %d of %d loss level(s) (max excess %.3g); "
                "clipping OEP to AEP. This is an artefact of fitting the two GPD "
                "tails independently far beyond the %d-year sample.",
                int(crossing.sum()),
                out.size,
                float(np.max(out[crossing] - ceiling[crossing])),
                self.n_years,
            )
            out = np.minimum(out, ceiling)
        return out

    # -- public API --------------------------------------------------------
    def aep(self, loss: ArrayLike) -> NDArray[np.float64]:
        """Aggregate exceedance probability at one or more loss levels.

        Parameters
        ----------
        loss : array_like
            Loss threshold(s).

        Returns
        -------
        numpy.ndarray
            Probability that a year's total loss exceeds each threshold.

        Examples
        --------
        >>> ExceedanceProbability([1.0, 2.0, 3.0, 4.0]).aep([0.0, 3.0]).tolist()
        [1.0, 0.5]
        """
        return self._exceedance(loss, "aep")

    def oep(self, loss: ArrayLike) -> NDArray[np.float64]:
        """Occurrence exceedance probability at one or more loss levels.

        The result is capped at :meth:`aep`, since a year's largest event can
        never exceed that year's total loss. A warning is logged whenever the
        independently fitted tails cross and the cap binds.

        Parameters
        ----------
        loss : array_like
            Loss threshold(s).

        Returns
        -------
        numpy.ndarray
            Probability that a year's largest single event exceeds each
            threshold, never above :meth:`aep` at the same level.

        Raises
        ------
        ValueError
            If the curve was built without ``max_event_losses``.
        """
        return self._exceedance(loss, "oep")

    def aep_empirical(self, loss: ArrayLike) -> NDArray[np.float64]:
        """Purely empirical AEP, with no GPD tail extrapolation.

        Parameters
        ----------
        loss : array_like
            Loss threshold(s).

        Returns
        -------
        numpy.ndarray
            Empirical exceedance frequency.
        """
        query = np.atleast_1d(np.asarray(loss, dtype=np.float64))
        return self._empirical(self._sample("aep"), query).astype(np.float64)

    def oep_empirical(self, loss: ArrayLike) -> NDArray[np.float64]:
        """Purely empirical OEP, with no GPD tail extrapolation.

        Parameters
        ----------
        loss : array_like
            Loss threshold(s).

        Returns
        -------
        numpy.ndarray
            Empirical exceedance frequency.
        """
        query = np.atleast_1d(np.asarray(loss, dtype=np.float64))
        return self._empirical(self._sample("oep"), query).astype(np.float64)

    def return_period(self, loss: ArrayLike, kind: str = "aep") -> NDArray[np.float64]:
        """Return period corresponding to one or more loss levels.

        Parameters
        ----------
        loss : array_like
            Loss threshold(s).
        kind : {'aep', 'oep'}, default 'aep'
            Which curve to invert.

        Returns
        -------
        numpy.ndarray
            Return period in years; ``inf`` where the exceedance probability
            is zero.

        Examples
        --------
        >>> ExceedanceProbability([1.0, 2.0, 3.0, 4.0]).return_period(3.0).tolist()
        [2.0]
        """
        probability = self._exceedance(loss, kind)
        with np.errstate(divide="ignore"):
            return np.where(probability > 0, 1.0 / probability, np.inf)

    def loss_at_return_period(self, rp: ArrayLike, kind: str = "aep") -> NDArray[np.float64]:
        """Loss level associated with one or more return periods.

        Below the tail threshold the empirical quantile (linearly interpolated)
        is used; above it the fitted GPD is inverted analytically.  The OEP loss
        is capped by the AEP loss at the same return period, mirroring the cap
        applied to the probabilities themselves.

        Parameters
        ----------
        rp : array_like
            Return period(s) in years; must be ``>= 1``.
        kind : {'aep', 'oep'}, default 'aep'
            Which curve to invert.

        Returns
        -------
        numpy.ndarray
            Loss level for each return period.

        Raises
        ------
        ValueError
            If any return period is smaller than 1.
        """
        out = self._raw_loss_at_return_period(rp, kind)
        if kind.lower() != "oep":
            return out

        ceiling = self._raw_loss_at_return_period(rp, "aep")
        slack = _CLIP_TOLERANCE * np.maximum(np.abs(ceiling), 1.0)
        crossing = out > ceiling + slack
        if crossing.any():
            logger.warning(
                "OEP loss exceeded AEP loss at %d of %d return period(s); clipping "
                "OEP to AEP. This is an artefact of fitting the two GPD tails "
                "independently far beyond the %d-year sample.",
                int(crossing.sum()),
                out.size,
                self.n_years,
            )
            out = np.minimum(out, ceiling)
        return out

    def _raw_loss_at_return_period(self, rp: ArrayLike, kind: str) -> NDArray[np.float64]:
        """Invert one curve without the OEP-below-AEP cap."""
        periods = np.atleast_1d(np.asarray(rp, dtype=np.float64))
        if np.any(periods < 1.0):
            raise ValueError("Return periods must be >= 1 year.")

        values = self._sample(kind)
        probability = 1.0 / periods
        out = np.quantile(values, np.clip(1.0 - probability, 0.0, 1.0))

        fit = self._fit(kind)
        if fit is not None:
            from scipy.stats import genpareto

            shape, loc, scale, threshold, p_threshold = fit
            tail = probability < p_threshold
            if tail.any() and p_threshold > 0:
                out[tail] = threshold + genpareto.ppf(
                    1.0 - probability[tail] / p_threshold, shape, loc=loc, scale=scale
                )
        return out

    def curve(self, kind: str = "aep", n_points: int = 200) -> pd.DataFrame:
        """Tabulate the exceedance probability curve for plotting.

        Parameters
        ----------
        kind : {'aep', 'oep'}, default 'aep'
            Which curve to tabulate.
        n_points : int, default 200
            Number of points, spaced logarithmically in return period between
            about 1.05 and 1000 years.

        Returns
        -------
        pandas.DataFrame
            Columns ``loss``, ``probability`` and ``return_period``, sorted by
            increasing loss.

        Raises
        ------
        ValueError
            If ``n_points`` is smaller than 2.
        """
        if n_points < 2:
            raise ValueError(f"n_points must be >= 2, got {n_points}")

        periods = np.logspace(np.log10(1.05), 3.0, n_points)
        losses = self.loss_at_return_period(periods, kind=kind)
        probability = self._exceedance(losses, kind)
        with np.errstate(divide="ignore"):
            realised = np.where(probability > 0, 1.0 / probability, np.inf)

        return pd.DataFrame(
            {"loss": losses, "probability": probability, "return_period": realised}
        ).sort_values("loss", kind="stable", ignore_index=True)

    def tail_fit(self, kind: str = "aep") -> tuple[float, float, float, float]:
        """Return the fitted GPD tail parameters.

        Parameters
        ----------
        kind : {'aep', 'oep'}, default 'aep'
            Which curve's tail to report.

        Returns
        -------
        tuple of float
            ``(shape, loc, scale, threshold)``.

        Raises
        ------
        RuntimeError
            If too few excesses were available to fit a tail.
        """
        fit = self._fit(kind)
        if fit is None:
            raise RuntimeError(
                f"No GPD tail could be fitted for {kind.upper()}: too few excesses "
                f"above the {100 * self.tail_quantile:.0f}% threshold."
            )
        shape, loc, scale, threshold, _ = fit
        return shape, loc, scale, threshold

    @property
    def aal(self) -> float:
        """Average annual loss over the simulated years."""
        return float(np.mean(self.annual_losses))

    @property
    def n_years(self) -> int:
        """Number of simulated years behind the curve."""
        return int(self.annual_losses.size)

    def __repr__(self) -> str:
        return (
            f"ExceedanceProbability(n_years={self.n_years}, "
            f"tail_quantile={self.tail_quantile}, aal={self.aal:,.0f})"
        )


#: Backwards-compatible alias.
EPCurve = ExceedanceProbability
