"""Multi-year loss simulation over a synthetic event set."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..hazards.tropical_cyclone import TropicalCycloneHazard
from ..stochastic.catalog import SyntheticTCCatalog
from ..tracks.processing import (
    DEFAULT_TRACK_FREQ,
    add_heading,
    add_translation_velocity,
    interpolate_track,
    truncate_after_hurricane,
)
from ..vulnerability.base import VulnerabilityModel
from .calculator import LossCalculator

__all__ = ["SimulationResults", "LossSimulator"]

logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    """Per-year and per-event losses from a :class:`LossSimulator` run.

    Attributes
    ----------
    annual_losses : numpy.ndarray
        Shape ``(n_years,)`` aggregate loss for each simulated year.
    max_event_losses : numpy.ndarray
        Shape ``(n_years,)`` largest single-event loss per year; ``0`` for
        years with no landfalling storm.
    events : pandas.DataFrame
        One row per contributing event, columns ``year``, ``storm_id``,
        ``loss``.
    tracks : dict
        Maps ``(year, storm_id)`` to the processed track of every storm that
        reached the portfolio.
    n_years : int
        Number of simulated years.
    """

    annual_losses: np.ndarray
    max_event_losses: np.ndarray
    events: pd.DataFrame
    tracks: dict[tuple[int, str], pd.DataFrame] = field(default_factory=dict)
    n_years: int = 0

    @property
    def aal(self) -> float:
        """Average annual loss over the simulated years."""
        if self.n_years == 0:
            return 0.0
        return float(np.mean(self.annual_losses))

    def to_frame(self) -> pd.DataFrame:
        """Return the per-year losses as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns ``year``, ``annual_loss``, ``max_event_loss``.
        """
        return pd.DataFrame(
            {
                "year": np.arange(self.n_years),
                "annual_loss": self.annual_losses,
                "max_event_loss": self.max_event_losses,
            }
        )

    def __repr__(self) -> str:
        return (
            f"SimulationResults(n_years={self.n_years}, n_events={len(self.events)}, "
            f"aal={self.aal:,.0f})"
        )


class LossSimulator:
    """Run a synthetic catalog through the loss pipeline, year by year.

    Storms whose track never enters the portfolio bounding box (expanded by
    ``buffer_deg``) are skipped before any wind field is evaluated.

    Parameters
    ----------
    catalog : natcat.stochastic.SyntheticTCCatalog
        Fitted stochastic catalog.
    portfolio : pandas.DataFrame
        Exposure with ``latitude``, ``longitude`` and ``tiv``. Never mutated.
    vulnerability : natcat.vulnerability.VulnerabilityModel
        Damage function.
    buffer_deg : float, default 5.0
        Degrees of padding around the portfolio bounding box.
    freq : str, default DEFAULT_TRACK_FREQ ('5min')
        Time step the synthetic tracks are interpolated to before the wind
        field is evaluated. The footprint is a maximum over discrete track
        positions, so a coarse step under-samples compact storms and biases
        losses low: on the Florida LitPop portfolio a 1-hour step gives about
        25 % less AAL than 5 minutes (and a thinner tail) for roughly one
        eighth of the run time. Use ``'1h'`` for quick exploration only.
    truncate_after_hurricane : bool, default True
        Cut each synthetic track after its last fix at or above hurricane
        strength (``max_wind_speed_kt >= 64``) before evaluating the wind
        field, mirroring :func:`natcat.tracks.prepare_track`. Storms that
        never reach hurricane strength are kept in full.
    seed : int, optional
        If given, re-seeds the catalog's generator so the run is reproducible.

    Attributes
    ----------
    results : SimulationResults or None
        Output of the most recent :meth:`run`.

    Examples
    --------
    >>> simulator = LossSimulator(catalog, portfolio, vulnerability)  # doctest: +SKIP
    >>> results = simulator.run(1000)                                 # doctest: +SKIP
    >>> results.aal                                                   # doctest: +SKIP
    """

    def __init__(
        self,
        catalog: SyntheticTCCatalog,
        portfolio: pd.DataFrame,
        vulnerability: VulnerabilityModel,
        *,
        buffer_deg: float = 5.0,
        freq: str = DEFAULT_TRACK_FREQ,
        truncate_after_hurricane: bool = True,
        seed: int | None = None,
    ) -> None:
        self.catalog = catalog
        self.portfolio = portfolio
        self.vulnerability = vulnerability
        self.buffer_deg = float(buffer_deg)
        self.freq = freq
        self.truncate_after_hurricane = truncate_after_hurricane
        if seed is not None:
            self.catalog.rng = np.random.default_rng(seed)

        self.lat_min = float(portfolio["latitude"].min()) - self.buffer_deg
        self.lat_max = float(portfolio["latitude"].max()) + self.buffer_deg
        self.lon_min = float(portfolio["longitude"].min()) - self.buffer_deg
        self.lon_max = float(portfolio["longitude"].max()) + self.buffer_deg

        self.results: SimulationResults | None = None

    def _reaches_portfolio(self, track: pd.DataFrame) -> bool:
        """Whether any track point falls inside the buffered bounding box."""
        return bool(
            (
                track["latitude"].between(self.lat_min, self.lat_max)
                & track["longitude"].between(self.lon_min, self.lon_max)
            ).any()
        )

    def _process(self, track: pd.DataFrame) -> pd.DataFrame:
        """Truncate, interpolate a synthetic track and recompute its kinematics."""
        out = truncate_after_hurricane(track) if self.truncate_after_hurricane else track
        out = interpolate_track(out, freq=self.freq)
        out = add_translation_velocity(out)
        return add_heading(out)

    def run(self, n_years: int, *, progress: bool = True) -> SimulationResults:
        """Simulate ``n_years`` synthetic seasons and accumulate losses.

        Parameters
        ----------
        n_years : int
            Number of years to simulate.
        progress : bool, default True
            Show a tqdm progress bar over storms.

        Returns
        -------
        SimulationResults
            Per-year and per-event losses. Also stored on :attr:`results`.

        Raises
        ------
        ValueError
            If ``n_years`` is negative.
        """
        if n_years < 0:
            raise ValueError(f"n_years must be non-negative, got {n_years}")

        annual = np.zeros(n_years, dtype=np.float64)
        max_event = np.zeros(n_years, dtype=np.float64)
        rows: list[dict[str, object]] = []
        tracks: dict[tuple[int, str], pd.DataFrame] = {}

        catalog = self.catalog.generate(n_years=n_years)
        if not catalog.empty:
            groups = list(catalog.groupby(["year", "storm_id"], sort=True))
            iterator = groups
            if progress:
                from tqdm.auto import tqdm

                iterator = tqdm(groups, desc="Simulating storms", unit="storm")

            for (year, storm_id), storm in iterator:
                if not self._reaches_portfolio(storm):
                    continue
                processed = self._process(storm.reset_index(drop=True))
                hazard = TropicalCycloneHazard(processed)
                calculator = LossCalculator(hazard, self.vulnerability)
                calculator.compute(self.portfolio)
                loss = calculator.total_loss

                year = int(year)
                annual[year] += loss
                max_event[year] = max(max_event[year], loss)
                rows.append({"year": year, "storm_id": storm_id, "loss": loss})
                tracks[(year, str(storm_id))] = processed

        events = pd.DataFrame(rows, columns=["year", "storm_id", "loss"])
        self.results = SimulationResults(
            annual_losses=annual,
            max_event_losses=max_event,
            events=events,
            tracks=tracks,
            n_years=n_years,
        )
        logger.info("Simulated %d years, %d contributing events.", n_years, len(events))
        return self.results

    def __repr__(self) -> str:
        n = 0 if self.results is None else self.results.n_years
        return (
            f"LossSimulator(n_locations={len(self.portfolio)}, buffer_deg={self.buffer_deg}, "
            f"freq={self.freq!r}, years_run={n})"
        )
