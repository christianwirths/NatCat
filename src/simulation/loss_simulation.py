"""
Loss simulation module.
Runs the full synthetic TC loss pipeline across many simulated years. 
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from hazards.tropical_cyclone import TropicalCycloneHazard
from loss_model import LossCalculator
from utils.track import track_interpolation, cyclone_velocity, cyclone_bearing


class LossSimulator:
    """
    Runs the full synthetic TC loss simulation and stores results every year

    Usage:
        sim = LossSimulator(model, portfolio, vulnerability, buf=5.0)
        sim.run(n_years=1000)

        print(f"AAL: ${sim.aal:,.0f}")
        sim.get_year(42)          # losses + storm ids for year 42
        sim.get_storm_tracks(42)  # processed track DataFrames for year 42
    """

    def __init__(
        self,
        tc_model,
        portfolio: pd.DataFrame,
        vulnerability,
        portfolio_bounds: tuple = None,
        buf: float = 5.0,
        interp_time_step: str = '1h'
    ):
        """
        Input:
            tc_model: Fitted SyntheticTCCatalog instance.
            portfolio: DataFrame with latitude, longitude, tiv (and optionally construction) columns.
            vulnerability: WindVulnerability (or any VulnerabilityModel) instance.
            portfolio_bounds: (lat_min, lat_max, lon_min, lon_max) of the portfolio region.
                              If None, derived automatically from the portfolio itself.
            buf: Degrees buffer around portfolio bounds used for spatial pre-filtering.
            interp_time_step: Time step string passed to track_interpolation (e.g. '1h', '30min').
        """
        self.tc_model = tc_model
        self.portfolio = portfolio
        self.vulnerability = vulnerability
        self.buf = buf
        self.interp_time_step = interp_time_step

        # Derive bounding box from portfolio if not provided
        if portfolio_bounds is not None:
            lat_min, lat_max, lon_min, lon_max = portfolio_bounds
        else:
            lat_min = portfolio['latitude'].min()
            lat_max = portfolio['latitude'].max()
            lon_min = portfolio['longitude'].min()
            lon_max = portfolio['longitude'].max()

        self._lat_min = lat_min - buf
        self._lat_max = lat_max + buf
        self._lon_min = lon_min - buf
        self._lon_max = lon_max + buf

        # Results (populated by .run())
        self._results = None

    def run(self, n_years: int):
        """
        Run the loss simulation for n_years synthetic years.

        Input:
            n_years: Number of synthetic years to simulate.
        Returns:
            self (for chaining)
        """
        self._results = []

        for _ in tqdm(range(n_years), desc="Simulating years"):
            synthetic_tracks = self.tc_model.generate(n_years=1)

            year_losses = []
            year_storm_ids = []
            year_tracks = []

            for storm_id in synthetic_tracks['storm_id'].unique():
                storm_data = synthetic_tracks[synthetic_tracks['storm_id'] == storm_id].copy()

                # Skip storms outside of buffer zone 
                near_portfolio = (
                    storm_data['latitude'].between(self._lat_min, self._lat_max) &
                    storm_data['longitude'].between(self._lon_min, self._lon_max)
                ).any()
                if not near_portfolio:
                    continue

                # add timestamp for compatability 
                storm_data['timestamp'] = pd.Timestamp('1900-01-01') + pd.to_timedelta(
                    storm_data['hour'] * self.tc_model.time_step, unit='h'
                )

                # interpolate track for better spatial resolution in the wind field
                storm_data = track_interpolation(storm_data, time_step=self.interp_time_step)
                storm_data = cyclone_velocity(storm_data)
                storm_data = cyclone_bearing(storm_data)

                hazard = TropicalCycloneHazard(storm_data)
                loss = LossCalculator(hazard, self.vulnerability)
                loss.calculate_portfolio_loss(self.portfolio)

                year_losses.append(loss.total_loss)
                year_storm_ids.append(storm_id)
                year_tracks.append(storm_data)

            self._results.append({
                'losses': year_losses,
                'storm_ids': year_storm_ids,
                'storm_tracks': year_tracks
            })

        return self

    @property
    def losses_per_year(self) -> list:
        """Total loss per simulated year."""
        self._check_run()
        return [sum(yr['losses']) for yr in self._results]

    @property
    def aal(self) -> float:
        """Average Annual Loss (AAL) across all simulated years."""
        return float(np.mean(self.losses_per_year))

    def get_year(self, i: int) -> dict:
        """
        Returns loss and storm details for simulated year i.

        Input:
            i: Year index (0-based).
        Returns:
            dict with keys: losses, storm_ids, total_loss
        """
        self._check_run()
        yr = self._results[i]
        return {
            'total_loss': sum(yr['losses']),
            'losses': yr['losses'],
            'storm_ids': yr['storm_ids']
        }

    def get_storm_tracks(self, i: int) -> list:
        """
        Returns the processed (interpolated) track DataFrames for all storms 
        that hit the portfolio in simulated year i.

        Input:
            i: Year index (0-based).
        Returns:
            List of track DataFrames, one per contributing storm.
        """
        self._check_run()
        return self._results[i]['storm_tracks']

    def _check_run(self):
        if self._results is None:
            raise RuntimeError("Simulation not run yet. Call .run() first.")

    def __repr__(self):
        n = len(self._results) if self._results is not None else 0
        return (
            f"LossSimulator(buf={self.buf}, interp='{self.interp_time_step}', "
            f"portfolio={len(self.portfolio)} locations, years_run={n})"
        )
