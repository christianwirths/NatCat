#!/usr/bin/env python
"""Regenerate every natcat documentation figure and animation.

Usage
-----
    conda run -n NatCat python scripts/make_figures.py [--years N] [--quick]
        [--skip-litpop] [--only NAME[,NAME...]]

Each figure is produced by its own function registered in ``FIGURES``.
Expensive objects (the fitted stochastic catalog, the loss simulation, the
portfolio) are computed once and cached in memory across figures via
``FigureCache``. A failure in one figure is logged with its traceback and
does not stop the others; the script exits non-zero if any figure failed.

Notes
-----
This script depends on the ``natcat`` core package (``natcat.tracks``,
``natcat.hazards``, ``natcat.exposure``, ``natcat.loss``, ``natcat.stochastic``,
``natcat.financial``) as specified in the v1 refactor spec. Until that core
lands, running this script end to end is expected to fail; it can still be
imported and its CLI/figure registry exercised.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "docs" / "assets" / "figures"
ANIMATIONS_DIR = REPO_ROOT / "docs" / "assets" / "animations"
DATA_DIR = REPO_ROOT / "docs" / "assets" / "data"

logger = logging.getLogger("make_figures")

# Hurricane Michael (2018), AL142018.
MICHAEL_YEAR = 2018
MICHAEL_BASIN = "al"
MICHAEL_STORM_NUMBER = "14"
MICHAEL_TRACK_EXTENT = (-92, -72, 16, 38)
MICHAEL_FOOTPRINT_EXTENT = (-89, -79, 24.3, 33.2)

# LitPop USA bounds: (lat_min, lat_max, lon_min, lon_max).
LITPOP_USA_BOUNDS = (24.5, 32.0, -87.6, -80.0)
SYNTHETIC_PORTFOLIO_N = 3000

CATALOG_EXTENT = (-100, -10, 0, 50)
RETURN_PERIODS = (10, 50, 100, 250)

FIGURES: dict[str, Callable[[FigureCache], None]] = {}


def figure(name: str) -> Callable[[Callable[[FigureCache], None]], Callable[[FigureCache], None]]:
    """Register a function under ``name`` in the module-level figure registry."""

    def decorator(fn: Callable[[FigureCache], None]) -> Callable[[FigureCache], None]:
        FIGURES[name] = fn
        return fn

    return decorator


class FigureCache:
    """Lazily computes and caches objects shared across several figures.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (``years``, ``skip_litpop``, ...).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._michael_track = None
        self._portfolio = None
        self._catalog_model = None
        self._catalog = None
        self._simulation_results = None
        self._ep = None

    @property
    def michael_track(self) -> pd.DataFrame:
        """Processed best track for Hurricane Michael (2018)."""
        if self._michael_track is None:
            from natcat.tracks import load_best_track

            self._michael_track = load_best_track(MICHAEL_YEAR, MICHAEL_BASIN, MICHAEL_STORM_NUMBER)
        return self._michael_track

    @property
    def portfolio(self) -> pd.DataFrame:
        """Exposure portfolio: LitPop USA (default) or synthetic (``--skip-litpop``)."""
        if self._portfolio is None:
            if self.args.skip_litpop:
                from natcat.exposure import synthetic_portfolio

                lat_min, lat_max, lon_min, lon_max = LITPOP_USA_BOUNDS
                self._portfolio = synthetic_portfolio(
                    SYNTHETIC_PORTFOLIO_N,
                    bounds=(lat_min, lat_max, lon_min, lon_max),
                    seed=42,
                )
            else:
                from natcat.exposure import load_litpop_exposure

                self._portfolio = load_litpop_exposure("USA", bounds=LITPOP_USA_BOUNDS)
        return self._portfolio

    @property
    def catalog_model(self):
        """Fitted ``SyntheticTCCatalog`` (genesis + transitions + frequency)."""
        if self._catalog_model is None:
            from natcat.config import get_raw_data_dir
            from natcat.stochastic import SyntheticTCCatalog

            model = SyntheticTCCatalog(seed=42)
            model.fit(get_raw_data_dir())
            self._catalog_model = model
        return self._catalog_model

    @property
    def catalog(self) -> pd.DataFrame:
        """Generated synthetic catalog (long-format DataFrame) for ``self.args.years``."""
        if self._catalog is None:
            self._catalog = self.catalog_model.generate(n_years=self.args.years)
        return self._catalog

    @property
    def simulation_results(self):
        """Cached ``LossSimulator`` run over the requested number of years."""
        if self._simulation_results is None:
            import pickle

            from natcat.loss import LossSimulator
            from natcat.vulnerability import WindVulnerability

            cache = Path(self.args.cache) if self.args.cache else None
            if cache is not None and cache.exists():
                with cache.open("rb") as fh:
                    cached = pickle.load(fh)
                if cached.n_years == self.args.years:
                    logger.info("Loaded cached simulation results from %s", cache)
                    self._simulation_results = cached
                    return cached
                logger.info(
                    "Cache at %s has %d years, expected %d; re-running.",
                    cache,
                    cached.n_years,
                    self.args.years,
                )

            simulator = LossSimulator(
                self.catalog_model,
                self.portfolio,
                WindVulnerability(),
                seed=42,
            )
            self._simulation_results = simulator.run(self.args.years, progress=True)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                with cache.open("wb") as fh:
                    pickle.dump(self._simulation_results, fh)
                logger.info("Cached simulation results to %s", cache)
        return self._simulation_results

    @property
    def ep(self):
        """Cached ``ExceedanceProbability`` fitted on the simulation results."""
        if self._ep is None:
            from natcat.financial import ExceedanceProbability

            self._ep = ExceedanceProbability.from_simulation(self.simulation_results)
        return self._ep


@figure("michael_track")
def fig_michael_track(ctx: FigureCache) -> None:
    from natcat.plotting import add_source_note, plot_track, save

    fig, _ = plot_track(
        ctx.michael_track,
        extent=MICHAEL_TRACK_EXTENT,
        title="Hurricane Michael (2018) — best track",
    )
    add_source_note(fig, "Source: NHC ATCF best track (AL142018)")
    save(fig, FIGURES_DIR / "michael_track.png")


@figure("michael_footprint")
def fig_michael_footprint(ctx: FigureCache) -> None:
    from natcat.hazards import TropicalCycloneHazard
    from natcat.loss import LossCalculator
    from natcat.plotting import add_source_note, plot_footprint, save
    from natcat.vulnerability import WindVulnerability

    hazard = TropicalCycloneHazard(ctx.michael_track)
    calculator = LossCalculator(hazard, WindVulnerability())
    results = calculator.compute(ctx.portfolio)

    fig, _ = plot_footprint(
        results,
        extent=MICHAEL_FOOTPRINT_EXTENT,
        value="damage_ratio",
        portfolio=ctx.portfolio,
        track=ctx.michael_track,
        title="Hurricane Michael (2018) — modelled damage ratio",
    )
    exposure_source = "Synthetic exposure" if ctx.args.skip_litpop else "LitPop exposure (CLIMADA)"
    add_source_note(fig, f"Source: NHC ATCF best track; {exposure_source}")
    save(fig, FIGURES_DIR / "michael_footprint.png")


@figure("michael_animation")
def fig_michael_animation(ctx: FigureCache) -> None:
    from natcat.hazards import TropicalCycloneHazard
    from natcat.loss import LossCalculator
    from natcat.plotting import animate_footprint
    from natcat.vulnerability import WindVulnerability

    track = ctx.michael_track
    start = pd.Timestamp("2018-10-07")
    end = pd.to_datetime(track["time"]).max()
    times = pd.date_range(start, end, freq="6h")

    hazard = TropicalCycloneHazard(track)
    calculator = LossCalculator(hazard, WindVulnerability())
    history = calculator.compute_history(ctx.portfolio, times)

    animate_footprint(
        history,
        track=track,
        portfolio=ctx.portfolio,
        extent=MICHAEL_FOOTPRINT_EXTENT,
        value="damage_ratio",
        step="6h",
        fps=2,
        path=ANIMATIONS_DIR / "michael_damage_evolution.gif",
        title="Hurricane Michael (2018) — damage evolution",
    )


@figure("wind_profile")
def fig_wind_profile(ctx: FigureCache) -> None:
    from natcat.plotting import plot_wind_profile, save

    fig, _ = plot_wind_profile()
    save(fig, FIGURES_DIR / "wind_profile.png")


@figure("vulnerability_curves")
def fig_vulnerability_curves(ctx: FigureCache) -> None:
    from natcat.plotting import plot_vulnerability, save

    fig, _ = plot_vulnerability()
    save(fig, FIGURES_DIR / "vulnerability_curves.png")


@figure("rmw_heuristic")
def fig_rmw_heuristic(ctx: FigureCache) -> None:
    from natcat.plotting import plot_rmw_heuristic, save

    fig, _ = plot_rmw_heuristic()
    save(fig, FIGURES_DIR / "rmw_heuristic.png")


@figure("genesis_points")
def fig_genesis_points(ctx: FigureCache) -> None:
    # ASSUMPTION (verify once natcat.stochastic lands): GenesisModel exposes a
    # `.sample(n, rng=None) -> pd.DataFrame` method returning latitude/longitude
    # columns, and a `.kde` attribute holding the fitted scipy.stats.gaussian_kde
    # (fit on stacked [latitude, longitude], matching plot_genesis's expectation).
    from natcat.plotting import plot_genesis, save
    from natcat.stochastic.genesis import extract_genesis_points

    model = ctx.catalog_model
    historical = extract_genesis_points(model.tracks)
    rng = np.random.default_rng(42)
    synthetic = model.genesis.sample(300, rng=rng)
    fig, _ = plot_genesis(
        historical,
        synthetic,
        kde=getattr(model.genesis, "kde", None),
        title="Storm genesis — historical record vs. KDE samples",
    )
    save(fig, FIGURES_DIR / "genesis_points.png")


@figure("synthetic_tracks")
def fig_synthetic_tracks(ctx: FigureCache) -> None:
    from natcat.plotting import plot_catalog_comparison, save

    historical = ctx.catalog_model.tracks
    rng = np.random.default_rng(42)
    sample_size = min(150, len(historical))
    indices = rng.choice(len(historical), size=sample_size, replace=False)
    historical_sample = [historical[i] for i in indices]

    fig, _ = plot_catalog_comparison(
        historical_sample,
        ctx.catalog,
        extent=CATALOG_EXTENT,
        max_storms=150,
        title="Tropical cyclone tracks — historical sample vs. synthetic event set",
    )
    save(fig, FIGURES_DIR / "synthetic_tracks.png")


@figure("annual_loss_distribution")
def fig_annual_loss_distribution(ctx: FigureCache) -> None:
    from natcat.plotting import plot_annual_loss_distribution, save

    fig, _ = plot_annual_loss_distribution(ctx.simulation_results.annual_losses)
    save(fig, FIGURES_DIR / "annual_loss_distribution.png")


@figure("ep_curves")
def fig_ep_curves(ctx: FigureCache) -> None:
    from natcat.plotting import plot_ep_curve, save

    fig, _ = plot_ep_curve(ctx.ep, highlight_rps=RETURN_PERIODS)
    save(fig, FIGURES_DIR / "ep_curves.png")


def write_simulation_summary(ctx: FigureCache) -> None:
    """Write AAL and return-period losses to docs/assets/data/simulation_summary.json."""
    results = ctx.simulation_results
    ep = ctx.ep

    def _loss_at(rp: int, kind: str) -> float:
        return float(np.atleast_1d(ep.loss_at_return_period(rp, kind=kind))[0])

    summary = {
        "n_years": int(results.n_years),
        "n_events": len(results.events),
        "aal": float(results.aal),
        "return_periods": {
            str(rp): {"aep": _loss_at(rp, "aep"), "oep": _loss_at(rp, "oep")}
            for rp in RETURN_PERIODS
        },
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "simulation_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for make_figures.py."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--years", type=int, default=1000, help="Simulation years (default: 1000)")
    parser.add_argument(
        "--cache",
        default=None,
        help="Pickle path for the simulation results; reused on later runs with the same --years",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke run: caps --years at 200 for quick iteration",
    )
    parser.add_argument(
        "--skip-litpop",
        action="store_true",
        help="Use natcat.exposure.synthetic_portfolio instead of LitPop",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated figure names to generate. Known: {', '.join(sorted(FIGURES))}",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.years = min(args.years, 200)
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point: generate the requested figures, isolating failures."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ANIMATIONS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.only is None:
        names = list(FIGURES)
    else:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = sorted(set(names) - set(FIGURES))
    if unknown:
        logger.error("Unknown figure name(s): %s. Known: %s", unknown, sorted(FIGURES))
        return 1

    ctx = FigureCache(args)
    failures: list[str] = []

    for name in names:
        logger.info("Generating %s ...", name)
        start = time.perf_counter()
        try:
            FIGURES[name](ctx)
        except Exception:  # noqa: BLE001 - isolate failures so other figures still run
            logger.error("Figure %r failed:\n%s", name, traceback.format_exc())
            failures.append(name)
        else:
            logger.info("  %s done in %.1fs", name, time.perf_counter() - start)

    if {"ep_curves", "annual_loss_distribution"} & set(names):
        logger.info("Writing simulation_summary.json ...")
        try:
            write_simulation_summary(ctx)
        except Exception:  # noqa: BLE001 - isolate failures so other figures still run
            logger.error("Writing simulation_summary.json failed:\n%s", traceback.format_exc())
            failures.append("simulation_summary.json")

    if failures:
        logger.error("Failed: %s", failures)
        return 1
    logger.info("All figures generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
