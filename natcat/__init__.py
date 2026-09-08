"""natcat: a transparent, end-to-end tropical cyclone catastrophe model.

The package covers the full modelling chain -- ATCF best-track ingestion,
parametric wind hazard, vulnerability curves, exposure portfolios, ground-up
loss, a stochastic event set, and the financial risk metrics built on top of it.

Examples
--------
>>> import natcat
>>> track = natcat.load_best_track(2018, "al", "14")             # doctest: +SKIP
>>> hazard = natcat.TropicalCycloneHazard(track)                 # doctest: +SKIP
>>> portfolio = natcat.synthetic_portfolio(1000, seed=0)         # doctest: +SKIP
>>> calculator = natcat.LossCalculator(hazard, natcat.WindVulnerability())  # doctest: +SKIP
>>> calculator.compute(portfolio)["loss"].sum()                  # doctest: +SKIP
"""

from __future__ import annotations

from .exposure import load_litpop_exposure, synthetic_portfolio
from .financial import EPCurve, ExceedanceProbability
from .hazards import TropicalCycloneHazard
from .loss import LossCalculator, LossSimulator, SimulationResults
from .stochastic import SyntheticTCCatalog
from .tracks import load_best_track
from .vulnerability import WindVulnerability

__version__ = "0.1.0"

__all__ = [
    "EPCurve",
    "ExceedanceProbability",
    "LossCalculator",
    "LossSimulator",
    "SimulationResults",
    "SyntheticTCCatalog",
    "TropicalCycloneHazard",
    "WindVulnerability",
    "__version__",
    "load_best_track",
    "load_litpop_exposure",
    "synthetic_portfolio",
]
