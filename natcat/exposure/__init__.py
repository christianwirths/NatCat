"""Exposure portfolios: synthetic test data and CLIMADA LitPop loading."""

from .litpop import climada_to_portfolio, construction_type_heuristic, load_litpop_exposure
from .synthetic import synthetic_portfolio

__all__ = [
    "climada_to_portfolio",
    "construction_type_heuristic",
    "load_litpop_exposure",
    "synthetic_portfolio",
]
