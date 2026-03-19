""" 
Reinsurance module. This allows us to model different reinsurance structures (e.g. excess of loss, quota share) and compute losses for each layer.
"""

from .base import EventResults, PortfolioResults
from .ep import ExceedenceProbabilityCalculator

__all__ = ['EventResults', 'PortfolioResults', 'ExceedenceProbabilityCalculator']