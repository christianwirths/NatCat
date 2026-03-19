""" 
Base module containing datacalsses and base class for the reinsurance module.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class EventResults:
    """Data class to store results for a single simulated event."""
    gross_loss: float
    ceded_loss: float
    retained_loss: float
    recovery_ratio: float
    layer_exhausted: bool
    details: Dict[str, Any]  # Additional details about the event (e.g. storm ID, track, etc.)


@dataclass
class PortfolioResults:
    """Data class to store results for an entire portfolio across multiple events."""
    total_gross_loss: float
    total_ceded_loss: float
    total_retained_loss: float
    overall_recovery_ratio: float
    events: List[EventResults]  # List of results for each event in the simulation


