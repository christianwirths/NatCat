"""Small containers for event- and year-level financial results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["EventResult", "YearResult"]


@dataclass
class EventResult:
    """Loss outcome of a single modelled event.

    Attributes
    ----------
    storm_id : str
        Identifier of the event.
    year : int
        Simulated year the event belongs to.
    gross_loss : float
        Ground-up loss before any reinsurance structure.
    ceded_loss : float
        Portion recovered from reinsurance.
    details : dict
        Free-form extras, e.g. the layer that responded.
    """

    storm_id: str
    year: int
    gross_loss: float
    ceded_loss: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def retained_loss(self) -> float:
        """Loss retained after cession."""
        return self.gross_loss - self.ceded_loss


@dataclass
class YearResult:
    """Aggregate outcome of one simulated year.

    Attributes
    ----------
    year : int
        Simulated year index.
    events : list of EventResult
        Contributing events, in no particular order.
    """

    year: int
    events: list[EventResult] = field(default_factory=list)

    @property
    def aggregate_loss(self) -> float:
        """Sum of the gross losses of all events in the year."""
        return float(sum(event.gross_loss for event in self.events))

    @property
    def max_event_loss(self) -> float:
        """Largest single-event gross loss, or ``0`` for an event-free year."""
        return float(max((event.gross_loss for event in self.events), default=0.0))
