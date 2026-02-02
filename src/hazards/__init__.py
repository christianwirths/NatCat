"""
Hazard models for natural catastrophe risk assessment.
"""

from .base import HazardModel, Coordinates, IntensityValues
from .tropical_cyclone import TropicalCycloneHazard

__all__ = ["HazardModel", "Coordinates", "IntensityValues", "TropicalCycloneHazard"]