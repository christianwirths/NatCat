"""
Data preprocessing package for natural catastrophe modeling.

Organized by peril type:
- tropical_cyclone: TC track data, forecasts, exposure
- earthquake: (future) ground motion, fault data
- flood: (future) inundation, river gauge data
"""

from . import tropical_cyclone

__all__ = ['tropical_cyclone']
