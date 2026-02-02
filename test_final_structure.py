#!/usr/bin/env python
"""Test final structure after moving wind_fields into tropical_cyclone."""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np

# Test wind field functions are in tropical_cyclone
from hazards.tropical_cyclone import _rankine_vortex, _get_heuristic_rmw
print("✓ Wind field functions in tropical_cyclone module")

# Test main interface
from hazards import TropicalCycloneHazard

track = pd.DataFrame({
    'latitude': [28.0, 29.0],
    'longitude': [-86.0, -85.0],
    'max_wind_speed_kt': [100, 110],
    'radius_max_wind_nm': [25, 25],
    'velocity_kt': [10, 12],
    'bearing_deg': [45, 50]
})

hazard = TropicalCycloneHazard(track)
coords = np.array([[29.0, -85.5]])
winds = hazard.compute_intensity(coords)
print(f"✓ TropicalCycloneHazard: wind = {winds[0]:.1f} kt")

# Test utils.track (imports _get_heuristic_rmw internally)
from utils.track import prepare_track_data
print("✓ utils.track imports work (no circular dependency)")

print("\n✅ Final structure is optimal!")
print("\nStructure:")
print("  hazards/")
print("    ├── base.py           (HazardModel ABC)")
print("    └── tropical_cyclone.py  (TC hazard + wind physics)")
print("  utils/")
print("    ├── geo.py    (geographic calculations)")
print("    ├── units.py  (unit conversions)")
print("    └── track.py  (track processing)")
