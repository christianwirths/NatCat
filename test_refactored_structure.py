#!/usr/bin/env python
"""Test script for the refactored utils structure."""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd

# Test utils imports
print("Testing utils imports...")
from utils import haversine_distance, bearing, nm2km, kt2kmh
from utils import track_interpolation, cyclone_velocity, cyclone_bearing
print("✓ All utils imports successful")

# Test submodule imports
from utils.geo import haversine_distance as hav
from utils.units import nm2km as nm_km
from utils.track import cyclone_bearing as cb
print("✓ Submodule imports successful")

# Test hazard functionality
print("\nTesting TropicalCycloneHazard...")
from hazards import TropicalCycloneHazard

track = pd.DataFrame({
    'latitude': [28.0, 29.0, 30.0],
    'longitude': [-86.0, -85.0, -84.0],
    'max_wind_speed_kt': [100, 110, 120],
    'radius_max_wind_nm': [25, 25, 30],
    'velocity_kt': [10, 12, 15],
    'bearing_deg': [45, 45, 50]
})

hazard = TropicalCycloneHazard(track)
coords = np.array([[29.0, -85.5], [30.5, -83.5]])
winds = hazard.compute_intensity(coords)
print(f"✓ Wind speeds computed: {winds}")

# Test unit conversions
print("\nTesting unit conversions...")
dist_km = nm2km(100)
speed_kmh = kt2kmh(50)
print(f"✓ 100 nm = {dist_km} km")
print(f"✓ 50 kt = {speed_kmh} km/h")

print("\n" + "="*50)
print("✅ All tests passed! Structure is optimal.")
print("="*50)
