# Code Structure Reorganization

## Summary

Successfully reorganized the NatCat codebase to eliminate duplication, create a clean modular structure, and colocate TC-specific physics with the TC hazard model.

## Final Structure

```
src/
├── hazards/                      # Hazard models (polymorphic interface)
│   ├── base.py                   # HazardModel ABC
│   └── tropical_cyclone.py       # TC hazard + wind field physics
│
├── utils/                        # Reusable utilities (domain-organized)
│   ├── geo.py                    # Geographic calculations
│   ├── units.py                  # Unit conversions
│   └── track.py                  # Track data processing
│
└── vulnerability/                # Damage models
    ├── base.py                   # VulnerabilityFunction ABC  
    └── wind_damage.py            # Wind damage curves
```

## Module Responsibilities

### `hazards/tropical_cyclone.py` - TC Hazard + Wind Physics
**Private wind field functions (underscore prefix for internal use):**
- `_rankine_vortex()` - Rankine vortex wind profile
- `_get_heuristic_rmw()` - Heuristic RMW estimates based on wind speed
- `_max_wind_speeds_at_locations()` - Calculate maximum winds over track

**Public class:**
- `TropicalCycloneHazard(track_data, vortex_model='rankine')` - Main TC hazard model

### `utils/geo.py` - Geographic Utilities
- `haversine_distance(lat1, lon1, lat2_array, lon2_array, *, unit='nm')` - Great circle distance
- `bearing(lat1, lon1, lat2, lon2)` - Calculate bearing between points

### `utils/units.py` - Unit Conversions
- `nm2km(nm)` - Nautical miles → kilometers
- `kt2kmh(kt)` - Knots → km/h

### `utils/track.py` - Track Data Processing
- `track_interpolation(df, time_step='5min', kind='linear')` - Interpolate to finer resolution
- `extract_past_trajectory(df, timestamp, tech='OFCL')` - Extract past trajectory from A-deck
- `extract_future_trajectory(df, timestamp, tech='OFCL')` - Extract forecast from A-deck
- `cyclone_velocity(df)` - Calculate storm translation speed
- `cyclone_bearing(df)` - Calculate storm heading
- `prepare_track_data(df, timestamp, model, avail=1, past=True)` - Full preparation pipeline

### `hazards/base.py` - Hazard Interface
- `HazardModel` ABC with `compute_intensity(coordinates)` and `peril_type` property

## Import Examples

```python
# Recommended: Top-level imports
from utils import haversine_distance, bearing, nm2km, kt2kmh
from utils import track_interpolation, cyclone_velocity, prepare_track_data
from hazards import TropicalCycloneHazard

# Alternative: Submodule imports
from utils.geo import haversine_distance, bearing
from utils.units import nm2km, kt2kmh
from utils.track import cyclone_velocity, cyclone_bearing

# Hazard models
from hazards.base import HazardModel
from hazards import TropicalCycloneHazard

# Wind field functions (internal, not recommended for external use)
from hazards.tropical_cyclone import _rankine_vortex, _get_heuristic_rmw
```

## Key Design Decisions

1. **TC-specific physics colocated** - Wind field models moved into `tropical_cyclone.py` (private functions)
2. **No code duplication** - Single source of truth for each function
3. **Separation of concerns** - Utils for reusable functions, hazards for peril-specific models
4. **Clean imports** - Both top-level and submodule imports supported via `__init__.py`
5. **Type hints** - Modern Python with proper type annotations throughout utils
6. **Better naming** - `bearing` instead of `beraring`, explicit `unit` parameter
7. **Polymorphic design** - `HazardModel` ABC supports multiple peril types

## Migration from Old Structure

**Removed files:**
- `wind_fields.py` → moved into `hazards/tropical_cyclone.py` as private functions
- `utils.py` → reorganized into `utils/` package (backed up as `utils_old.py.bak`)

**Import changes:**
- `from wind_fields import max_wind_speeds_at_locations` → now `_max_wind_speeds_at_locations` (private)
- `from utils import haversine_vectorized` → `from utils import haversine_distance` (with `unit=` parameter)
- `from utils import beraring` → `from utils import bearing`

## Rationale: Why Wind Physics in tropical_cyclone.py?

Wind field models (Rankine vortex, RMW heuristics) are:
- **TC-specific** - Not reusable for other perils (earthquakes, floods, etc.)
- **Physics models** - Not general utilities
- **Tightly coupled** - Only used by `TropicalCycloneHazard`

By colocating them as private functions, we:
- Keep TC code self-contained
- Signal they're internal implementation details (underscore prefix)
- Reduce cognitive overhead (one file for all TC logic)
- Follow "high cohesion, low coupling" principle

## Testing

Run test files to verify structure:
```bash
python test_refactored_structure.py  # Original tests
python test_final_structure.py       # After wind_fields.py removal
```
