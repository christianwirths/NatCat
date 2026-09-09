"""Calibrate vulnerability parameters against observed storm losses.

Workflow::

    observed = normalise_losses(load_observed_losses(), reference_year=2018)
    cases = build_cases(observed)                      # footprint per storm, once
    result = Calibrator(cases, ValueDependentVulnerability()).fit(seed=0)
    print(result.summary())
"""

from .calibrator import CalibrationResult, Calibrator
from .cases import CONUS_BOUNDS, StormCase, build_cases, load_cases, save_cases, storm_region
from .observed import REQUIRED_COLUMNS, US_CPI, load_observed_losses, normalise_losses

__all__ = [
    "CONUS_BOUNDS",
    "REQUIRED_COLUMNS",
    "US_CPI",
    "CalibrationResult",
    "Calibrator",
    "StormCase",
    "build_cases",
    "load_cases",
    "load_observed_losses",
    "normalise_losses",
    "save_cases",
    "storm_region",
]
